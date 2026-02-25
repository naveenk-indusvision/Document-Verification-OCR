"""
Full document processing pipeline:
  1. OpenCV preprocessing (deskew, enhance, denoise)
  2. PaddleOCR layout analysis (detect tables vs text regions)
  3. img2table for dedicated table extraction
  4. pdfplumber for digital PDF table extraction
  5. Sarvam Vision for Indian document text (primary)
  6. OpenAI gpt-4o as fallback

All libraries used are Apache 2.0 or MIT — fully free for commercial use.
"""
import os
import io
import re
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Orchestrates the full extraction pipeline:
    preprocess → layout analysis → table extraction → text OCR → merge.
    """

    # Quality threshold — if Sarvam quality is below this, fall back to OpenAI
    FALLBACK_THRESHOLD = 0.50

    def __init__(self, sarvam_api_key: str = "", openai_api_key: str = "",
                 use_preprocessing: bool = True, use_layout_analysis: bool = True,
                 use_table_extraction: bool = True):
        self.sarvam_api_key = sarvam_api_key
        self.openai_api_key = openai_api_key
        self.use_preprocessing = use_preprocessing
        self.use_layout_analysis = use_layout_analysis
        self.use_table_extraction = use_table_extraction

        # Lazy-loaded components
        self._preprocessor = None
        self._paddle_ocr = None
        self._sarvam_extractor = None
        self._openai_extractor = None

    # ------------------------------------------------------------------
    # Lazy loaders (avoid heavy imports at module load time)
    # ------------------------------------------------------------------

    def _get_preprocessor(self):
        if self._preprocessor is None:
            from image_preprocessor import ImagePreprocessor
            self._preprocessor = ImagePreprocessor()
        return self._preprocessor

    def _get_paddle_ocr(self):
        if self._paddle_ocr is None:
            try:
                from paddleocr import PaddleOCR
                # PaddleOCR v3.x API
                self._paddle_ocr = PaddleOCR(
                    use_textline_orientation=True,
                    lang="en",
                )
            except (ImportError, Exception) as e:
                logger.warning(f"PaddleOCR init failed: {e}. Layout analysis disabled.")
                self._paddle_ocr = "unavailable"  # Sentinel to avoid retrying
        return self._paddle_ocr if self._paddle_ocr != "unavailable" else None

    def _get_sarvam(self):
        if self._sarvam_extractor is None and self.sarvam_api_key:
            from sarvam_extractor import SarvamTextExtractor
            self._sarvam_extractor = SarvamTextExtractor(api_key=self.sarvam_api_key)
        return self._sarvam_extractor

    def _get_openai(self):
        if self._openai_extractor is None and self.openai_api_key:
            from openai_text_extractor import OpenAITextExtractor
            self._openai_extractor = OpenAITextExtractor(api_key=self.openai_api_key)
        return self._openai_extractor

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_document(self, uploaded_file) -> Dict:
        """
        Full pipeline: preprocess → extract tables → OCR text → merge.

        Returns dict with:
            status, content, tables_html, extraction_method, quality_score,
            pages_processed, sarvam_blocks, low_confidence_blocks,
            used_fallback, pipeline_steps
        """
        if uploaded_file is None:
            return self._error("No file provided")

        file_name = uploaded_file.name.lower()
        is_pdf = file_name.endswith(".pdf")
        is_image = any(file_name.endswith(ext) for ext in [".jpg", ".jpeg", ".png"])

        if not (is_pdf or is_image):
            return self._error("Unsupported file type. Supported: PDF, JPG, JPEG, PNG")

        pipeline_steps = []
        tables_html = []

        try:
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()

            # ==============================================================
            # STEP 1: Convert to images (for preprocessing + layout analysis)
            # ==============================================================
            images = self._to_images(file_bytes, file_name)
            pipeline_steps.append("file_loaded")

            # ==============================================================
            # STEP 2: OpenCV Preprocessing (deskew, enhance, denoise)
            # ==============================================================
            preprocessed_images = []
            if self.use_preprocessing and images:
                preprocessor = self._get_preprocessor()
                for img in images:
                    try:
                        enhanced = preprocessor.preprocess_pil_image(img)
                        preprocessed_images.append(enhanced)
                    except Exception as e:
                        logger.warning(f"Preprocessing failed for a page: {e}")
                        preprocessed_images.append(img)  # Use original on failure
                pipeline_steps.append("opencv_preprocessed")
            else:
                preprocessed_images = images if images else []

            # ==============================================================
            # STEP 3: Table extraction with img2table
            # ==============================================================
            if self.use_table_extraction:
                # For digital PDFs — try pdfplumber first (more accurate)
                if is_pdf:
                    pdf_tables = self._extract_tables_pdfplumber(file_bytes)
                    if pdf_tables:
                        tables_html.extend(pdf_tables)
                        pipeline_steps.append("pdfplumber_tables")

                # For scanned docs / images — use img2table
                if not tables_html and preprocessed_images:
                    img_tables = self._extract_tables_img2table(preprocessed_images)
                    if img_tables:
                        tables_html.extend(img_tables)
                        pipeline_steps.append("img2table_tables")

            # ==============================================================
            # STEP 4: Layout analysis with PaddleOCR (optional)
            # ==============================================================
            paddle_text = ""
            if self.use_layout_analysis and preprocessed_images:
                paddle_result = self._paddle_ocr_extract(preprocessed_images)
                if paddle_result:
                    paddle_text = paddle_result
                    pipeline_steps.append("paddleocr_layout")

            # ==============================================================
            # STEP 5: Primary OCR with Sarvam (on preprocessed file)
            # ==============================================================
            sarvam_result = None
            if self.sarvam_api_key:
                sarvam_result = self._extract_with_sarvam(
                    file_bytes, file_name, preprocessed_images
                )
                if sarvam_result and sarvam_result.get("status") == "success":
                    pipeline_steps.append("sarvam_ocr")

            # ==============================================================
            # STEP 6: OpenAI fallback if needed
            # ==============================================================
            openai_result = None
            used_fallback = False

            sarvam_failed = (
                sarvam_result is None
                or sarvam_result.get("status") != "success"
            )
            sarvam_low_quality = (
                sarvam_result is not None
                and sarvam_result.get("quality_score") is not None
                and sarvam_result["quality_score"] < self.FALLBACK_THRESHOLD
            )

            if (sarvam_failed or sarvam_low_quality) and self.openai_api_key:
                reason = "low accuracy" if sarvam_low_quality else "extraction failed"
                openai_result = self._extract_with_openai(preprocessed_images)
                if openai_result and openai_result.get("status") == "success":
                    used_fallback = True
                    pipeline_steps.append(f"openai_fallback ({reason})")

            # ==============================================================
            # STEP 7: Merge all results
            # ==============================================================
            return self._merge_results(
                sarvam_result=sarvam_result,
                openai_result=openai_result,
                paddle_text=paddle_text,
                tables_html=tables_html,
                used_fallback=used_fallback,
                pipeline_steps=pipeline_steps,
            )

        except Exception as e:
            logger.exception("Document processing failed")
            return self._error(f"Processing error: {str(e)}")

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _to_images(self, file_bytes: bytes, file_name: str) -> List[Image.Image]:
        """Convert file to a list of PIL Images (one per page)."""
        if file_name.endswith(".pdf"):
            return self._pdf_to_images(file_bytes)
        else:
            img = Image.open(io.BytesIO(file_bytes))
            if img.mode == "RGBA":
                img = img.convert("RGB")
            return [img]

    def _pdf_to_images(self, pdf_bytes: bytes) -> List[Image.Image]:
        """Convert PDF pages to images using pdf2image."""
        try:
            from pdf2image import convert_from_bytes
            poppler_paths = [
                None,
                r"C:\Program Files\poppler-25.07.0\Library\bin",
                r"C:\poppler\bin",
                r"C:\poppler-25.07.0\Library\bin",
                r"C:\Program Files\poppler\bin",
            ]
            for pp in poppler_paths:
                try:
                    kwargs = {"pdf_file": pdf_bytes, "dpi": 300, "first_page": 1, "last_page": 10}
                    if pp:
                        kwargs["poppler_path"] = pp
                    return convert_from_bytes(**kwargs)
                except Exception:
                    continue

            # Fallback: PyMuPDF
            return self._pdf_to_images_pymupdf(pdf_bytes)
        except ImportError:
            return self._pdf_to_images_pymupdf(pdf_bytes)

    def _pdf_to_images_pymupdf(self, pdf_bytes: bytes) -> List[Image.Image]:
        """Fallback PDF → images using PyMuPDF."""
        try:
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            images = []
            for page_num in range(min(len(doc), 10)):
                page = doc[page_num]
                mat = fitz.Matrix(300 / 72, 300 / 72)  # 300 DPI
                pix = page.get_pixmap(matrix=mat)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                images.append(img)
            doc.close()
            return images
        except Exception as e:
            logger.warning(f"PyMuPDF fallback failed: {e}")
            return []

    def _extract_tables_pdfplumber(self, pdf_bytes: bytes) -> List[str]:
        """Extract tables from digital PDFs using pdfplumber. Returns HTML strings."""
        try:
            import pdfplumber
            tables_html = []
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page_num, page in enumerate(pdf.pages[:10]):
                    extracted_tables = page.extract_tables()
                    for table in extracted_tables:
                        if not table or len(table) < 2:
                            continue
                        html = self._table_to_html(table, page_num + 1)
                        if html:
                            tables_html.append(html)
            return tables_html
        except Exception as e:
            logger.warning(f"pdfplumber table extraction failed: {e}")
            return []

    def _extract_tables_img2table(self, images: List[Image.Image]) -> List[str]:
        """Extract tables from images using img2table. Returns HTML strings."""
        try:
            from img2table.document import Image as Img2TableImage
            from img2table.ocr import TesseractOCR
            import tempfile

            tables_html = []

            # Try Tesseract OCR backend (widely available)
            try:
                ocr = TesseractOCR(lang="eng")
            except Exception:
                ocr = None  # img2table can detect table structure without OCR

            for page_idx, pil_img in enumerate(images):
                # img2table needs a file path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    pil_img.save(tmp.name, format="PNG")
                    tmp_path = tmp.name

                try:
                    img_doc = Img2TableImage(src=tmp_path)
                    extracted = img_doc.extract_tables(ocr=ocr)

                    for table in extracted:
                        if table and hasattr(table, 'df') and table.df is not None:
                            df = table.df
                            if len(df) > 0:
                                html = df.to_html(index=False, border=1, classes="extracted-table")
                                tables_html.append(
                                    f"<!-- Table from page {page_idx + 1} -->\n{html}"
                                )
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

            return tables_html
        except ImportError as e:
            logger.warning(f"img2table not available: {e}")
            return []
        except Exception as e:
            logger.warning(f"img2table extraction failed: {e}")
            return []

    def _paddle_ocr_extract(self, images: List[Image.Image]) -> str:
        """Use PaddleOCR for layout-aware text extraction."""
        paddle = self._get_paddle_ocr()
        if paddle is None:
            return ""

        all_text_parts = []
        for page_idx, pil_img in enumerate(images):
            try:
                img_array = np.array(pil_img)

                # PaddleOCR v3 uses .predict(), older versions use .ocr()
                if hasattr(paddle, 'predict'):
                    result = paddle.predict(img_array)
                else:
                    result = paddle.ocr(img_array, cls=True)

                if not result:
                    continue

                page_lines = []

                # Handle different PaddleOCR output formats
                # v3: result is a list of dicts or a structured result object
                # v2: result is [[bbox, (text, conf)], ...]
                page_data = result
                if isinstance(result, list) and result:
                    # Could be a list of pages or a list of line results
                    if isinstance(result[0], list):
                        page_data = result[0]  # First page
                    else:
                        page_data = result

                if not page_data:
                    continue

                for line in page_data:
                    try:
                        if isinstance(line, dict):
                            # PaddleOCR v3 dict format: {"text": ..., "score": ..., "rec_box": ...}
                            text = str(line.get("text", line.get("rec_text", "")))
                            confidence = float(line.get("score", line.get("rec_score", 0)))
                        elif isinstance(line, (list, tuple)) and len(line) >= 2:
                            # v2 format: [bbox, (text, confidence)]
                            text_info = line[1]
                            if isinstance(text_info, (list, tuple)):
                                text = str(text_info[0])
                                confidence = float(text_info[1]) if len(text_info) > 1 else 0
                            elif isinstance(text_info, dict):
                                text = str(text_info.get("text", ""))
                                confidence = float(text_info.get("score", 0))
                            else:
                                text = str(text_info)
                                confidence = 0
                        else:
                            continue
                        if confidence > 0.5 and text.strip():
                            page_lines.append(text.strip())
                    except (TypeError, ValueError, IndexError):
                        continue

                if page_lines:
                    if page_idx > 0:
                        all_text_parts.append(f"\n--- Page {page_idx + 1} ---\n")
                    all_text_parts.append("\n".join(page_lines))

            except Exception as e:
                logger.warning(f"PaddleOCR failed on page {page_idx + 1}: {e}")
                continue

        return "\n\n".join(all_text_parts)

    def _extract_with_sarvam(self, file_bytes: bytes, file_name: str,
                              preprocessed_images: List[Image.Image]) -> Optional[Dict]:
        """Run Sarvam extraction — always use ORIGINAL file bytes.

        Sarvam has its own internal OCR pipeline that works best with original
        documents. Preprocessing (deskew/enhance) is used for OpenAI/PaddleOCR
        fallback only.
        """
        sarvam = self._get_sarvam()
        if not sarvam:
            return None

        try:
            uploaded_file = _BytesUploadedFile(file_bytes, file_name)
            return sarvam.extract_text_from_uploaded_file(uploaded_file)
        except Exception as e:
            logger.warning(f"Sarvam extraction failed: {e}")
            return None

    # Phrases that indicate OpenAI refused to process the document
    _REFUSAL_PHRASES = [
        "i can't assist",
        "i cannot assist",
        "i'm unable to",
        "i am unable to",
        "i can't help",
        "i cannot help",
        "sorry, i can't",
        "sorry, i cannot",
        "i'm not able to",
    ]

    def _is_refusal(self, text: str) -> bool:
        """Check if OpenAI returned a refusal instead of actual content."""
        lower = text.strip().lower()
        return any(phrase in lower for phrase in self._REFUSAL_PHRASES)

    def _extract_with_openai(self, images: List[Image.Image]) -> Optional[Dict]:
        """Run OpenAI Vision extraction on preprocessed images."""
        openai_ext = self._get_openai()
        if not openai_ext:
            return None

        try:
            all_text = []
            total_tokens = 0

            for page_idx, img in enumerate(images):
                result = openai_ext.extract_text_from_image(img)
                if result.get("status") == "success":
                    text = result.get("text", "")
                    # Skip refusal responses
                    if self._is_refusal(text):
                        logger.warning(f"OpenAI refused page {page_idx + 1}, skipping")
                        continue
                    if page_idx > 0:
                        all_text.append(f"\n--- Page {page_idx + 1} ---\n")
                    all_text.append(text)
                    total_tokens += result.get("tokens_used", 0)

            if all_text:
                combined = "\n\n".join(all_text)
                # Final check — if the entire output is a refusal, return None
                if self._is_refusal(combined):
                    logger.warning("OpenAI refused entire document")
                    return None
                return {
                    "status": "success",
                    "content": combined,
                    "extraction_method": "OpenAI Vision (preprocessed)",
                    "pages_processed": len(images),
                    "total_tokens": total_tokens,
                    "sarvam_blocks": None,
                    "quality_score": None,
                    "low_confidence_blocks": None,
                }
            return None
        except Exception as e:
            logger.warning(f"OpenAI extraction failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Result merging
    # ------------------------------------------------------------------

    def _merge_results(self, sarvam_result: Optional[Dict],
                       openai_result: Optional[Dict],
                       paddle_text: str,
                       tables_html: List[str],
                       used_fallback: bool,
                       pipeline_steps: List[str]) -> Dict:
        """Merge outputs from all pipeline stages into a single result."""

        # Pick the primary text source
        if used_fallback and openai_result and openai_result.get("status") == "success":
            primary = openai_result
            method = "OpenAI Vision (fallback, preprocessed)"
        elif sarvam_result and sarvam_result.get("status") == "success":
            primary = sarvam_result
            method = "Sarvam Document Intelligence (preprocessed)"
        elif paddle_text.strip():
            # PaddleOCR as last resort
            primary = {
                "status": "success",
                "content": paddle_text,
                "extraction_method": "PaddleOCR",
                "pages_processed": 1,
                "total_tokens": 0,
                "sarvam_blocks": None,
                "quality_score": None,
                "low_confidence_blocks": None,
            }
            method = "PaddleOCR (fallback)"
        else:
            return self._error("All extraction methods failed. Please re-upload a clearer document.")

        content = primary.get("content", "")

        # Append extracted tables at the end if they're not already in the content
        if tables_html:
            tables_section = "\n\n--- EXTRACTED TABLES ---\n\n" + "\n\n".join(tables_html)
            # Only add if tables aren't already represented in the content
            if "<table" not in content.lower():
                content = content + tables_section
            else:
                # Replace existing tables with the dedicated extraction (likely more accurate)
                content = content + "\n\n--- DEDICATED TABLE EXTRACTION ---\n\n" + "\n\n".join(tables_html)

        return {
            "status": "success",
            "content": content,
            "tables_html": tables_html,
            "extraction_method": method,
            "quality_score": primary.get("quality_score"),
            "pages_processed": primary.get("pages_processed", 0),
            "total_tokens": primary.get("total_tokens", 0),
            "sarvam_blocks": primary.get("sarvam_blocks"),
            "low_confidence_blocks": primary.get("low_confidence_blocks"),
            "used_fallback": used_fallback,
            "pipeline_steps": pipeline_steps,
            "paddle_text": paddle_text if paddle_text.strip() else None,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _table_to_html(table_data: list, page_num: int = 0) -> str:
        """Convert a pdfplumber table (list of lists) to an HTML table."""
        if not table_data:
            return ""

        html = f'<table border="1" class="extracted-table">\n'
        # First row as header
        html += "  <thead><tr>"
        for cell in table_data[0]:
            cell_text = str(cell).strip() if cell else ""
            html += f"<th>{cell_text}</th>"
        html += "</tr></thead>\n"

        # Remaining rows
        html += "  <tbody>\n"
        for row in table_data[1:]:
            html += "    <tr>"
            for cell in row:
                cell_text = str(cell).strip() if cell else ""
                html += f"<td>{cell_text}</td>"
            html += "</tr>\n"
        html += "  </tbody>\n</table>"

        return f"<!-- Page {page_num} -->\n{html}"

    @staticmethod
    def _error(message: str) -> Dict:
        return {
            "status": "error",
            "content": message,
            "tables_html": [],
            "extraction_method": "Pipeline Failed",
            "quality_score": None,
            "pages_processed": 0,
            "total_tokens": 0,
            "sarvam_blocks": None,
            "low_confidence_blocks": None,
            "used_fallback": False,
            "pipeline_steps": [],
            "paddle_text": None,
        }


class _BytesUploadedFile:
    """Minimal file-like wrapper so Sarvam/OpenAI extractors can consume raw bytes."""

    def __init__(self, data: bytes, name: str):
        self._data = data
        self.name = name
        self.size = len(data)
        self.type = self._guess_type(name)
        self._buf = io.BytesIO(data)

    def read(self, *args):
        return self._buf.read(*args)

    def seek(self, *args):
        return self._buf.seek(*args)

    def getvalue(self):
        return self._data

    @staticmethod
    def _guess_type(name: str) -> str:
        name = name.lower()
        if name.endswith(".pdf"):
            return "application/pdf"
        elif name.endswith(".png"):
            return "image/png"
        elif name.endswith((".jpg", ".jpeg")):
            return "image/jpeg"
        return "application/octet-stream"
