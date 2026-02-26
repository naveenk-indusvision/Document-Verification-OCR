"""
Fast document processing pipeline:
  1. PDF/Image → PIL Images at 150 DPI (PyMuPDF primary, pdf2image fallback)
  2. Single GPT-4o Vision call with all pages

No preprocessing, no PaddleOCR, no Sarvam, no table-extraction libraries.
GPT-4o handles skew, noise, tables, and multi-language text natively.
"""
import io
import logging
from typing import Dict, List, Optional
from PIL import Image

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Lean pipeline: convert to images → single OpenAI GPT-4o Vision call.
    """

    def __init__(self, openai_api_key: str = ""):
        self.openai_api_key = openai_api_key
        self._openai_extractor = None

    def _get_openai(self):
        if self._openai_extractor is None and self.openai_api_key:
            from openai_text_extractor import OpenAITextExtractor
            self._openai_extractor = OpenAITextExtractor(api_key=self.openai_api_key)
        return self._openai_extractor

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_document(self, uploaded_file, doc_type: str = None) -> Dict:
        """
        Fast pipeline: images at 150 DPI → single GPT-4o Vision call.
        When doc_type is provided, combines OCR + structured extraction in one call.
        """
        if uploaded_file is None:
            return self._error("No file provided")

        file_name = uploaded_file.name.lower()
        is_pdf = file_name.endswith(".pdf")
        is_image = any(file_name.endswith(ext) for ext in [".jpg", ".jpeg", ".png"])

        if not (is_pdf or is_image):
            return self._error("Unsupported file type. Supported: PDF, JPG, JPEG, PNG")

        pipeline_steps = []

        try:
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()

            # STEP 1: Convert to images at 150 DPI
            images = self._to_images(file_bytes, file_name)
            pipeline_steps.append("file_loaded")

            if not images:
                return self._error("Failed to convert document to images")

            # STEP 2: Single GPT-4o Vision call
            openai_ext = self._get_openai()
            if not openai_ext:
                return self._error("OpenAI API key not configured")

            # Combined OCR + extraction when doc_type is provided (faster)
            if doc_type:
                result = openai_ext.extract_text_and_data(images, doc_type)
                pipeline_steps.append("openai_vision_ocr_and_extract")
            else:
                result = openai_ext.extract_text_from_images(images)
                pipeline_steps.append("openai_vision_ocr")

            if result.get("status") != "success":
                return self._error(result.get("text", "OpenAI extraction failed"))

            text = result.get("text", "")
            if self._is_refusal(text):
                return self._error("OpenAI refused to process this document. Try re-uploading.")

            return {
                "status": "success",
                "content": text,
                "tables_html": [],
                "extraction_method": "OpenAI Vision (GPT-4o)",
                "quality_score": None,
                "pages_processed": len(images),
                "total_tokens": result.get("tokens_used", 0),
                "sarvam_blocks": None,
                "low_confidence_blocks": None,
                "used_fallback": False,
                "pipeline_steps": pipeline_steps,
                "paddle_text": None,
                "extracted_data": result.get("extracted_data"),
            }

        except Exception as e:
            logger.exception("Document processing failed")
            return self._error(f"Processing error: {str(e)}")

    # ------------------------------------------------------------------
    # Image conversion
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
        """Convert PDF to images at 150 DPI. PyMuPDF primary, pdf2image fallback."""
        # Try PyMuPDF first (fast, no external binaries)
        images = self._pdf_to_images_pymupdf(pdf_bytes)
        if images:
            return images
        # Fallback to pdf2image (needs Poppler)
        return self._pdf_to_images_pdf2image(pdf_bytes)

    def _pdf_to_images_pymupdf(self, pdf_bytes: bytes) -> List[Image.Image]:
        """PDF → images using PyMuPDF at 150 DPI."""
        try:
            import fitz
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            images = []
            for page_num in range(min(len(doc), 10)):
                page = doc[page_num]
                mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
                pix = page.get_pixmap(matrix=mat)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                images.append(img)
            doc.close()
            return images
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {e}")
            return []

    def _pdf_to_images_pdf2image(self, pdf_bytes: bytes) -> List[Image.Image]:
        """Fallback PDF → images using pdf2image + Poppler."""
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
                    kwargs = {"pdf_file": pdf_bytes, "dpi": 150, "first_page": 1, "last_page": 10}
                    if pp:
                        kwargs["poppler_path"] = pp
                    return convert_from_bytes(**kwargs)
                except Exception:
                    continue
            return []
        except ImportError:
            return []

    # ------------------------------------------------------------------
    # Refusal detection
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
    """Minimal file-like wrapper so extractors can consume raw bytes."""

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
