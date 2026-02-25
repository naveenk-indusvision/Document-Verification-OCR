"""
Sarvam Document Intelligence text extractor.
Replaces OpenAI Vision as the primary OCR engine.
"""
import os
import re
import json
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional


class SarvamTextExtractor:
    """Extracts text from documents using Sarvam Document Intelligence API."""

    SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
    MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB
    LOW_CONFIDENCE_THRESHOLD = 0.70

    def __init__(self, api_key: str, language: str = "en-IN"):
        if not api_key:
            raise ValueError("Sarvam API key is required. Set the SARVAM_API_KEY environment variable.")
        from sarvamai import SarvamAI
        self.client = SarvamAI(api_subscription_key=api_key)
        self.language = language

    def extract_text_from_uploaded_file(self, uploaded_file, file_type: str = None) -> Dict:
        """
        Extract text from an uploaded file using Sarvam Document Intelligence.

        Returns a dict with:
            status, content, extraction_method, pages_processed, total_tokens,
            sarvam_blocks, quality_score, low_confidence_blocks
        """
        error = self._validate_file(uploaded_file)
        if error:
            return self._error_result(error)

        suffix = Path(uploaded_file.name).suffix.lower()
        tmp_path = None
        tmp_zip_path = None

        try:
            # Save uploaded file to disk (Sarvam SDK needs a file path)
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                uploaded_file.seek(0)
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Run Sarvam job
            from sarvamai.core.api_error import ApiError

            job = self.client.document_intelligence.create_job(
                language=self.language,
                output_format="md"
            )
            job.upload_file(tmp_path)
            job.start()
            status = job.wait_until_complete()

            if hasattr(status, "job_state") and status.job_state == "Failed":
                msg = getattr(status, "error_message", "Unknown error")
                return self._error_result(f"Sarvam processing failed: {msg}")

            # PartiallyCompleted — still try to extract whatever succeeded
            if hasattr(status, "job_state") and status.job_state == "PartiallyCompleted":
                pass  # proceed to download partial results

            # Download output ZIP
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as ztmp:
                tmp_zip_path = ztmp.name
            job.download_output(tmp_zip_path)

            # Parse the ZIP contents
            parsed = self._parse_sarvam_zip(tmp_zip_path)
            if parsed is None:
                return self._error_result("Failed to parse Sarvam output ZIP.")

            markdown = parsed["markdown"]
            blocks = parsed["blocks"]
            page_count = parsed["page_count"]

            # Build raw text from blocks (preserves exact content + tables as HTML)
            raw_text = self._build_raw_text_from_blocks(blocks)

            # Also keep cleaned markdown as a backup
            cleaned_markdown = self._clean_markdown(markdown)

            # Use raw block text if available, fall back to cleaned markdown
            content = raw_text if raw_text.strip() else cleaned_markdown

            if len(content.strip()) < 20:
                return self._error_result(
                    "Extraction produced minimal or no text. "
                    "The document may be blank, heavily redacted, or of very poor quality. "
                    "Please re-upload with a clearer scan."
                )

            # Compute quality metrics
            quality = self._compute_quality_metrics(blocks)

            return {
                "status": "success",
                "content": content,
                "extraction_method": "Sarvam Document Intelligence",
                "pages_processed": page_count,
                "total_tokens": 0,
                "sarvam_blocks": blocks,
                "quality_score": quality["average_confidence"],
                "low_confidence_blocks": quality["low_confidence_blocks"],
            }

        except ImportError:
            return self._error_result(
                "sarvamai package not installed. Run: pip install sarvamai"
            )
        except Exception as e:
            # Catch ApiError and anything else
            return self._error_result(f"Sarvam extraction error: {str(e)}")
        finally:
            # Cleanup temp files
            for path in [tmp_path, tmp_zip_path]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_file(self, uploaded_file) -> Optional[str]:
        """Pre-validate file before sending to Sarvam. Returns error message or None."""
        if uploaded_file is None:
            return "No file provided."

        if hasattr(uploaded_file, "size") and uploaded_file.size == 0:
            return "File is empty."

        if hasattr(uploaded_file, "size") and uploaded_file.size > self.MAX_FILE_SIZE:
            return f"File exceeds {self.MAX_FILE_SIZE // (1024*1024)} MB limit."

        ext = Path(uploaded_file.name).suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            return (
                f"Unsupported file format '{ext}'. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        return None

    def _parse_sarvam_zip(self, zip_path: str) -> Optional[Dict]:
        """Extract markdown and block metadata from Sarvam output ZIP."""
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                names = zf.namelist()

                # Read document.md
                markdown = ""
                md_files = [n for n in names if n.endswith(".md")]
                if md_files:
                    markdown = zf.read(md_files[0]).decode("utf-8", errors="replace")

                # Read page metadata JSONs
                all_blocks: List[Dict] = []
                page_count = 0
                json_files = sorted(
                    [n for n in names if n.endswith(".json") and "metadata" in n.lower()]
                )
                for jf in json_files:
                    page_count += 1
                    try:
                        page_data = json.loads(zf.read(jf).decode("utf-8", errors="replace"))
                        blocks = page_data.get("blocks", [])
                        for block in blocks:
                            block["_page_num"] = page_data.get("page_num", page_count)
                        all_blocks.extend(blocks)
                    except (json.JSONDecodeError, KeyError):
                        continue

                # If no metadata found, still count pages from markdown
                if page_count == 0:
                    page_count = 1

                return {
                    "markdown": markdown,
                    "blocks": all_blocks,
                    "page_count": page_count,
                }
        except zipfile.BadZipFile:
            return None
        except Exception:
            return None

    def _compute_quality_metrics(self, blocks: List[Dict]) -> Dict:
        """Compute quality scores from block confidence values."""
        if not blocks:
            return {
                "average_confidence": 0.0,
                "min_confidence": 0.0,
                "low_confidence_blocks": [],
            }

        confidences = [b.get("confidence", 0.0) for b in blocks]
        avg = sum(confidences) / len(confidences)
        min_conf = min(confidences)

        low_blocks = [
            b for b in blocks
            if b.get("confidence", 0.0) < self.LOW_CONFIDENCE_THRESHOLD
        ]

        return {
            "average_confidence": round(avg, 4),
            "min_confidence": round(min_conf, 4),
            "low_confidence_blocks": low_blocks,
        }

    def _clean_markdown(self, markdown: str) -> str:
        """Strip base64 embedded images but preserve ALL text and table content exactly."""
        cleaned = re.sub(r'!\[Image\]\(data:image/[^)]+\)', '', markdown)
        # Also remove any other base64 data URIs
        cleaned = re.sub(r'!\[[^\]]*\]\(data:[^)]+\)', '', cleaned)
        # Collapse multiple blank lines (but preserve structure)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

    def _build_raw_text_from_blocks(self, blocks: list) -> str:
        """Reconstruct raw document text from Sarvam blocks, preserving exact content.

        This outputs every block's text in reading order without layout labels,
        keeping tables as HTML so downstream consumers can parse them.
        """
        if not blocks:
            return ""
        sorted_blocks = sorted(blocks, key=lambda b: (b.get("_page_num", 1), b.get("reading_order", 0)))
        parts = []
        current_page = None
        for block in sorted_blocks:
            page = block.get("_page_num", 1)
            if current_page is not None and page != current_page:
                parts.append(f"\n--- Page {page} ---\n")
            current_page = page

            text = block.get("text", "").strip()
            if not text:
                continue
            layout_tag = block.get("layout_tag", "")
            # Skip image-only blocks
            if layout_tag == "image":
                continue
            parts.append(text)
        return "\n\n".join(parts)

    @staticmethod
    def _error_result(message: str) -> Dict:
        """Build a standardized error response dict."""
        return {
            "status": "error",
            "content": message,
            "extraction_method": "Sarvam Document Intelligence",
            "pages_processed": 0,
            "total_tokens": 0,
            "sarvam_blocks": None,
            "quality_score": None,
            "low_confidence_blocks": None,
        }
