import openai
import base64
import io
import os
import json
import time
import logging
from typing import List, Dict, Optional
from PIL import Image
import streamlit as st

logger = logging.getLogger(__name__)

# Field schemas for combined OCR + structured extraction per document type
EXTRACTION_SCHEMAS = {
    "annexure": {
        "cardholder_name": "Name of the cardholder",
        "passport_number": "Passport number",
        "date_of_birth": "Date of birth",
        "date_of_issuance": "Passport date of issuance",
        "date_of_expiry": "Passport date of expiry",
        "mothers_name": "Mother's name",
        "travel_date": "Travel date",
        "pan_number": "PAN number",
        "destination": "Travel destination",
        "date_of_travel": "Date of travel",
    },
    "pan": {
        "pan_number": "PAN number",
        "full_name": "Full name on PAN card",
        "fathers_name": "Father's name",
        "date_of_birth": "Date of birth",
    },
    "passport": {
        "passport_number": "Passport number",
        "full_name": "Full name on passport",
        "mothers_name": "Mother's name",
        "date_of_birth": "Date of birth",
        "date_of_issuance": "Date of issuance",
        "date_of_expiry": "Date of expiry",
    },
    "visa": {
        "full_name": "Full name on visa",
        "visa_expiry_date": "Visa expiry date",
        "country_destination": "Country or destination",
    },
    "ticket": {
        "full_name": "Full name",
        "date_of_travel": "Date of travel",
        "date_of_return": "Date of return or exit date from country",
    },
}


class OpenAITextExtractor:
    def __init__(self, api_key: str = None):
        """Initialize OpenAI text extractor with API key"""
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY', '')
            if not api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            self.client = openai.OpenAI(api_key=api_key)

    SYSTEM_PROMPT = (
        "You are an OCR engine used by an authorized financial services company "
        "for their internal KYC document digitization. Your job is to extract all "
        "visible text from document images exactly as written. This includes ID cards, "
        "passports, PAN cards, bank statements, and forms. Extract ALL text faithfully "
        "— for tables use HTML table tags. Never skip, summarize, or rephrase any content."
    )

    OCR_PROMPT = """You are an OCR tool used by an authorized financial compliance team to digitize their own business documents for KYC (Know Your Customer) verification. This is a legitimate, authorized business process.

Extract ALL text from this document image EXACTLY as written.

RULES:
1. Copy every single word, number, date, name, address, and field value EXACTLY as it appears
2. Do NOT summarize, paraphrase, or skip any content
3. Do NOT classify or label content (no "Header:", "Paragraph:" labels)
4. For TABLES: reproduce them as HTML tables with <table>, <tr>, <th>, <td> tags preserving all rows, columns, headers and cell values exactly
5. If there are form fields with labels and values, write them as "Label: Value" on each line
6. If there are multiple columns, read left to right, top to bottom
7. Preserve all line breaks, spacing, and structure as they appear in the document
8. Include ALL numbers, dates, names, addresses, reference numbers, amounts - miss nothing
9. If text appears rotated, faded, or at an angle, still extract it
10. Do NOT add any commentary, explanation, or formatting beyond what exists in the document

Return ONLY the raw text content exactly as it appears in the document."""

    COMBINED_SYSTEM_PROMPT = (
        "You are an OCR and data extraction engine used by an authorized financial services "
        "company for their internal KYC document digitization. You extract all visible text "
        "AND structured fields from document images. Always respond with valid JSON only."
    )

    def encode_image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffer, format='JPEG', quality=75)
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')

    def _call_with_retry(self, api_call, max_retries=3):
        """Execute an API call with exponential backoff retry for transient errors."""
        for attempt in range(max_retries + 1):
            try:
                return api_call()
            except (openai.RateLimitError, openai.APITimeoutError,
                    openai.InternalServerError, openai.APIConnectionError) as e:
                if attempt == max_retries:
                    raise
                wait = 2 ** attempt
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)

    def _get_detail_level(self, num_images: int) -> str:
        """Use 'auto' for single-page docs (PAN/Passport), 'high' for multi-page."""
        return "auto" if num_images == 1 else "high"

    def _get_max_tokens(self, num_images: int, combined: bool = False) -> int:
        """Adaptive max_tokens based on page count and mode."""
        if combined:
            if num_images == 1:
                return 8192
            elif num_images <= 3:
                return 12288
            else:
                return 16384
        else:
            if num_images == 1:
                return 4096
            elif num_images <= 3:
                return 8192
            else:
                return 16384

    def extract_text_and_data(self, images: List[Image.Image], doc_type: str) -> Dict:
        """Combined OCR + structured extraction in a SINGLE GPT-4o Vision call.

        Returns both raw text and structured fields, eliminating the need for
        a separate GPT-4o-mini extraction call.
        """
        if not images:
            return {"status": "error", "text": "", "extracted_data": None, "tokens_used": 0}

        schema = EXTRACTION_SCHEMAS.get(doc_type.lower())
        if not schema:
            return {"status": "error", "text": f"Unknown doc type: {doc_type}", "extracted_data": None, "tokens_used": 0}

        try:
            fields_list = "\n".join(f"- {k}: {v}" for k, v in schema.items())
            fields_json = ",\n    ".join(f'"{k}": "value or null"' for k in schema.keys())

            page_note = ""
            if len(images) > 1:
                page_note = f"This document has {len(images)} pages shown as {len(images)} images below. Extract from ALL pages in order.\n\n"

            prompt = f"""{page_note}Perform TWO tasks on this document and return the result as a single JSON object.

**TASK 1 - RAW TEXT EXTRACTION:**
Extract ALL text from the document EXACTLY as written.
- Copy every word, number, date, name, address, and field value EXACTLY
- For TABLES: use HTML <table>, <tr>, <th>, <td> tags
- Preserve line breaks, spacing, and structure
- Include ALL content - miss nothing
- If text is rotated, faded, or angled, still extract it

**TASK 2 - STRUCTURED DATA EXTRACTION:**
From the same document, extract these specific fields:
{fields_list}

Return ONLY this JSON object:
{{
  "raw_text": "the complete OCR text exactly as it appears",
  "extracted_data": {{
    {fields_json}
  }}
}}

Set any field to null if not found in the document."""

            detail = self._get_detail_level(len(images))
            content_parts = [{"type": "text", "text": prompt}]
            for image in images:
                b64 = self.encode_image_to_base64(image)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail}
                })

            max_tokens = self._get_max_tokens(len(images), combined=True)

            def api_call():
                return self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": self.COMBINED_SYSTEM_PROMPT},
                        {"role": "user", "content": content_parts}
                    ],
                    max_tokens=max_tokens,
                    temperature=0,
                    response_format={"type": "json_object"}
                )

            response = self._call_with_retry(api_call)
            result_text = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0

            parsed = json.loads(result_text)
            raw_text = parsed.get("raw_text", "")
            extracted = parsed.get("extracted_data", {})

            return {
                "status": "success",
                "text": raw_text,
                "extracted_data": {
                    "status": "success",
                    "data": extracted,
                    "document_type": doc_type
                },
                "tokens_used": tokens
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse combined JSON response: {e}")
            return {"status": "error", "text": "", "extracted_data": None, "tokens_used": 0}
        except Exception as e:
            return {
                "status": "error",
                "text": f"Error extracting text with OpenAI: {str(e)}",
                "extracted_data": None,
                "tokens_used": 0
            }

    def extract_text_from_images(self, images: List[Image.Image], custom_prompt: str = None) -> Dict:
        """Extract text from multiple images in a SINGLE API call (OCR only).

        Kept for backward compatibility. For new code, prefer extract_text_and_data().
        """
        if not images:
            return {"status": "error", "text": "No images provided", "tokens_used": 0}

        try:
            prompt = custom_prompt or self.OCR_PROMPT
            if len(images) > 1:
                prompt = (
                    f"This document has {len(images)} pages, shown as {len(images)} images below. "
                    f"Extract text from ALL pages in order.\n\n{prompt}"
                )

            detail = self._get_detail_level(len(images))
            content_parts = [{"type": "text", "text": prompt}]
            for image in images:
                base64_image = self.encode_image_to_base64(image)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": detail
                    }
                })

            max_tokens = self._get_max_tokens(len(images), combined=False)

            def api_call():
                return self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": content_parts}
                    ],
                    max_tokens=max_tokens,
                    temperature=0
                )

            response = self._call_with_retry(api_call)
            extracted_text = response.choices[0].message.content
            return {
                "status": "success",
                "text": extracted_text,
                "tokens_used": response.usage.total_tokens if response.usage else 0
            }

        except Exception as e:
            return {
                "status": "error",
                "text": f"Error extracting text with OpenAI: {str(e)}",
                "tokens_used": 0
            }

    def extract_text_from_image(self, image: Image.Image, custom_prompt: str = None) -> Dict[str, str]:
        """Extract text from a single image using OpenAI Vision API"""
        return self.extract_text_from_images([image], custom_prompt)


# Example usage and testing
if __name__ == "__main__":
    try:
        extractor = OpenAITextExtractor()
        print("OpenAI Text Extractor initialized successfully!")
        print("Ready to extract text from images and PDFs using GPT-4o Vision.")
    except Exception as e:
        print(f"Error initializing OpenAI Text Extractor: {e}")
        print("Make sure to set your OPENAI_API_KEY environment variable.")
