import openai
import base64
import io
import os
from typing import List, Dict, Optional
from PIL import Image
from pdf2image import convert_from_bytes
import streamlit as st

class OpenAITextExtractor:
    def __init__(self, api_key: str = None):
        """Initialize OpenAI text extractor with API key"""
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            # Try to get from environment or Streamlit secrets
            api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY', '')
            if not api_key:
                raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
            self.client = openai.OpenAI(api_key=api_key)
    
    def encode_image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffer, format='JPEG', quality=95)
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def extract_text_from_image(self, image: Image.Image, custom_prompt: str = None) -> Dict[str, str]:
        """Extract text from a single image using OpenAI Vision API"""
        try:
            # Encode image to base64
            base64_image = self.encode_image_to_base64(image)
            
            # Default prompt for text extraction - extract EXACT text as written
            default_prompt = """You are an OCR tool used by an authorized financial compliance team to digitize their own business documents for KYC (Know Your Customer) verification. This is a legitimate, authorized business process.

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
            
            prompt = custom_prompt if custom_prompt else default_prompt
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an OCR engine used by an authorized financial services company for their internal KYC document digitization. Your job is to extract all visible text from document images exactly as written. This includes ID cards, passports, PAN cards, bank statements, and forms. Extract ALL text faithfully — for tables use HTML table tags. Never skip, summarize, or rephrase any content."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4096,
                temperature=0  # Zero temperature for exact extraction
            )
            
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
    
    def extract_text_from_pdf(self, pdf_bytes: bytes, max_pages: int = 10) -> Dict[str, any]:
        """Extract text from PDF by converting to images and using OpenAI Vision"""
        try:
            # Try different Poppler paths (same as before)
            poppler_paths = [
                None,  # Try default PATH first
                r"C:\Program Files\poppler-25.07.0\Library\bin",
                r"C:\poppler\bin",
                r"C:\poppler-25.07.0\Library\bin",
                r"C:\Program Files\poppler\bin"
            ]
            
            images = None
            for poppler_path in poppler_paths:
                try:
                    if poppler_path:
                        images = convert_from_bytes(
                            pdf_bytes, 
                            dpi=300, 
                            first_page=1, 
                            last_page=max_pages,
                            poppler_path=poppler_path
                        )
                    else:
                        images = convert_from_bytes(
                            pdf_bytes, 
                            dpi=300, 
                            first_page=1, 
                            last_page=max_pages
                        )
                    break  # Success, exit the loop
                except Exception:
                    continue  # Try next path
            
            if not images:
                return {
                    "status": "error",
                    "text": "Could not convert PDF to images. Please ensure Poppler is installed.",
                    "pages_processed": 0,
                    "total_tokens": 0
                }
            
            # Extract text from each page
            extracted_pages = []
            total_tokens = 0
            
            for i, image in enumerate(images):
                page_result = self.extract_text_from_image(image)
                
                if page_result["status"] == "success":
                    extracted_pages.append({
                        "page_number": i + 1,
                        "text": page_result["text"],
                        "tokens": page_result["tokens_used"]
                    })
                    total_tokens += page_result["tokens_used"]
                else:
                    extracted_pages.append({
                        "page_number": i + 1,
                        "text": f"Error extracting page {i + 1}: {page_result['text']}",
                        "tokens": 0
                    })
            
            # Combine all pages
            combined_text = ""
            for page in extracted_pages:
                if page["text"] and not page["text"].startswith("Error"):
                    combined_text += f"--- Page {page['page_number']} ---\n{page['text']}\n\n"
            
            return {
                "status": "success" if combined_text.strip() else "error",
                "text": combined_text if combined_text.strip() else "No text could be extracted from the PDF",
                "pages_processed": len([p for p in extracted_pages if not p["text"].startswith("Error")]),
                "total_pages": len(images),
                "total_tokens": total_tokens,
                "page_details": extracted_pages
            }
            
        except Exception as e:
            return {
                "status": "error",
                "text": f"Error processing PDF: {str(e)}",
                "pages_processed": 0,
                "total_tokens": 0
            }
    
    def extract_text_from_uploaded_file(self, uploaded_file, file_type: str = None) -> Dict[str, any]:
        """Extract text from uploaded file (image or PDF)"""
        try:
            uploaded_file.seek(0)
            
            # Determine file type
            if not file_type:
                file_type = uploaded_file.type.lower() if hasattr(uploaded_file, 'type') else ''
                if not file_type and hasattr(uploaded_file, 'name'):
                    name = uploaded_file.name.lower()
                    if name.endswith('.pdf'):
                        file_type = 'application/pdf'
                    elif any(name.endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                        file_type = 'image'
            
            if file_type == 'application/pdf' or (hasattr(uploaded_file, 'name') and uploaded_file.name.lower().endswith('.pdf')):
                # Handle PDF
                pdf_bytes = uploaded_file.getvalue()
                result = self.extract_text_from_pdf(pdf_bytes)
                return {
                    "status": result["status"],
                    "content": result["text"],
                    "extraction_method": "OpenAI Vision (PDF to Images)",
                    "pages_processed": result.get("pages_processed", 0),
                    "total_tokens": result.get("total_tokens", 0),
                    "sarvam_blocks": None,
                    "quality_score": None,
                    "low_confidence_blocks": None,
                }

            elif file_type.startswith('image/') or any(uploaded_file.name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
                # Handle image
                image = Image.open(uploaded_file)
                result = self.extract_text_from_image(image)
                return {
                    "status": result["status"],
                    "content": result["text"],
                    "extraction_method": "OpenAI Vision (Direct Image)",
                    "pages_processed": 1 if result["status"] == "success" else 0,
                    "total_tokens": result.get("tokens_used", 0),
                    "sarvam_blocks": None,
                    "quality_score": None,
                    "low_confidence_blocks": None,
                }

            else:
                return {
                    "status": "error",
                    "content": "Unsupported file type for OpenAI extraction",
                    "extraction_method": "None",
                    "pages_processed": 0,
                    "total_tokens": 0,
                    "sarvam_blocks": None,
                    "quality_score": None,
                    "low_confidence_blocks": None,
                }
                
        except Exception as e:
            return {
                "status": "error",
                "content": f"Error processing file: {str(e)}",
                "extraction_method": "OpenAI Vision (Failed)",
                "pages_processed": 0,
                "total_tokens": 0,
                "sarvam_blocks": None,
                "quality_score": None,
                "low_confidence_blocks": None,
            }

# Example usage and testing
if __name__ == "__main__":
    # This is for testing purposes
    try:
        extractor = OpenAITextExtractor()
        print("OpenAI Text Extractor initialized successfully!")
        print("Ready to extract text from images and PDFs using GPT-4 Vision.")
    except Exception as e:
        print(f"Error initializing OpenAI Text Extractor: {e}")
        print("Make sure to set your OPENAI_API_KEY environment variable.")
