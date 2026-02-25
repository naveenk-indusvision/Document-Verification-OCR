from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import json

class AnnexureInfo(BaseModel):
    cardholder_name: Optional[str] = Field(description="Name of the cardholder", default=None)
    passport_number: Optional[str] = Field(description="Passport number", default=None)
    date_of_birth: Optional[str] = Field(description="Date of birth", default=None)
    date_of_issuance: Optional[str] = Field(description="Passport date of issuance", default=None)
    date_of_expiry: Optional[str] = Field(description="Passport date of expiry", default=None)
    mothers_name: Optional[str] = Field(description="Mother's name", default=None)
    travel_date: Optional[str] = Field(description="Travel date", default=None)
    pan_number: Optional[str] = Field(description="PAN number", default=None)
    destination: Optional[str] = Field(description="Travel destination", default=None)
    date_of_travel: Optional[str] = Field(description="Date of travel", default=None)

class PANInfo(BaseModel):
    pan_number: Optional[str] = Field(description="PAN number", default=None)
    full_name: Optional[str] = Field(description="Full name on PAN card", default=None)
    fathers_name: Optional[str] = Field(description="Father's name", default=None)
    date_of_birth: Optional[str] = Field(description="Date of birth", default=None)

class PassportInfo(BaseModel):
    passport_number: Optional[str] = Field(description="Passport number", default=None)
    full_name: Optional[str] = Field(description="Full name on passport", default=None)
    mothers_name: Optional[str] = Field(description="Mother's name", default=None)
    date_of_birth: Optional[str] = Field(description="Date of birth", default=None)
    date_of_issuance: Optional[str] = Field(description="Date of issuance", default=None)
    date_of_expiry: Optional[str] = Field(description="Date of expiry", default=None)

class VisaInfo(BaseModel):
    full_name: Optional[str] = Field(description="Full name on visa", default=None)
    visa_expiry_date: Optional[str] = Field(description="Visa expiry date", default=None)
    country_destination: Optional[str] = Field(description="Country or destination", default=None)

class TicketInfo(BaseModel):
    full_name: Optional[str] = Field(description="Full name (from Annexure)", default=None)
    date_of_travel: Optional[str] = Field(description="Date of travel (from Annexure)", default=None)
    date_of_return: Optional[str] = Field(description="Date of return or exit date from country", default=None)

class DataExtractor:
    def __init__(self, api_key: str):
        """Initialize the data extractor with OpenAI API key"""
        self.chat = ChatOpenAI(
            model_name="gpt-5",
            openai_api_key=api_key,
            temperature=0  # Low temperature for consistent extraction
        )
    
    def extract_annexure_data(self, text: str) -> Dict[str, Any]:
        """Extract structured data from Annexure document"""
        parser = PydanticOutputParser(pydantic_object=AnnexureInfo)
        
        prompt = f"""
Extract the following details from this Annexure/mother document text:
- Cardholder name
- Passport number
- Date of birth
- Date of issuance (passport)
- Date of expiry (passport)
- Mother's name
- Travel date
- PAN number
- Destination
- Date of travel

Return the output as JSON matching the specified format.
If a field is not found, return null for that field.

{parser.get_format_instructions()}

Text:
\"\"\"
{text}
\"\"\"
"""
        
        result = self._extract_with_parser(prompt, parser, "Annexure")
        
        # Clean PAN number after extraction
        if result["status"] == "success" and result["data"].get("pan_number"):
            import re
            # Remove all spaces and special characters, keep only English letters and numbers
            cleaned_pan = re.sub(r'[^A-Z0-9]', '', str(result["data"]["pan_number"]).upper())
            result["data"]["pan_number"] = cleaned_pan
        
        return result
    
    def extract_pan_data(self, text: str) -> Dict[str, Any]:
        """Extract structured data from PAN card"""
        parser = PydanticOutputParser(pydantic_object=PANInfo)
        
        prompt = f"""
Extract the following details from this PAN card text:
- PAN number
- Full name
- Father's name
- Date of birth

Return the output as JSON matching the specified format.
If a field is not found, return null for that field.

{parser.get_format_instructions()}

Text:
\"\"\"
{text}
\"\"\"
"""
        
        result = self._extract_with_parser(prompt, parser, "PAN")
        
        # Clean PAN number after extraction
        if result["status"] == "success" and result["data"].get("pan_number"):
            import re
            # Remove all spaces and special characters, keep only English letters and numbers
            cleaned_pan = re.sub(r'[^A-Z0-9]', '', str(result["data"]["pan_number"]).upper())
            result["data"]["pan_number"] = cleaned_pan
        
        return result
    
    def extract_passport_data(self, text: str) -> Dict[str, Any]:
        """Extract structured data from Passport"""
        parser = PydanticOutputParser(pydantic_object=PassportInfo)
        
        prompt = f"""
Extract the following details from this passport text:
- Passport number
- Full name
- Mother's name
- Date of birth
- Date of issuance
- Date of expiry

Return the output as JSON matching the specified format.
If a field is not found, return null for that field.

{parser.get_format_instructions()}

Text:
\"\"\"
{text}
\"\"\"
"""
        
        return self._extract_with_parser(prompt, parser, "Passport")
    
    def extract_visa_data(self, text: str) -> Dict[str, Any]:
        """Extract structured data from Visa"""
        parser = PydanticOutputParser(pydantic_object=VisaInfo)
        
        prompt = f"""
Extract the following details from this visa text:
- Full name
- Visa expiry date
- Country or destination

Return the output as JSON matching the specified format.
If a field is not found, return null for that field.

{parser.get_format_instructions()}

Text:
\"\"\"
{text}
\"\"\"
"""
        
        return self._extract_with_parser(prompt, parser, "Visa")
    
    def extract_ticket_data(self, text: str) -> Dict[str, Any]:
        """Extract structured data from Ticket"""
        parser = PydanticOutputParser(pydantic_object=TicketInfo)
        
        prompt = f"""
Extract the following details from this ticket text:
- Full name (should match Annexure)
- Date of travel (should match Annexure)
- Date of return or exit date from country (validate with Visa)

Return the output as JSON matching the specified format.
If a field is not found, return null for that field.

{parser.get_format_instructions()}

Text:
\"\"\"
{text}
\"\"\"
"""
        
        return self._extract_with_parser(prompt, parser, "Ticket")
    
    def _extract_with_parser(self, prompt: str, parser: PydanticOutputParser, doc_type: str) -> Dict[str, Any]:
        """Internal method to extract data using LangChain"""
        try:
            messages = [
                SystemMessage(content=f"You are an expert at extracting structured data from {doc_type} documents. Extract only the information that is clearly visible in the text."),
                HumanMessage(content=prompt)
            ]
            
            response = self.chat.invoke(messages)
            
            # Parse the response to Python object
            parsed_data = parser.parse(response.content)
            
            return {
                "status": "success",
                "data": parsed_data.dict(),
                "document_type": doc_type
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "data": {},
                "document_type": doc_type
            }
    
    def extract_data_by_document_type(self, text: str, document_type: str) -> Dict[str, Any]:
        """Extract data based on document type"""
        document_type = document_type.lower()
        
        if document_type == "annexure":
            return self.extract_annexure_data(text)
        elif document_type == "pan":
            return self.extract_pan_data(text)
        elif document_type == "passport":
            return self.extract_passport_data(text)
        elif document_type == "visa":
            return self.extract_visa_data(text)
        elif document_type == "ticket":
            return self.extract_ticket_data(text)
        else:
            return {
                "status": "error",
                "error": f"Unknown document type: {document_type}",
                "data": {},
                "document_type": document_type
            }

# Example usage and testing
if __name__ == "__main__":
    # This is for testing purposes
    api_key = "your-api-key-here"
    
    try:
        extractor = DataExtractor(api_key)
        print("Data Extractor initialized successfully!")
        print("Ready to extract structured data from documents.")
    except Exception as e:
        print(f"Error initializing Data Extractor: {e}")
