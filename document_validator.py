from fuzzywuzzy import fuzz
from typing import Dict, List, Optional, Any
import re
from datetime import datetime

class DocumentValidator:
    def __init__(self, similarity_threshold: int = 80):
        """
        Initialize document validator
        
        Args:
            similarity_threshold: Minimum similarity score for fuzzy matching (0-100)
        """
        self.similarity_threshold = similarity_threshold
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        # Convert to lowercase, remove extra spaces, special characters
        normalized = re.sub(r'[^\w\s]', '', str(text).lower().strip())
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized
    
    def normalize_date(self, date_str: str) -> str:
        """Normalize date to DD/MM/YYYY format with smart format detection"""
        if not date_str:
            return ""
        
        # Clean the input string - remove extra spaces
        clean_date = str(date_str).strip()
        
        # Replace any separator with /
        normalized = re.sub(r'[-.\s]', '/', clean_date)
        
        # Split by separator
        parts = normalized.split('/')
        
        if len(parts) == 3:
            # Try different permutations since day is never in the middle
            # We know the format is one of: DD/MM/YYYY, MM/DD/YYYY, YYYY/MM/DD, YYYY/DD/MM
            permutations = [
                (parts[0], parts[1], parts[2]),  # DD/MM/YYYY or MM/DD/YYYY
                (parts[2], parts[0], parts[1]),  # YYYY/DD/MM
                (parts[2], parts[1], parts[0]),  # YYYY/MM/DD
            ]
            
            for day_candidate, month_candidate, year_candidate in permutations:
                try:
                    day = int(day_candidate)
                    month = int(month_candidate)
                    year = int(year_candidate)
                    
                    # Handle 2-digit years (assume 20xx for years 00-30, 19xx for 31-99)
                    if year < 100:
                        if year <= 30:
                            year += 2000
                        else:
                            year += 1900
                    
                    # Validate ranges
                    if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100:
                        # Return standardized format with leading zeros
                        return f"{day:02d}/{month:02d}/{year}"
                        
                except ValueError:
                    continue
        
        # If can't parse, return original
        return clean_date
    
    def fuzzy_match(self, text1: str, text2: str) -> Dict[str, Any]:
        """Strict matching - only exact matches pass, everything else requires manual verification or fails"""
        if not text1 or not text2:
            return {"match": False, "score": 0, "reason": "Missing data", "match_type": "missing", "requires_manual": False}
        
        # Normalize for exact comparison (remove extra spaces, case insensitive)
        norm1 = re.sub(r'\s+', ' ', str(text1).strip().lower())
        norm2 = re.sub(r'\s+', ' ', str(text2).strip().lower())
        
        # STRICT: Only exact match after normalization counts as "match"
        if norm1 == norm2:
            return {"match": True, "score": 100, "reason": "Matched", "match_type": "exact", "requires_manual": False}
        
        # Calculate similarity scores for determining manual verification
        ratio_score = fuzz.ratio(norm1, norm2)
        partial_score = fuzz.partial_ratio(norm1, norm2)
        token_sort_score = fuzz.token_sort_ratio(norm1, norm2)
        token_set_score = fuzz.token_set_ratio(norm1, norm2)
        
        # Use the highest score
        best_score = max(ratio_score, partial_score, token_sort_score, token_set_score)
        
        # STRICT LOGIC: If somewhat similar (60%+), require manual verification
        if best_score >= 60:
            return {
                "match": False,
                "score": best_score,
                "reason": f"Somewhat similar - requires manual verification",
                "match_type": "partial",
                "requires_manual": True,
                "clean_values": {"text1": text1, "text2": text2},
                "scores": {
                    "ratio": ratio_score,
                    "partial": partial_score,
                    "token_sort": token_sort_score,
                    "token_set": token_set_score
                }
            }
        else:
            # Complete mismatch - no manual verification needed
            return {
                "match": False,
                "score": best_score,
                "reason": "Mismatch",
                "match_type": "no_match",
                "requires_manual": False,
                "clean_values": {"text1": text1, "text2": text2}
            }
    
    def exact_match(self, value1: str, value2: str) -> Dict[str, Any]:
        """Perform exact matching for numbers, IDs, etc."""
        if not value1 or not value2:
            return {"match": False, "score": 0, "reason": "Missing data", "match_type": "missing", "requires_manual": False}
        
        # Normalize by removing spaces and converting to uppercase
        norm1 = re.sub(r'\s+', '', str(value1).upper().strip())
        norm2 = re.sub(r'\s+', '', str(value2).upper().strip())
        
        if norm1 == norm2:
            return {"match": True, "score": 100, "reason": "Matched", "match_type": "exact", "requires_manual": False}
        
        return {
            "match": False, 
            "score": 0, 
            "reason": "Mismatch", 
            "match_type": "no_match",
            "requires_manual": False
        }
    
    def date_match(self, date1: str, date2: str) -> Dict[str, Any]:
        """Date matching with manual verification for close dates and different formats"""
        if not date1 or not date2:
            return {"match": False, "score": 0, "reason": "Missing data", "match_type": "missing", "requires_manual": False}
        
        # Normalize both dates to DD/MM/YYYY format
        norm_date1 = self.normalize_date(date1)
        norm_date2 = self.normalize_date(date2)
        
        # If both normalized successfully and match exactly
        if norm_date1 and norm_date2 and norm_date1 == norm_date2:
            return {"match": True, "score": 100, "reason": "Matched", "match_type": "exact", "requires_manual": False}
        
        # Check if dates are the same but in different formats
        if norm_date1 and norm_date2:
            try:
                from datetime import datetime
                
                # Parse normalized dates
                date1_obj = datetime.strptime(norm_date1, '%d/%m/%Y')
                date2_obj = datetime.strptime(norm_date2, '%d/%m/%Y')
                
                # Check if they represent the same date
                if date1_obj == date2_obj:
                    return {
                        "match": False,
                        "score": 0,
                        "reason": "Same date, different format - requires manual verification",
                        "match_type": "partial",
                        "requires_manual": True,
                        "clean_values": {"date1": date1, "date2": date2, "normalized1": norm_date1, "normalized2": norm_date2}
                    }
                
                # Calculate difference for close dates
                date_diff = abs((date1_obj - date2_obj).days)
                
                # If dates are within 1 day or 1 month (30 days), require manual verification
                if date_diff <= 30:
                    return {
                        "match": False,
                        "score": 0,
                        "reason": "Close dates - requires manual verification",
                        "match_type": "partial",
                        "requires_manual": True,
                        "clean_values": {"date1": date1, "date2": date2, "normalized1": norm_date1, "normalized2": norm_date2, "difference_days": date_diff}
                    }
                
            except ValueError:
                pass  # If date parsing fails, continue to invalid case
        
        # If dates don't match and are not close - they are Invalid
        return {
            "match": False,
            "score": 0,
            "reason": "Invalid",
            "match_type": "no_match",
            "requires_manual": False
        }
    
    def pan_match(self, pan1: str, pan2: str) -> Dict[str, Any]:
        """Special matching for PAN numbers with partial match detection"""
        if not pan1 or not pan2:
            return {"match": False, "score": 0, "reason": "Missing PAN data", "match_type": "missing", "requires_manual": False}
        
        # Clean PAN numbers - remove ALL spaces (including between characters) and special characters
        # Only keep English letters (A-Z) and numbers (0-9)
        clean_pan1 = re.sub(r'[^A-Z0-9]', '', str(pan1).upper())
        clean_pan2 = re.sub(r'[^A-Z0-9]', '', str(pan2).upper())
        
        # Exact match check
        if clean_pan1 == clean_pan2:
            return {
                "match": True, 
                "score": 100, 
                "reason": "Matched", 
                "match_type": "exact",
                "requires_manual": False,
                "clean_values": {"pan1": clean_pan1, "pan2": clean_pan2}
            }
        
        # No match if lengths are very different (PAN should be 10 characters)
        if abs(len(clean_pan1) - len(clean_pan2)) > 2:
            return {
                "match": False, 
                "score": 0, 
                "reason": "Mismatch", 
                "match_type": "no_match",
                "requires_manual": False,
                "clean_values": {"pan1": clean_pan1, "pan2": clean_pan2}
            }
        
        # Check for partial match (character by character)
        matching_chars = sum(1 for a, b in zip(clean_pan1, clean_pan2) if a == b)
        max_length = max(len(clean_pan1), len(clean_pan2))
        partial_score = (matching_chars / max_length) * 100 if max_length > 0 else 0
        
        # If partial match is significant (60-99%), require manual verification
        if 60 <= partial_score < 100:
            return {
                "match": False,  # Not auto-matched
                "score": partial_score,
                "reason": f"Somewhat similar - requires manual verification",
                "match_type": "partial",
                "requires_manual": True,
                "clean_values": {"pan1": clean_pan1, "pan2": clean_pan2},
                "matching_positions": [i for i, (a, b) in enumerate(zip(clean_pan1, clean_pan2)) if a == b]
            }
        
        # Low similarity - no match
        return {
            "match": False,
            "score": partial_score,
            "reason": "Mismatch",
            "match_type": "no_match",
            "requires_manual": False,
            "clean_values": {"pan1": clean_pan1, "pan2": clean_pan2}
        }
    
    def validate_pan_against_annexure(self, pan_data: Dict, annexure_data: Dict) -> Dict[str, Any]:
        """Validate PAN card data against Annexure"""
        validations = {}
        
        # Check PAN number with special PAN matching logic
        if pan_data.get("pan_number") and annexure_data.get("pan_number"):
            validations["pan_number"] = self.pan_match(
                pan_data["pan_number"], 
                annexure_data["pan_number"]
            )
        
        # Check full name (fuzzy match since names can vary slightly)
        if pan_data.get("full_name") and annexure_data.get("cardholder_name"):
            validations["full_name"] = self.fuzzy_match(
                pan_data["full_name"], 
                annexure_data["cardholder_name"]
            )
        
        # Check date of birth using simple date matching
        if pan_data.get("date_of_birth") and annexure_data.get("date_of_birth"):
            validations["date_of_birth"] = self.date_match(
                pan_data["date_of_birth"], 
                annexure_data["date_of_birth"]
            )
        
        # Calculate overall match percentage
        total_checks = len(validations)
        if total_checks > 0:
            matches = sum(1 for v in validations.values() if v["match"])
            overall_score = (matches / total_checks) * 100
        else:
            overall_score = 0
        
        return {
            "document_type": "PAN vs Annexure",
            "overall_match": overall_score >= 70,  # 70% threshold for overall match
            "overall_score": overall_score,
            "field_validations": validations,
            "summary": f"{len([v for v in validations.values() if v['match']])}/{total_checks} fields match"
        }
    
    def validate_passport_against_annexure(self, passport_data: Dict, annexure_data: Dict) -> Dict[str, Any]:
        """Validate Passport data against Annexure"""
        validations = {}
        
        # Check passport number
        if passport_data.get("passport_number") and annexure_data.get("passport_number"):
            validations["passport_number"] = self.exact_match(
                passport_data["passport_number"], 
                annexure_data["passport_number"]
            )
        
        # Check full name
        if passport_data.get("full_name") and annexure_data.get("cardholder_name"):
            validations["full_name"] = self.fuzzy_match(
                passport_data["full_name"], 
                annexure_data["cardholder_name"]
            )
        
        # Check date of birth using simple date matching
        if passport_data.get("date_of_birth") and annexure_data.get("date_of_birth"):
            validations["date_of_birth"] = self.date_match(
                passport_data["date_of_birth"], 
                annexure_data["date_of_birth"]
            )
        
        # Check mother's name
        if passport_data.get("mothers_name") and annexure_data.get("mothers_name"):
            validations["mothers_name"] = self.fuzzy_match(
                passport_data["mothers_name"], 
                annexure_data["mothers_name"]
            )
        
        # Check passport dates using simple date matching
        if passport_data.get("date_of_issuance") and annexure_data.get("date_of_issuance"):
            validations["date_of_issuance"] = self.date_match(
                passport_data["date_of_issuance"], 
                annexure_data["date_of_issuance"]
            )
        
        if passport_data.get("date_of_expiry") and annexure_data.get("date_of_expiry"):
            validations["date_of_expiry"] = self.date_match(
                passport_data["date_of_expiry"], 
                annexure_data["date_of_expiry"]
            )
        
        # Calculate overall match percentage
        total_checks = len(validations)
        if total_checks > 0:
            matches = sum(1 for v in validations.values() if v["match"])
            overall_score = (matches / total_checks) * 100
        else:
            overall_score = 0
        
        return {
            "document_type": "Passport vs Annexure",
            "overall_match": overall_score >= 70,
            "overall_score": overall_score,
            "field_validations": validations,
            "summary": f"{len([v for v in validations.values() if v['match']])}/{total_checks} fields match"
        }
    
    def validate_all_documents(self, documents_data: Dict[str, Dict]) -> Dict[str, Any]:
        """Validate all documents against Annexure"""
        results = {}
        
        annexure_data = documents_data.get("Annexure", {}).get("data", {})
        
        if not annexure_data:
            return {
                "status": "error",
                "message": "Annexure data not available for validation",
                "validations": {}
            }
        
        # Validate PAN against Annexure
        if "PAN" in documents_data and documents_data["PAN"].get("data"):
            results["PAN"] = self.validate_pan_against_annexure(
                documents_data["PAN"]["data"], 
                annexure_data
            )
        
        # Validate Passport against Annexure
        if "Passport" in documents_data and documents_data["Passport"].get("data"):
            results["Passport"] = self.validate_passport_against_annexure(
                documents_data["Passport"]["data"], 
                annexure_data
            )
        
        # Calculate overall validation score
        if results:
            total_validations = len(results)
            successful_validations = sum(1 for v in results.values() if v["overall_match"])
            overall_success_rate = (successful_validations / total_validations) * 100
        else:
            overall_success_rate = 0
        
        return {
            "status": "success",
            "overall_success_rate": overall_success_rate,
            "validations": results,
            "summary": f"{len([v for v in results.values() if v['overall_match']])}/{len(results)} documents validated successfully"
        }

# Example usage
if __name__ == "__main__":
    validator = DocumentValidator()
    print("Document Validator initialized successfully!")
    print("Ready to validate documents against Annexure.")
