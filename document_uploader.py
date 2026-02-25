import copy
import streamlit as st
import os
import re
import pandas as pd
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from data_extractor import DataExtractor
from document_validator import DocumentValidator

load_dotenv()

# API keys
SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

@st.cache_resource
def get_document_processor():
    """Get or create the full document processing pipeline."""
    try:
        return DocumentProcessor(
            sarvam_api_key=SARVAM_API_KEY,
            openai_api_key=OPENAI_API_KEY,
            use_preprocessing=True,
            use_layout_analysis=True,
            use_table_extraction=True,
        )
    except Exception as e:
        st.error(f"Failed to initialize document processor: {e}")
        return None

@st.cache_resource
def get_data_extractor():
    """Get or create data extractor"""
    try:
        return DataExtractor(api_key=OPENAI_API_KEY)
    except Exception as e:
        st.error(f"Failed to initialize data extractor: {e}")
        return None

@st.cache_resource
def get_document_validator():
    """Get or create document validator"""
    try:
        return DocumentValidator(similarity_threshold=80)
    except Exception as e:
        st.error(f"Failed to initialize document validator: {e}")
        return None


def parse_document(uploaded_file) -> Dict:
    """
    Full extraction pipeline:
      OpenCV preprocessing → PaddleOCR layout → img2table/pdfplumber tables
      → Sarvam OCR (primary) → OpenAI fallback → merged result.
    """
    if uploaded_file is None:
        return {"status": "error", "content": "No file provided"}

    file_type = uploaded_file.type.lower() if uploaded_file.type else ""
    file_name = uploaded_file.name.lower()

    is_pdf = file_type == "application/pdf" or file_name.endswith('.pdf')
    is_image = (
        file_type in ["image/jpeg", "image/jpg", "image/png"]
        or any(file_name.endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])
    )

    if not (is_pdf or is_image):
        return {
            "filename": uploaded_file.name,
            "file_type": file_type,
            "file_size": f"{uploaded_file.size / 1024:.2f} KB",
            "status": "unsupported",
            "content": "Unsupported file type. Supported: PDF, JPG, JPEG, PNG",
            "sarvam_blocks": None,
            "quality_score": None,
            "low_confidence_blocks": None,
            "extraction_method": None,
        }

    processor = get_document_processor()
    if not processor:
        return {
            "filename": uploaded_file.name,
            "file_type": file_type,
            "file_size": f"{uploaded_file.size / 1024:.2f} KB",
            "status": "error",
            "content": "Document processor not available. Check API keys.",
            "sarvam_blocks": None,
            "quality_score": None,
            "low_confidence_blocks": None,
            "extraction_method": None,
        }

    # Run the full pipeline
    result = processor.process_document(uploaded_file)

    # Build unified parsing_result
    parsing_result = {
        "filename": uploaded_file.name,
        "file_type": file_type,
        "file_size": f"{uploaded_file.size / 1024:.2f} KB",
        "status": result.get("status", "error"),
        "content": result.get("content", ""),
        "sarvam_blocks": result.get("sarvam_blocks"),
        "quality_score": result.get("quality_score"),
        "low_confidence_blocks": result.get("low_confidence_blocks"),
        "extraction_method": result.get("extraction_method", ""),
        "used_fallback": result.get("used_fallback", False),
        "tables_html": result.get("tables_html", []),
        "pipeline_steps": result.get("pipeline_steps", []),
        "paddle_text": result.get("paddle_text"),
    }

    # Flag as error if content starts with "Error"
    if parsing_result["content"].startswith("Error"):
        parsing_result["status"] = "error"

    return parsing_result

def render_quality_banner(quality_score, doc_name, extraction_method=None, pipeline_steps=None):
    """Show extraction quality with color-coded indicator, engine info, and pipeline steps."""
    if extraction_method:
        st.info(f"🔧 Extraction Engine: {extraction_method}")
    if pipeline_steps:
        step_icons = {
            "file_loaded": "📄",
            "opencv_preprocessed": "🖼️",
            "pdfplumber_tables": "📊",
            "img2table_tables": "📊",
            "paddleocr_layout": "📐",
            "sarvam_ocr": "🇮🇳",
        }
        step_labels = []
        for step in pipeline_steps:
            icon = step_icons.get(step, "✅")
            step_labels.append(f"{icon} {step}")
        st.caption(f"Pipeline: {' → '.join(step_labels)}")
    if quality_score is None:
        return
    pct = f"{quality_score:.0%}"
    if quality_score >= 0.85:
        st.success(f"Extraction Quality: {pct} — High confidence")
    elif quality_score >= 0.70:
        st.warning(f"Extraction Quality: {pct} — Some fields may need review")
    elif quality_score >= 0.50:
        st.warning(f"Extraction Quality: {pct} — Low confidence, review carefully")
    else:
        st.error(f"Extraction Quality: {pct} — Very low confidence, consider re-uploading with a clearer image")


def render_block_view(blocks, doc_name):
    """Render Sarvam blocks with layout labels, confidence colors, and HTML tables."""
    sorted_blocks = sorted(blocks, key=lambda b: (b.get("_page_num", 1), b.get("reading_order", 0)))

    tag_labels = {
        "table": "Table",
        "paragraph": "Paragraph",
        "header": "Header",
        "image": "Image",
    }

    for i, block in enumerate(sorted_blocks):
        layout_tag = block.get("layout_tag", "unknown")
        confidence = block.get("confidence", 0)
        text = block.get("text", "").strip()

        # Skip image blocks and empty blocks
        if layout_tag == "image" or not text:
            continue

        # Block header: layout tag + confidence
        col_label, col_conf = st.columns([4, 1])
        with col_label:
            label = tag_labels.get(layout_tag, layout_tag.title())
            st.markdown(f"**{label}**")
        with col_conf:
            if confidence >= 0.85:
                st.markdown(f":green[{confidence:.0%}]")
            elif confidence >= 0.70:
                st.markdown(f":orange[{confidence:.0%}]")
            else:
                st.markdown(f":red[{confidence:.0%}]")

        # Render content based on layout type
        if layout_tag == "table":
            st.markdown(text, unsafe_allow_html=True)
        elif layout_tag == "header":
            clean_header = re.sub(r'<[^>]+>', '', text)
            st.markdown(f"### {clean_header}")
        else:
            st.markdown(text)

        st.markdown("---")


def render_document_preview(parsed_result, doc_name, compact=False):
    """Render document preview: pipeline info, blocks/text, and dedicated tables."""
    quality_score = parsed_result.get("quality_score")
    blocks = parsed_result.get("sarvam_blocks")
    extraction_method = parsed_result.get("extraction_method")
    pipeline_steps = parsed_result.get("pipeline_steps")
    tables_html = parsed_result.get("tables_html", [])

    render_quality_banner(quality_score, doc_name, extraction_method, pipeline_steps)

    # --- Show Sarvam block view if available ---
    if blocks:
        render_block_view(blocks, doc_name)

        low_blocks = parsed_result.get("low_confidence_blocks") or []
        if low_blocks:
            with st.expander(f"⚠️ Low Confidence Blocks ({len(low_blocks)})"):
                for block in low_blocks:
                    tag = block.get("layout_tag", "unknown")
                    conf = block.get("confidence", 0)
                    snippet = block.get("text", "")[:120]
                    st.warning(f"**{tag.title()}** (confidence: {conf:.0%}): {snippet}...")
    else:
        # No Sarvam blocks — show raw text (from OpenAI / PaddleOCR)
        content = parsed_result.get("content", "")
        if compact:
            content = content[:500] + ("..." if len(content) > 500 else "")

        if "<table" in content.lower():
            st.markdown(content, unsafe_allow_html=True)
        else:
            height = 100 if compact else 400
            st.text_area("Extracted Text", content, height=height, disabled=True, key=f"preview_plain_{doc_name}")

    # --- Show dedicated table extractions (from pdfplumber / img2table) ---
    if tables_html:
        with st.expander(f"📊 Extracted Tables ({len(tables_html)})", expanded=True):
            for idx, table_html in enumerate(tables_html):
                st.markdown(f"**Table {idx + 1}**")
                st.markdown(table_html, unsafe_allow_html=True)
                st.markdown("---")


def main():
    st.set_page_config(
        page_title="Document Uploader",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("Document Upload")
    st.markdown("Upload the required documents below:")
    
    # Define required documents for validation
    required_documents = {
        "Annexure": {"uploaded": False, "file": None, "parsed_content": None, "extracted_data": None},
        "PAN": {"uploaded": False, "file": None, "parsed_content": None, "extracted_data": None},
        "Passport": {"uploaded": False, "file": None, "parsed_content": None, "extracted_data": None}
    }
    
    # Initialize manual verification state
    if 'manual_verifications' not in st.session_state:
        st.session_state.manual_verifications = {}
    
    # Initialize session state (deep copy to avoid shared inner dict references)
    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = copy.deepcopy(required_documents)
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Document upload section — each doc is isolated via session state
        for doc_name in required_documents.keys():
            uploaded_file = st.file_uploader(
                f"{doc_name}",
                type=['pdf', 'jpg', 'jpeg', 'png'],
                key=f"uploader_{doc_name}"
            )

            if uploaded_file is not None:
                # Check if this is a new file or already processed
                current_file_info = st.session_state.uploaded_documents[doc_name]
                needs_processing = (
                    not current_file_info["uploaded"]
                    or current_file_info["file"] is None
                    or current_file_info["file"].name != uploaded_file.name
                    or current_file_info["parsed_content"] is None
                )

                if needs_processing:
                    st.session_state.uploaded_documents[doc_name]["uploaded"] = True
                    st.session_state.uploaded_documents[doc_name]["file"] = uploaded_file
                    # Clear stale data for THIS doc before re-processing
                    st.session_state.uploaded_documents[doc_name]["parsed_content"] = None
                    st.session_state.uploaded_documents[doc_name]["extracted_data"] = None

                    # Parse document content
                    with st.spinner(f"Processing {doc_name}..."):
                        result = parse_document(uploaded_file)
                        st.session_state.uploaded_documents[doc_name]["parsed_content"] = result

                        # Extract structured data if text extraction was successful
                        if result["status"] == "success":
                            with st.spinner(f"Extracting data from {doc_name}..."):
                                data_extractor = get_data_extractor()
                                if data_extractor:
                                    extracted_data = data_extractor.extract_data_by_document_type(
                                        result["content"],
                                        doc_name
                                    )
                                    st.session_state.uploaded_documents[doc_name]["extracted_data"] = extracted_data
                else:
                    # File already processed, just update the file reference
                    st.session_state.uploaded_documents[doc_name]["file"] = uploaded_file

                # --- Always read THIS doc's result from session state (never from a local var) ---
                doc_parsed = st.session_state.uploaded_documents[doc_name]["parsed_content"]

                if doc_parsed is not None:
                    # Show success/warning banner
                    if doc_parsed["status"] == "success":
                        engine = doc_parsed.get("extraction_method", "")
                        used_fallback = doc_parsed.get("used_fallback", False)
                        quality = doc_parsed.get("quality_score")

                        if used_fallback:
                            st.warning(f"⚠️ {doc_name} processed using OpenAI fallback (Sarvam had low accuracy)")
                        elif quality is not None and quality < 0.50:
                            st.error(
                                f"❌ {doc_name} extracted but quality is very low ({quality:.0%}). "
                                "Consider re-uploading with a clearer scan or higher resolution."
                            )
                        elif quality is not None and quality < 0.70:
                            st.warning(
                                f"⚠️ {doc_name} processed but some content may be inaccurate ({quality:.0%} confidence). "
                                "Review extracted data carefully."
                            )
                        else:
                            st.success(f"✅ {doc_name} processed successfully via {engine}")
                    else:
                        st.error(f"❌ {doc_name} processing failed: {doc_parsed.get('content', '')}")

                    # Display file details for THIS document
                    st.markdown(f"**File Details:**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.text(f"Name: {uploaded_file.name}")
                        st.text(f"Size: {uploaded_file.size / 1024:.2f} KB")
                    with col_b:
                        st.text(f"Type: {uploaded_file.type}")
                        st.text(f"Status: {doc_parsed['status']}")

                    # Show parsing preview for THIS document only
                    if doc_parsed["status"] == "success" and doc_parsed.get("content"):
                        with st.expander(f"Preview {doc_name}", expanded=False):
                            render_document_preview(doc_parsed, f"inline_{doc_name}", compact=True)
                else:
                    st.info(f"⏳ {doc_name} is being processed...")

            else:
                st.session_state.uploaded_documents[doc_name]["uploaded"] = False
                st.session_state.uploaded_documents[doc_name]["file"] = None
                st.session_state.uploaded_documents[doc_name]["parsed_content"] = None
                st.session_state.uploaded_documents[doc_name]["extracted_data"] = None

            st.markdown("---")
    
    with col2:
        # Status panel
        st.subheader("Upload Status")
        
        # Calculate completion status
        uploaded_count = sum(1 for doc in st.session_state.uploaded_documents.values() if doc["uploaded"])
        total_count = len(required_documents)
        completion_percentage = (uploaded_count / total_count) * 100
        
        # Progress bar
        st.progress(completion_percentage / 100)
        st.write(f"{uploaded_count}/{total_count} documents uploaded")
        
        # Document checklist
        for doc_name, doc_info in st.session_state.uploaded_documents.items():
            if doc_info["uploaded"]:
                st.markdown(f"✅ {doc_name}")
            else:
                st.markdown(f"◯ {doc_name}")
        
        # Completion status
        if uploaded_count == total_count:
            st.success("All documents uploaded!")
            
            # Show download/save option
            if st.button("Save Documents", type="primary"):
                save_documents(st.session_state.uploaded_documents)
    
    # Document content display section
    if uploaded_count > 0:
        st.markdown("---")
        st.subheader("Document Contents")
    
        # Check if any documents have been parsed
        parsed_docs = {doc_name: doc_info for doc_name, doc_info in st.session_state.uploaded_documents.items() 
                       if doc_info["uploaded"] and doc_info["parsed_content"]}
        
        if parsed_docs:
            # Create tabs for each parsed document
            tab_names = list(parsed_docs.keys())
            tabs = st.tabs(tab_names)
            
            for i, (doc_name, doc_info) in enumerate(parsed_docs.items()):
                with tabs[i]:
                    parsed_content = doc_info["parsed_content"]

                    # Document content
                    if parsed_content["status"] == "success":
                        render_document_preview(parsed_content, f"tab_{doc_name}")

                        # Download button for raw text
                        st.download_button(
                            label=f"Download {doc_name} Text",
                            data=parsed_content["content"],
                            file_name=f"{doc_name}_extracted.txt",
                            mime="text/plain",
                            key=f"download_{doc_name}"
                        )
                    else:
                        st.error(f"Failed to parse {doc_name}")
        
        # Display extracted structured data
        st.markdown("---")
        st.subheader("Extracted Data")
        
        extracted_docs = {doc_name: doc_info for doc_name, doc_info in st.session_state.uploaded_documents.items() 
                         if doc_info["uploaded"] and doc_info.get("extracted_data")}
        
        if extracted_docs:
            # Create tabs for each document's extracted data
            tab_names = list(extracted_docs.keys())
            data_tabs = st.tabs([f"{name} Data" for name in tab_names])
            
            for i, (doc_name, doc_info) in enumerate(extracted_docs.items()):
                with data_tabs[i]:
                    extracted_data = doc_info["extracted_data"]
                    
                    if extracted_data["status"] == "success":
                        data = extracted_data["data"]
                        
                        # Display extracted fields in a nice format
                        st.markdown(f"**{doc_name} Information:**")
                        
                        # Create two columns for better layout
                        col_left, col_right = st.columns(2)
                        
                        fields = list(data.items())
                        mid_point = len(fields) // 2
                        
                        with col_left:
                            for key, value in fields[:mid_point]:
                                if value:
                                    st.text(f"{key.replace('_', ' ').title()}: {value}")
                                else:
                                    st.text(f"{key.replace('_', ' ').title()}: -")
                        
                        with col_right:
                            for key, value in fields[mid_point:]:
                                if value:
                                    st.text(f"{key.replace('_', ' ').title()}: {value}")
                                else:
                                    st.text(f"{key.replace('_', ' ').title()}: -")
                        
                        # Download structured data as JSON
                        st.download_button(
                            label=f"Download {doc_name} Data (JSON)",
                            data=json.dumps(data, indent=2),
                            file_name=f"{doc_name}_extracted_data.json",
                            mime="application/json",
                            key=f"download_data_{doc_name}"
                        )
                    else:
                        st.error(f"Failed to extract data from {doc_name}: {extracted_data.get('error', 'Unknown error')}")
        
        # Document Validation Section
        st.markdown("---")
        st.subheader("Document Validation")
        
        # Check if we have enough data for validation
        annexure_ready = (
            st.session_state.uploaded_documents.get("Annexure", {}).get("extracted_data") and
            st.session_state.uploaded_documents["Annexure"]["extracted_data"]["status"] == "success"
        )
        
        pan_ready = (
            st.session_state.uploaded_documents.get("PAN", {}).get("extracted_data") and
            st.session_state.uploaded_documents["PAN"]["extracted_data"]["status"] == "success"
        )
        
        passport_ready = (
            st.session_state.uploaded_documents.get("Passport", {}).get("extracted_data") and
            st.session_state.uploaded_documents["Passport"]["extracted_data"]["status"] == "success"
        )
        
        # Show validation status
        col1, col2, col3 = st.columns(3)
        with col1:
            if annexure_ready:
                st.success("✅ Annexure Ready")
            else:
                st.error("❌ Annexure Required")
        
        with col2:
            if pan_ready:
                st.success("✅ PAN Ready")
            else:
                st.info("⚪ PAN Optional")
        
        with col3:
            if passport_ready:
                st.success("✅ Passport Ready")
            else:
                st.info("⚪ Passport Optional")
        
        # Validate button
        validation_possible = annexure_ready and (pan_ready or passport_ready)
        
        if validation_possible:
            if st.button("🔍 Validate Documents", type="primary"):
                st.session_state.show_validation = True
                # Clear any previous validation results to force fresh validation
                if 'validation_results' in st.session_state:
                    del st.session_state.validation_results
            
            # Show validation results if button was clicked
            if st.session_state.get("show_validation", False):
                # Prepare data for validation
                documents_for_validation = {}
                
                for doc_name in ["Annexure", "PAN", "Passport"]:
                    doc_info = st.session_state.uploaded_documents.get(doc_name)
                    if doc_info and doc_info.get("extracted_data") and doc_info["extracted_data"]["status"] == "success":
                        documents_for_validation[doc_name] = doc_info["extracted_data"]
                
                if len(documents_for_validation) >= 2:  # Need at least Annexure + one other document
                    # Check if validation results are already cached
                    if 'validation_results' not in st.session_state:
                        validator = get_document_validator()
                        if validator:
                            with st.spinner("Validating documents..."):
                                st.session_state.validation_results = validator.validate_all_documents(documents_for_validation)
                    
                    validation_results = st.session_state.validation_results
                    
                    if validation_results and validation_results["status"] == "success":
                                                # Display clean validation results
                        for doc_type, validation in validation_results["validations"].items():
                            st.markdown(f"**{validation['document_type']}**")
                            
                            # Create clean validation table
                            validation_data = []
                            for field_name, field_validation in validation["field_validations"].items():
                                if field_validation["match"]:
                                    status = "✅ Valid"
                                elif field_validation.get("requires_manual", False):
                                    status = "⚠️ Need Manual Verification"
                                else:
                                    status = "❌ Not Valid"
                                
                                validation_data.append({
                                    "Variable Name": field_name.replace('_', ' ').title(),
                                    "Status": status
                                })
                            
                            # Display validation table
                            if validation_data:
                                st.table(validation_data)
                            
                            # Manual verification section (only if needed)
                            manual_fields = [f for f, v in validation["field_validations"].items() if v.get("requires_manual", False)]
                            if manual_fields:
                                st.markdown("**Manual Verification Required:**")
                                
                                for field_name in manual_fields:
                                    field_validation = validation["field_validations"][field_name]
                                    verification_key = f"{doc_type}_{field_name}"
                                    
                                    with st.expander(f"{field_name.replace('_', ' ').title()}"):
                                        # Show values being compared
                                        if "clean_values" in field_validation:
                                            clean_vals = field_validation['clean_values']
                                            
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.markdown("**Annexure Value:**")
                                                if 'pan2' in clean_vals:
                                                    st.text(clean_vals['pan2'])
                                                elif 'text2' in clean_vals:
                                                    st.text(clean_vals['text2'])
                                                elif 'date2' in clean_vals:
                                                    st.text(clean_vals['date2'])
                                            
                                            with col2:
                                                st.markdown(f"**{doc_type} Value:**")
                                                if 'pan1' in clean_vals:
                                                    st.text(clean_vals['pan1'])
                                                elif 'text1' in clean_vals:
                                                    st.text(clean_vals['text1'])
                                                elif 'date1' in clean_vals:
                                                    st.text(clean_vals['date1'])
                                        
                                        # Accept/Reject buttons
                                        col_approve, col_reject, col_status = st.columns([1, 1, 2])
                                        
                                        with col_approve:
                                            if st.button("✅ Accept", key=f"approve_{verification_key}"):
                                                st.session_state.manual_verifications[verification_key] = "approved"
                                                st.rerun()
                                        
                                        with col_reject:
                                            if st.button("❌ Reject", key=f"reject_{verification_key}"):
                                                st.session_state.manual_verifications[verification_key] = "rejected"
                                                st.rerun()
                                        
                                        with col_status:
                                            if verification_key in st.session_state.manual_verifications:
                                                decision = st.session_state.manual_verifications[verification_key]
                                                if decision == "approved":
                                                    st.success("✅ Accepted")
                                                elif decision == "rejected":
                                                    st.error("❌ Rejected")
                                            else:
                                                st.info("⏳ Awaiting decision")
                            
                            st.markdown("---")
                            
                        # Download validation report with unique key
                        import time
                        st.download_button(
                            label="📋 Download Validation Report (JSON)",
                            data=json.dumps(validation_results, indent=2),
                            file_name="document_validation_report.json",
                            mime="application/json",
                            key=f"download_validation_report_{int(time.time())}"
                        )
                    else:
                        st.error(f"Validation failed: {validation_results['message']}")
                else:
                    st.error("Document validator not available")
            else:
                st.info("📋 Upload and process Annexure + at least one other document (PAN/Passport) to enable validation")
        else:
            st.info("📋 Upload and process Annexure document to enable validation")


def save_documents(uploaded_documents: Dict):
    """Save uploaded documents to a folder"""
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = "uploaded_documents"
        os.makedirs(upload_dir, exist_ok=True)
        
        saved_files = []
        for doc_name, doc_info in uploaded_documents.items():
            if doc_info["uploaded"] and doc_info["file"] is not None:
                file_path = os.path.join(upload_dir, doc_info["file"].name)
                with open(file_path, "wb") as f:
                    f.write(doc_info["file"].getbuffer())
                saved_files.append(doc_info["file"].name)
        
        if saved_files:
            st.success(f"✅ Successfully saved {len(saved_files)} documents to '{upload_dir}' folder!")
            st.info("Saved files: " + ", ".join(saved_files))
        else:
            st.error("❌ No documents to save!")
            
    except Exception as e:
        st.error(f"❌ Error saving documents: {str(e)}")

if __name__ == "__main__":
    main()
