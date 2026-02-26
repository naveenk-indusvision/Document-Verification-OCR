import copy
import streamlit as st
import os
import re
import pandas as pd
import json
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from data_extractor import DataExtractor
from document_validator import DocumentValidator

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


@st.cache_resource
def get_document_processor():
    """Get or create the document processing pipeline (OpenAI-only)."""
    try:
        return DocumentProcessor(openai_api_key=OPENAI_API_KEY)
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


def parse_document(uploaded_file, doc_type: str = None) -> Dict:
    """
    Extract text via: PDF/Image → 150 DPI images → single GPT-4o Vision call.
    When doc_type is provided, OCR + structured extraction happen in one call.
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
            "extraction_method": None,
        }

    result = processor.process_document(uploaded_file, doc_type=doc_type)

    return {
        "filename": uploaded_file.name,
        "file_type": file_type,
        "file_size": f"{uploaded_file.size / 1024:.2f} KB",
        "status": result.get("status", "error"),
        "content": result.get("content", ""),
        "extraction_method": result.get("extraction_method", ""),
        "used_fallback": result.get("used_fallback", False),
        "tables_html": result.get("tables_html", []),
        "pipeline_steps": result.get("pipeline_steps", []),
        "quality_score": result.get("quality_score"),
        "pages_processed": result.get("pages_processed", 0),
        "total_tokens": result.get("total_tokens", 0),
        "sarvam_blocks": None,
        "low_confidence_blocks": None,
        "paddle_text": None,
        "extracted_data": result.get("extracted_data"),
    }


def _clean_extracted_pan(extracted: Dict, doc_name: str):
    """Clean PAN number in combined extraction results."""
    if (
        extracted
        and extracted.get("status") == "success"
        and extracted.get("data", {}).get("pan_number")
    ):
        import re
        extracted["data"]["pan_number"] = re.sub(
            r'[^A-Z0-9]', '', str(extracted["data"]["pan_number"]).upper()
        )


def process_single_document(doc_name: str, uploaded_file) -> Dict:
    """Process one document: OCR + structured extraction. Thread-safe."""
    parsed = parse_document(uploaded_file, doc_type=doc_name)
    extracted = None
    if parsed["status"] == "success":
        if parsed.get("extracted_data"):
            # Combined OCR+extraction already done in one call — no extra API call
            extracted = parsed["extracted_data"]
            _clean_extracted_pan(extracted, doc_name)
        else:
            # Fallback: separate extraction call via GPT-4o-mini
            data_extractor = get_data_extractor()
            if data_extractor:
                extracted = data_extractor.extract_data_by_document_type(
                    parsed["content"], doc_name
                )
    return {"parsed": parsed, "extracted": extracted}


def process_all_documents_parallel(documents: Dict) -> Dict[str, Dict]:
    """Process all uploaded documents in parallel using ThreadPoolExecutor."""
    results = {}

    # Collect docs that need processing
    to_process = {}
    for doc_name, doc_info in documents.items():
        if doc_info.get("file") is not None:
            to_process[doc_name] = doc_info["file"]

    if not to_process:
        return results

    with ThreadPoolExecutor(max_workers=min(3, len(to_process))) as executor:
        futures = {
            executor.submit(process_single_document, name, f): name
            for name, f in to_process.items()
        }
        for future in as_completed(futures):
            doc_name = futures[future]
            try:
                results[doc_name] = future.result()
            except Exception as e:
                results[doc_name] = {
                    "parsed": {
                        "status": "error",
                        "content": f"Processing error: {str(e)}",
                        "extraction_method": "Failed",
                    },
                    "extracted": None,
                }

    return results


def render_quality_banner(quality_score, doc_name, extraction_method=None, pipeline_steps=None):
    """Show extraction quality info (kept minimal)."""
    pass


def render_document_preview(parsed_result, doc_name, compact=False):
    """Render document preview: pipeline info and extracted text."""
    extraction_method = parsed_result.get("extraction_method")
    pipeline_steps = parsed_result.get("pipeline_steps")
    quality_score = parsed_result.get("quality_score")

    render_quality_banner(quality_score, doc_name, extraction_method, pipeline_steps)

    content = parsed_result.get("content", "")
    if compact:
        content = content[:500] + ("..." if len(content) > 500 else "")

    if "<table" in content.lower():
        st.markdown(content, unsafe_allow_html=True)
    else:
        height = 100 if compact else 400
        st.text_area("Extracted Text", content, height=height, disabled=True, key=f"preview_plain_{doc_name}")


def main():
    st.set_page_config(
        page_title="Document Uploader",
        page_icon="doc",
        layout="wide"
    )

    st.title("Document Upload")
    st.markdown("Upload the required documents below:")

    # Define required documents
    required_documents = {
        "Annexure": {"uploaded": False, "file": None, "parsed_content": None, "extracted_data": None},
        "PAN": {"uploaded": False, "file": None, "parsed_content": None, "extracted_data": None},
        "Passport": {"uploaded": False, "file": None, "parsed_content": None, "extracted_data": None}
    }

    if 'manual_verifications' not in st.session_state:
        st.session_state.manual_verifications = {}

    if 'uploaded_documents' not in st.session_state:
        st.session_state.uploaded_documents = copy.deepcopy(required_documents)

    # Create two columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # File uploaders — just collect files, no auto-processing
        for doc_name in required_documents.keys():
            uploaded_file = st.file_uploader(
                f"{doc_name}",
                type=['pdf', 'jpg', 'jpeg', 'png'],
                key=f"uploader_{doc_name}"
            )

            if uploaded_file is not None:
                st.session_state.uploaded_documents[doc_name]["uploaded"] = True
                st.session_state.uploaded_documents[doc_name]["file"] = uploaded_file

                # Show file info
                st.caption(f"{uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")

                # Show previously processed results if available
                doc_parsed = st.session_state.uploaded_documents[doc_name].get("parsed_content")
                if doc_parsed and doc_parsed["status"] == "success":
                    st.success(f"{doc_name} processed successfully")
                    with st.expander(f"Preview {doc_name}", expanded=False):
                        render_document_preview(doc_parsed, f"inline_{doc_name}", compact=True)
                elif doc_parsed and doc_parsed["status"] == "error":
                    st.error(f"{doc_name} failed: {doc_parsed.get('content', '')}")
            else:
                st.session_state.uploaded_documents[doc_name]["uploaded"] = False
                st.session_state.uploaded_documents[doc_name]["file"] = None
                st.session_state.uploaded_documents[doc_name]["parsed_content"] = None
                st.session_state.uploaded_documents[doc_name]["extracted_data"] = None

            st.markdown("---")

        # --- Process All Documents button ---
        uploaded_files = {
            name: info for name, info in st.session_state.uploaded_documents.items()
            if info.get("file") is not None
        }
        has_unprocessed = any(
            info.get("parsed_content") is None
            for info in uploaded_files.values()
        )

        if uploaded_files and has_unprocessed:
            if st.button("Process All Documents", type="primary", use_container_width=True):
                with st.spinner(f"Processing {len(uploaded_files)} document(s) in parallel..."):
                    results = process_all_documents_parallel(st.session_state.uploaded_documents)

                    for doc_name, result in results.items():
                        st.session_state.uploaded_documents[doc_name]["parsed_content"] = result["parsed"]
                        st.session_state.uploaded_documents[doc_name]["extracted_data"] = result["extracted"]

                st.rerun()

    with col2:
        # Status panel
        st.subheader("Upload Status")

        uploaded_count = sum(1 for doc in st.session_state.uploaded_documents.values() if doc["uploaded"])
        total_count = len(required_documents)
        completion_percentage = (uploaded_count / total_count) * 100

        st.progress(completion_percentage / 100)
        st.write(f"{uploaded_count}/{total_count} documents uploaded")

        for doc_name, doc_info in st.session_state.uploaded_documents.items():
            parsed = doc_info.get("parsed_content")
            if parsed and parsed["status"] == "success":
                st.markdown(f"Done: {doc_name}")
            elif doc_info["uploaded"]:
                st.markdown(f"Uploaded: {doc_name}")
            else:
                st.markdown(f"Pending: {doc_name}")

        if uploaded_count == total_count:
            st.success("All documents uploaded!")
            if st.button("Save Documents", type="primary"):
                save_documents(st.session_state.uploaded_documents)

    # Document content display section
    parsed_docs = {
        doc_name: doc_info
        for doc_name, doc_info in st.session_state.uploaded_documents.items()
        if doc_info["uploaded"] and doc_info.get("parsed_content")
    }

    if parsed_docs:
        st.markdown("---")
        st.subheader("Document Contents")

        tab_names = list(parsed_docs.keys())
        tabs = st.tabs(tab_names)

        for i, (doc_name, doc_info) in enumerate(parsed_docs.items()):
            with tabs[i]:
                parsed_content = doc_info["parsed_content"]

                if parsed_content["status"] == "success":
                    render_document_preview(parsed_content, f"tab_{doc_name}")
                    st.download_button(
                        label=f"Download {doc_name} Text",
                        data=parsed_content["content"],
                        file_name=f"{doc_name}_extracted.txt",
                        mime="text/plain",
                        key=f"download_{doc_name}"
                    )
                else:
                    st.error(f"Failed to parse {doc_name}")

        # Extracted Data section
        st.markdown("---")
        st.subheader("Extracted Data")

        extracted_docs = {
            doc_name: doc_info
            for doc_name, doc_info in st.session_state.uploaded_documents.items()
            if doc_info["uploaded"] and doc_info.get("extracted_data")
        }

        if extracted_docs:
            tab_names = list(extracted_docs.keys())
            data_tabs = st.tabs([f"{name} Data" for name in tab_names])

            for i, (doc_name, doc_info) in enumerate(extracted_docs.items()):
                with data_tabs[i]:
                    extracted_data = doc_info["extracted_data"]

                    if extracted_data["status"] == "success":
                        data = extracted_data["data"]
                        st.markdown(f"**{doc_name} Information:**")

                        col_left, col_right = st.columns(2)
                        fields = list(data.items())
                        mid_point = len(fields) // 2

                        with col_left:
                            for key, value in fields[:mid_point]:
                                st.text(f"{key.replace('_', ' ').title()}: {value or '-'}")

                        with col_right:
                            for key, value in fields[mid_point:]:
                                st.text(f"{key.replace('_', ' ').title()}: {value or '-'}")

                        st.download_button(
                            label=f"Download {doc_name} Data (JSON)",
                            data=json.dumps(data, indent=2),
                            file_name=f"{doc_name}_extracted_data.json",
                            mime="application/json",
                            key=f"download_data_{doc_name}"
                        )
                    else:
                        st.error(f"Failed to extract data: {extracted_data.get('error', 'Unknown error')}")

        # Document Validation Section
        st.markdown("---")
        st.subheader("Document Validation")

        annexure_ready = (
            st.session_state.uploaded_documents.get("Annexure", {}).get("extracted_data")
            and st.session_state.uploaded_documents["Annexure"]["extracted_data"]["status"] == "success"
        )
        pan_ready = (
            st.session_state.uploaded_documents.get("PAN", {}).get("extracted_data")
            and st.session_state.uploaded_documents["PAN"]["extracted_data"]["status"] == "success"
        )
        passport_ready = (
            st.session_state.uploaded_documents.get("Passport", {}).get("extracted_data")
            and st.session_state.uploaded_documents["Passport"]["extracted_data"]["status"] == "success"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            if annexure_ready:
                st.success("Annexure Ready")
            else:
                st.error("Annexure Required")
        with col2:
            if pan_ready:
                st.success("PAN Ready")
            else:
                st.info("PAN Optional")
        with col3:
            if passport_ready:
                st.success("Passport Ready")
            else:
                st.info("Passport Optional")

        validation_possible = annexure_ready and (pan_ready or passport_ready)

        if validation_possible:
            if st.button("Validate Documents", type="primary"):
                st.session_state.show_validation = True
                if 'validation_results' in st.session_state:
                    del st.session_state.validation_results

            if st.session_state.get("show_validation", False):
                documents_for_validation = {}
                for doc_name in ["Annexure", "PAN", "Passport"]:
                    doc_info = st.session_state.uploaded_documents.get(doc_name)
                    if doc_info and doc_info.get("extracted_data") and doc_info["extracted_data"]["status"] == "success":
                        documents_for_validation[doc_name] = doc_info["extracted_data"]

                if len(documents_for_validation) >= 2:
                    if 'validation_results' not in st.session_state:
                        validator = get_document_validator()
                        if validator:
                            with st.spinner("Validating documents..."):
                                st.session_state.validation_results = validator.validate_all_documents(documents_for_validation)

                    validation_results = st.session_state.validation_results

                    if validation_results and validation_results["status"] == "success":
                        for doc_type, validation in validation_results["validations"].items():
                            st.markdown(f"**{validation['document_type']}**")

                            validation_data = []
                            for field_name, field_validation in validation["field_validations"].items():
                                if field_validation["match"]:
                                    status = "Valid"
                                elif field_validation.get("requires_manual", False):
                                    status = "Need Manual Verification"
                                else:
                                    status = "Not Valid"
                                validation_data.append({
                                    "Variable Name": field_name.replace('_', ' ').title(),
                                    "Status": status
                                })

                            if validation_data:
                                st.table(validation_data)

                            # Manual verification section
                            manual_fields = [
                                f for f, v in validation["field_validations"].items()
                                if v.get("requires_manual", False)
                            ]
                            if manual_fields:
                                st.markdown("**Manual Verification Required:**")
                                for field_name in manual_fields:
                                    field_validation = validation["field_validations"][field_name]
                                    verification_key = f"{doc_type}_{field_name}"

                                    with st.expander(f"{field_name.replace('_', ' ').title()}"):
                                        if "clean_values" in field_validation:
                                            clean_vals = field_validation['clean_values']
                                            c1, c2 = st.columns(2)
                                            with c1:
                                                st.markdown("**Annexure Value:**")
                                                if 'pan2' in clean_vals:
                                                    st.text(clean_vals['pan2'])
                                                elif 'text2' in clean_vals:
                                                    st.text(clean_vals['text2'])
                                                elif 'date2' in clean_vals:
                                                    st.text(clean_vals['date2'])
                                            with c2:
                                                st.markdown(f"**{doc_type} Value:**")
                                                if 'pan1' in clean_vals:
                                                    st.text(clean_vals['pan1'])
                                                elif 'text1' in clean_vals:
                                                    st.text(clean_vals['text1'])
                                                elif 'date1' in clean_vals:
                                                    st.text(clean_vals['date1'])

                                        col_approve, col_reject, col_status = st.columns([1, 1, 2])
                                        with col_approve:
                                            if st.button("Accept", key=f"approve_{verification_key}"):
                                                st.session_state.manual_verifications[verification_key] = "approved"
                                                st.rerun()
                                        with col_reject:
                                            if st.button("Reject", key=f"reject_{verification_key}"):
                                                st.session_state.manual_verifications[verification_key] = "rejected"
                                                st.rerun()
                                        with col_status:
                                            if verification_key in st.session_state.manual_verifications:
                                                decision = st.session_state.manual_verifications[verification_key]
                                                if decision == "approved":
                                                    st.success("Accepted")
                                                elif decision == "rejected":
                                                    st.error("Rejected")
                                            else:
                                                st.info("Awaiting decision")

                            st.markdown("---")

                        import time
                        st.download_button(
                            label="Download Validation Report (JSON)",
                            data=json.dumps(validation_results, indent=2),
                            file_name="document_validation_report.json",
                            mime="application/json",
                            key=f"download_validation_report_{int(time.time())}"
                        )
                    else:
                        st.error(f"Validation failed: {validation_results.get('message', 'Unknown error')}")
                else:
                    st.error("Document validator not available")
        else:
            st.info("Upload and process Annexure + at least one other document (PAN/Passport) to enable validation")


def save_documents(uploaded_documents: Dict):
    """Save uploaded documents to a folder"""
    try:
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
            st.success(f"Successfully saved {len(saved_files)} documents to '{upload_dir}' folder!")
            st.info("Saved files: " + ", ".join(saved_files))
        else:
            st.error("No documents to save!")

    except Exception as e:
        st.error(f"Error saving documents: {str(e)}")


if __name__ == "__main__":
    main()
