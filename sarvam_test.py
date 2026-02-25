"""
Sarvam Document Intelligence - Standalone test page.
Run with: streamlit run sarvam_test.py
Kept separate from main document_uploader for testing.
"""
import streamlit as st
import tempfile
import zipfile
import os
from pathlib import Path

st.set_page_config(
    page_title="Sarvam Doc Intelligence Test",
    page_icon="📄",
    layout="wide"
)

st.title("Sarvam Document Intelligence – Test")
st.caption("Standalone test page – enter your API key and process documents with Sarvam Vision")

# Configuration sidebar
with st.sidebar:
    st.subheader("Configuration")
    
    api_key = st.text_input(
        "Sarvam API Key",
        type="password",
        placeholder="Enter your API key",
        help="Get your key from dashboard.sarvam.ai"
    )
    
    st.divider()
    st.subheader("Processing options")
    
    languages = {
        "Hindi": "hi-IN",
        "English": "en-IN",
        "Bengali": "bn-IN",
        "Gujarati": "gu-IN",
        "Kannada": "kn-IN",
        "Malayalam": "ml-IN",
        "Marathi": "mr-IN",
        "Odia": "or-IN",
        "Punjabi": "pa-IN",
        "Tamil": "ta-IN",
        "Telugu": "te-IN",
        "Urdu": "ur-IN",
        "Assamese": "as-IN",
        "Bodo": "bodo-IN",
        "Dogri": "doi-IN",
        "Kashmiri": "ks-IN",
        "Konkani": "kok-IN",
        "Maithili": "mai-IN",
        "Manipuri": "mni-IN",
        "Nepali": "ne-IN",
        "Sanskrit": "sa-IN",
        "Santali": "sat-IN",
        "Sindhi": "sd-IN",
    }
    
    selected_lang_label = st.selectbox(
        "Document language",
        options=list(languages.keys()),
        index=0,
        help="Primary language of the document (BCP-47)"
    )
    language = languages[selected_lang_label]
    
    output_formats = {
        "Markdown": "md",
        "HTML": "html",
        "JSON": "json",
    }
    
    selected_format_label = st.selectbox(
        "Output format",
        options=list(output_formats.keys()),
        index=0,
        help="Output delivered as ZIP (html/md/json files)"
    )
    output_format = output_formats[selected_format_label]
    
    st.info("Powered by Sarvam Vision (3B multimodal)")

# Main content
if not api_key:
    st.warning("Enter your Sarvam API key in the sidebar to start.")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload document",
    type=["pdf", "png", "jpg", "jpeg", "zip"],
    help="PDF, PNG, JPEG, or ZIP (flat archive of page images)"
)

if uploaded_file:
    col_info, col_process = st.columns([1, 1])
    
    with col_info:
        st.write("**File:**", uploaded_file.name)
        st.write("**Size:**", f"{uploaded_file.size / 1024:.1f} KB")
    
    with col_process:
        process_btn = st.button("Process with Sarvam", type="primary", use_container_width=True)

    if process_btn:
        try:
            with st.spinner("Processing document..."):
                from sarvamai import SarvamAI
                from sarvamai.core.api_error import ApiError
                
                client = SarvamAI(api_subscription_key=api_key)
                
                job = client.document_intelligence.create_job(
                    language=language,
                    output_format=output_format
                )
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    job.upload_file(tmp_path)
                    job.start()
                    status = job.wait_until_complete()
                    
                    output_dir = Path("sarvam_output")
                    output_dir.mkdir(exist_ok=True)
                    output_zip = output_dir / f"sarvam_output_{uploaded_file.name}.zip"
                    job.download_output(str(output_zip))
                    
                    with open(output_zip, "rb") as f:
                        zip_bytes = f.read()
                    
                    st.session_state.sarvam_result = {
                        "status": status.job_state,
                        "metrics": job.get_page_metrics(),
                        "zip_bytes": zip_bytes,
                        "file_name": output_zip.name,
                        "preview_files": [],
                    }
                    try:
                        with zipfile.ZipFile(output_zip, "r") as zf:
                            names = zf.namelist()
                            text_files = [n for n in names if n.endswith((".md", ".html", ".txt", ".json"))]
                            for name in text_files[:3]:
                                content = zf.read(name).decode("utf-8", errors="replace")
                                st.session_state.sarvam_result["preview_files"].append((name, content[:5000] + ("..." if len(content) > 5000 else "")))
                    except Exception:
                        pass
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                        
        except ApiError as e:
            st.error(f"API error {e.status_code}: {e.body}")
        except ImportError:
            st.error("sarvamai package not installed. Run: pip install sarvamai")
        except Exception as e:
            st.exception(e)

if "sarvam_result" in st.session_state:
    r = st.session_state.sarvam_result
    st.success(f"Job completed: **{r['status']}**")
    if r.get("metrics"):
        st.json(r["metrics"])
    st.download_button(
        "Download output (ZIP)",
        data=r["zip_bytes"],
        file_name=r["file_name"],
        mime="application/zip",
        use_container_width=True
    )
    if r.get("preview_files"):
        with st.expander("Output preview"):
            for name, content in r["preview_files"]:
                st.text_area(name, content, height=200, disabled=True, key=f"preview_{name.replace('/', '_')}")
