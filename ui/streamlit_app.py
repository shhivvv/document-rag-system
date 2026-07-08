import streamlit as st
import requests
import os
import time
from pathlib import Path

# Config
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="RAG Document QA System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS Styling (Dark Theme & Glassmorphism)
st.markdown("""
<style>
    /* Main Layout */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1c24 100%);
        color: #e0e0e8;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    .main-title {
        font-size: 2.8rem;
        background: linear-gradient(45deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
        text-align: center;
        font-weight: 800;
    }
    
    /* Cards and Glassmorphism Container */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Chat bubbles */
    .chat-bubble {
        padding: 15px 20px;
        border-radius: 15px;
        margin-bottom: 10px;
        max-width: 85%;
        line-height: 1.5;
    }
    
    .chat-user {
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 2px;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3);
    }
    
    .chat-assistant {
        background: #252836;
        color: #e0e0e8;
        margin-right: auto;
        border-bottom-left-radius: 2px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Metadata and source badge */
    .source-badge {
        display: inline-block;
        background: rgba(0, 242, 254, 0.1);
        border: 1px solid rgba(0, 242, 254, 0.3);
        color: #00f2fe;
        padding: 2px 8px;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 5px;
        margin-top: 5px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.4);
    }
    
    /* Latency indicator */
    .latency-tag {
        font-size: 0.8rem;
        color: #888899;
        text-align: right;
        margin-top: -10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "health_status" not in st.session_state:
    st.session_state.health_status = {}

# Retrieve server health metrics
def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            st.session_state.health_status = r.json()
            return True
    except Exception:
        st.session_state.health_status = {"status": "offline", "groq_api_configured": False, "total_documents": 0, "total_chunks": 0}
    return False

is_online = check_api_health()

# SIDEBAR: Control Panel and File Uploader
with st.sidebar:
    st.markdown("### 🤖 System Dashboard")
    if is_online:
        st.success("API Backend: Connected")
        status_data = st.session_state.health_status
        st.markdown(f"**Total Documents:** `{status_data.get('total_documents', 0)}`")
        st.markdown(f"**Total Text Chunks:** `{status_data.get('total_chunks', 0)}`")
        st.markdown(f"**Embedding Model:** `{status_data.get('embedding_model', 'None')}`")
        if status_data.get("groq_api_configured"):
            st.success("Groq API: Configured")
        else:
            st.warning("Groq API: Missing API Key")
    else:
        st.error("API Backend: Offline")
        st.info("Start the FastAPI server: `uvicorn app.main:app --reload`")

    st.markdown("---")
    st.markdown("### 📤 Upload Documents")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, TXT or Markdown files",
        type=["pdf", "docx", "txt", "md", "markdown"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        if st.button("Process & Index Document", use_container_width=True):
            with st.spinner("Analyzing document structure & indexing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    r = requests.post(f"{API_URL}/upload", files=files, timeout=60)
                    if r.status_code == 200:
                        res = r.json()
                        st.success(f"Indexed successfully! Added {res['chunks_added']} chunks. Skipped {res['duplicates_skipped']} duplicates.")
                        check_api_health()
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {r.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Error connecting to server: {str(e)}")

    st.markdown("---")
    st.markdown("### ⚙️ Retrieval Parameters")
    use_mmr = st.checkbox("Enable MMR Diversity Search", value=False, help="Maximal Marginal Relevance balances keyword similarity and content diversity in retrieval.")
    
    lang_options = ["Auto-Detect", "English", "Hindi", "French", "Spanish", "German", "Japanese", "Marathi"]
    selected_lang = st.selectbox("Force Target Language", options=lang_options)

    if st.button("🗑️ Clear Vector Database", use_container_width=True, type="secondary"):
        try:
            # We don't have a direct route for delete collection in current API specification,
            # but we can implement it as a reset or simply notify the user.
            # Wait, let's write a simple reset script or just make a request.
            # Since the API doesn't expose DELETE /collection, we could just clear the documents or advise.
            st.info("Database clearing must be performed via server administrator or code reset.")
        except Exception as e:
            st.error(str(e))

# MAIN BODY LAYOUT
st.markdown("<div class='main-title'>Document Intelligence Portal</div>", unsafe_allow_html=True)

# Tabs
tab_qa, tab_contradict, tab_documents = st.tabs(["💬 Context QA Assistant", "🔍 Contradiction Detection", "📚 Indexed Documents"])

# TAB 1: Context QA Assistant
with tab_qa:
    # Quick info banner
    if not st.session_state.health_status.get("groq_api_configured", False):
        st.warning("⚠️ LLM (Groq) is not configured. Ask questions will fail. Set GROQ_API_KEY in your .env file.")

    # Chat history viewport
    chat_container = st.container()
    with chat_container:
        for chat in st.session_state.chat_history:
            # User Message
            st.markdown(f"<div class='chat-bubble chat-user'>{chat['question']}</div>", unsafe_allow_html=True)
            # Assistant Message
            st.markdown(f"<div class='chat-bubble chat-assistant'>{chat['answer']}</div>", unsafe_allow_html=True)
            # Latency / Metadata
            st.markdown(f"<div class='latency-tag'>Language: {chat['language']} | Latency: {chat['latency']}s</div>", unsafe_allow_html=True)
            
            # Citations (expandable)
            if chat.get("citations"):
                with st.expander("🔍 View Supporting Citations", expanded=False):
                    for idx, cit in enumerate(chat["citations"]):
                        st.markdown(
                            f"**[{idx+1}] Source:** `{cit['source_file']}` | **Page:** `{cit['page'] or 'N/A'}` | "
                            f"**Match Score:** `{cit.get('similarity_score', 'N/A')}`"
                        )
                        st.caption(f"**Snippet Preview:** *\"{cit['snippet'][:300]}...\"*")
                        st.markdown("---")

    # Spacer
    st.write("")
    
    # Input Area
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([6, 1])
        with col_input:
            user_question = st.text_input("Ask a question about the indexed documents...", placeholder="e.g. What is the policy on maternity leave?", label_visibility="collapsed")
        with col_btn:
            submit_btn = st.form_submit_button("Send 🚀", use_container_width=True)

    if submit_btn and user_question.strip():
        # Prepare payload
        payload = {"question": user_question, "use_mmr": use_mmr}
        if selected_lang != "Auto-Detect":
            payload["target_language"] = selected_lang

        with st.spinner("Searching document index & drafting response..."):
            try:
                r = requests.post(f"{API_URL}/ask", json=payload, timeout=60)
                if r.status_code == 200:
                    res = r.json()
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": res["answer"],
                        "language": res["language"],
                        "latency": res["latency_seconds"],
                        "citations": res["citations"]
                    })
                    st.rerun()
                else:
                    st.error(f"Error answering question: {r.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Network error communicating with API: {str(e)}")

    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("🧹 Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# TAB 2: Contradiction Detection
with tab_contradict:
    st.markdown("### 🔍 Policy Audit & Contradiction Checker")
    st.write("Compare the semantic context of two documents on a specific topic to audit alignments or identify conflicts.")
    
    # Get available documents from health status (or query local folder)
    try:
        # Fetch file names
        # Standard chroma retrieval for names could also be done. We will retrieve filenames from uploaded folder
        # or simply from vector store health metadata if we add file names. Let's make an API call.
        # But wait, we can just look up files in the vector DB metadata using a custom endpoint or fallback.
        # Let's write a simple HTTP request to health to get the names if available.
        # Wait, how does Streamlit know the list of documents?
        # We can read files in `data/documents/` directly using python, because Streamlit runs locally!
        doc_dir = Path("./data/documents")
        available_files = []
        if doc_dir.exists():
            available_files = [f.name for f in doc_dir.iterdir() if f.is_file() and f.name != ".gitkeep"]
    except Exception:
        available_files = []

    if len(available_files) < 2:
        st.info("⚠️ Please upload at least 2 documents in the sidebar to run contradiction auditing.")
    else:
        with st.form("contradict_form"):
            col_doc1, col_doc2 = st.columns(2)
            with col_doc1:
                doc1 = st.selectbox("Select Document 1", options=available_files)
            with col_doc2:
                doc2 = st.selectbox("Select Document 2", options=available_files)
                
            topic = st.text_input("Enter Topic to Compare", placeholder="e.g. Remote work, leave limits, stipend")
            submit_audit = st.form_submit_button("Run Contradiction Audit 🔍")

        if submit_audit:
            if doc1 == doc2:
                st.warning("Please select two different documents to compare.")
            elif not topic.strip():
                st.warning("Please specify a topic for the comparison.")
            else:
                with st.spinner("Analyzing document alignments..."):
                    try:
                        payload = {"document1": doc1, "document2": doc2, "topic": topic}
                        r = requests.post(f"{API_URL}/contradict", json=payload, timeout=60)
                        
                        if r.status_code == 200:
                            res = r.json()
                            status = res["status"]
                            confidence = res["confidence"]
                            
                            # Render beautiful audit results card
                            st.markdown("### Audit Results")
                            
                            if status == "Conflict":
                                st.error(f"🔴 Conflict Detected (Confidence: {confidence:.2%})")
                            elif status == "Partial Conflict":
                                st.warning(f"🟡 Partial Conflict / Discrepancy (Confidence: {confidence:.2%})")
                            else:
                                st.success(f"🟢 Aligned / No Conflict (Confidence: {confidence:.2%})")
                                
                            st.markdown(f"<div class='glass-card'><strong>Reasoning:</strong><br>{res['reasoning']}</div>", unsafe_allow_html=True)
                            
                            col_ev1, col_ev2 = st.columns(2)
                            with col_ev1:
                                st.markdown(f"#### Evidence from `{doc1}`")
                                st.markdown(f"<div class='glass-card' style='font-size:0.9rem; max-height: 400px; overflow-y: auto;'>{res['document1_evidence']}</div>", unsafe_allow_html=True)
                            with col_ev2:
                                st.markdown(f"#### Evidence from `{doc2}`")
                                st.markdown(f"<div class='glass-card' style='font-size:0.9rem; max-height: 400px; overflow-y: auto;'>{res['document2_evidence']}</div>", unsafe_allow_html=True)
                        else:
                            st.error(f"Audit failed: {r.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Error contacting API backend: {str(e)}")

# TAB 3: Indexed Documents
with tab_documents:
    st.markdown("### 📚 Managed Document Library")
    
    doc_dir = Path("./data/documents")
    if doc_dir.exists():
        files = [f for f in doc_dir.iterdir() if f.is_file() and f.name != ".gitkeep"]
        if not files:
            st.info("No documents are currently stored in the library.")
        else:
            for f in files:
                size_kb = f.stat().st_size / 1024
                st.markdown(f"""
                <div class='glass-card' style='margin-bottom: 10px;'>
                    <span style='font-size: 1.1rem; font-weight: 600;'>📄 {f.name}</span><br>
                    <span style='color: #888899; font-size: 0.85rem;'>Path: {f.absolute()} | Size: {size_kb:.2f} KB</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No documents are currently stored in the library.")
