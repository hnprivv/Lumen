import streamlit as st
import streamlit.components.v1 as components
import time
import os
import tomllib

def get_max_upload_size() -> int:
    config_path = os.path.join(os.path.dirname(__file__), ".streamlit", "config.toml")
    try:
        with open(config_path, "rb") as f:
            config = tomllib.load(f)
        return config.get("server", {}).get("maxUploadSize", 200)
    except Exception:
        return 200

MAX_UPLOAD_MB = get_max_upload_size()
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from google import genai
from google.genai import types

# ── Page config (must be first) ───────────────────────────────────────────────
st.set_page_config(page_title="Lumen", page_icon="icon.png", layout="centered")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide sidebar and hamburger */
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none !important; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* App background */
.stApp { background-color: #0d1117; color: #e6edf3; }

/* Landing */
.landing-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 80px 20px 40px;
    text-align: center;
}
.lumen-title {
    font-size: 64px;
    font-weight: 800;
    color: #f0883e;
    letter-spacing: -3px;
    line-height: 1;
    margin-bottom: 12px;
}
.lumen-sub {
    font-size: 17px;
    color: #8b949e;
    margin-bottom: 48px;
    max-width: 400px;
}
.upload-card {
    background: #161b22;
    border: 2px dashed #30363d;
    border-radius: 16px;
    padding: 40px 48px;
    width: 100%;
    max-width: 440px;
    transition: border-color 0.2s;
    text-align: center;
    cursor: pointer;
}
.upload-card:hover { border-color: #f0883e; }
.upload-icon { font-size: 40px; margin-bottom: 12px; }
.upload-label { font-size: 15px; color: #e6edf3; margin-bottom: 4px; }
.upload-hint { font-size: 13px; color: #8b949e; }

/* Hide file uploader button and instructions; keep the input alive */
[data-testid="stFileUploaderDropzone"] button { display: none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    min-height: 0 !important;
}
[data-testid="stFileUploader"] { background: transparent !important; }

/* Chat header */
.chat-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding-bottom: 14px;
    border-bottom: 1px solid #21262d;
    margin-bottom: 20px;
}
.chat-badge {
    font-size: 20px;
    font-weight: 800;
    color: #f0883e;
    letter-spacing: -1px;
}
.chat-doc {
    font-size: 13px;
    color: #8b949e;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 3px 10px;
    font-family: monospace;
}

/* Justify assistant messages */
[data-testid="stChatMessage"] .stMarkdown p { text-align: justify !important; }

/* Page citation */
.source-tag {
    display: inline-block;
    font-size: 12px;
    color: #f0883e;
    background: rgba(240,136,62,0.1);
    border: 1px solid rgba(240,136,62,0.25);
    border-radius: 4px;
    padding: 2px 8px;
    margin-top: 6px;
}

/* Buttons */
.stButton > button {
    background: transparent !important;
    border: 1px solid #30363d !important;
    color: #8b949e !important;
    border-radius: 6px !important;
    font-size: 13px !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    border-color: #f0883e !important;
    color: #f0883e !important;
}

/* Footer */
.lumen-footer {
    text-align: center;
    font-size: 12px;
    color: #484f58;
    padding: 40px 0 20px;
    border-top: 1px solid #21262d;
    margin-top: 40px;
}
.lumen-footer a {
    color: #484f58;
    text-decoration: none;
    transition: color 0.15s;
}
.lumen-footer a:hover { color: #f0883e; }

/* Chat input */
[data-testid="stChatInput"] {
    background-color: transparent !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    outline: none !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #30363d !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    color: #e6edf3 !important;
    caret-color: #e6edf3 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "doc_pages" not in st.session_state:
    st.session_state.doc_pages = 0

# ── Gemini client ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_gemini_client():
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")
    if not api_key:
        st.error("Set GOOGLE_API_KEY in your environment or Streamlit secrets.")
        st.stop()
    return genai.Client(api_key=api_key)

# ── RAG core ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vector_store(uploaded_file) -> tuple:
    reader = PdfReader(uploaded_file)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"page": page_num}))

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=f"doc_{int(time.time())}",
    )
    return vector_db, len(reader.pages)

def build_prompt(query: str, vector_db, history: list) -> tuple[str, list[int]]:
    results = vector_db.similarity_search(query, k=4)
    context = "\n\n".join(
        f"[Page {r.metadata['page']}]: {r.page_content}" for r in results
    )
    pages = sorted({r.metadata["page"] for r in results})

    history_text = ""
    for msg in history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are a strict document analysis assistant. Rules:
1. Answer ONLY using the document context provided below.
2. If the answer is not in the context, say exactly: "I cannot find this in the document."
3. Make sure you always add (PBUH) after the Prophet Muhammad's name.
4. Do not use outside knowledge.
5. Be concise and accurate.

CONVERSATION HISTORY:
{history_text}
DOCUMENT CONTEXT:
{context}

USER QUESTION: {query}

ANSWER:"""
    return prompt, pages

def stream_response(prompt: str):
    client = get_gemini_client()
    for chunk in client.models.generate_content_stream(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=3000,
        ),
    ):
        if chunk.text:
            yield chunk.text

# ── Landing page ──────────────────────────────────────────────────────────────
def show_landing():
    st.markdown("""
    <div class="landing-wrap">
        <div class="lumen-title">Lumen</div>
        <div class="lumen-sub">Upload a PDF and ask questions about it.<br>Get precise answers with page-level sources.</div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown(
            f'<div class="upload-card" id="upload-card">'
            f'<div class="upload-icon">📄</div>'
            f'<div class="upload-label">Drop your PDF here</div>'
            f'<div class="upload-hint">or click to browse</div>'
            f'<div class="upload-hint" style="margin-top:10px;">PDF &nbsp;·&nbsp; Max {MAX_UPLOAD_MB} MB</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader("pdf", type=["pdf"], label_visibility="collapsed")

    # Forward clicks on the card to the hidden file input
    components.html("""
    <script>
    (function() {
        function init() {
            try {
                const doc = window.parent.document;
                const card = doc.getElementById("upload-card");
                const input = doc.querySelector('input[type="file"]');
                if (card && input) {
                    card.addEventListener("click", () => input.click());
                } else {
                    setTimeout(init, 200);
                }
            } catch(e) {}
        }
        init();
    })();
    </script>
    """, height=0)

    return uploaded

# ── Chat page ─────────────────────────────────────────────────────────────────
def show_chat():
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(f"""
        <div class="chat-header">
            <span class="chat-badge">Lumen</span>
            <span class="chat-doc">📄 {st.session_state.doc_name} &nbsp;·&nbsp; {st.session_state.doc_pages} pages</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("↩ New", use_container_width=True):
            st.session_state.vector_db = None
            st.session_state.doc_name = None
            st.session_state.doc_pages = 0
            st.session_state.messages = []
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("pages"):
                pages_str = ", ".join(str(p) for p in msg["pages"])
                st.markdown(f'<span class="source-tag">📖 page(s) {pages_str}</span>', unsafe_allow_html=True)

    if query := st.chat_input("Ask a question about the document…"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        prompt, pages = build_prompt(query, st.session_state.vector_db, st.session_state.messages)

        with st.chat_message("assistant"):
            answer = st.write_stream(stream_response(prompt))
            pages_str = ", ".join(str(p) for p in pages)
            st.markdown(f'<span class="source-tag">📖 page(s) {pages_str}</span>', unsafe_allow_html=True)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "pages": pages}
        )

# ── Main ──────────────────────────────────────────────────────────────────────
if not st.session_state.vector_db:
    uploaded = show_landing()
    if uploaded:
        with st.spinner("Indexing document…"):
            vector_db, page_count = build_vector_store(uploaded)
            st.session_state.vector_db = vector_db
            st.session_state.doc_name = uploaded.name
            st.session_state.doc_pages = page_count
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": f"I've read **{uploaded.name}** ({page_count} pages). What would you like to know?",
                }
            ]
        st.rerun()
else:
    show_chat()

st.markdown(
    '<div class="lumen-footer">'
    '© 2026 Lumen by <a href="https://github.com/hnprivv">Huzaifa Najam</a>.<br><br>'
    'Relevant excerpts from your document are sent to Google Gemini for answer generation and are not stored. <br>'
    'Do not upload documents containing sensitive personal data.'
    '</div>',
    unsafe_allow_html=True,
)
