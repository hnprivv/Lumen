import streamlit as st
import time
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from google import genai
from google.genai import types

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

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

def build_vector_store(uploaded_file):
    reader = PdfReader(uploaded_file)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata={"page": page_num}))

    return Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        collection_name=f"doc_{int(time.time())}",
    )

def answer_question(query: str, vector_db, history: list) -> tuple[str, list[int]]:
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
3. Do not use outside knowledge.
4. Be concise and accurate.

CONVERSATION HISTORY:
{history_text}
DOCUMENT CONTEXT:
{context}

USER QUESTION: {query}

ANSWER:"""

    client = get_gemini_client()
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=3000,
        ),
    )
    return response.text, pages

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="DocChat", page_icon="📄", layout="centered")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("DocChat")
    st.caption("Upload a PDF and ask questions about it.")
    st.divider()

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded:
        if st.session_state.doc_name != uploaded.name:
            with st.spinner("Indexing document…"):
                st.session_state.vector_db = build_vector_store(uploaded)
                st.session_state.doc_name = uploaded.name
                st.session_state.messages = []
            st.success(f"Ready! Ask me anything about **{uploaded.name}**")
        else:
            st.success(f"Loaded: **{uploaded.name}**")

    if st.session_state.vector_db:
        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("RAG Document Chatbot")

if not st.session_state.vector_db:
    st.info("Upload a PDF in the sidebar to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("pages"):
                st.caption(f"Source: page(s) {', '.join(str(p) for p in msg['pages'])}")

    if query := st.chat_input("Ask a question about the document…"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching document…"):
                answer, pages = answer_question(
                    query, st.session_state.vector_db, st.session_state.messages
                )
            st.markdown(answer)
            st.caption(f"Source: page(s) {', '.join(str(p) for p in pages)}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "pages": pages}
        )
