# Lumen

An AI-powered PDF chat tool built with Streamlit and Google Gemini. Upload any PDF and ask questions about it — Lumen chunks and embeds the document locally, retrieves the most relevant passages per query, and streams precise answers with exact page-number citations.

No hallucinations from outside knowledge. Every answer is grounded strictly in the document you upload.

## What it does

- Parses and chunks your PDF into overlapping segments for accurate retrieval
- Embeds all chunks locally using `all-MiniLM-L6-v2` — no data leaves your machine for embedding
- Stores vectors in ChromaDB and retrieves the top 4 most relevant passages per question
- Streams answers via Google Gemini, strictly limited to document content
- Cites the exact page numbers every answer was drawn from
- Maintains conversation history so follow-up questions have context

## Why answers stay grounded

Most document chat tools let the model answer from general knowledge when the document doesn't cover something. Lumen doesn't. The prompt explicitly forbids outside knowledge — if the answer isn't in the retrieved passages, the model says so. Every response also surfaces the page numbers it pulled from, so you can verify claims directly in the source.

## Tech stack

- **Streamlit** — UI and deployment
- **Google Gemini 3.1 Flash Lite** — answer generation
- **LangChain** — text splitting and vector store orchestration
- **ChromaDB** — local vector database
- **HuggingFace sentence-transformers** (`all-MiniLM-L6-v2`) — local embeddings
- **pypdf** — PDF parsing

## Running locally

```bash
git clone https://github.com/hnprivv/Lumen
cd Lumen
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

Add your Gemini API key to `.streamlit/secrets.toml`:
```toml
GOOGLE_API_KEY = "your-key-here"
```

Get a free key at [aistudio.google.com](https://aistudio.google.com).

```bash
streamlit run app.py
```

## Deploying to Streamlit Cloud

1. Push the repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo
3. Set main file to `app.py`
4. Under Advanced settings → Secrets, add your `GOOGLE_API_KEY`
5. Deploy

---

Built by [Huzaifa Najam](https://github.com/hnprivv).
