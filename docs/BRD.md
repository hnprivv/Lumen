# Business Requirements Document (BRD) — Lumen

**Project:** Lumen — RAG-Powered PDF Chat Tool  
**Author:** Huzaifa Najam  
**Status:** Active

---

## Overview

Lumen addresses a core limitation of general-purpose AI chat tools: they answer from training data, not from the specific document in front of the user. When someone needs to interrogate a dense PDF — a research paper, a legal contract, a technical manual — a generic LLM will hallucinate, paraphrase from memory, or fail to locate specific details. Lumen eliminates this by restricting every answer strictly to the content of the uploaded document, retrieved through semantic search and cited back to the exact page it came from.

The tool operates with no accounts, no storage, and no configuration. Upload a PDF, ask questions, get grounded answers.

## Core Business Objectives

1. **Groundedness:** Every answer must be derived exclusively from the uploaded document. The system must not draw on outside knowledge under any circumstances. If an answer is not found in the document, the model must say so explicitly rather than speculate.

2. **Verifiability:** Every response must surface the exact page numbers the answer was drawn from, allowing the user to cross-reference claims directly in the source document.

3. **Privacy:** Document embeddings must be generated locally on the server — no raw document content is transmitted to any third-party service beyond the specific retrieved passages included in the generation prompt. The user must be informed of this clearly.

4. **Responsiveness:** Answers must stream token-by-token in real time. The user should see the response begin appearing within seconds of submitting a query, not after full generation completes.

5. **Conversational continuity:** The system must retain context across multiple questions within a session. Follow-up questions must be interpretable in light of earlier exchanges without the user needing to repeat context.

6. **Frictionless access:** No user accounts, no sign-up, no configuration. A user arriving at the application for the first time must be able to upload a document and receive an answer in a single uninterrupted flow.

7. **Portfolio quality:** Demonstrate production-grade engineering practices through clean architecture, documented requirements, and a polished, custom UI — not a default Streamlit interface.

## Functional Scope

**Included:**
- Single PDF upload per session
- Local chunking and embedding of document content
- Natural language querying with semantic retrieval
- Streaming answer generation strictly limited to document content
- Page-level citations on every response
- Multi-turn conversation with history context
- Session reset to upload a new document
- Data handling disclosure visible at all times

**Excluded:**
- User accounts or authentication
- Persistent document or conversation storage
- Multi-document querying
- Non-PDF file formats
- Document annotation or editing
- Export of chat history
- Batch processing
- ATS or third-party integrations

## Success Metrics

- Every answer is sourced exclusively from retrieved document passages — outside knowledge never appears in responses
- Every answer surfaces the specific page numbers it was drawn from
- Document embeddings are generated locally with no raw content transmitted externally
- Responses begin streaming within seconds of query submission
- Follow-up questions succeed without the user needing to restate context
- A first-time user can upload, query, and receive a grounded answer without any instruction or configuration

## Operating Context

Lumen is a single-developer portfolio project. It runs on Streamlit Community Cloud with a dependency on the Google Gemini API for answer generation and on HuggingFace sentence-transformers for local embedding. There is no dedicated support infrastructure, SLA, or monitoring. The application is designed to be stateless and single-session by intent — not as a constraint to be worked around.
