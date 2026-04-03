---
title: LangGraph RAG Agent
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# langgraph-rag-agent

A production-oriented RAG chatbot built with **LangGraph**, **LiteLLM**, and **Chainlit**. Upload documents and ask questions — the agent retrieves relevant context from your files and answers with source attribution, or falls back to live web search when no relevant documents are found.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-0.x-orange)
![Chainlit](https://img.shields.io/badge/Chainlit-UI-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Features

- **Multi-model support** — switch between Gemini 2.5 Flash and Llama 3.3 70b (via Groq) at runtime from the UI
- **RAG pipeline** — upload PDF, TXT, or Markdown files; documents are split, embedded, and stored in Chroma
- **Source attribution** — every answer from a document includes the filename and page number
- **Web search fallback** — if no uploaded document contains relevant content, Tavily search is used automatically
- **Streaming** — token-by-token output via LiteLLM's async streaming interface
- **Transparent reasoning** — Chainlit `Step` blocks show whether the agent is searching documents or the web
- **Tested** — unit and integration tests covering nodes, tools, and graph compilation

---
 
## Live Demo

🚀 **[Try it on Hugging Face Spaces](https://huggingface.co/spaces/Alex-Resch/langgraph-rag-agent)** — no setup required, runs in your browser.
 
---

---

## Try it out

Download the sample document and upload it to test the RAG pipeline:

📄 [attention-is-all-you-need.pdf](https://arxiv.org/pdf/1706.03762) — the original Transformer paper
Example questions:
- "Who are the authors?"
- "What problem do Transformers solve compared to RNNs?"
- "What is the role of the attention mechanism?"

---

## Architecture

```
User message
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                   LangGraph Graph                   │
│                                                     │
│  ┌─────────────────────┐    ┌─────────────────────┐ │
│  │   search_pipeline   │───▶│     call_llm        │ │
│  │                     │    │                     │ │
│  │  1. search_documents│    │  ChatLiteLLM        │ │
│  │     (Chroma / cosine│    │  (Gemini / Llama)   │ │
│  │      similarity)    │    │  streaming=True     │ │
│  │                     │    │                     │ │
│  │  2. web_search_     │    │  SystemMessage +    │ │
│  │     fallback        │    │  full history       │ │
│  │     (Tavily)        │    │                     │ │
│  └─────────────────────┘    └─────────────────────┘ │
└─────────────────────────────────────────────────────┘
      │
      ▼
Streamed answer with [source, page N]
```

**Control flow:** The search node always runs first. It queries the Chroma vectorstore with cosine similarity. Results with a score ≥ the configured threshold are injected as a `SystemMessage` into the graph state. If no relevant chunks are found (`NO_DOCUMENTS_FOUND`), Tavily web search runs instead. The LLM node then receives the full conversation history plus the retrieved context and streams a response.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM interface | [LiteLLM](https://github.com/BerriAI/litellm) via `ChatLiteLLM` |
| Models | Gemini 2.5 Flash · Llama 3.3 70b (Groq) |
| Vector store | [Chroma](https://www.trychroma.com/) |
| Embeddings | [SentenceTransformers](https://www.sbert.net/) |
| Web search | [Tavily](https://tavily.com/) |
| UI | [Chainlit](https://chainlit.io/) |
| Document loaders | LangChain (PyPDFLoader, TextLoader, UnstructuredMarkdownLoader) |
| Testing | pytest · pytest-asyncio · unittest.mock |

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/Alex-Resch/langgraph-rag-agent.git
cd langgraph-rag-agent
uv sync
source .venv/bin/activate
```

### 2. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```env
GOOGLE_API_KEY=...        # Gemini 2.5 Flash
GROQ_API_KEY=...          # Llama 3.3 70b via Groq
TAVILY_API_KEY=...        # Web search fallback
```

All three services have a free tier that you can use for testing 
without providing payment information:
- **Groq** — https://console.groq.com
- **Google AI Studio** — https://aistudio.google.com/apikey
- **Tavily** — https://app.tavily.com

### 3. Run

```bash
chainlit run main.py -w
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Usage

1. Select a model from the settings panel (Gemini 2.5 Flash or Llama 3.3 70b)
2. Optionally upload one or more PDF, TXT, or Markdown files via the attachment button
3. Ask your question — the agent will search your documents first and fall back to the web if needed
4. Answers citing documents include `[filename, page N]` references

---

## Development & Testing

This project uses `uv` for dependency management and a `Makefile` for common tasks:

```bash
make test       # Run pytest
make lint       # Run ruff formatting and linting
make typecheck  # Run pyright typechecking
make all        # Run lint, typecheck, and test
```

The test suite covers:

- **`test_tools.py`** — document loader dispatch, chunking, vectorstore interaction, similarity threshold logic, web search error handling
- **`test_nodes.py`** — system message injection, model propagation, conversation history forwarding, search/fallback routing
- **`test_graph.py`** — graph compilation, node presence, instance isolation, end-to-end `ainvoke` with mocked dependencies

All async tests run automatically via `asyncio_mode = auto`.

---

## Configuration

Key constants in `config.py`:

| Constant | Default | Description |
|---|---------|---|
| `CHUNK_SIZE` | 500     | Max characters per document chunk |
| `CHUNK_OVERLAP` | 50      | Overlap between consecutive chunks |
| `SIMILARITY_THRESHOLD` | 0.5     | Minimum cosine similarity score for a chunk to be considered relevant |
| `TAVILY_MAX_RESULTS` | 10      | Number of web results to retrieve |