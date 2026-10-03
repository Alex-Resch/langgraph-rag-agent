---
title: LangGraph RAG Agent
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# langgraph-rag-agent

A production-oriented RAG chatbot built with **LangGraph**, **LiteLLM** and **Chainlit**. Upload documents and ask questions — a tool-calling agent searches your files and answers with source attribution, or searches the web when the answer is not in your documents.

Beyond the agent itself, the project covers what it takes to run an LLM app in production: **observability** with Langfuse, **automated evals** in CI, **guardrails** against prompt injection and PII leaks, and **infrastructure as code** for AWS.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-1.x-orange)
![Chainlit](https://img.shields.io/badge/Chainlit-UI-green)
![Terraform](https://img.shields.io/badge/Terraform-AWS-purple?logo=terraform)

---

## Features

- **Tool-calling agent** — the LLM decides when to search the uploaded documents and when to fall back to Tavily web search
- **RAG pipeline** — upload PDF, TXT or Markdown files; documents are split, embedded (`BAAI/bge-base-en-v1.5`) and stored in Chroma
- **Source attribution** — answers from documents cite the file and page; web answers say explicitly that they are not from the document
- **Input guardrail** — structured PII (emails, phone numbers, IBANs, credit cards, …) is masked with Presidio, prompt injection and harmful requests are blocked by an LLM safety check
- **Observability** — every request is traced in Langfuse with token usage, costs and session IDs; PII is masked before traces leave the app
- **Evals** — RAG quality and guardrail detection are measured with DeepEval on every relevant pull request
- **Infrastructure as code** — Terraform for AWS ECS Fargate, deployed by GitHub Actions without stored AWS keys (OIDC)
- **Streaming** — token-by-token output in the Chainlit UI
- **MCP server** — the same document search as a knowledge base for Claude Code and other MCP clients, served over Streamable HTTP

---

## Live Demo

🚀 **[Try it on Hugging Face Spaces](https://huggingface.co/spaces/Alex-Resch/langgraph-rag-agent)** — no setup required, runs in your browser.

### Try it out

Upload a PDF, for example the evaluation report [`evals/data/nordheide_solar_park_report.pdf`](evals/data/nordheide_solar_park_report.pdf) (a fictional 6-page report), and ask:
- "Which bank provided the loan and at what interest rate?"
- "What was the investment per household supplied by the park?" (calculated from two pages)
- "Who is the mayor of Nordheide?" (not in the document → web search)

---

## Architecture

```
User message
      │
      ▼
┌──────────────────────────────────────────────────────────────┐
│                       LangGraph Graph                        │
│                                                              │
│  ┌──────────────┐ blocked ┌─────┐                            │
│  │ input_guard  │────────▶│ END │  "I can't help with that"  │
│  │              │         └─────┘                            │
│  │ Presidio PII │                                            │
│  │ + LLM safety │                                            │
│  └──────┬───────┘                                            │
│         │ safe                                               │
│         ▼                                                    │
│  ┌──────────────┐  tool calls  ┌──────────────────────────┐  │
│  │   call_llm   │─────────────▶│          tools           │  │
│  │              │◀─────────────│ search_documents (Chroma)│  │
│  │ Gemini 2.5   │   results    │ web_search_fallback      │  │
│  │ Flash        │              │ (Tavily)                 │  │
│  └──────┬───────┘              └──────────────────────────┘  │
│         │ final answer                                       │
└─────────┼────────────────────────────────────────────────────┘
          ▼
Streamed answer with sources ──── traces ───▶ Langfuse
```

- **`input_guard`** masks structured PII in all user messages and asks `gemini-2.5-flash-lite` (structured output) whether the latest message is a prompt injection or a harmful request. Off-topic questions are allowed on purpose.
- **`call_llm`** runs Gemini 2.5 Flash with the tools bound. It loops with the **`tools`** node until it can answer.
- The vectorstore is passed to the tools via LangGraph's runtime context, so the agent has no dependency on the UI.

### Why there is no output guardrail

The input guardrail is enough for this use case, so an output check was left out deliberately:
- Users only ever see their own uploaded documents, so answers cannot leak someone else's data.
- Answers are streamed — an output check would either show PII before it runs or remove streaming.
- Traces are masked separately before they are exported to Langfuse.

---

## Observability

The app sends every chat message and document summary to [Langfuse](https://langfuse.com) via the LangChain callback handler:

- traces grouped by Chainlit session, with every LangGraph node, tool call and LLM generation
- token usage (also while streaming) and costs per request
- PII masked in all span attributes before export (`mask_otel_spans`)

Eval runs are traced too, in a separate Langfuse environment, with each metric stored as a score on its trace.

---

## Evals

Evals run with [DeepEval](https://deepeval.com) and `gemini-2.5-flash-lite` as the judge (`uv run pytest evals -s`).

| Suite | Test set | Metrics | Gate | Latest result |
|---|---|---|---|---|
| RAG | 20 questions on a fictional 6-page report (single page, late page, multi-page, calculations, not in document) | Correctness (GEval), Answer Relevancy, Faithfulness, Contextual Precision | ≥ 90 % of cases pass | 20/20 |
| Guardrail | 15 attacks, 10 harmless edge cases | detection rate, false positives | ≥ 90 % detected, ≤ 1 false positive | 15/15 detected, 0/10 false positives |

The evals workflow runs on every pull request that touches the agent, the config or the evals, and pushes scores to Langfuse.

---

## MCP Server

`mcp_server/` makes the document search available to any [MCP](https://modelcontextprotocol.io) client, for example Claude Code. It reuses the agent's retrieval code but has its own knowledge base on disk, so documents are ingested once instead of per chat.

| Tool | Returns |
|---|---|
| `search_documents(query)` | the most relevant passages with file name and page |
| `list_documents()` | the files in the knowledge base |

The server calls no LLM, so it needs no API keys.

```bash
uv run python -m mcp_server.ingest path/to/documents   # rebuild the knowledge base from a folder (PDF, TXT, MD)
uv run python -m mcp_server.server                     # serve it at http://127.0.0.1:8001/mcp
claude mcp add --transport http knowledge-base http://127.0.0.1:8001/mcp
```

Run the ingest command again whenever documents change: it rebuilds the knowledge base from scratch, so removed files disappear from the search. Restart the server afterwards.

**In production** a knowledge base server would also need what this demo leaves out on purpose: authentication (MCP supports OAuth), a deployment as a container, and a scheduled ingest from the real source, such as Confluence or SharePoint, instead of a local folder. The server only listens on `127.0.0.1` for that reason.

---

## Deployment on AWS

`infra/` contains Terraform for a complete AWS setup in `eu-central-1`:

| Component | Purpose |
|---|---|
| ECS Fargate (ARM64) + Application Load Balancer | runs the container, sticky sessions for Chainlit's WebSocket |
| ECR | container registry |
| SSM Parameter Store (SecureString) | API keys, injected into the container at start |
| IAM roles | task execution role + GitHub OIDC role limited to this repository's `main` branch, this ECR repository and this ECS service |
| CloudWatch Logs | container logs |
| AWS Budgets | email alert as soon as monthly costs exceed 1 $ |

The **Deploy** workflow (`workflow_dispatch`) logs in to AWS via OIDC, builds the ARM64 image on GitHub and rolls it out to ECS.

```bash
cp infra/terraform.tfvars.example infra/terraform.tfvars   # set budget_alert_email
export TF_VAR_app_secrets='{"GEMINI_API_KEY":"...","TAVILY_API_KEY":"...","LANGFUSE_PUBLIC_KEY":"...","LANGFUSE_SECRET_KEY":"...","LANGFUSE_BASE_URL":"https://cloud.langfuse.com"}'
terraform -chdir=infra init
terraform -chdir=infra apply
gh variable set AWS_ROLE_ARN --body "$(terraform -chdir=infra output -raw github_deploy_role_arn)"
gh workflow run deploy.yml
terraform -chdir=infra output app_url
terraform -chdir=infra destroy                              # when done
```

The setup has been deployed, tested end to end and destroyed again — it is meant for on-demand deployments, the permanent demo runs on Hugging Face Spaces.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM interface | [LiteLLM](https://github.com/BerriAI/litellm) via `ChatLiteLLM` |
| Models | Gemini 2.5 Flash (agent) · Gemini 2.5 Flash Lite (guardrail, eval judge) |
| Vector store | [Chroma](https://www.trychroma.com/) |
| Embeddings | `BAAI/bge-base-en-v1.5` via HuggingFace |
| Web search | [Tavily](https://tavily.com/) |
| UI | [Chainlit](https://chainlit.io/) |
| Guardrails | [Presidio](https://microsoft.github.io/presidio/) · LLM safety check |
| MCP | [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) (FastMCP, Streamable HTTP) |
| Observability | [Langfuse](https://langfuse.com) |
| Evals | [DeepEval](https://deepeval.com) |
| Infrastructure | Terraform · AWS ECS Fargate · Docker |
| CI | GitHub Actions · ruff · pyright · pytest · pip-audit · Dependabot |

---

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/Alex-Resch/langgraph-rag-agent.git
cd langgraph-rag-agent
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in your keys in `.env`:

```env
GEMINI_API_KEY=...        # https://aistudio.google.com/apikey
TAVILY_API_KEY=...        # https://app.tavily.com
# Optional – tracing is disabled without these:
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

All services have a free tier that works without payment information.

### 3. Run

```bash
uv run chainlit run main.py -w
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Development & Testing

```bash
uv run ruff check . && uv run ruff format --check .   # lint + format
uv run pyright                                        # type check
uv run pytest                                         # 54 unit tests, no API calls
uv run pytest evals -s                                # evals, needs API keys
```

The CI workflow runs all of these checks except the evals on every push to `main` and every pull request, plus `pip-audit` and `terraform validate`. Dependabot updates dependencies and GitHub Actions monthly; all actions are pinned by commit SHA.

---

## Configuration

Key constants in `config.py`:

| Constant | Default | Description |
|---|---|---|
| `DEFAULT_MODEL` | `gemini/gemini-2.5-flash` | Agent model |
| `GUARD_MODEL` | `gemini/gemini-2.5-flash-lite` | Model for the safety check |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Embedding model for the vectorstore |
| `CHUNK_SIZE` | 500 | Max characters per document chunk |
| `CHUNK_OVERLAP` | 50 | Overlap between consecutive chunks |
| `SIMILARITY_THRESHOLD` | 0.2 | Minimum relevance score for a chunk to be returned |
| `TAVILY_MAX_RESULTS` | 5 | Number of web results to retrieve |
