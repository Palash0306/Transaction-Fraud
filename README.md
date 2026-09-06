# 🏦 Bank Filings & Earnings Call Q&A Agent

A retrieval-augmented, citation-grounded Q&A agent over real SEC 10-K filings — with a built-in evidence verifier, a human-approval gate for low-confidence answers, and an MCP server so the retrieval engine can be called from any MCP-compatible client, not just this app.

Built end-to-end as an independent portfolio project, on a 100% free stack, targeting AI/ML Engineer and Data Scientist roles in banking and financial services.

---

## Why this project exists

Credit analysts and risk teams spend real hours manually reading dense regulatory filings to answer specific questions — *"what did the bank say about loan loss provisions this quarter?"* This project builds that tool.

The core idea isn't just "an LLM that answers questions about PDFs." It's a system that treats **an ungrounded answer about regulatory disclosures as a compliance risk, not just an inconvenience.** So every answer is:

- **Retrieved** from real filing text, not generated from the model's training data
- **Verified** by a second LLM pass that explicitly checks whether the retrieved evidence actually supports a confident answer
- **Escalated to human review** automatically when confidence is low, instead of guessing

That verification layer is the actual point of the project — not a bolt-on afterthought.

## What I set out to prove with this project

I wanted to build a genuinely complete, working RAG system from scratch — not a notebook demo — to demonstrate real engineering competence across the full stack: data ingestion, embeddings, vector search, agentic orchestration, verification logic, tool interoperability (MCP), a real API and UI, and automated testing with CI. Every piece had to actually work end-to-end on real data, not toy examples.

---

## Architecture

```
User question
      │
      ▼
┌─────────────────────┐
│  LangGraph orchestrator  │
└─────────────────────┘
      │
      ▼
Node 1: retrieve_context ──► Chroma (vector DB) ──► top-k real filing chunks
      │
      ▼
Node 2: verify_evidence ──► LLM judges: is this evidence actually sufficient?
      │                      (low confidence → human-approval gate)
      ▼
Node 3: synthesize_answer ──► LLM writes a cited answer from real evidence
      │
      ▼
Final answer + source citations (ticker, filing date)
```

`retrieve_context()` is also exposed as an **MCP server tool**, so any MCP client (e.g. Claude Desktop) can query the same filings index directly — completely independent of this app's own UI.

**Important scope note:** the MCP server exposes only the *retrieval* step — raw, ranked filing excerpts — not the full verify/synthesize graph. This is a deliberate design choice: MCP tools are meant to be narrow, composable capabilities that a *client* reasons over, not a wrapper around an entire agentic pipeline. The full retrieve → verify → synthesize flow, including the human-approval gate, only runs through this app's own FastAPI/Streamlit path.

---

## Tech stack

| Layer | Tool | Notes |
|---|---|---|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Runs locally on Apple Silicon via PyTorch/MPS — no CUDA, no cloud cost |
| Vector store | Chroma | Local, persistent, free |
| Orchestration | LangGraph | Retrieve → verify → synthesize, with conditional routing for the human-approval gate |
| LLM | Groq (`qwen/qwen3.6-27b`, `openai/gpt-oss-120b`) | Free-tier hosted inference — no local GPU needed for generation |
| Tool exposure | MCP (Model Context Protocol) | Custom server wrapping the retrieval function for any MCP client |
| Backend | FastAPI | Serves the full LangGraph pipeline over HTTP |
| Frontend | Streamlit | Lightweight chat UI with citations and confidence warnings |
| Data source | SEC EDGAR | Real 10-K filings, fetched via the public submissions API |
| Testing | pytest + `unittest.mock` | Real retrieval tests, mocked LLM logic tests, full graph routing tests, API contract tests |
| CI | GitHub Actions | Lint + full test suite on every push |

---

## What actually works, end to end

- [x] SEC EDGAR ingestion pipeline — real 10-K filings for JPMorgan, Bank of America, and Wells Fargo
- [x] Chunking + local embedding pipeline (batched, MPS-accelerated)
- [x] Persistent Chroma vector index with distance-threshold filtering for irrelevant queries
- [x] LangGraph orchestrator: retrieve → verify → synthesize, with a real conditional human-approval gate
- [x] MCP server exposing retrieval (not the full verify/synthesize graph) — tested live through Claude Desktop, returning real cited financial figures via Claude's own reasoning over the retrieved chunks
- [x] FastAPI backend + Streamlit chat frontend, fully wired to the LangGraph pipeline
- [x] 15+ automated tests covering retrieval, verifier logic, graph routing, and API contracts
- [x] GitHub Actions CI running the full suite on every push

---

## Engineering challenges I actually hit (and how I solved them)

Real bugs found through real testing — good interview material, since these show debugging process, not just a finished result:

1. **Silent early return in chunk loading** — a `return` statement indented one level too deep inside a `for` loop caused only the *first* file in a directory to ever load, with no error thrown. Fixed by tracing exact indentation and adding a regression test.
2. **Reasoning-model output breaking JSON parsing** — the verifier LLM sometimes kept "thinking out loud" in plain text *after* its `<think>` block closed, before finally printing valid JSON. A naive `json.loads()` failed on the surrounding text even though valid JSON was present. Fixed with a targeted regex that extracts the JSON object containing the expected key, ignoring any surrounding text.
3. **Verifier miscalibration** — the verifier initially rejected perfectly good evidence because it was tabular/numeric rather than "qualitative," flagging strong answers as low-confidence. Diagnosed by inspecting raw retrieved chunks directly, then fixed by explicitly instructing the verifier that financial tables and dollar figures count as sufficient evidence.
4. **Query pollution from meta-instructions** — embedding a question that included wrapper text like *"using the X tool"* measurably degraded retrieval quality, since the embedding model has no way to separate instruction from intent. Diagnosed with a side-by-side retrieval comparison script.
5. **CI path and branch mismatches** — the workflow file was initially nested one folder too deep (wrong repo root) and watching the wrong git branch, so GitHub Actions silently never ran. Fixed by tracing the actual repo root from git's own output and aligning the trigger branch.

---

## Setup

```bash
git clone <this-repo>
cd bank-filings-qa-agent
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt

# configure environment variables
cp .env.example .env
# fill in: GROQ_API_KEY

# ingest real filings
python3 phase2_full_pipeline_fixed.py

# build the vector index
python3 phase3_store_in_chroma.py

# run the API
uvicorn phase6_api:app --reload

# run the demo UI (separate terminal)
streamlit run phase6_streamlit_app.py

# run the MCP server (for use with Claude Desktop or another MCP client)
python3 phase5_mcp_server.py
```

## Running tests

```bash
python3 -m pytest tests/ -v
```

---

## What this project is not

This is a portfolio project, not a production banking system. It doesn't handle real client data, isn't compliance-reviewed, and the verification agent reduces — but does not eliminate — the risk of an incorrect answer. Any real deployment would need additional review, monitoring, and legal sign-off before use in an actual decision-making context.

## Roadmap / stretch goals

- [ ] Expand to earnings call transcripts alongside filings
- [ ] Drift monitoring on retrieval quality over time
- [ ] Swap in AWS S3/RDS as an optional cloud storage layer

---

## Skills demonstrated

Agentic workflow design (LangGraph) · MCP server development · RAG pipeline design (chunking, embeddings, retrieval) · multi-agent verifier/critic pattern · human-in-the-loop escalation logic · vector database usage (Chroma) · backend API design (FastAPI) · automated testing with mocking · CI/CD (GitHub Actions) · local ML inference on Apple Silicon (PyTorch/MPS)