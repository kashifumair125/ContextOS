# ContextOS
### Long-Document Reasoning Memory for LLMs

> **RAG gives you chunks. ContextOS gives you understanding.**

---

## The Problem

Modern LLMs forget. RAG systems retrieve similar-looking text. Neither approach handles what professionals actually need when working with 100-page legal contracts, research papers, or large codebases:

| Need | RAG | ContextOS |
|------|-----|-----------|
| Find relevant section | ✅ | ✅ |
| Understand argument chains | ❌ | ✅ |
| Detect contradictions between sections | ❌ | ✅ |
| Track cross-references across 80 pages | ❌ | ✅ |
| Hierarchical document understanding | ❌ | ✅ |
| Multi-hop reasoning ("why does §4 matter for §12?") | ❌ | ✅ |

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │         ContextOS Pipeline       │
                    └─────────────────────────────────┘
                                    │
         ┌──────────────────────────▼──────────────────────────┐
         │                  Document Ingestion                  │
         │  PDF / Markdown / Plain Text / Code files            │
         │  → Hierarchical section/subsection/paragraph nodes   │
         └──────────────────────────┬──────────────────────────┘
                                    │
         ┌──────────────────────────▼──────────────────────────┐
         │               Knowledge Graph Builder                │
         │                                                      │
         │  1. PARENT_OF edges   (document structure)           │
         │  2. LLM Summaries     (bottom-up pyramid)            │
         │  3. REFERENCES edges  (explicit cross-refs)          │
         │  4. CONTRADICTS edges (LLM-detected conflicts)       │
         │  5. ELABORATES edges  (expansion relationships)      │
         │  6. SUPPORTS edges    (evidential relationships)      │
         └──────────────────────────┬──────────────────────────┘
                                    │
         ┌──────────────────────────▼──────────────────────────┐
         │              Multi-Hop Reasoning Engine              │
         │                                                      │
         │  Query classification → Graph traversal             │
         │  → Context assembly → LLM reasoning                 │
         │  → Structured answer with graph evidence            │
         └──────────────────────────────────────────────────────┘
```

### Edge Types

| Type | Description | Example |
|------|-------------|---------|
| `PARENT_OF` | Structural hierarchy | Document → Section → Paragraph |
| `REFERENCES` | Explicit cross-reference in text | "See Section 4.2" |
| `CONTRADICTS` | Conflicting claims detected by LLM | §4 says X, §12 says not-X |
| `ELABORATES` | One section expands another | §6 details the concept from §2 |
| `SUPPORTS` | Evidential relationship | §9 provides proof for claim in §3 |

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Set your API key

Get a free key at [console.groq.com](https://console.groq.com) — no credit card needed.

```bash
export GROQ_API_KEY=your_key_here
```

### 3. Run the server

```bash
uvicorn main:app --reload
# API docs at http://localhost:8000/docs
```

### 4. Ingest a document

```bash
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "your document text here", "detect_relations": true}'
```

Response:
```json
{
  "doc_id": "a3f9c2d1b4e8",
  "stats": {"nodes": 47, "edges": 63, "edge_types": {"PARENT_OF": 46, "CONTRADICTS": 2, "REFERENCES": 8, "ELABORATES": 7}},
  "elapsed_seconds": 12.4,
  "message": "Built graph with 47 nodes, 63 edges"
}
```

### 5. Query with multi-hop reasoning

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "a3f9c2d1b4e8", "question": "Does section 4 contradict section 12?"}'
```

Response:
```json
{
  "query_type": "CONTRADICTION",
  "answer": "Yes. Section 4 establishes that liability is capped at $50,000, while Section 12 states that in cases of gross negligence, liability is unlimited. This creates a direct contradiction...",
  "evidence_nodes": [...],
  "graph_insight": "[CONTRADICTS] '4. Liability Cap' → '12. Gross Negligence': Section 4 cap conflicts with unlimited liability in Section 12",
  "confidence": 0.9
}
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest/text` | Ingest raw text → build graph |
| `POST` | `/ingest/file` | Upload PDF/text file → build graph |
| `POST` | `/query` | Multi-hop reasoning query |
| `GET` | `/graph/{doc_id}` | Full graph as JSON |
| `GET` | `/nodes/{doc_id}` | List all nodes |
| `GET` | `/node/{doc_id}/{node_id}` | Single node + all edges |
| `GET` | `/contradictions/{doc_id}` | All detected contradictions |
| `DELETE` | `/graph/{doc_id}` | Remove graph from memory |

Interactive docs: `http://localhost:8000/docs`

---

## Use Cases

- **Legal tech**: "Does the indemnification clause in Article 8 override the limitation in Article 14?"
- **Research**: "What is the argument chain from the hypothesis to the conclusion?"
- **Compliance**: "Which sections impose obligations on the counterparty?"
- **Codebase analysis**: "Where is function X defined and which modules reference it?"
- **Medical**: "Does the contraindication in Section 3 apply given the patient profile in Section 7?"

---

## Stack

- **Language**: Python 3.11+
- **Graph**: NetworkX (plug-in Neo4j for production scale)
- **LLM**: Groq API — `llama-3.3-70b-versatile` (free tier, ~500 tokens/s)
- **API**: FastAPI
- **PDF parsing**: pdfplumber

---

## Roadmap

- [ ] Embedding-based `SEMANTICALLY_SIMILAR` edges
- [ ] Neo4j backend for production scale
- [ ] Streaming responses for long reasoning chains
- [ ] Batch document comparison ("how do Contract A and B differ?")
- [ ] Code-aware parser for Python/JS/TS codebases
- [ ] Web UI with interactive graph visualization

---

## Author

Built by **Umair Kashif** — MCA (MIT Manipal), researcher in transformer-based NLP (ICIST 2025).  
[GitHub](https://github.com/umair) · [LinkedIn](https://linkedin.com/in/umair)
