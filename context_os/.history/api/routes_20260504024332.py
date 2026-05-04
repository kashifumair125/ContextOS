"""
ContextOS — FastAPI Routes
==========================
REST API exposing document ingestion, graph inspection, and multi-hop reasoning.

Endpoints:
  POST /ingest          — Upload document, build knowledge graph
  GET  /graph/{doc_id}  — Inspect the knowledge graph
  POST /query           — Multi-hop reasoning query
  GET  /nodes/{doc_id}  — List all graph nodes
  GET  /node/{node_id}  — Get a specific node + its edges
  POST /contradiction   — Find all contradictions in a document
  GET  /health          — Health check
"""

import os
import time
import tempfile
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from context_os.ingestion.parser import DocumentParser
from context_os.graph.builder import GraphBuilder, KnowledgeGraph
from context_os.reasoning.engine import ReasoningEngine


# ─── In-memory graph store (swap with Redis/DB in production) ──────────────
graph_store: dict[str, KnowledgeGraph] = {}

parser = DocumentParser()
builder = GraphBuilder()
engine = ReasoningEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 ContextOS started")
    yield
    print("🛑 ContextOS shutting down")


app = FastAPI(
    title="ContextOS",
    description="Long-Document Reasoning Memory for LLMs — hierarchical knowledge graphs with multi-hop reasoning",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ─────────────────────────────────────────────

class IngestTextRequest(BaseModel):
    text: str
    doc_id: Optional[str] = None
    detect_relations: bool = True


class QueryRequest(BaseModel):
    doc_id: str
    question: str


class IngestResponse(BaseModel):
    doc_id: str
    stats: dict
    elapsed_seconds: float
    message: str


class QueryResponse(BaseModel):
    doc_id: str
    query: str
    query_type: str
    answer: str
    reasoning_path: list[str]
    evidence_nodes: list[dict]
    confidence: float
    graph_insight: str


# ─── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "graphs_loaded": len(graph_store),
        "version": "0.1.0"
    }


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(req: IngestTextRequest):
    """
    Ingest raw text, build a hierarchical knowledge graph.
    Returns a doc_id you use for subsequent queries.
    """
    start = time.time()
    try:
        nodes = parser.parse(req.text)
        if not nodes:
            raise HTTPException(400, "No nodes extracted from document")

        kg = await builder.build_async(nodes, detect_relations=req.detect_relations)
        doc_id = req.doc_id or kg.doc_hash
        graph_store[doc_id] = kg

        return IngestResponse(
            doc_id=doc_id,
            stats=kg.stats(),
            elapsed_seconds=round(time.time() - start, 2),
            message=f"Built graph with {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges"
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...), detect_relations: bool = True):
    """
    Upload a PDF or text file. Saves temporarily, parses, builds graph.
    """
    start = time.time()
    tmp_path = os.path.join(tempfile.gettempdir(), file.filename)
    try:
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)

        nodes = parser.parse(tmp_path)
        if not nodes:
            raise HTTPException(400, "No content extracted from file")

        kg = await builder.build_async(nodes, detect_relations=detect_relations)
        doc_id = kg.doc_hash
        graph_store[doc_id] = kg

        return IngestResponse(
            doc_id=doc_id,
            stats=kg.stats(),
            elapsed_seconds=round(time.time() - start, 2),
            message=f"Processed '{file.filename}' → {kg.graph.number_of_nodes()} nodes"
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/query", response_model=QueryResponse)
async def query_graph(req: QueryRequest):
    """
    Multi-hop reasoning query over a previously ingested document.
    
    Example questions:
      - "Does section 4 contradict section 12?"
      - "What is the argument chain supporting the main thesis?"
      - "Which sections cover liability limitations?"
      - "How does the indemnification clause relate to the liability cap?"
    """
    kg = _get_graph(req.doc_id)
    result = engine.query(kg, req.question)

    return QueryResponse(
        doc_id=req.doc_id,
        query=result.query,
        query_type=result.query_type,
        answer=result.answer,
        reasoning_path=result.reasoning_path,
        evidence_nodes=result.evidence_nodes,
        confidence=result.confidence,
        graph_insight=result.graph_insight
    )


@app.get("/graph/{doc_id}")
async def get_graph(doc_id: str):
    """Returns the full knowledge graph as a JSON-serializable dict."""
    kg = _get_graph(doc_id)
    return kg.to_dict()


@app.get("/nodes/{doc_id}")
async def list_nodes(doc_id: str, level: Optional[int] = None):
    """List all nodes in a graph, optionally filtered by level."""
    kg = _get_graph(doc_id)
    nodes = [
        {
            "id": n.id,
            "level": n.level,
            "title": n.title,
            "word_count": n.word_count,
            "parent_id": n.parent_id,
            "summary": kg.summaries.get(n.id, ""),
            "page_num": n.page_num,
        }
        for n in kg.nodes.values()
        if level is None or n.level == level
    ]
    return {"doc_id": doc_id, "count": len(nodes), "nodes": nodes}


@app.get("/node/{doc_id}/{node_id}")
async def get_node(doc_id: str, node_id: str):
    """Get a specific node with its full content, summary, and all edges."""
    kg = _get_graph(doc_id)
    if node_id not in kg.nodes:
        raise HTTPException(404, f"Node '{node_id}' not found")

    node = kg.nodes[node_id]
    edges_out = [
        {"target": t, "type": d.get('type'), "evidence": d.get('evidence', '')}
        for _, t, d in kg.graph.edges(node_id, data=True)
    ]
    edges_in = [
        {"source": s, "type": d.get('type'), "evidence": d.get('evidence', '')}
        for s, _, d in kg.graph.in_edges(node_id, data=True)
    ]

    return {
        "id": node.id,
        "level": node.level,
        "title": node.title,
        "content": node.content,
        "summary": kg.summaries.get(node_id, ""),
        "word_count": node.word_count,
        "page_num": node.page_num,
        "parent_id": node.parent_id,
        "edges_out": edges_out,
        "edges_in": edges_in,
    }


@app.get("/contradictions/{doc_id}")
async def find_contradictions(doc_id: str):
    """Returns all detected contradictions in the document."""
    kg = _get_graph(doc_id)
    contradictions = []
    for source, target, data in kg.graph.edges(data=True):
        if data.get('type') == 'CONTRADICTS':
            src = kg.nodes.get(source)
            tgt = kg.nodes.get(target)
            contradictions.append({
                "section_a": {"id": source, "title": src.title if src else source,
                               "summary": kg.summaries.get(source, "")},
                "section_b": {"id": target, "title": tgt.title if tgt else target,
                               "summary": kg.summaries.get(target, "")},
                "confidence": data.get('weight', 0),
                "evidence": data.get('evidence', ''),
            })
    return {"doc_id": doc_id, "count": len(contradictions), "contradictions": contradictions}


@app.delete("/graph/{doc_id}")
async def delete_graph(doc_id: str):
    if doc_id not in graph_store:
        raise HTTPException(404, f"Graph '{doc_id}' not found")
    del graph_store[doc_id]
    return {"message": f"Graph '{doc_id}' deleted"}


# ─── Helpers ───────────────────────────────────────────────────────────────

def _get_graph(doc_id: str) -> KnowledgeGraph:
    if doc_id not in graph_store:
        raise HTTPException(404, f"No graph found for doc_id '{doc_id}'. Call /ingest first.")
    return graph_store[doc_id]
