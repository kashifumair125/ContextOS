"""
ContextOS — Knowledge Graph Builder
Core innovation: builds a HIERARCHICAL graph (not flat chunks)
with LLM-generated summaries at every level and typed relationship edges.

Edge types:
  - PARENT_OF      : structural hierarchy
  - REFERENCES     : explicit cross-references ("see section 4")
  - CONTRADICTS    : detected logical contradictions
  - ELABORATES     : one section expands on another
  - SUPPORTS       : one section supports a claim in another
  - SEMANTICALLY_SIMILAR : high cosine similarity (embedding-based)
"""

import re
import os
import json
import asyncio
import hashlib
from typing import Optional
from dataclasses import dataclass, field

import networkx as nx
from groq import Groq

from context_os.ingestion.parser import DocumentNode


@dataclass
class GraphEdge:
    source: str
    target: str
    edge_type: str
    weight: float = 1.0
    evidence: str = ""


@dataclass
class KnowledgeGraph:
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    nodes: dict[str, DocumentNode] = field(default_factory=dict)
    summaries: dict[str, str] = field(default_factory=dict)
    doc_hash: str = ""

    def get_node(self, node_id: str) -> Optional[DocumentNode]:
        return self.nodes.get(node_id)

    def get_summary(self, node_id: str) -> str:
        return self.summaries.get(node_id, self.nodes[node_id].content[:300] if node_id in self.nodes else "")

    def neighbors_of(self, node_id: str, edge_type: Optional[str] = None) -> list[str]:
        edges = self.graph.edges(node_id, data=True)
        if edge_type:
            return [t for _, t, d in edges if d.get('type') == edge_type]
        return [t for _, t, _ in edges]

    def find_path(self, source_id: str, target_id: str) -> list[str]:
        try:
            return nx.shortest_path(self.graph.to_undirected(), source_id, target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def stats(self) -> dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "edge_types": self._edge_type_counts(),
            "depth": self._max_depth(),
        }

    def _edge_type_counts(self) -> dict:
        counts = {}
        for _, _, d in self.graph.edges(data=True):
            t = d.get('type', 'unknown')
            counts[t] = counts.get(t, 0) + 1
        return counts

    def _max_depth(self) -> int:
        try:
            return max(self.nodes[n].level for n in self.graph.nodes if n in self.nodes)
        except (ValueError, KeyError):
            return 0

    def to_dict(self) -> dict:
        """Serializes to JSON-safe dict for API responses."""
        return {
            "doc_hash": self.doc_hash,
            "stats": self.stats(),
            "nodes": [
                {
                    "id": n.id, "level": n.level, "title": n.title,
                    "word_count": n.word_count, "parent_id": n.parent_id,
                    "summary": self.summaries.get(n.id, ""),
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {"source": s, "target": t, "type": d.get("type"), "evidence": d.get("evidence", "")}
                for s, t, d in self.graph.edges(data=True)
            ]
        }


class GraphBuilder:
    """
    Ingests a list of DocumentNodes and builds a KnowledgeGraph.

    Steps:
      1. Add all nodes to the graph
      2. Add PARENT_OF edges (structural hierarchy)
      3. Generate LLM summaries for each node (bottom-up)
      4. Detect REFERENCES edges (explicit cross-refs in text)
      5. Detect CONTRADICTS and ELABORATES edges (LLM-based)
      6. Optionally add SEMANTICALLY_SIMILAR edges (embedding cosine)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def build(self, nodes: list[DocumentNode], detect_relations: bool = True) -> KnowledgeGraph:
        """Synchronous build — runs async internally."""
        return asyncio.get_event_loop().run_until_complete(
            self.build_async(nodes, detect_relations)
        )

    async def build_async(self, nodes: list[DocumentNode], detect_relations: bool = True) -> KnowledgeGraph:
        kg = KnowledgeGraph()
        kg.doc_hash = self._hash_nodes(nodes)

        # Step 1 & 2: Add nodes + structural edges
        for node in nodes:
            kg.graph.add_node(node.id, level=node.level, title=node.title)
            kg.nodes[node.id] = node
            if node.parent_id and node.parent_id in kg.nodes:
                kg.graph.add_edge(node.parent_id, node.id, type="PARENT_OF", weight=1.0, evidence="")

        # Step 3: Generate summaries bottom-up (leaf nodes first)
        await self._generate_summaries(kg, nodes)

        if detect_relations:
            # Step 4: Detect explicit cross-references
            self._detect_reference_edges(kg, nodes)
            # Step 5: LLM-based contradiction + elaboration detection
            await self._detect_semantic_edges(kg, nodes)

        return kg

    # ─── Summarization ────────────────────────────────────────────────

    async def _generate_summaries(self, kg: KnowledgeGraph, nodes: list[DocumentNode]):
        """
        Bottom-up summarization:
        - Leaf nodes: summarize their raw content
        - Parent nodes: summarize by combining children's summaries
        This creates a "summary pyramid" — the document root has a
        single coherent summary of everything.
        """
        # Sort by level descending = leaves first
        sorted_nodes = sorted(nodes, key=lambda n: -n.level)

        # Process in batches of 5 to respect rate limits
        batch_size = 5
        for i in range(0, len(sorted_nodes), batch_size):
            batch = sorted_nodes[i:i + batch_size]
            tasks = [self._summarize_node(kg, node) for node in batch]
            await asyncio.gather(*tasks)

    async def _summarize_node(self, kg: KnowledgeGraph, node: DocumentNode):
        """Generate a concise summary for one node."""
        # Determine input: leaf uses raw content, parent uses children's summaries
        children_ids = [n for n in kg.graph.successors(node.id)
                        if kg.graph[node.id][n].get('type') == 'PARENT_OF']

        if children_ids:
            child_summaries = "\n".join(
                f"[{kg.nodes[c].title}]: {kg.summaries.get(c, kg.nodes[c].content[:200])}"
                for c in children_ids if c in kg.nodes
            )
            prompt = f"""You are building a knowledge graph of a long document.
Summarize this section titled "{node.title}" based on its subsections.
Keep it to 2-3 sentences. Focus on the KEY claims, arguments, or rules.

Subsection summaries:
{child_summaries}

Summary:"""
        else:
            content = node.content[:2000] if node.content else node.title
            if not content.strip():
                kg.summaries[node.id] = f"[Empty section: {node.title}]"
                return
            prompt = f"""You are building a knowledge graph of a long document.
Summarize this section titled "{node.title}" in 2-3 sentences.
Focus on KEY claims, rules, definitions, or arguments — not filler.

Content:
{content}

Summary:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            kg.summaries[node.id] = response.choices[0].message.content.strip()
        except Exception as e:
            kg.summaries[node.id] = node.content[:200] if node.content else node.title

    # ─── Reference Detection ───────────────────────────────────────────

    def _detect_reference_edges(self, kg: KnowledgeGraph, nodes: list[DocumentNode]):
        """
        Scans text for explicit cross-references like:
        "see Section 4", "as defined in clause 12", "per Article 3"
        Maps them to actual nodes and creates REFERENCES edges.
        """
        ref_patterns = [
            r'(?:see|per|as per|pursuant to|refer to|defined in|described in|outlined in)\s+(?:section|clause|article|paragraph|chapter|part)\s+(\d+(?:\.\d+)*)',
            r'(?:section|clause|article|§)\s*(\d+(?:\.\d+)*)',
        ]

        # Build a lookup: normalized_title -> node_id
        title_index: dict[str, str] = {}
        for node in nodes:
            # Index by leading number e.g. "4.2" -> node_id
            num_match = re.match(r'^(\d+(?:\.\d+)*)', node.title)
            if num_match:
                title_index[num_match.group(1)] = node.id

        for node in nodes:
            if not node.content:
                continue
            found_refs = set()
            for pattern in ref_patterns:
                for match in re.finditer(pattern, node.content, re.IGNORECASE):
                    ref_num = match.group(1)
                    if ref_num in title_index and title_index[ref_num] != node.id:
                        found_refs.add(title_index[ref_num])

            for target_id in found_refs:
                if not kg.graph.has_edge(node.id, target_id):
                    kg.graph.add_edge(
                        node.id, target_id,
                        type="REFERENCES",
                        weight=0.8,
                        evidence=f"Explicit cross-reference from '{node.title}'"
                    )

    # ─── Semantic Edge Detection ───────────────────────────────────────

    async def _detect_semantic_edges(self, kg: KnowledgeGraph, nodes: list[DocumentNode]):
        """
        Uses the LLM to find CONTRADICTS and ELABORATES relationships
        between top-level sections. We compare section summaries pairwise
        (only for level=1 nodes to keep API calls manageable).
        """
        top_nodes = [n for n in nodes if n.level == 1 and n.id in kg.summaries]
        if len(top_nodes) < 2:
            return

        # Build comparison pairs (avoid O(n²) — only compare sections that
        # share keyword overlap, as a cheap pre-filter)
        pairs = self._candidate_pairs(kg, top_nodes)

        tasks = [self._classify_pair(kg, a, b) for a, b in pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, GraphEdge):
                if not kg.graph.has_edge(result.source, result.target):
                    kg.graph.add_edge(
                        result.source, result.target,
                        type=result.edge_type,
                        weight=result.weight,
                        evidence=result.evidence
                    )

    def _candidate_pairs(self, kg: KnowledgeGraph, nodes: list[DocumentNode]) -> list[tuple]:
        """Pre-filter pairs by shared keyword overlap to reduce LLM calls."""
        pairs = []
        summaries = {n.id: set(re.findall(r'\b\w{5,}\b', kg.summaries.get(n.id, "").lower())) for n in nodes}

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                overlap = summaries[a.id] & summaries[b.id]
                if len(overlap) >= 3:  # At least 3 shared keywords
                    pairs.append((a, b))

        return pairs[:20]  # Cap at 20 pairs per document

    async def _classify_pair(self, kg: KnowledgeGraph, a: DocumentNode, b: DocumentNode) -> Optional[GraphEdge]:
        """Ask LLM to classify the relationship between two sections."""
        summary_a = kg.summaries.get(a.id, "")
        summary_b = kg.summaries.get(b.id, "")

        prompt = f"""You are analyzing a document's knowledge graph.
Compare these two sections and classify their relationship.

Section A — "{a.title}":
{summary_a}

Section B — "{b.title}":
{summary_b}

Respond ONLY in this JSON format (no markdown, no explanation):
{{
  "relationship": "CONTRADICTS" | "ELABORATES" | "SUPPORTS" | "NONE",
  "direction": "A_TO_B" | "B_TO_A" | "BIDIRECTIONAL",
  "confidence": 0.0-1.0,
  "evidence": "one sentence explaining why"
}}

Rules:
- CONTRADICTS: A and B make conflicting claims about the same topic
- ELABORATES: One section expands/details a claim from the other
- SUPPORTS: One section provides evidence for a claim in the other
- NONE: No meaningful relationship
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = response.choices[0].message.content.strip()
            data = json.loads(raw)

            if data["relationship"] == "NONE" or data["confidence"] < 0.6:
                return None

            source, target = (a.id, b.id) if data["direction"] in ("A_TO_B", "BIDIRECTIONAL") else (b.id, a.id)
            return GraphEdge(
                source=source, target=target,
                edge_type=data["relationship"],
                weight=data["confidence"],
                evidence=data.get("evidence", "")
            )
        except Exception:
            return None

    def _hash_nodes(self, nodes: list[DocumentNode]) -> str:
        content = "".join(n.content for n in nodes)
        return hashlib.md5(content.encode()).hexdigest()[:12]
