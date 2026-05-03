"""
ContextOS — Multi-Hop Reasoning Engine
=======================================
This is the core differentiator from naive RAG.

Instead of "retrieve similar chunks → answer",
we do: "traverse the graph → assemble context → reason over relationships"

Query types supported:
  - CONTRADICTION : "Does section 4 contradict section 12?"
  - COMPARISON    : "How do clauses A and B differ on liability?"
  - CHAIN         : "What is the argument chain from premise X to conclusion Y?"
  - SCOPE         : "What sections cover topic Z?"
  - DEFINITION    : "How is term T defined and where is it used?"
  - FREEFORM      : General multi-hop reasoning
"""

import re
import json
from typing import Optional
from dataclasses import dataclass

from groq import Groq

from context_os.graph.builder import KnowledgeGraph


@dataclass
class ReasoningResult:
    query: str
    query_type: str
    answer: str
    reasoning_path: list[str]       # List of node_ids traversed
    evidence_nodes: list[dict]      # [{id, title, summary, role}]
    confidence: float
    graph_insight: str              # e.g. "Found CONTRADICTS edge between §4 and §12"


class ReasoningEngine:
    """
    Multi-hop reasoning over a KnowledgeGraph.
    Uses the graph structure to FIND relevant nodes,
    then uses the LLM to REASON over their relationships.
    """

    QUERY_CLASSIFIERS = {
        "CONTRADICTION": [
            r'contradict', r'conflict', r'inconsisten', r'contradict',
            r'oppose', r'differ from', r'contrary to', r'disagree'
        ],
        "COMPARISON": [
            r'compar', r'differ', r'vs\.?', r'versus', r'between',
            r'distinction', r'similar', r'same as', r'unlike'
        ],
        "CHAIN": [
            r'how does.+lead', r'argument chain', r'reasoning behind',
            r'what leads to', r'step[s]? to', r'why does'
        ],
        "DEFINITION": [
            r'defin', r'what (is|are|does)', r'mean[s]?', r'term',
            r'concept of', r'explain'
        ],
        "SCOPE": [
            r'which section', r'where (is|are|does)', r'all mention',
            r'covers?', r'about', r'topic of', r'related to'
        ],
    }

    def __init__(self, api_key: Optional[str] = None):
        import os
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"

    def query(self, kg: KnowledgeGraph, question: str) -> ReasoningResult:
        """Main entry point for multi-hop reasoning."""
        query_type = self._classify_query(question)
        relevant_nodes = self._retrieve_nodes(kg, question, query_type)
        graph_insight = self._extract_graph_insight(kg, relevant_nodes, query_type)
        answer, confidence = self._reason(kg, question, query_type, relevant_nodes, graph_insight)

        return ReasoningResult(
            query=question,
            query_type=query_type,
            answer=answer,
            reasoning_path=[n["id"] for n in relevant_nodes],
            evidence_nodes=relevant_nodes,
            confidence=confidence,
            graph_insight=graph_insight
        )

    # ─── Query Classification ──────────────────────────────────────────

    def _classify_query(self, question: str) -> str:
        q_lower = question.lower()
        for qtype, patterns in self.QUERY_CLASSIFIERS.items():
            if any(re.search(p, q_lower) for p in patterns):
                return qtype
        return "FREEFORM"

    # ─── Graph Traversal for Context Retrieval ─────────────────────────

    def _retrieve_nodes(self, kg: KnowledgeGraph, question: str, query_type: str) -> list[dict]:
        """
        Retrieves relevant nodes using graph-aware strategies:
        - CONTRADICTION: fetch all CONTRADICTS edges, return those node pairs
        - COMPARISON: find nodes matching both sides of "X vs Y"
        - SCOPE: keyword search across all summaries
        - CHAIN: BFS from seed node
        - FREEFORM/DEFINITION: scored keyword search
        """
        if query_type == "CONTRADICTION":
            return self._get_contradiction_nodes(kg)
        elif query_type == "COMPARISON":
            return self._get_comparison_nodes(kg, question)
        elif query_type == "CHAIN":
            return self._get_chain_nodes(kg, question)
        else:
            return self._keyword_search(kg, question, top_k=8)

    def _get_contradiction_nodes(self, kg: KnowledgeGraph) -> list[dict]:
        """Returns all node pairs connected by CONTRADICTS edges."""
        results = []
        seen = set()
        for source, target, data in kg.graph.edges(data=True):
            if data.get('type') == 'CONTRADICTS':
                for nid in [source, target]:
                    if nid not in seen and nid in kg.nodes:
                        node = kg.nodes[nid]
                        results.append({
                            "id": nid,
                            "title": node.title,
                            "summary": kg.get_summary(nid),
                            "level": node.level,
                            "role": f"CONTRADICTS partner of {target if nid == source else source}",
                            "edge_evidence": data.get('evidence', '')
                        })
                        seen.add(nid)
        return results

    def _get_comparison_nodes(self, kg: KnowledgeGraph, question: str) -> list[dict]:
        """
        For "how do X and Y differ", tries to identify both X and Y
        and fetch their nodes + any direct edges between them.
        """
        # Extract potential section references
        section_refs = re.findall(r'section\s+(\d+(?:\.\d+)*)|clause\s+(\d+(?:\.\d+)*)', question, re.IGNORECASE)
        target_nums = [r[0] or r[1] for r in section_refs]

        results = []
        for node in kg.nodes.values():
            for num in target_nums:
                if node.title.startswith(num):
                    results.append({
                        "id": node.id, "title": node.title,
                        "summary": kg.get_summary(node.id),
                        "level": node.level, "role": "comparison_target",
                        "edge_evidence": ""
                    })

        if len(results) < 2:
            # Fall back to keyword search
            results = self._keyword_search(kg, question, top_k=6)

        return results

    def _get_chain_nodes(self, kg: KnowledgeGraph, question: str) -> list[dict]:
        """
        For chain/argument queries, does a BFS traversal from seed nodes
        following ELABORATES and SUPPORTS edges.
        """
        seeds = self._keyword_search(kg, question, top_k=2)
        if not seeds:
            return []

        seed_id = seeds[0]["id"]
        visited = set()
        result = []
        queue = [seed_id]

        while queue and len(result) < 8:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            if current in kg.nodes:
                node = kg.nodes[current]
                result.append({
                    "id": current, "title": node.title,
                    "summary": kg.get_summary(current),
                    "level": node.level, "role": "chain_step",
                    "edge_evidence": ""
                })

            # Follow ELABORATES and SUPPORTS edges
            for _, target, data in kg.graph.edges(current, data=True):
                if data.get('type') in ('ELABORATES', 'SUPPORTS', 'PARENT_OF') and target not in visited:
                    queue.append(target)

        return result

    def _keyword_search(self, kg: KnowledgeGraph, question: str, top_k: int = 8) -> list[dict]:
        """
        Scored keyword search over node summaries and titles.
        Scores = (keyword hits in summary) + (keyword hits in title * 3)
        Also boosts nodes with non-structural edges (REFERENCES, CONTRADICTS, etc.)
        """
        # Extract meaningful query keywords (skip stopwords)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'does', 'do', 'did', 'what', 'which', 'how', 'where', 'when',
                     'this', 'that', 'these', 'those', 'and', 'or', 'of', 'in', 'to'}
        keywords = [w.lower() for w in re.findall(r'\b\w{3,}\b', question)
                    if w.lower() not in stopwords]

        scored = []
        for node_id, node in kg.nodes.items():
            summary = kg.summaries.get(node_id, "").lower()
            title = node.title.lower()
            score = sum(summary.count(kw) for kw in keywords)
            score += sum(title.count(kw) * 3 for kw in keywords)

            # Boost nodes with semantic edges
            for _, _, d in kg.graph.edges(node_id, data=True):
                if d.get('type') not in ('PARENT_OF',):
                    score += 1

            if score > 0:
                scored.append((score, node_id))

        scored.sort(reverse=True)
        results = []
        for score, node_id in scored[:top_k]:
            node = kg.nodes[node_id]
            results.append({
                "id": node_id, "title": node.title,
                "summary": kg.get_summary(node_id),
                "level": node.level, "role": "keyword_match",
                "score": score, "edge_evidence": ""
            })
        return results

    # ─── Graph Insight Extraction ──────────────────────────────────────

    def _extract_graph_insight(self, kg: KnowledgeGraph, nodes: list[dict], query_type: str) -> str:
        """
        Extracts graph-structural insights about the retrieved nodes.
        E.g. "Found CONTRADICTS edge between §4 and §12 with confidence 0.87"
        This is the KEY thing RAG can't do.
        """
        node_ids = {n["id"] for n in nodes}
        insights = []

        for source, target, data in kg.graph.edges(data=True):
            edge_type = data.get('type', '')
            if edge_type == 'PARENT_OF':
                continue
            if source in node_ids and target in node_ids:
                src_title = kg.nodes[source].title if source in kg.nodes else source
                tgt_title = kg.nodes[target].title if target in kg.nodes else target
                evidence = data.get('evidence', '')
                insights.append(
                    f"[{edge_type}] '{src_title}' → '{tgt_title}'"
                    + (f": {evidence}" if evidence else "")
                )

        # Path insights
        if len(nodes) >= 2:
            path = kg.find_path(nodes[0]["id"], nodes[-1]["id"])
            if len(path) > 2:
                path_titles = [kg.nodes[p].title if p in kg.nodes else p for p in path]
                insights.append(f"Reasoning path: {' → '.join(path_titles)}")

        return "\n".join(insights) if insights else "No direct graph relationships found between retrieved nodes."

    # ─── LLM Reasoning ────────────────────────────────────────────────

    def _reason(
        self,
        kg: KnowledgeGraph,
        question: str,
        query_type: str,
        nodes: list[dict],
        graph_insight: str
    ) -> tuple[str, float]:
        """
        Final LLM call: given the graph context + insights, answer the question.
        """
        if not nodes:
            return "No relevant sections found in the document for this query.", 0.1

        context_blocks = []
        for n in nodes:
            context_blocks.append(
                f"### {n['title']} (Level {n['level']})\n"
                f"Summary: {n['summary']}\n"
                f"Role: {n['role']}"
                + (f"\nEdge evidence: {n['edge_evidence']}" if n.get('edge_evidence') else "")
            )

        context_str = "\n\n".join(context_blocks)

        system_prompt = """You are ContextOS, an advanced document reasoning system.
You have access to a hierarchical knowledge graph of a long document — not just text chunks,
but structured nodes with typed relationships (CONTRADICTS, REFERENCES, ELABORATES, SUPPORTS).

Your job: answer the user's question using BOTH the section content AND the graph relationships.
Always explicitly reference which sections you're reasoning over.
If a CONTRADICTS relationship exists, name both sections and explain the contradiction.
Be precise, cite section titles, and reason step-by-step."""

        user_prompt = f"""Question: {question}

Query Type: {query_type}

=== GRAPH STRUCTURAL INSIGHTS ===
{graph_insight}

=== RELEVANT DOCUMENT SECTIONS ===
{context_str}

=== INSTRUCTIONS ===
Answer the question using both the section content and the graph relationships.
Format your response with:
1. Direct answer (2-3 sentences)
2. Evidence (which sections and relationships support this)
3. Confidence: HIGH / MEDIUM / LOW

Respond now:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=800,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            answer = response.choices[0].message.content.strip()

            # Parse confidence from response
            confidence = 0.7
            if 'HIGH' in answer.upper():
                confidence = 0.9
            elif 'LOW' in answer.upper():
                confidence = 0.4

            return answer, confidence

        except Exception as e:
            return f"Reasoning error: {str(e)}", 0.0
