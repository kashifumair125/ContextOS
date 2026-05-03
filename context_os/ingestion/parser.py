"""
ContextOS — Document Parser
Supports: PDF, plain text, markdown, Python/JS/TS code files
Extracts structure: title, sections, subsections, paragraphs
"""

import re
import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class DocumentNode:
    """
    Represents a single structural unit of a document.
    Could be a section, subsection, or paragraph.
    """
    id: str
    level: int          # 0=doc, 1=section, 2=subsection, 3=paragraph
    title: str
    content: str
    parent_id: Optional[str]
    page_num: Optional[int]
    children: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    def __repr__(self):
        return f"<Node [{self.level}] '{self.title[:40]}' ({self.word_count}w)>"


class DocumentParser:
    """
    Parses long documents into a hierarchical tree of DocumentNodes.
    Supports PDF (via pdfplumber), Markdown, plain text, and code files.
    """

    HEADING_PATTERNS = [
        # Markdown headings
        (r'^(#{1,6})\s+(.+)$', 'markdown'),
        # Numbered sections: 1. Title / 1.1 Subtitle / Section 4:
        (r'^((?:\d+\.)+\d*)\s+([A-Z].{2,80})$', 'numbered'),
        (r'^Section\s+(\d+)[:\.\s]+(.+)$', 'section_keyword'),
        (r'^CHAPTER\s+(\d+)[:\.\s]+(.+)$', 'chapter'),
        # ALL CAPS titles (legal docs)
        (r'^([A-Z][A-Z\s]{4,60})$', 'caps_title'),
    ]

    def parse(self, source: str) -> list[DocumentNode]:
        """
        Main entry point. Accepts a file path or raw text string.
        Returns a flat list of DocumentNodes in document order.
        """
        if os.path.isfile(source):
            return self._parse_file(source)
        else:
            return self._parse_text(source, source_name="inline_text")

    def _parse_file(self, filepath: str) -> list[DocumentNode]:
        ext = Path(filepath).suffix.lower()
        if ext == '.pdf':
            text = self._extract_pdf(filepath)
        else:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
        return self._parse_text(text, source_name=Path(filepath).name)

    def _extract_pdf(self, filepath: str) -> str:
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(filepath) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    pages.append(f"[PAGE {i+1}]\n{text}")
            return "\n\n".join(pages)
        except ImportError:
            raise ImportError("Install pdfplumber: pip install pdfplumber")

    def _parse_text(self, text: str, source_name: str) -> list[DocumentNode]:
        lines = text.split('\n')
        nodes: list[DocumentNode] = []
        node_counter = [0]

        def make_id() -> str:
            node_counter[0] += 1
            return f"node_{node_counter[0]:04d}"

        # Root document node
        root_id = make_id()
        root = DocumentNode(
            id=root_id, level=0,
            title=source_name, content="",
            parent_id=None, page_num=None,
            metadata={"source": source_name}
        )
        nodes.append(root)

        # Stack tracks (level, node_id) for parent resolution
        stack: list[tuple[int, str]] = [(0, root_id)]

        current_content_lines: list[str] = []
        current_node: Optional[DocumentNode] = None
        current_page: Optional[int] = None

        def flush_content():
            if current_node and current_content_lines:
                current_node.content = '\n'.join(current_content_lines).strip()
            current_content_lines.clear()

        for raw_line in lines:
            line = raw_line.rstrip()

            # Track page markers (from PDF extraction)
            page_match = re.match(r'^\[PAGE (\d+)\]$', line)
            if page_match:
                current_page = int(page_match.group(1))
                continue

            # Try to detect a heading
            heading_info = self._detect_heading(line)

            if heading_info:
                flush_content()
                h_level, h_title = heading_info
                # Determine parent by walking up the stack
                while stack and stack[-1][0] >= h_level:
                    stack.pop()
                parent_id = stack[-1][1] if stack else root_id

                node_id = make_id()
                node = DocumentNode(
                    id=node_id,
                    level=h_level,
                    title=h_title.strip(),
                    content="",
                    parent_id=parent_id,
                    page_num=current_page,
                )
                nodes.append(node)
                stack.append((h_level, node_id))
                current_node = node
            else:
                if line.strip():
                    # If no current section node, attach to root
                    if current_node is None:
                        current_node = root
                    current_content_lines.append(line)
                elif current_content_lines:
                    # Blank line = paragraph break — optionally split here
                    current_content_lines.append("")

        flush_content()

        # Post-process: break large leaf nodes into paragraph sub-nodes
        nodes = self._split_large_nodes(nodes, node_counter)
        return nodes

    def _detect_heading(self, line: str) -> Optional[tuple[int, str]]:
        """Returns (level, title) if line looks like a heading, else None."""
        stripped = line.strip()
        if not stripped or len(stripped) > 120:
            return None

        # Markdown heading
        md = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if md:
            return len(md.group(1)), md.group(2)

        # Numbered section: 1., 1.1, 1.1.1 etc
        num = re.match(r'^(\d+(?:\.\d+)*)\s{1,4}([A-Z].{1,80})$', stripped)
        if num:
            depth = num.group(1).count('.') + 1
            return min(depth + 1, 3), f"{num.group(1)} {num.group(2)}"

        # ALL CAPS (typical in legal contracts)
        if re.match(r'^[A-Z][A-Z\s]{4,60}$', stripped) and stripped.isupper():
            return 1, stripped.title()

        # "Article X" / "Section X" / "CHAPTER X" pattern
        kw = re.match(r'^(ARTICLE|SECTION|CHAPTER|PART)\s+(\w+)[:\.]?\s*(.*)', stripped, re.IGNORECASE)
        if kw:
            return 1, f"{kw.group(1).title()} {kw.group(2)}: {kw.group(3)}".strip()

        return None

    def _split_large_nodes(self, nodes: list[DocumentNode], counter: list[int]) -> list[DocumentNode]:
        """
        Splits leaf nodes with >500 words into paragraph-level child nodes.
        This ensures the graph has fine-grained nodes for reasoning.
        """
        result = []
        for node in nodes:
            result.append(node)
            if node.word_count > 500 and not any(n.parent_id == node.id for n in nodes):
                paragraphs = [p.strip() for p in node.content.split('\n\n') if p.strip()]
                if len(paragraphs) > 1:
                    for i, para in enumerate(paragraphs):
                        counter[0] += 1
                        child = DocumentNode(
                            id=f"node_{counter[0]:04d}",
                            level=node.level + 1,
                            title=f"{node.title} ¶{i+1}",
                            content=para,
                            parent_id=node.id,
                            page_num=node.page_num,
                        )
                        result.append(child)
        return result
