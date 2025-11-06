"""
Enhanced citation formatter for Founders AI (region-agnostic).

Purpose
- Format startup-grounded answers with a clear, compact “Citations” and “Sources” section.
- Normalize and deduplicate metric citations (ARR, MRR, MAU, DAU, GMV, funding rounds, currency amounts).
- Group document sources by company and file, merging pages and known key metrics.

Input contracts
- format_response(answer: str, citations: List[str], sources: List[Dict], sector: str = "Other") -> str
  citations: free-text metric strings (e.g., "$10M ARR", "50M users", "Series A $10M")
  sources: list of dicts with fields (best-effort; formatter tolerates missing fields):
    {
        "company": str,
        "title": Optional[str],
        "filename": Optional[str],
        "page": Optional[int],
        "url": Optional[str],
        "key_metrics": Optional[List[str]]
    }

Output structure
Answer
<answer>

Sector
<sector>

Citations
- <citation 1>
- <citation 2>

Sources
- <Company> — "<Title>", page X[, Y, Z] (filename.pdf)
  Metrics: <m1>, <m2>, ...

Region-agnostic notes
- Supports common currency symbols and abbreviations: $, €, £, ¥; K, M, B; “million”, “billion”.
- Does not assume country-specific units like “crore” or “lakh”.
- Uppercases metric labels consistently (ARR, MRR, MAU, DAU, GMV).
"""

from typing import List, Dict, Any, Tuple, Optional


class StartupCitationFormatter:
    """
    Formats startup citations and sources for Founders AI responses (global-ready).
    """

    LABELS = ("ARR", "MRR", "MAU", "DAU", "GMV")

    def format_response(
        self,
        answer: str,
        citations: List[str],
        sources: List[Dict[str, Any]],
        sector: str = "Other"
    ) -> str:
        """
        Render a complete, human-readable block that includes the answer,
        sector tag, normalized citations, and grouped sources.

        Args:
            answer: The final generated text to present to the user.
            citations: Free-text metric strings extracted from the answer.
            sources: Source metadata used for the answer.
            sector: Optional business sector tag.

        Returns:
            A clean, textual section suitable for app display or Markdown.
        """
        parts: List[str] = []

        # 1) Answer
        if answer:
            parts.append(answer.strip())

        # 2) Sector
        if sector:
            parts.append("\nSector")
            parts.append(sector.strip())

        # 3) Citations
        norm_citations = self.normalize_citations(citations)
        if norm_citations:
            parts.append("\nCitations")
            parts.extend([f"- {c}" for c in norm_citations])

        # 4) Sources
        sources_block = self.format_sources(sources)
        if sources_block:
            parts.append("\nSources")
            parts.append(sources_block)

        return "\n\n".join(p for p in parts if p).strip()

    # -------------------------
    # Citation normalization
    # -------------------------

    def normalize_citations(self, citations: List[str]) -> List[str]:
        """
        Normalize and deduplicate citations in a region-agnostic way.

        Steps:
        - Trim and collapse whitespace
        - Uppercase labels (ARR, MRR, MAU, DAU, GMV)
        - Preserve global currency and magnitude formats (e.g., $, €, £, ¥, K/M/B, million/billion)
        """
        if not citations:
            return []
        out: List[str] = []
        seen = set()
        for c in citations:
            v = (c or "").strip()
            if not v:
                continue
            # Collapse whitespace
            v = " ".join(v.split())
            # Normalize common labels to uppercase
            v = self._uppercase_labels(v)
            # Keep other tokens as-is for global compatibility
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def _uppercase_labels(self, text: str) -> str:
        # Replace common metric labels regardless of original case
        for label in self.LABELS:
            text = self._case_replace(text, label.lower(), label)
            text = self._case_replace(text, label.capitalize(), label)
            text = self._case_replace(text, label, label)
        return text

    def _case_replace(self, text: str, old: str, new: str) -> str:
        # Simple string replace without regex
        return text.replace(old, new)

    # -------------------------
    # Source grouping and rendering
    # -------------------------

    def format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """
        Group sources by (company, filename), merge titles/pages/metrics,
        and render in a compact bullet list.

        Input fields tolerated as optional; formatter degrades gracefully.
        """
        if not sources:
            return ""
        grouped = self._group_sources(sources)
        lines: List[str] = []
        for _, data in grouped:
            company = data.get("company") or "Unknown Company"
            title = (data.get("title") or "").strip()
            filename = (data.get("filename") or "").strip()
            pages = data.get("pages", [])
            key_metrics = self.normalize_citations(data.get("key_metrics", []))

            # First line
            head = f"- {company}"
            if title:
                head += f' — "{title}"'
            if pages:
                if len(pages) == 1:
                    head += f", page {pages[0]}"
                else:
                    head += f", pages {', '.join(str(p) for p in pages)}"
            if filename:
                head += f" ({filename})"
            lines.append(head)

            # Metrics line
            if key_metrics:
                lines.append(f"  Metrics: {', '.join(key_metrics)}")

        return "\n".join(lines).strip()

    def _group_sources(
        self,
        sources: List[Dict[str, Any]]
    ) -> List[Tuple[Tuple[str, str], Dict[str, Any]]]:
        """
        Group by (company, filename). Merge:
        - title (prefer longest non-empty)
        - pages (deduplicated, sorted)
        - key_metrics (union)
        Returns a list sorted by company then filename.
        """
        bucket: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for s in sources:
            company = ((s.get("company") or "").strip()) or "Unknown Company"
            filename = (s.get("filename") or "").strip()
            key = (company, filename)

            entry = bucket.get(key)
            if entry is None:
                entry = {
                    "company": company,
                    "filename": filename,
                    "title": (s.get("title") or "").strip(),
                    "pages": [],
                    "key_metrics": set()
                }
                bucket[key] = entry

            # Merge title: keep longer non-empty
            incoming_title = (s.get("title") or "").strip()
            if incoming_title and len(incoming_title) > len(entry.get("title", "")):
                entry["title"] = incoming_title

            # Merge pages
            p = s.get("page")
            if isinstance(p, int):
                entry["pages"].append(p)

            # Merge metrics
            for m in (s.get("key_metrics") or []):
                m_norm = self._uppercase_labels((" ".join((m or "").split())).strip())
                if m_norm:
                    entry["key_metrics"].add(m_norm)

        # Finalize pages and order
        grouped: List[Tuple[Tuple[str, str], Dict[str, Any]]] = []
        for key, data in bucket.items():
            # Deduplicate and sort pages
            if data.get("pages"):
                pages_sorted = sorted(set([p for p in data["pages"] if isinstance(p, int)]))
                data["pages"] = pages_sorted
            else:
                data["pages"] = []
            # Convert metrics to list and sort for stable display
            data["key_metrics"] = list(sorted(data["key_metrics"]))
            grouped.append((key, data))

        # Sort by company, then filename for stable display
        grouped.sort(key=lambda x: (x[0][0].lower(), x[0][1].lower()))
        return grouped


# Global instance for easy import across the app
startup_citation_formatter = StartupCitationFormatter()
