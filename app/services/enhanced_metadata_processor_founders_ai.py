"""
Enhanced metadata processor for startup case studies.

Provides intelligent metadata extraction from startup documents including:
- Company name and document type detection
- Business metrics extraction (ARR, MRR, MAU, DAU, GMV, funding rounds)
- Sector and stage classification
- Structured metadata for better retrieval and citation accuracy

This processor is region-agnostic and supports global startup case studies.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class DocumentTitleExtractor:
    """Extracts meaningful titles from startup documents."""

    def __init__(self):
        # Startup document patterns
        self.title_patterns = [
            # Case study and success story patterns
            r"(?:Case Study|Success Story|Company Profile)\s*:?\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
            r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:Case Study|Success Story|Profile)",

            # Company-specific patterns
            r"(?:How|Why)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:Built|Scaled|Grew|Achieved)",
            r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s*-\s*(?:Building|Scaling|Growing)",

            # Generic business document patterns
            r"(?:Business Model|Strategy|Growth|Expansion)\s+of\s+([A-Z][a-zA-Z]+)",
            r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+Overview",
            r"Funding Round\s+(?:Report|Details)\s*:?\s*([A-Z][a-zA-Z]+)",
        ]

        self.compiled_patterns = [re.compile(pattern) for pattern in self.title_patterns]

    def extract_title(self, content: str, filename: str) -> str:
        """
        Extract a meaningful title from document content and filename.

        Strategy:
        1. Try to match startup-specific patterns in content
        2. Parse company name from filename
        3. Fall back to cleaned filename

        Args:
            content: Document content (first 500 chars typically sufficient)
            filename: Original filename

        Returns:
            Human-readable document title
        """
        # Try content patterns first
        for pattern in self.compiled_patterns:
            match = pattern.search(content[:1000])
            if match:
                return match.group(0).strip()

        # Try filename parsing
        if filename:
            base_name = Path(filename).stem
            # Remove common suffixes
            base_name = re.sub(r'(?i)[-_](case[-_]?study|success[-_]?story|profile|overview)$', '', base_name)
            # Convert separators to spaces
            clean_name = re.sub(r'[_-]+', ' ', base_name)
            # Capitalize
            clean_name = ' '.join(word.capitalize() for word in clean_name.split())
            if clean_name and len(clean_name) > 3:
                return clean_name

        # Fallback
        return "Startup Document"


class StartupMetricExtractor:
    """Extracts business metrics from startup documents (region-agnostic)."""

    def __init__(self):
        # Common metric patterns (support $ € £ ¥ and K/M/B notation)
        self.metric_patterns = [
            # Revenue metrics
            r"(?:ARR|Annual Recurring Revenue)\s*:?\s*([\$€£¥]?\s*[0-9][0-9.,]*\s*(?:K|M|B|Million|Billion)?)",
            r"(?:MRR|Monthly Recurring Revenue)\s*:?\s*([\$€£¥]?\s*[0-9][0-9.,]*\s*(?:K|M|B|Million|Billion)?)",
            r"(?:GMV|Gross Merchandise Value)\s*:?\s*([\$€£¥]?\s*[0-9][0-9.,]*\s*(?:K|M|B|Million|Billion)?)",
            r"(?:Revenue)\s*:?\s*([\$€£¥]?\s*[0-9][0-9.,]*\s*(?:K|M|B|Million|Billion)?)",

            # User metrics
            r"(?:MAU|Monthly Active Users)\s*:?\s*([0-9][0-9.,]*\s*(?:K|M|B|Million|Billion)?)",
            r"(?:DAU|Daily Active Users)\s*:?\s*([0-9][0-9.,]*\s*(?:K|M|B|Million|Billion)?)",
            r"([0-9][0-9.,]+\s*(?:M|Million|B|Billion)?)\s+(?:users|customers|downloads)",

            # Funding metrics
            r"(?:Series [A-K]|Seed|Pre-Seed)\s+(?:Round)?\s*:?\s*([\$€£¥]?\s*[0-9][0-9.,]*\s*(?:K|M|B|Million|Billion)?)",
            r"(?:Valuation)\s*:?\s*([\$€£¥]?\s*[0-9][0-9.,]*\s*(?:K|M|B|Million|Billion)?)",

            # Growth metrics
            r"([0-9]+(?:\.[0-9]+)?%)\s+(?:growth|retention|churn|conversion|CAGR)",
        ]

        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.metric_patterns]

    def extract_metrics(self, content: str) -> List[str]:
        """
        Extract business metrics from document content.

        Returns unique, normalized metric strings sorted by length.
        """
        metrics = set()
        for pattern in self.compiled_patterns:
            matches = pattern.findall(content)
            for match in matches:
                # Handle tuple results from groups
                if isinstance(match, tuple):
                    match = match[0]
                normalized = self._normalize_metric(match)
                if normalized:
                    metrics.add(normalized)
        return sorted(list(metrics), key=lambda x: (len(x), x))

    def _normalize_metric(self, metric: str) -> str:
        """Normalize metric string (trim, collapse whitespace)."""
        return " ".join((metric or "").split()).strip()


class DocumentMetadataEnhancer:
    """
    Enhances startup document metadata for better retrieval and citation.

    Extracts:
    - Company name and document title
    - Business metrics (ARR, MAU, funding rounds, etc.)
    - Document type (case study, success story, funding report, etc.)
    - Content statistics and preview
    """

    def __init__(self):
        self.title_extractor = DocumentTitleExtractor()
        self.metric_extractor = StartupMetricExtractor()

    def enhance_metadata(
        self,
        content: str,
        filename: str,
        chunk_index: int,
        total_chunks: int,
        file_path: str,
        source: str = "uploaded-pdf"
    ) -> Dict[str, Any]:
        """
        Enhance document metadata with structured startup information.

        This method processes a document chunk and returns enriched metadata
        suitable for vector storage and retrieval.

        Args:
            content: Document text content
            filename: Original filename
            chunk_index: Zero-based chunk index
            total_chunks: Total number of chunks for this document
            file_path: Full file path
            source: Source identifier (default: "uploaded-pdf")

        Returns:
            Dictionary containing enhanced metadata with startup-specific fields
        """
        # Extract title
        title = self.title_extractor.extract_title(content, filename)

        # Extract metrics
        metrics = self.metric_extractor.extract_metrics(content)

        # Generate document ID
        doc_id = self._generate_document_id(filename, file_path)

        # Determine document type
        doc_type = self._determine_document_type(filename, content)

        # Extract key information
        key_info = self._extract_key_information(content, title)

        # Build enhanced metadata
        enhanced_metadata = {
            # Basic document info
            "document_id": doc_id,
            "title": title,
            "filename": filename,
            "file_path": file_path,
            "source": source,
            "document_type": doc_type,

            # Chunk information
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "chunk_size": len(content),
            "chunk_ratio": f"{chunk_index + 1}/{total_chunks}",

            # Business metrics
            "startup_metrics": metrics,
            "metric_count": len(metrics),
            "key_metrics": metrics[:5],  # Top 5 most prominent

            # Content analysis
            "word_count": len(content.split()),
            "character_count": len(content),
            "has_metrics": len(metrics) > 0,
            "content_preview": content[:200] + "..." if len(content) > 200 else content,

            # Key information
            "key_information": key_info,

            # Processing metadata
            "processed_at": datetime.utcnow().isoformat(),
            "metadata_version": "2.0",
            "enhanced": True
        }

        return enhanced_metadata

    def _generate_document_id(self, filename: str, file_path: str) -> str:
        """Generate a consistent unique document ID from filename and path."""
        base_string = f"{filename}_{file_path}"
        return hashlib.md5(base_string.encode()).hexdigest()[:12]

    def _determine_document_type(self, filename: str, content: str) -> str:
        """
        Determine the type of startup document.

        Common types:
        - case_study: Detailed company analysis
        - success_story: Growth narrative
        - funding_round: Investment details
        - company_profile: Overview
        - strategy_report: Business strategy analysis
        """
        filename_lower = filename.lower()
        content_lower = content.lower()

        if "case" in filename_lower and "study" in filename_lower:
            return "case_study"
        elif "success" in filename_lower or "story" in filename_lower:
            return "success_story"
        elif "funding" in filename_lower or "round" in filename_lower:
            return "funding_round"
        elif "profile" in filename_lower or "overview" in filename_lower:
            return "company_profile"
        elif "strategy" in content_lower or "business model" in content_lower:
            return "strategy_report"
        else:
            return "startup_document"

    def _extract_key_information(self, content: str, title: str) -> Dict[str, Any]:
        """
        Extract key thematic flags from document content.

        These flags help with filtering and personalization during retrieval.
        """
        content_lower = content.lower()

        key_info = {
            "document_title": title,
            "is_b2b": any(keyword in content_lower for keyword in [
                "b2b", "enterprise", "saas", "business-to-business"
            ]),
            "is_b2c": any(keyword in content_lower for keyword in [
                "b2c", "consumer", "retail", "business-to-consumer"
            ]),
            "mentions_funding": any(keyword in content_lower for keyword in [
                "series a", "series b", "seed", "funding", "raised", "valuation"
            ]),
            "mentions_growth": any(keyword in content_lower for keyword in [
                "growth", "scaling", "expansion", "user acquisition", "retention"
            ]),
            "mentions_product": any(keyword in content_lower for keyword in [
                "product-market fit", "pmf", "launch", "mvp", "iteration"
            ]),
            "has_metrics": any(keyword in content_lower for keyword in [
                "arr", "mrr", "mau", "dau", "gmv", "revenue"
            ]),
        }

        return key_info


class MetadataProcessor:
    """
    Main metadata processor for startup document ingestion.

    This is the primary interface for processing documents during ingestion.
    Use process_document_metadata for each document chunk.
    """

    def __init__(self):
        self.enhancer = DocumentMetadataEnhancer()

    def process_document_metadata(
        self,
        content: str,
        filename: str,
        chunk_index: int,
        total_chunks: int,
        file_path: str,
        source: str = "uploaded-pdf"
    ) -> Dict[str, Any]:
        """
        Process and enhance document metadata for a single chunk.

        This method wraps the enhancer with error handling and logging.
        If enhancement fails, returns basic fallback metadata.

        Args:
            content: Document text
            filename: Original filename
            chunk_index: Zero-based index
            total_chunks: Total chunks in document
            file_path: Full file path
            source: Source identifier

        Returns:
            Enhanced metadata dictionary or fallback on error
        """
        try:
            enhanced_metadata = self.enhancer.enhance_metadata(
                content, filename, chunk_index, total_chunks, file_path, source
            )
            logger.info(f"Enhanced metadata for {filename} chunk {chunk_index + 1}/{total_chunks}")
            return enhanced_metadata

        except Exception as e:
            logger.error(f"Error processing metadata for {filename}: {e}")
            # Return basic fallback metadata
            return {
                "title": filename,
                "filename": filename,
                "file_path": file_path,
                "source": source,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "document_type": "startup_document",
                "startup_metrics": [],
                "enhanced": False,
                "error": str(e)
            }


# Global instance for easy importing
enhanced_metadata_processor = MetadataProcessor()