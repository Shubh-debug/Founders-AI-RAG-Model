"""
Query complexity detection and routing system for Founders AI.

Automatically detects complex founder queries that require multi-hop reasoning and
routes them to the appropriate processing pipeline.
"""

import logging
import re
from typing import Dict, Any, Tuple, List
from enum import Enum

from .multi_hop_reasoning_founders_ai import ReasoningComplexity, QueryComplexityAnalyzer

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Types of startup/founder queries based on complexity and structure"""
    SIMPLE_FACTUAL = "simple_factual"
    DEFINITION = "definition"
    COMPARATIVE = "comparative"
    ANALYTICAL = "analytical"
    MULTI_ASPECT = "multi_aspect"
    PROCEDURAL = "procedural"
    INTERPRETATIVE = "interpretative"


class QueryComplexityDetector:
    """Advanced query complexity detection with startup domain awareness and routing."""

    def __init__(self):
        self.complexity_analyzer = QueryComplexityAnalyzer()

        # Startup-specific complexity patterns (bounded to avoid backtracking)
        self.startup_complexity_patterns = {
            "multi_metric": [
                r"\b(arr|mrr|gmv|mau|dau|ltv|cac|churn|retention)\b.*?\b(arr|mrr|gmv|mau|dau|ltv|cac|churn|retention)\b",
                r"\b(series\s+[a-k]\+?)\b.*?\b(valuation|dilution|round)\b",
            ],
            "conditional_reasoning": [
                r"\bif\b.{1,60}\bthen\b",
                r"\bwhen\b.{1,60}\bshould\b",
                r"\bunless\b.{1,60}\bconsider\b",
                r"\bin\s+case\b.{1,60}\bwhere\b",
            ],
            "comparative_analysis": [
                r"\bcompare\b.{1,60}\bwith\b",
                r"\bdifference\s+between\b",
                r"\bsimilarit(?:y|ies)\b.{1,60}\band\b",
                r"\bversus\b.{1,10}\bvs\b",
            ],
            "cross_company": [
                r"\bacross\b.{1,60}\b(companies|markets|segments)\b",
                r"\bthroughout\b.{1,60}\b(cohorts|stages)\b",
                r"\bin\s+all\b.{1,60}\b(segments|channels)\b",
                r"\bvarious\b.{1,60}\b(business models|geographies)\b",
            ],
            "procedural_complexity": [
                r"\bstep\s+by\s+step\b",
                r"\bprocess\b.{1,60}\bprocedure\b",
                r"\bworkflow\b.{1,60}\bsequence\b",
                r"\btimeline\b.{1,60}\bstages\b",
            ],
            "financial_reasoning": [
                r"\b(ltv\/?cac|payback|gross\s+margin|burn\s+multiple|magic\s+number)\b",
                r"\b(cohort|retention\s+curve|net\s+revenue\s+retention|nrr|grr)\b",
            ],
        }

        # Query type classification patterns (startup phrasing)
        self.query_type_patterns = {
            QueryType.SIMPLE_FACTUAL: [
                r"\bwhat\s+is\b",
                r"\bdefine\b",
                r"\bwhat\s+does.*?mean\b",
            ],
            QueryType.DEFINITION: [
                r"\bdefinition\s+of\b",
                r"\bmeaning\s+of\b",
                r"\bwhat\s+is.*?defined\s+as\b",
            ],
            QueryType.COMPARATIVE: [
                r"\bcompare\b.*?\band\b",
                r"\bdifference\s+between\b",
                r"\bsimilarit(?:y|ies)\b.*?\band\b",
                r"\bversus\b.*?\bvs\b",
            ],
            QueryType.ANALYTICAL: [
                r"\banalyze\b.*?\bimplications\b",
                r"\bevaluate\b.*?\beffectiveness\b",
                r"\bassess\b.*?\bimpact\b",
                r"\bexamine\b.*?\bconsequences\b",
                r"\bhow\b.*?\baffect(s)?\b.*?\bgrowth|retention|unit\s+economics\b",
            ],
            QueryType.MULTI_ASPECT: [
                r"\ball\s+aspects\s+of\b",
                r"\bvarious.*?\belements\b",
                r"\bdifferent.*?\bcomponents\b",
                r"\bmultiple.*?\bfactors\b",
            ],
            QueryType.PROCEDURAL: [
                r"\bhow\s+to\b.*?\bprocess\b",
                r"\bsteps\s+involved\b",
                r"\bprocedure\s+for\b",
                r"\bworkflow.*?\bsequence\b",
                r"\bplaybook\b|\bros?admap\b",
            ],
            QueryType.INTERPRETATIVE: [
                r"\binterpretation\s+of\b",
                r"\bhow\s+to\s+interpret\b",
                r"\bmeaning\s+and\s+scope\b",
                r"\bwhat\s+does\b.*?\bimply\b",
            ],
        }

    def detect_complexity_and_type(self, query: str) -> Tuple[ReasoningComplexity, QueryType, Dict[str, Any]]:
        """
        Detect query complexity and type with detailed analysis.

        Args:
            query: The founder query to analyze

        Returns:
            Tuple of (complexity_level, query_type, analysis_details)
        """
        query_lower = query.lower().strip()

        # Base complexity from analyzer (already startup-aware)
        complexity_level, base_analysis = self.complexity_analyzer.analyze_complexity(query)

        # Detect startup-specific complexity patterns
        startup_patterns_found = self._detect_startup_patterns(query_lower)

        # Classify query type
        query_type = self._classify_query_type(query_lower)

        # Enhanced score with startup signals
        enhanced_score = self._calculate_enhanced_complexity_score(
            base_analysis, startup_patterns_found, query_type
        )

        # Final level tuned for startup/multi-company synthesis
        final_complexity = self._determine_final_complexity(enhanced_score, startup_patterns_found)

        analysis_details = {
            **base_analysis,
            "startup_patterns_found": startup_patterns_found,
            "query_type": query_type.value,
            "enhanced_complexity_score": enhanced_score,
            "final_complexity_level": final_complexity.value,
            "requires_multi_hop": final_complexity in [ReasoningComplexity.COMPLEX, ReasoningComplexity.VERY_COMPLEX],
            "recommended_approach": self._get_recommended_approach(final_complexity, query_type),
        }

        logger.info(f"Query analysis: {final_complexity.value} complexity, {query_type.value} type")
        return final_complexity, query_type, analysis_details

    def _detect_startup_patterns(self, query_lower: str) -> Dict[str, List[str]]:
        """Detect startup-specific complexity patterns in the query"""
        patterns_found: Dict[str, List[str]] = {}

        for category, patterns in self.startup_complexity_patterns.items():
            found: List[str] = []
            for pattern in patterns:
                matches = re.findall(pattern, query_lower, re.IGNORECASE)
                if matches:
                    # Flatten tuples from grouped regexes
                    for m in matches:
                        if isinstance(m, tuple):
                            found.append(" ".join([x for x in m if x]))
                        else:
                            found.append(m)
            if found:
                patterns_found[category] = found

        return patterns_found

    def _classify_query_type(self, query_lower: str) -> QueryType:
        """Classify the type of startup/founder query"""
        type_scores: Dict[QueryType, int] = {}

        for qtype, patterns in self.query_type_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    score += 1
            type_scores[qtype] = score

        return max(type_scores.items(), key=lambda x: x[1])[0] if type_scores else QueryType.SIMPLE_FACTUAL

    def _calculate_enhanced_complexity_score(
        self,
        base_analysis: Dict[str, Any],
        startup_patterns: Dict[str, List[str]],
        query_type: QueryType,
    ) -> float:
        """Calculate enhanced complexity score considering startup patterns and query type"""
        base_score = base_analysis.get("complexity_score", 0)

        # Pattern bonuses
        pattern_bonus = 0
        for category, items in startup_patterns.items():
            weight = 2 if category in {"multi_metric", "cross_company", "financial_reasoning"} else 1
            pattern_bonus += len(items) * weight

        # Query type complexity
        type_bonus_map = {
            QueryType.SIMPLE_FACTUAL: 0,
            QueryType.DEFINITION: 1,
            QueryType.COMPARATIVE: 3,
            QueryType.ANALYTICAL: 4,
            QueryType.MULTI_ASPECT: 5,
            QueryType.PROCEDURAL: 3,
            QueryType.INTERPRETATIVE: 4,
        }
        type_bonus = type_bonus_map.get(query_type, 0)

        # Length/structure bonus
        length_bonus = 0
        word_count = base_analysis.get("word_count", 0)
        if word_count > 120:
            length_bonus = 3
        elif word_count > 60:
            length_bonus = 2
        elif word_count > 35:
            length_bonus = 1

        return base_score + pattern_bonus + type_bonus + length_bonus

    def _determine_final_complexity(
        self,
        enhanced_score: float,
        startup_patterns: Dict[str, List[str]],
    ) -> ReasoningComplexity:
        """Determine final complexity with startup-aware thresholds"""
        if "cross_company" in startup_patterns or "multi_metric" in startup_patterns:
            if enhanced_score >= 7:
                return ReasoningComplexity.VERY_COMPLEX
            elif enhanced_score >= 5:
                return ReasoningComplexity.COMPLEX
        elif "comparative_analysis" in startup_patterns or "financial_reasoning" in startup_patterns:
            if enhanced_score >= 8:
                return ReasoningComplexity.VERY_COMPLEX
            elif enhanced_score >= 5:
                return ReasoningComplexity.COMPLEX
        else:
            if enhanced_score >= 9:
                return ReasoningComplexity.VERY_COMPLEX
            elif enhanced_score >= 6:
                return ReasoningComplexity.COMPLEX
            elif enhanced_score >= 3:
                return ReasoningComplexity.MODERATE

        return ReasoningComplexity.SIMPLE

    def _get_recommended_approach(self, complexity: ReasoningComplexity, query_type: QueryType) -> str:
        """Get recommended processing approach based on complexity and type"""
        if complexity == ReasoningComplexity.VERY_COMPLEX:
            return "multi_hop_reasoning_with_cross_company_synthesis"
        elif complexity == ReasoningComplexity.COMPLEX:
            if query_type in [QueryType.COMPARATIVE, QueryType.ANALYTICAL, QueryType.MULTI_ASPECT]:
                return "multi_hop_reasoning"
            else:
                return "enhanced_rag_with_agent"
        elif complexity == ReasoningComplexity.MODERATE:
            if query_type in [QueryType.PROCEDURAL, QueryType.INTERPRETATIVE]:
                return "enhanced_rag"
            else:
                return "standard_rag"
        else:
            return "standard_rag"

    def should_use_multi_hop_reasoning(self, query: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if a founder query should use multi-hop reasoning.
        """
        complexity, query_type, analysis = self.detect_complexity_and_type(query)
        return analysis.get("requires_multi_hop", False), analysis


# Global instance
query_complexity_detector = QueryComplexityDetector()
