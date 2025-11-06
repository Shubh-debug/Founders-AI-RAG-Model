"""
Multi-hop reasoning system for complex startup and founder queries.

Provides iterative query decomposition, intermediate reasoning steps, and
comprehensive synthesis of complex startup questions requiring multiple reasoning hops.

This system handles sophisticated startup strategy questions by breaking them into
sub-questions, reasoning through each independently, and synthesizing insights
from multiple case studies into actionable founder guidance.
"""

import asyncio
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------- ENUM DEFINITIONS ----------------
class ReasoningComplexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class ReasoningStepType(str, Enum):
    QUERY_DECOMPOSITION = "query_decomposition"
    SUB_QUERY_EXECUTION = "sub_query_execution"
    INFORMATION_SYNTHESIS = "information_synthesis"
    CONFLICT_RESOLUTION = "conflict_resolution"
    FINAL_SYNTHESIS = "final_synthesis"


# ---------------- DATA CLASSES ----------------
@dataclass
class ReasoningStep:
    step_id: str
    step_type: ReasoningStepType
    input_query: str
    output_result: str
    sources_used: List[Dict[str, Any]]
    confidence_score: float
    execution_time: float
    metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class ReasoningChain:
    chain_id: str
    original_query: str
    complexity_level: ReasoningComplexity
    steps: List[ReasoningStep]
    final_answer: str
    total_execution_time: float
    overall_confidence: float
    citations: List[str]
    metadata: Dict[str, Any]
    created_at: datetime


# ---------------- ANALYZER ----------------
class QueryComplexityAnalyzer:
    """Analyzes query complexity for multi-hop reasoning using linguistic and startup indicators."""

    COMPLEXITY_INDICATORS = {
        "multiple_concepts": [
            "and", "also", "furthermore", "additionally", "moreover",
            "in addition", "as well as", "along with", "together with"
        ],
        "conditional_reasoning": [
            "if", "when", "unless", "provided that", "in case",
            "assuming", "supposing", "given that"
        ],
        "comparative_analysis": [
            "compare", "contrast", "difference", "similarity",
            "versus", "vs", "between", "among"
        ],
        "causal_reasoning": [
            "because", "due to", "as a result", "consequently",
            "therefore", "thus", "hence", "causes", "leads to"
        ],
        "startup_complexity": [
            "series", "round", "funding", "valuation", "revenue",
            "arr", "mrr", "mau", "dau", "growth",
            "sector", "market", "pmf", "unicorn", "exit", "pivot"
        ],
        "multi_document": [
            "across", "throughout", "in all", "various", "different",
            "multiple", "several", "both", "each", "respective"
        ],
    }

    @classmethod
    def analyze_complexity(cls, query: str) -> Tuple[ReasoningComplexity, Dict[str, Any]]:
        query_lower = query.lower()
        complexity_score = 0
        detected_indicators: Dict[str, List[str]] = {}

        for category, indicators in cls.COMPLEXITY_INDICATORS.items():
            found = [ind for ind in indicators if ind in query_lower]
            if found:
                detected_indicators[category] = found
                complexity_score += len(found)

        word_count = len(query.split())
        sentence_count = sum(query.count(p) for p in ".?!")
        startup_terms = sum(1 for term in cls.COMPLEXITY_INDICATORS["startup_complexity"] if term in query_lower)

        if word_count > 50:
            complexity_score += 2
        if sentence_count > 2:
            complexity_score += 1
        if startup_terms > 3:
            complexity_score += 2

        if complexity_score >= 8:
            level = ReasoningComplexity.VERY_COMPLEX
        elif complexity_score >= 5:
            level = ReasoningComplexity.COMPLEX
        elif complexity_score >= 3:
            level = ReasoningComplexity.MODERATE
        else:
            level = ReasoningComplexity.SIMPLE

        return level, {
            "complexity_score": complexity_score,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "startup_terms_count": startup_terms,
            "detected_indicators": detected_indicators,
            "requires_multi_hop": level in [ReasoningComplexity.COMPLEX, ReasoningComplexity.VERY_COMPLEX],
        }


# ---------------- QUERY DECOMPOSER ----------------
class QueryDecomposer:
    """Decomposes complex queries into manageable sub-queries."""

    def __init__(self):
        self.openai_client = None

    async def initialize(self):
        """Initialize OpenAI client for query decomposition."""
        if self.openai_client:
            return
        try:
            import openai
            # ✅ FIX: safer client setup (no blocking)
            self.openai_client = openai.AsyncOpenAI()
        except Exception as e:
            logger.warning(f"OpenAI not available: {e}")
            self.openai_client = None

    async def decompose_query(self, query: str, analysis: Dict[str, Any]) -> List[str]:
        """Break a complex query into focused sub-queries."""
        await self.initialize()
        if not self.openai_client:
            return [query]

        prompt = f"""
Decompose the founder query below into 2–4 actionable, focused sub-queries.

Query: {query}

Detected indicators: {analysis.get('detected_indicators', {})}

Each sub-query should:
- Be concise (max 1 sentence)
- Contain a question mark
- Be independently answerable
- Focus on practical startup aspects
        """

        try:
            # ✅ FIX: added async + timeout
            response = await asyncio.wait_for(
                self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a startup research expert breaking down complex founder questions."},
                        {"role": "user", "content": prompt.strip()},
                    ],
                    max_tokens=400,
                    temperature=0.1,
                ),
                timeout=25,
            )

            msg = getattr(response.choices[0].message, "content", "") if response.choices else ""
            sub_queries = [q.strip() for q in msg.split("\n") if q.strip() and "?" in q]

            return sub_queries or [query]

        except asyncio.TimeoutError:
            logger.warning("Decomposition timed out — fallback to single query.")
            return [query]
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return [query]


# ---------------- MULTI-HOP REASONING ENGINE ----------------
class MultiHopReasoningEngine:
    """Handles multi-hop reasoning for complex founder queries."""

    def __init__(self):
        self.complexity_analyzer = QueryComplexityAnalyzer()
        self.query_decomposer = QueryDecomposer()
        self.reasoning_chains: Dict[str, ReasoningChain] = {}

    async def process_complex_query(self, query: str, session_id: Optional[str] = None) -> ReasoningChain:
        start = time.time()
        chain_id = str(uuid.uuid4())

        try:
            level, analysis = self.complexity_analyzer.analyze_complexity(query)
            sub_queries = (
                await self.query_decomposer.decompose_query(query, analysis)
                if analysis["requires_multi_hop"]
                else [query]
            )

            steps, all_sources, all_citations, confidences = [], [], [], []

            for i, sq in enumerate(sub_queries):
                step = await self._execute_reasoning_step(chain_id, sq, i, len(sub_queries))
                steps.append(step)
                all_sources.extend(step.sources_used)
                all_citations.extend(self._extract_citations(step.output_result))
                confidences.append(step.confidence_score)

            synthesis_start = time.time()
            final_answer = await self._synthesize_final_answer(query, steps, all_sources)
            synthesis_time = time.time() - synthesis_start

            synthesis_step = ReasoningStep(
                step_id=f"{chain_id}_synthesis",
                step_type=ReasoningStepType.FINAL_SYNTHESIS,
                input_query="Final synthesis of all reasoning steps",
                output_result=final_answer,
                sources_used=all_sources,
                confidence_score=sum(confidences) / len(confidences) if confidences else 0.0,
                execution_time=synthesis_time,
                metadata={"synthesis_type": "multi_step_integration"},
                timestamp=datetime.now(),
            )

            steps.append(synthesis_step)

            total_time = time.time() - start
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            chain = ReasoningChain(
                chain_id=chain_id,
                original_query=query,
                complexity_level=level,
                steps=steps,
                final_answer=final_answer,
                total_execution_time=total_time,
                overall_confidence=avg_conf,
                citations=list(set(all_citations)),
                metadata={"analysis_details": analysis, "sub_queries_count": len(sub_queries), "session_id": session_id},
                created_at=datetime.now(),
            )
            self.reasoning_chains[chain_id] = chain
            logger.info(f"✅ Reasoning completed in {total_time:.2f}s, confidence={avg_conf:.2f}")
            return chain

        except Exception as e:
            logger.exception("❌ Multi-hop reasoning failed.")
            return ReasoningChain(
                chain_id=chain_id,
                original_query=query,
                complexity_level=ReasoningComplexity.SIMPLE,
                steps=[],
                final_answer=f"Error: {e}",
                total_execution_time=time.time() - start,
                overall_confidence=0.0,
                citations=[],
                metadata={"error": str(e)},
                created_at=datetime.now(),
            )

    async def _execute_reasoning_step(self, chain_id: str, sub_query: str, index: int, total: int) -> ReasoningStep:
        start = time.time()
        step_id = f"{chain_id}_step_{index+1}"

        try:
            from lightweight_llm_rag_founders_ai import lightweight_llm_rag

            rag_result = await asyncio.wait_for(
                lightweight_llm_rag.process_query(query=sub_query, top_k=8),
                timeout=40,
            )

            response = rag_result.get("response", "")
            sources = rag_result.get("sources", [])
            confidence = self._calculate_step_confidence(sources, response)

            try:
                from startup_classifier import startup_classifier
                domain = startup_classifier.classify(sub_query)
            except Exception:
                domain = {"category": "Other", "confidence": 0.0}

            return ReasoningStep(
                step_id=step_id,
                step_type=ReasoningStepType.SUB_QUERY_EXECUTION,
                input_query=sub_query,
                output_result=response,
                sources_used=sources,
                confidence_score=confidence,
                execution_time=time.time() - start,
                metadata={"step_index": index, "total_steps": total, **domain},
                timestamp=datetime.now(),
            )

        except asyncio.TimeoutError:
            logger.warning(f"Step {index+1} timed out for query: {sub_query}")
            return ReasoningStep(
                step_id=step_id,
                step_type=ReasoningStepType.SUB_QUERY_EXECUTION,
                input_query=sub_query,
                output_result="Timeout: No response received.",
                sources_used=[],
                confidence_score=0.0,
                execution_time=time.time() - start,
                metadata={"error": "timeout"},
                timestamp=datetime.now(),
            )
        except Exception as e:
            logger.error(f"Step execution failed: {e}")
            return ReasoningStep(
                step_id=step_id,
                step_type=ReasoningStepType.SUB_QUERY_EXECUTION,
                input_query=sub_query,
                output_result=f"Error: {e}",
                sources_used=[],
                confidence_score=0.0,
                execution_time=time.time() - start,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
            )

    def _calculate_step_confidence(self, sources: List[Dict[str, Any]], response: str) -> float:
        if not sources or not response:
            return 0.0
        avg_sim = sum(s.get("similarity_score", 0.5) for s in sources) / len(sources)
        indicators = {
            "has_citations": len(self._extract_citations(response)) > 0,
            "enough_text": len(response.split()) > 20,
            "has_terms": any(t in response.lower() for t in ["arr", "mrr", "mau", "revenue", "growth"]),
            "multi_source": len(sources) > 1,
        }
        bonus = sum(indicators.values()) * 0.1
        return round(min(1.0, avg_sim + bonus), 3)

    async def _synthesize_final_answer(self, query: str, steps: List[ReasoningStep], all_sources: List[Dict[str, Any]]) -> str:
        try:
            await self.query_decomposer.initialize()
            if not self.query_decomposer.openai_client:
                return self._fallback_synthesis(steps)

            context = "\n".join(
                f"Step {i+1}: {s.input_query}\nAnswer: {s.output_result}\n"
                for i, s in enumerate(steps)
            )

            prompt = f"""
You are a startup research expert. Synthesize the following reasoning steps into a direct, actionable founder answer.

Query: {query}

Steps:
{context}

Rules:
- Lead with the answer
- Combine insights across steps
- Keep it factual and concise
- Include startup examples and metrics
"""

            resp = await self.query_decomposer.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You summarize multi-step reasoning into concise founder advice."},
                    {"role": "user", "content": prompt.strip()},
                ],
                max_tokens=800,
                temperature=0.2,
            )
            return getattr(resp.choices[0].message, "content", "").strip() or self._fallback_synthesis(steps)

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return self._fallback_synthesis(steps)

    def _fallback_synthesis(self, steps: List[ReasoningStep]) -> str:
        return "\n".join(
            f"{i+1}. {s.output_result}" for i, s in enumerate(steps)
        ) or "No synthesis available."

    def _extract_citations(self, text: str) -> List[str]:
        metrics = ["arr", "mrr", "mau", "dau", "gmv", "series", "funding", "revenue"]
        return [m.upper() for m in metrics if m in text.lower()]

    def get_reasoning_chain(self, chain_id: str) -> Optional[ReasoningChain]:
        return self.reasoning_chains.get(chain_id)

    def get_reasoning_chains_by_session(self, session_id: str) -> List[ReasoningChain]:
        return [
            chain for chain in self.reasoning_chains.values()
            if chain.metadata.get("session_id") == session_id
        ]


# ✅ Global instance
multi_hop_reasoning_engine = MultiHopReasoningEngine()
