"""
Founders AI Adaptive RAG Orchestrator

Coordinates between different RAG strategies based on query intent,
complexity, and context. Uses modular subcomponents like:
- QueryIntentClassifier
- Lightweight LLM RAG
- Multi-hop Reasoning Engine
- Hallucination Validator
- Enhanced Metadata Processor
"""

import asyncio
import logging
import time
from typing import Dict, Any

from .lightweight_llm_rag_founders_ai import lightweight_llm_rag
from .multi_hop_reasoning_founders_ai import multi_hop_reasoning_engine
from .enhanced_metadata_processor_founders_ai import enhanced_metadata_processor
from .enhanced_citation_formatter_founders_ai import startup_citation_formatter
from .hallucination_validator_founders_ai import hallucination_validator
from .query_intent_classifier import query_intent_classifier
from .prompt_templates_founders_ai import PromptTemplate
from .query_complexity_detector import query_complexity_detector
from .reasoning_chain_storage import reasoning_chain_storage
from .startup_tools import extract_startup_citations
from .startup_classifier import startup_classifier

logger = logging.getLogger(__name__)


class FoundersAIOrchestrator:
    """
    Adaptive orchestrator that intelligently selects and coordinates
    between RAG components based on query intent, type, and complexity.
    """

    async def process_query(self, query: str, user_preferences: Dict[str, Any] = None):
        start_time = time.time()

        # -------------------------
        # Step 1: Intent Detection
        # -------------------------
        try:
            if hasattr(query_intent_classifier, "classify"):
                intent_result = query_intent_classifier.classify(query)
            elif callable(query_intent_classifier):
                intent_result = query_intent_classifier(query)
            else:
                raise TypeError(
                    "Invalid query_intent_classifier — missing classify() or callable interface"
                )

            if isinstance(intent_result, dict):
                intent = intent_result.get("intent", "unknown")
                confidence = intent_result.get("confidence", 1.0)
            else:
                intent = getattr(intent_result, "intent", "unknown")
                confidence = getattr(intent_result, "confidence", 1.0)

        except Exception as e:
            logger.warning(f"[Orchestrator] Intent classification failed: {e}")
            intent, confidence = "unknown", 0.0

        # -------------------------
        # Step 2: Complexity Detection
        # -------------------------
        try:
            use_multi_hop, complexity = query_complexity_detector.should_use_multi_hop_reasoning(
                query
            )
        except Exception as e:
            logger.warning(f"[Orchestrator] Complexity detection failed: {e}")
            use_multi_hop, complexity = False, {"error": str(e)}

        # -------------------------
        # Step 3: Engine Routing
        # -------------------------
        try:
            if use_multi_hop:
                logger.info(f"[Orchestrator] Using multi-hop reasoning for: {query}")
                engine_result = await multi_hop_reasoning_engine.process_complex_query(
                    query=query
                )
                response = engine_result.final_answer
                sources = engine_result.citations
                metadata = engine_result.metadata
            else:
                logger.info(f"[Orchestrator] Using lightweight RAG for: {query}")
                engine_result = await lightweight_llm_rag.process_query(query=query)
                response = engine_result.get("response", "")
                sources = engine_result.get("sources", [])
                metadata = engine_result.get("metadata", {})

        except Exception as e:
            logger.error(f"[Orchestrator] Engine processing failed: {e}")
            response, sources, metadata = f"Error: {e}", [], {"error": str(e)}

        # -------------------------
        # Step 4: Validation & Enhancement
        # -------------------------
        try:
            validated_response = hallucination_validator.validate(response, sources)
            formatted_response = enhanced_metadata_processor.add_metadata(
                validated_response, query=query
            )
            citations = startup_citation_formatter.format_sources(sources)
        except Exception as e:
            logger.warning(f"[Orchestrator] Response validation failed: {e}")
            formatted_response = response
            citations = sources

        # -------------------------
        # Step 5: Store Reasoning Chain
        # -------------------------
        try:
            reasoning_chain_storage.store(
                query=query,
                response=formatted_response,
                metadata={
                    "intent": intent,
                    "confidence": confidence,
                    "complexity": complexity,
                    **metadata,
                },
            )
        except Exception as e:
            logger.warning(f"[Orchestrator] Could not store reasoning chain: {e}")

        # -------------------------
        # Step 6: Final Assembly
        # -------------------------
        processing_time = time.time() - start_time

        return type("AdaptiveResult", (), {})(
            response=formatted_response,
            sources=citations,
            intent=intent,
            confidence=confidence,
            metadata={
                "complexity": complexity,
                "components_used": [
                    "query_intent_classifier",
                    "query_complexity_detector",
                    "multi_hop_reasoning_engine" if use_multi_hop else "lightweight_llm_rag",
                    "hallucination_validator",
                    "enhanced_metadata_processor",
                    "enhanced_citation_formatter",
                ],
            },
            processing_time=processing_time,
        )


# Singleton instance for import
founders_ai_orchestrator = FoundersAIOrchestrator()
