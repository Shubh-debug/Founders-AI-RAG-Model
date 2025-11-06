# app/services/__init__.py
"""
Expose service singletons / factories in a package-safe manner.
Avoid heavy imports at package import time to reduce circular imports.
"""

# keep this file minimal — import actual modules lazily elsewhere
__all__ = [
    "lightweight_llm_rag",
    "multi_hop_reasoning_engine",
    "pdf_ingestion_service",
    "feedback_system",
    "query_complexity_detector",
    "query_intent_classifier",
]

