"""
Unified model package initializer for Founders AI.

Exports:
- Pydantic request/response models from `requests.py`
- SQLAlchemy ORM models from `document.py`

This allows clean imports like:
    from app.models import FounderQueryRequest, StartupCaseStudy
"""

from .requests import (
    FounderQueryRequest,
    FounderQueryResponse,
    ReasoningStepResponse,
    MultiHopReasoningResponse,
    PDFIngestionRequest,
    PDFIngestionResponse,
    HealthCheckResponse,
    ServiceInfoResponse,
    ReasoningChainRequest,
)

from .document import (
    StartupCaseStudy,
    FounderConversationHistory,
    DocumentProcessingStatus,
)

__all__ = [
    # Pydantic Models
    "FounderQueryRequest",
    "FounderQueryResponse",
    "ReasoningStepResponse",
    "MultiHopReasoningResponse",
    "PDFIngestionRequest",
    "PDFIngestionResponse",
    "HealthCheckResponse",
    "ServiceInfoResponse",
    "ReasoningChainRequest",
    
    # SQLAlchemy Models
    "StartupCaseStudy",
    "FounderConversationHistory",
    "DocumentProcessingStatus",
]
