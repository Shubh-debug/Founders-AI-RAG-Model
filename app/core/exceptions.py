"""
Custom exception classes for Founders AI RAG System.

Provides structured, typed exception handling for:
- Multi-hop reasoning failures
- RAG retrieval issues
- Embedding generation
- PDF ingestion
- Database/cache issues
- Orchestrator routing errors
- Query validation & complexity detection
"""

from typing import Optional, Dict, Any


# ----------------------------------------------------------------------
# BASE EXCEPTION
# ----------------------------------------------------------------------
class FoundersAIException(Exception):
    """
    Base exception for all Founders AI backend errors.

    Includes:
    - readable message
    - standardized error_code
    - contextual metadata for debugging
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "FOUNDERS_AI_ERROR"
        self.context = context or {}

class DocumentProcessingError(FoundersAIException):
    """Raised when a PDF, document, or text fails processing."""

    def __init__(
        self,
        message: str,
        document_id: Optional[str] = None,
        processing_stage: Optional[str] = None,
    ):
        context = {}
        if document_id:
            context["document_id"] = document_id
        if processing_stage:
            context["stage"] = processing_stage

        super().__init__(
            message=message,
            error_code="DOCUMENT_PROCESSING_ERROR",
            context=context,
        )

# ----------------------------------------------------------------------
# PDF INGESTION / DOCUMENT PROCESSING
# ----------------------------------------------------------------------
class PDFIngestionError(FoundersAIException):
    """Raised during PDF upload, parsing, or chunking."""

    def __init__(
        self,
        message: str,
        filename: Optional[str] = None,
        stage: Optional[str] = None,
    ):
        context = {}
        if filename:
            context["filename"] = filename
        if stage:
            context["stage"] = stage

        super().__init__(
            message=message,
            error_code="PDF_INGESTION_ERROR",
            context=context,
        )


# ----------------------------------------------------------------------
# EMBEDDINGS
# ----------------------------------------------------------------------
class EmbeddingError(FoundersAIException):
    """Raised during embedding generation or model invocation failures."""

    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        text_length: Optional[int] = None,
    ):
        context = {}
        if model:
            context["model"] = model
        if text_length:
            context["text_length"] = text_length

        super().__init__(
            message=message,
            error_code="EMBEDDING_ERROR",
            context=context,
        )


# ----------------------------------------------------------------------
# RAG RETRIEVAL & LIGHTWEIGHT RAG
# ----------------------------------------------------------------------
class RAGRetrievalError(FoundersAIException):
    """Raised for retrieval or similarity search failures."""

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
    ):
        context = {}
        if query:
            context["query"] = query[:120] + "..." if len(query) > 120 else query
        if top_k:
            context["top_k"] = top_k

        super().__init__(
            message=message,
            error_code="RAG_RETRIEVAL_ERROR",
            context=context,
        )


# ----------------------------------------------------------------------
# MULTI-HOP REASONING
# ----------------------------------------------------------------------
class MultiHopReasoningError(FoundersAIException):
    """
    Raised when any step of multi-hop reasoning fails:
    - decomposition
    - subquery execution
    - synthesis
    """

    def __init__(
        self,
        message: str,
        step_id: Optional[str] = None,
        phase: Optional[str] = None,
    ):
        context = {}
        if step_id:
            context["step_id"] = step_id
        if phase:
            context["phase"] = phase

        super().__init__(
            message=message,
            error_code="MULTI_HOP_REASONING_ERROR",
            context=context,
        )


# ----------------------------------------------------------------------
# ORCHESTRATION / ADAPTIVE RAG
# ----------------------------------------------------------------------
class OrchestrationError(FoundersAIException):
    """Raised during adaptive RAG orchestration, intent routing, or pipeline selection."""

    def __init__(
        self,
        message: str,
        algorithm: Optional[str] = None,
        intent: Optional[str] = None,
    ):
        context = {}
        if algorithm:
            context["algorithm"] = algorithm
        if intent:
            context["intent"] = intent

        super().__init__(
            message=message,
            error_code="ORCHESTRATION_ERROR",
            context=context,
        )


# ----------------------------------------------------------------------
# QUERY PROCESSING / VALIDATION
# ----------------------------------------------------------------------
class QueryProcessingError(FoundersAIException):
    """Raised during query validation, complexity detection, or processing."""

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        complexity: Optional[str] = None,
    ):
        context = {}
        if query:
            context["query"] = query[:150] + "..." if len(query) > 150 else query
        if complexity:
            context["complexity"] = complexity

        super().__init__(
            message=message,
            error_code="QUERY_PROCESSING_ERROR",
            context=context,
        )


# ----------------------------------------------------------------------
# DATABASE
# ----------------------------------------------------------------------
class DatabaseError(FoundersAIException):
    """Raised for connection failures, query errors, or database unavailability."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        pool_size: Optional[int] = None,
    ):
        context = {}
        if operation:
            context["operation"] = operation
        if pool_size:
            context["pool_size"] = pool_size

        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            context=context,
        )


# ----------------------------------------------------------------------
# CACHE (Redis)
# ----------------------------------------------------------------------
class CacheError(FoundersAIException):
    """Raised for Redis failures, serialization issues, or cache key problems."""

    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        context = {}
        if cache_key:
            context["cache_key"] = cache_key
        if operation:
            context["operation"] = operation

        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            context=context,
        )


# ----------------------------------------------------------------------
# RATE LIMITS
# ----------------------------------------------------------------------
class RateLimitExceededError(FoundersAIException):
    """Raised when the user exceeds allowed API request limits."""

    def __init__(
        self,
        message: str,
        client_ip: Optional[str] = None,
        request_count: Optional[int] = None,
        limit: Optional[int] = None,
    ):
        context = {}
        if client_ip:
            context["client_ip"] = client_ip
        if request_count:
            context["request_count"] = request_count
        if limit:
            context["limit"] = limit

        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            context=context,
        )


# ----------------------------------------------------------------------
# CONFIGURATION / ENV / SETTINGS
# ----------------------------------------------------------------------
class ConfigurationError(FoundersAIException):
    """Raised for invalid environment variables, missing OpenAI keys, misconfigurations."""

    def __init__(
        self,
        message: str,
        setting: Optional[str] = None,
        expected: Optional[str] = None,
    ):
        context = {}
        if setting:
            context["setting"] = setting
        if expected:
            context["expected"] = expected

        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            context=context,
        )
