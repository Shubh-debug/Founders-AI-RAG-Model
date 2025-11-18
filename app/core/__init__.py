"""
Core package for the Founders AI RAG system.

This package contains core functionality including configuration,
database management, exceptions, utilities, and lifecycle management.
"""

from .config import settings, get_settings
from .database import db_manager, get_database_manager, init_database

# Updated imports — match the new Founders AI exception names
from .exceptions import (
    FoundersAIException,
    PDFIngestionError,
    EmbeddingError,
    DatabaseError,
    CacheError,
    OrchestrationError,
    QueryProcessingError,
    RateLimitExceededError,
    ConfigurationError,
    MultiHopReasoningError,
    RAGRetrievalError
)

from .rate_limiter import rate_limiter, RateLimiter
from .response_formatter import response_formatter, ResponseFormatter
from .utils import (
    text_processor,
    validation_utils,
    data_transformer,
    TextProcessor,
    PerformanceTimer,
    ValidationUtils,
    DataTransformer,
    time_operation
)

__all__ = [
    # Configuration
    "settings",
    "get_settings",

    # Database
    "db_manager",
    "get_database_manager",
    "init_database",

    # Exceptions (Final, Founders AI–specific)
    "FoundersAIException",
    "PDFIngestionError",
    "EmbeddingError",
    "DatabaseError",
    "CacheError",
    "OrchestrationError",
    "QueryProcessingError",
    "RateLimitExceededError",
    "ConfigurationError",
    "MultiHopReasoningError",
    "RAGRetrievalError",

    # Rate limiting
    "rate_limiter",
    "RateLimiter",

    # Response formatting
    "response_formatter",
    "ResponseFormatter",

    # Utilities
    "text_processor",
    "validation_utils",
    "data_transformer",
    "TextProcessor",
    "PerformanceTimer",
    "ValidationUtils",
    "DataTransformer",
    "time_operation"
]
