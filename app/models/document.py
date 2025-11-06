"""
SQLAlchemy database models for Founders AI startup case studies and founder conversation history.

Provides database models for startup case studies with vector embeddings and conversation
history tracking with proper indexing and relationship management for founder queries.
"""

from sqlalchemy import Column, Integer, String, Text, JSON, DateTime, Float
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

Base = declarative_base()


# -------------------------------------------------------------------
# ENUM DEFINITIONS
# -------------------------------------------------------------------
class DocumentProcessingStatus(Enum):
    """Processing status for startup case study documents."""
    PENDING = "pending"
    PROCESSED = "processed"
    ERROR = "error"


# -------------------------------------------------------------------
# STARTUP CASE STUDY MODEL
# -------------------------------------------------------------------
class StartupCaseStudy(Base):
    """Database model for startup case studies with vector embeddings and metadata."""
    
    __tablename__ = "startup_case_studies"
    
    # Primary key and content
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False, index=True)
    title = Column(String(500), nullable=True, index=True)
    company = Column(String(255), nullable=True, index=True)
    
    # Processing status
    status = Column(
        String(50),
        nullable=False,
        default=DocumentProcessingStatus.PROCESSED.value,
        index=True
    )
    
    # Startup-specific metadata and embeddings
    case_study_metadata = Column("metadata", JSON, nullable=True)
    embedding = Column(Vector(1536), nullable=True)
    similarity_score = Column(Float, nullable=True)
    
    # Startup metrics
    sector = Column(String(100), nullable=True, index=True)
    stage = Column(String(50), nullable=True, index=True)
    arr = Column(String(50), nullable=True)
    mau = Column(String(50), nullable=True)
    dau = Column(String(50), nullable=True)
    gmv = Column(String(50), nullable=True)
    funding_round = Column(String(50), nullable=True, index=True)
    valuation = Column(String(50), nullable=True)
    growth_rate = Column(String(50), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    def __repr__(self) -> str:
        return f"<StartupCaseStudy(id={self.id}, company='{self.company}', sector='{self.sector}')>"
    
    def to_dictionary(self) -> Dict[str, Any]:
        return {
            "id": getattr(self, "id", None),
            "content": getattr(self, "content", ""),
            "title": getattr(self, "title", ""),
            "company": getattr(self, "company", ""),
            "status": getattr(self, "status", ""),
            "metadata": getattr(self, "case_study_metadata", {}),
            "sector": getattr(self, "sector", ""),
            "stage": getattr(self, "stage", ""),
            "arr": getattr(self, "arr", None),
            "mau": getattr(self, "mau", None),
            "dau": getattr(self, "dau", None),
            "gmv": getattr(self, "gmv", None),
            "funding_round": getattr(self, "funding_round", ""),
            "valuation": getattr(self, "valuation", ""),
            "growth_rate": getattr(self, "growth_rate", ""),
            "similarity_score": getattr(self, "similarity_score", None),
            "created_at": self.created_at.isoformat() if getattr(self, "created_at", None) else None,
            "updated_at": self.updated_at.isoformat() if getattr(self, "updated_at", None) else None
        }

    def is_processed(self) -> bool:
        """Check if case study has been processed."""
        status_value = getattr(self, "status", None)
        return isinstance(status_value, str) and status_value == DocumentProcessingStatus.PROCESSED.value

    def has_embedding(self) -> bool:
        """Check if case study has vector embedding."""
        return getattr(self, "embedding", None) is not None

    def get_processing_status(self) -> DocumentProcessingStatus:
        """Get processing status enum."""
        status_value = getattr(self, "status", None)
        if isinstance(status_value, str):
            try:
                return DocumentProcessingStatus(status_value)
            except ValueError:
                return DocumentProcessingStatus.ERROR
        return DocumentProcessingStatus.ERROR

    def has_startup_metrics(self) -> bool:
        """Check if case study has startup metrics."""
        return any([
            getattr(self, "arr", None),
            getattr(self, "mau", None),
            getattr(self, "dau", None),
            getattr(self, "gmv", None),
            getattr(self, "valuation", None)
        ])
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of available startup metrics."""
        return {
            "arr": getattr(self, "arr", None),
            "mau": getattr(self, "mau", None),
            "dau": getattr(self, "dau", None),
            "gmv": getattr(self, "gmv", None),
            "funding_round": getattr(self, "funding_round", None),
            "valuation": getattr(self, "valuation", None),
            "growth_rate": getattr(self, "growth_rate", None)
        }


# -------------------------------------------------------------------
# FOUNDER CONVERSATION HISTORY MODEL
# -------------------------------------------------------------------
class FounderConversationHistory(Base):
    """Database model for storing founder query history and context."""
    
    __tablename__ = "founder_conversations"
    
    # Primary key and session tracking
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    
    # Conversation content
    founder_query = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    
    # Context and tools used
    rag_context = Column(JSON, nullable=True)
    startup_cases_used = Column(JSON, nullable=True)
    reasoning_approach = Column(String(100), nullable=True)
    sectors_analyzed = Column(JSON, nullable=True)
    
    # Performance metrics
    response_time_ms = Column(Integer, nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Query metadata
    query_complexity = Column(String(50), nullable=True)
    query_intent = Column(String(50), nullable=True)
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self) -> str:
        """Readable debug representation that avoids Pylance type errors."""
        query_preview = (
            self.founder_query[:60] + "..."
            if isinstance(self.founder_query, str) and len(self.founder_query) > 60
            else self.founder_query
        )
        return (
            f"<FounderConversationHistory(id={self.id}, "
            f"session_id='{self.session_id}', query_preview={query_preview!r})>"
        )
    
    def to_dictionary(self) -> Dict[str, Any]:
        return {
            "id": getattr(self, "id", None),
            "session_id": getattr(self, "session_id", None),
            "founder_query": getattr(self, "founder_query", None),
            "assistant_response": getattr(self, "assistant_response", None),
            "rag_context": getattr(self, "rag_context", None),
            "startup_cases_used": getattr(self, "startup_cases_used", None),
            "reasoning_approach": getattr(self, "reasoning_approach", None),
            "sectors_analyzed": getattr(self, "sectors_analyzed", None),
            "response_time_ms": getattr(self, "response_time_ms", None),
            "confidence_score": getattr(self, "confidence_score", None),
            "query_complexity": getattr(self, "query_complexity", None),
            "query_intent": getattr(self, "query_intent", None),
            "created_at": self.created_at.isoformat() if getattr(self, "created_at", None) else None
        }
    
    def get_response_time_seconds(self) -> Optional[float]:
        """Get response time in seconds."""
        time_ms = getattr(self, "response_time_ms", None)
        return time_ms / 1000.0 if isinstance(time_ms, (int, float)) else None
    
    def has_rag_context(self) -> bool:
        """Check if response used RAG context."""
        rag_ctx = getattr(self, "rag_context", None)
        return isinstance(rag_ctx, list) and len(rag_ctx) > 0
    
    def get_startup_cases_count(self) -> int:
        """Get number of startup case studies referenced."""
        startup_cases = getattr(self, "startup_cases_used", None)
        return len(startup_cases) if isinstance(startup_cases, list) else 0
    
    def get_sectors_analyzed_count(self) -> int:
        """Get number of sectors analyzed in response."""
        sectors = getattr(self, "sectors_analyzed", None)
        return len(sectors) if isinstance(sectors, list) else 0
    
    def used_multi_hop_reasoning(self) -> bool:
        """Check if multi-hop reasoning was used."""
        reasoning_type = getattr(self, "reasoning_approach", "")
        return reasoning_type in [
            "multi_hop_reasoning",
            "multi_hop_reasoning_with_cross_company_synthesis",
        ]
    
    def get_confidence_percentage(self) -> Optional[float]:
        """Get confidence score as percentage."""
        score = getattr(self, "confidence_score", None)
        return score * 100 if isinstance(score, (int, float)) else None
