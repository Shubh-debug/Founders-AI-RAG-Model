"""
Pydantic models for Founders AI API requests and responses.

Defines structured data models for RAG, adaptive RAG, multi-hop reasoning,
PDF ingestion, and feedback handling in a startup intelligence context.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from enum import Enum


# ------------------------------------------------------------
# ENUMS
# ------------------------------------------------------------
class RAGAlgorithm(str, Enum):
    HYBRID = "hybrid"
    VECTOR_ONLY = "vector_only"
    KEYWORD_ONLY = "keyword_only"


class ResponseLength(str, Enum):
    SHORT = "short"
    NORMAL = "normal"
    DETAILED = "detailed"


# ------------------------------------------------------------
# QUERY REQUEST & RESPONSE MODELS
# ------------------------------------------------------------
class FounderQueryRequest(BaseModel):
    """Request model for Founders AI RAG and reasoning queries."""

    query: str = Field(..., min_length=1, max_length=10000, description="Founder or startup query to process")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of top relevant startup documents to retrieve")
    algorithm: RAGAlgorithm = Field(default=RAGAlgorithm.HYBRID, description="Retrieval algorithm to use")
    similarity_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum similarity threshold for relevance")
    enable_multi_hop_reasoning: bool = Field(default=True, description="Enable multi-hop reasoning for complex queries")
    force_multi_hop: bool = Field(default=False, description="Force multi-hop reasoning even for simple queries")
    response_length: ResponseLength = Field(default=ResponseLength.NORMAL, description="Preferred response length")
    use_agent: bool = Field(default=False, description="Use LangChain or tool-based agent for reasoning")
    session_id: Optional[str] = Field(default=None, description="Session identifier for this query")
    text_only: bool = Field(default=False, description="Return only plain text response")
    intent: Optional[str] = Field(default=None, description="Predicted intent of the query (classified by system)")

    @field_validator("query")
    def validate_query_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query cannot be empty or whitespace.")
        return value.strip()


class FounderQueryResponse(BaseModel):
    """Standardized response for startup or founder queries."""

    response: str = Field(..., description="The generated founder-focused insight or analysis")
    query: str = Field(..., description="Original founder query that was processed")
    context: List[Dict[str, Any]] = Field(default_factory=list, description="Startup case studies or documents retrieved")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata (intent, confidence, algorithm, etc.)")
    source: str = Field(..., description="Source module (lightweight_rag, adaptive_rag, or multi_hop_reasoning)")
    response_time_ms: int = Field(..., description="Response time in milliseconds")


# ------------------------------------------------------------
# MULTI-HOP REASONING
# ------------------------------------------------------------
class ReasoningStepResponse(BaseModel):
    step_id: str = Field(..., description="Unique identifier for this reasoning step")
    step_type: str = Field(..., description="Type of step (retrieval, generation, analysis, etc.)")
    input_query: str = Field(..., description="Input query or subquery for this step")
    output_result: str = Field(..., description="Generated or retrieved output for this step")
    confidence_score: float = Field(..., description="Confidence score for this step")
    execution_time: float = Field(..., description="Time taken for this step (in seconds)")
    sources_used: List[Dict[str, Any]] = Field(default_factory=list, description="Sources used during this reasoning step")


class MultiHopReasoningResponse(BaseModel):
    chain_id: str = Field(..., description="Unique identifier for the reasoning chain")
    original_query: str = Field(..., description="Original complex founder query")
    complexity_level: str = Field(..., description="Detected query complexity level")
    final_answer: str = Field(..., description="Final synthesized startup insight")
    reasoning_steps: List[ReasoningStepResponse] = Field(default_factory=list, description="Step-by-step reasoning trace")
    total_execution_time: float = Field(..., description="Total reasoning time (in seconds)")
    overall_confidence: float = Field(..., description="Overall confidence score of reasoning chain")
    citations: List[str] = Field(default_factory=list, description="Referenced startups or data sources")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata from reasoning chain")
    source: str = Field(default="multi_hop_reasoning", description="Source of this reasoning response")


# ------------------------------------------------------------
# PDF INGESTION
# ------------------------------------------------------------
class PDFIngestionRequest(BaseModel):
    source: str = Field(default="uploaded-pdf", description="Source identifier for uploaded startup documents")

    @field_validator("source")
    def validate_source(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Source cannot be empty")
        return value.strip()


class PDFIngestionResponse(BaseModel):
    message: str = Field(..., description="Status message for ingestion process")
    document_ids: List[str] = Field(default_factory=list, description="IDs of successfully ingested documents")
    status: str = Field(..., description="Overall status of ingestion (success or failure)")


# ------------------------------------------------------------
# HEALTH & SERVICE INFO
# ------------------------------------------------------------
class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Overall system health status")
    timestamp: float = Field(..., description="Unix timestamp of the health check")
    database: str = Field(..., description="Database connection status")
    services: Dict[str, str] = Field(default_factory=dict, description="Status of internal components")
    error: Optional[str] = Field(default=None, description="Error details if health check failed")


class ServiceInfoResponse(BaseModel):
    service: str = Field(..., description="Name of the service (Founders AI)")
    version: str = Field(..., description="Service version")
    status: str = Field(..., description="Current service status")
    endpoints: Dict[str, str] = Field(default_factory=dict, description="Available API endpoints")


# ------------------------------------------------------------
# REASONING CHAIN FETCH
# ------------------------------------------------------------
class ReasoningChainRequest(BaseModel):
    chain_id: Optional[str] = Field(default=None, description="Specific reasoning chain ID")
    session_id: Optional[str] = Field(default=None, description="Session ID to fetch all chains")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of chains to retrieve")


# ------------------------------------------------------------
# USER RESPONSES SAVE/RETRIEVE
# ------------------------------------------------------------
class FormResponseItem(BaseModel):
    # Flexible schema: accept either (id,type,answer) or (question,answer)
    id: Optional[int] = None
    type: Optional[str] = None
    question: Optional[str] = None
    answer: Any


class SaveResponsesRequest(BaseModel):
    userId: str = Field(..., description="Unique user identifier from frontend/auth")
    responses: List[FormResponseItem]

    @field_validator("userId")
    def validate_user_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("userId is required")
        return v.strip()

    @field_validator("responses")
    def validate_responses(cls, items: List[FormResponseItem]):
        if not items:
            raise ValueError("responses cannot be empty")
        for it in items:
            if it.answer is None:
                raise ValueError("each response must include 'answer'")
        return items


class SaveResponsesResponse(BaseModel):
    assessmentId: str


class UserGuidelineResponse(BaseModel):
    userId: str
    guideline: str
    assessmentId: Optional[str] = None
