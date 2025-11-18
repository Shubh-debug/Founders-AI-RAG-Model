"""
Unified API endpoints for Founders AI RAG System.

Integrates adaptive RAG, lightweight LLM RAG, and multi-hop reasoning engines.
Handles query routing, feedback, and document ingestion.
"""

import time
import logging
import json
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Body, Request, HTTPException

from ..core.config import settings
from ..core.rate_limiter import rate_limiter
from ..core.response_formatter import response_formatter
from ..core.exceptions import RateLimitExceededError

# Core RAG components
from ..services.lightweight_llm_rag_founders_ai import lightweight_llm_rag
from ..services.adaptive_rag_orchestrator import founders_ai_orchestrator as adaptive_rag_orchestrator
from ..services.multi_hop_reasoning_founders_ai import multi_hop_reasoning_engine
from ..services.query_complexity_detector import query_complexity_detector
from ..services.pdf_ingestion import pdf_ingestion_service
from ..services.feedback_system import feedback_system, UserFeedback, FeedbackType

# Models
from ..models.requests import (
    FounderQueryRequest,
    FounderQueryResponse,
    PDFIngestionResponse,
    HealthCheckResponse,
    ServiceInfoResponse,
    MultiHopReasoningResponse,
    ReasoningStepResponse,
    SaveResponsesRequest,
    SaveResponsesResponse,
    UserGuidelineResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------
# BASIC INFO / HEALTH CHECK
# ---------------------------------------------------------------------
@router.get("/", response_model=ServiceInfoResponse)
async def get_service_info():
    """Display available endpoints."""
    return ServiceInfoResponse(
        service="Founders AI RAG System",
        version="1.0.0",
        status="running",
        endpoints={
            "query": "/query",
            "adaptive": "/adaptive-query",
            "multi-hop": "/multi-hop",
            "feedback": "/feedback",
            "ingest-pdfs": "/ingest-pdfs",
            "save-responses": "/save-responses",
            "user-guideline": "/user-guideline/{user_id}",
        },
    )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Simple health check for backend components."""
    try:
        return HealthCheckResponse(
            status="healthy",
            timestamp=time.time(),
            database="connected",
            services={
                "lightweight_rag": "available",
                "adaptive_rag": "available",
                "multi_hop_reasoning": "available",
            },
        )
    except Exception as e:
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=time.time(),
            database="unknown",
            services={},
            error=str(e),
        )


# ---------------------------------------------------------------------
# LIGHTWEIGHT RAG QUERY
# ---------------------------------------------------------------------
@router.post("/query", response_model=FounderQueryResponse)
async def process_founder_query(request: Request, payload: FounderQueryRequest = Body(...)):
    """
    Process founder query via Lightweight LLM RAG engine.
    Routes to multi-hop or adaptive RAG if required.
    """
    client_ip = getattr(request.client, "host", "unknown")

    try:
        rate_limiter.check_and_record_request(client_ip)
    except RateLimitExceededError as e:
        raise HTTPException(status_code=429, detail=e.message)

    try:
        should_use_multi_hop, complexity = query_complexity_detector.should_use_multi_hop_reasoning(payload.query)

        if payload.force_multi_hop or (payload.enable_multi_hop_reasoning and should_use_multi_hop):

            multi_hop_result = await _process_multi_hop_query(payload, complexity)

            # Convert Pydantic → dict
            result = multi_hop_result.model_dump()

            return FounderQueryResponse(
                response=result["final_answer"],
                query=payload.query,
                context=[{"citation": c} for c in result["citations"]],
                metadata={
                    "algorithm": "multi_hop_reasoning",
                    "complexity": result["complexity_level"],
                    "processing_time": result["total_execution_time"],
                    "confidence": result["overall_confidence"],
                },
                source="multi-hop",
                response_time_ms=int(result["total_execution_time"] * 1000),
            )
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------
# ADAPTIVE RAG QUERY
# ---------------------------------------------------------------------
@router.post("/adaptive-query", response_model=FounderQueryResponse)
async def process_adaptive_query(request: Request, payload: FounderQueryRequest = Body(...)):
    """Process query via Adaptive RAG with intent-based orchestration."""
    client_ip = getattr(request.client, "host", "unknown")
    try:
        if rate_limiter.is_rate_limit_exceeded(client_ip):
            raise RateLimitExceededError("Rate limit exceeded for adaptive queries.")

        result = await adaptive_rag_orchestrator.process_query(
            query=payload.query,
            user_preferences={
                "response_length": getattr(payload, "response_length", "normal"),
                "retrieval_count": payload.top_k,
            },
        )

        # Ensure result is always dict-like for safety
        result_data = result if isinstance(result, dict) else getattr(result, "__dict__", {})

        return FounderQueryResponse(
            response=result_data.get("response", ""),
            query=payload.query,
            context=result_data.get("sources", []),
            metadata={
                **result_data.get("metadata", {}),
                "intent": (
                    getattr(getattr(result, "intent", None), "value", None)
                    if not isinstance(result, dict)
                    else result_data.get("intent", "")
                ),
                "confidence": result_data.get("confidence", 0),
                "processing_time_ms": int(result_data.get("processing_time", 0) * 1000),
            },
            source="adaptive_rag",
            response_time_ms=int(result_data.get("processing_time", 0) * 1000),
        )

    except RateLimitExceededError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        logger.error(f"Adaptive query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------
# MULTI-HOP REASONING
# ---------------------------------------------------------------------
async def _process_multi_hop_query(payload: FounderQueryRequest, complexity_analysis: Dict[str, Any]):
    """Process multi-hop reasoning with structured trace."""
    try:
        chain = await multi_hop_reasoning_engine.process_complex_query(
            query=payload.query, session_id=payload.session_id
        )

        steps = [
            ReasoningStepResponse(
                step_id=s.step_id,
                step_type=s.step_type.value,
                input_query=s.input_query,
                output_result=s.output_result,
                confidence_score=s.confidence_score,
                execution_time=s.execution_time,
                sources_used=s.sources_used,
            )
            for s in chain.steps
        ]

        return MultiHopReasoningResponse(
            chain_id=chain.chain_id,
            original_query=chain.original_query,
            complexity_level=chain.complexity_level.value,
            final_answer=chain.final_answer,
            reasoning_steps=steps,
            total_execution_time=chain.total_execution_time,
            overall_confidence=chain.overall_confidence,
            citations=chain.citations,
            metadata={"complexity_analysis": complexity_analysis, **chain.metadata},
        )
    except Exception as e:
        logger.error(f"Multi-hop reasoning failed: {e}")
        return MultiHopReasoningResponse(
            chain_id="error",
            original_query=payload.query,
            complexity_level="error",
            final_answer=f"Error: {str(e)}",
            reasoning_steps=[],
            total_execution_time=0,
            overall_confidence=0,
            citations=[],
            metadata={"error": str(e)},
        )


# ---------------------------------------------------------------------
# PDF INGESTION
# ---------------------------------------------------------------------
@router.post("/ingest-pdfs", response_model=PDFIngestionResponse)
async def ingest_pdf_documents(files: List[UploadFile] = File(...)):
    """Upload and ingest PDFs into the knowledge base."""
    try:
        paths = []
        for file in files:
            if not (file.filename or "").endswith(".pdf"):
                continue
            path = f"/tmp/{file.filename}"
            with open(path, "wb") as buffer:
                buffer.write(await file.read())
            paths.append(path)

        doc_ids = await pdf_ingestion_service.ingest_multiple_pdfs(paths, source="uploaded-pdf")

        return PDFIngestionResponse(
            message=f"Successfully ingested {len(doc_ids)} PDFs.",
            document_ids=doc_ids,
            status="success",
        )
    except Exception as e:
        logger.error(f"PDF ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------
# FEEDBACK SYSTEM
# ---------------------------------------------------------------------
@router.post("/feedback")
async def submit_feedback(feedback_data: Dict[str, Any] = Body(...)):
    """Submit user feedback for adaptive or RAG responses."""
    try:
        import uuid

        fb = UserFeedback(
            feedback_id=str(uuid.uuid4()),
            query=feedback_data.get("query", ""),
            response=feedback_data.get("response", ""),
            intent_classified=feedback_data.get("intent_classified", ""),
            feedback_type=FeedbackType(feedback_data.get("feedback_type", "rating")),
            rating=feedback_data.get("rating"),
            correction=feedback_data.get("correction"),
            comments=feedback_data.get("comments"),
            user_id=feedback_data.get("user_id"),
            session_id=feedback_data.get("session_id"),
        )
        success = await feedback_system.submit_feedback(fb)
        if success:
            return {"message": "Feedback submitted", "feedback_id": fb.feedback_id}
        raise HTTPException(status_code=500, detail="Failed to submit feedback")
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------
# USER RESPONSES SAVE & PERSONALIZATION GUIDELINE
# ---------------------------------------------------------------------
@router.post("/save-responses", response_model=SaveResponsesResponse)
async def save_user_responses(payload: SaveResponsesRequest = Body(...)):
    """Persist the user's assessment responses and return an assessmentId."""
    try:
        from ..services.user_profile_service import save_user_form_responses

        assessment_id = await save_user_form_responses(payload.userId, [r.model_dump() for r in payload.responses])
        return SaveResponsesResponse(assessmentId=assessment_id)
    except Exception as e:
        logger.error(f"Saving user responses failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user-guideline/{user_id}", response_model=UserGuidelineResponse)
async def get_user_guideline(user_id: str):
    """Return a formatted guideline string derived from latest user responses."""
    try:
        from ..services.user_profile_service import get_latest_user_form_responses, format_user_response_guideline

        row = await get_latest_user_form_responses(user_id)
        if not row:
            return UserGuidelineResponse(userId=user_id, guideline="", assessmentId=None)

        guideline = format_user_response_guideline(row.get("responses", []))
        return UserGuidelineResponse(userId=user_id, guideline=guideline, assessmentId=str(row.get("assessment_id")))
    except Exception as e:
        logger.error(f"Fetching user guideline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
