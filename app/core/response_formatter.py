"""
Response formatting utilities for Founders AI RAG System.

Cleans model outputs, formats RAG + multi-hop responses,
and produces consistent API-safe JSON responses.
"""

import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """
    Standard formatter for all Founders-AI responses.

    - Cleans LLM output
    - Formats RAG responses
    - Formats multi-hop responses
    - Formats orchestrator / adaptive pipeline responses
    """

    # ---------------------------------------------------------
    # TEXT CLEANING
    # ---------------------------------------------------------
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text for display."""
        if not text:
            return ""

        # Normalize escaped characters
        text = text.replace("\\n", "\n").replace('\\"', '"').replace('\\\\', '\\')

        # Remove excessive whitespace
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n[ \t]+", "\n", text)
        text = re.sub(r"[ \t]+\n", "\n", text)

        # Fix sentence spacing (Make sure punctuation is followed by a space)
        text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)

        # Normalize long punctuation
        text = re.sub(r"[.]{3,}", "...", text)
        text = re.sub(r"[-]{4,}", "---", text)

        # Fix missing newline after bullet points or numbered lists
        text = re.sub(r"\n(\d+\.)", r"\n\n\1", text)
        text = re.sub(r"\n([\-*])", r"\n\n\1", text)

        # Remove multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    # ---------------------------------------------------------
    # RAG RESPONSE FORMATTER
    # ---------------------------------------------------------
    @staticmethod
    def format_rag_response(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format RAG engine response for API output.

        Expected input keys:
        - response
        - query
        - context
        - metadata
        - processing_time
        """
        try:
            formatted_text = ResponseFormatter.clean_text(
                data.get("response", "")
            )

            return {
                "response": formatted_text,
                "query": data.get("query", ""),
                "context": data.get("context", []),
                "metadata": data.get("metadata", {}),
                "source": "founders_ai_rag_engine",
                "response_time_ms": int(data.get("processing_time", 0) * 1000)
            }

        except Exception as e:
            logger.error(f"Failed to format RAG response: {e}")
            return data

    # ---------------------------------------------------------
    # MULTI-HOP RESPONSE FORMATTER
    # ---------------------------------------------------------
    @staticmethod
    def format_multi_hop_response(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format multi-hop reasoning output.

        Expected input keys:
        - final_answer
        - steps
        - execution_time
        - combined_context
        """
        try:
            answer = ResponseFormatter.clean_text(
                data.get("final_answer", "")
            )

            return {
                "response": answer,
                "steps": data.get("steps", []),
                "context": data.get("combined_context", []),
                "metadata": {
                    "algorithm": "multi_hop_reasoning",
                },
                "source": "founders_ai_multi_hop",
                "response_time_ms": int(data.get("execution_time", 0) * 1000)
            }

        except Exception as e:
            logger.error(f"Failed to format multi-hop response: {e}")
            return data

    # ---------------------------------------------------------
    # AGENT / ORCHESTRATION FORMATTER
    # ---------------------------------------------------------
    @staticmethod
    def format_orchestrator_response(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format adaptive reasoning / orchestrator output.
        """
        try:
            return {
                "response": ResponseFormatter.clean_text(data.get("response", "")),
                "query": data.get("query", ""),
                "context": data.get("context", []),
                "metadata": data.get("metadata", {}),
                "pipeline_used": data.get("pipeline_used", "unknown"),
                "source": "founders_ai_orchestrator",
            }

        except Exception as e:
            logger.error(f"Failed to format orchestrator response: {e}")
            return data

    # ---------------------------------------------------------
    # ERROR FORMATTER
    # ---------------------------------------------------------
    @staticmethod
    def format_error(error_message: str, error_code: Optional[str] = None) -> Dict[str, Any]:
        """Format errors from exceptions into consistent API output."""
        error_response = {
            "error": True,
            "message": error_message,
            "source": "founders_ai_backend"
        }

        if error_code:
            error_response["error_code"] = error_code

        return error_response


# Global instance used everywhere
response_formatter = ResponseFormatter()
