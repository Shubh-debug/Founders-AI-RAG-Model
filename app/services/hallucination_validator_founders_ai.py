"""
Hallucination validation service for Founders AI.

Prevents false startup claims, hallucinated metrics, and ungrounded company data.
Validates that all citations, company references, and numerical data are present
in the source knowledge base before allowing the response to be returned.

This validator ensures founder guidance is always backed by real startup examples
and prevents the LLM from inventing metrics, funding rounds, or company achievements.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of hallucination validation."""
    is_valid: bool
    confidence: float
    issues: List[str]
    suggestions: List[str]


class HallucinationValidator:
    """Validates startup responses to prevent hallucinated content."""

    def __init__(self):
        # Startup metric patterns that should only appear if grounded in source
        self.startup_metric_patterns = [
            r"ARR|Annual Recurring Revenue",
            r"MRR|Monthly Recurring Revenue",
            r"MAU|Monthly Active Users",
            r"DAU|Daily Active Users",
            r"GMV|Gross Merchandise Value",
            r"Series [A-K]",
            r"\$[\d,]+[MBK]?|\$[\d,]+\s+(?:million|billion)",
            r"€[\d,]+[MBK]?|€[\d,]+\s+(?:million|billion)",
            r"£[\d,]+[MBK]?|£[\d,]+\s+(?:million|billion)",
        ]

        # Company names from your startup case studies
        self.known_companies = [
            "PolicyBazaar",
            "Groww",
            "CoinDCX",
            "ShareChat",
            "Fractal",
            "FirstCry",
            "Good Glamm Group",
            "Good Glamm",
        ]

        # Startup terminology that should be grounded in source
        self.startup_terms = [
            "product-market fit",
            "PMF",
            "unicorn",
            "valuation",
            "revenue",
            "raised",
            "funding",
            "Series A",
            "Series B",
            "Series C",
            "burn rate",
            "runway",
            "churn",
            "retention",
            "CAC",
            "LTV",
            "CAGR",
            "user acquisition",
            "conversion rate",
        ]

    def validate_response(
        self,
        response: str,
        context: str,
        query: str
    ) -> ValidationResult:
        """
        Validate a response for potential hallucination.

        This runs multiple validation checks to ensure the response is grounded
        in the source context. Returns a ValidationResult with confidence score
        and any issues found.

        Args:
            response: The generated response to validate
            context: The source context used for generation
            query: The original founder query

        Returns:
            ValidationResult with confidence score and issues
        """
        issues = []
        suggestions = []
        confidence = 1.0

        # Run validation checks
        citation_issues = self._validate_metrics(response, context)
        issues.extend(citation_issues)

        company_issues = self._validate_company_references(response, context, query)
        issues.extend(company_issues)

        content_issues = self._validate_startup_terminology(response, context)
        issues.extend(content_issues)

        intent_issues = self._validate_response_structure(response, query)
        issues.extend(intent_issues)

        # Calculate confidence
        if issues:
            confidence = max(0.0, 1.0 - (len(issues) * 0.2))
            suggestions.append("Review the response to ensure all startup data is from provided case studies.")
            suggestions.append("If information is not available in the knowledge base, state this clearly.")
            suggestions.append("Avoid generating metrics or company claims not present in source documents.")

        is_valid = len(issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            issues=issues,
            suggestions=suggestions
        )

    def _validate_metrics(self, response: str, context: str) -> List[str]:
        """Validate that all mentioned metrics exist in the context."""
        issues = []

        # Find all metric mentions in response
        for pattern in self.startup_metric_patterns:
            try:
                response_matches = re.findall(pattern, response, re.IGNORECASE)

                # For each metric found, check if it or similar exists in context
                for match in response_matches:
                    # Extract the numeric portion if present
                    if any(char.isdigit() for char in match):
                        # For numeric metrics, check for exact or similar occurrence
                        if match.lower() not in context.lower():
                            # Check if the metric label is at least present
                            metric_label = re.sub(r'[\d$€£,\s]+', '', match)
                            if metric_label and metric_label.lower() not in context.lower():
                                issues.append(f"Metric '{match}' mentioned but not found in source context")
            except re.error:
                continue

        return issues

    def _validate_company_references(
        self,
        response: str,
        context: str,
        query: str
    ) -> List[str]:
        """Validate that referenced companies are in the knowledge base."""
        issues = []

        # Check for company mentions in response
        for company in self.known_companies:
            if company in response:
                # Verify company is in context
                if company not in context:
                    issues.append(f"Company '{company}' mentioned in response but not found in context")

        # Check if query specifically requests a company
        for company in self.known_companies:
            if company.lower() in query.lower():
                # Verify case study data is available
                if company not in context:
                    issues.append(
                        f"Query requests {company} information but case study not available in context"
                    )

        return issues

    def _validate_startup_terminology(self, response: str, context: str) -> List[str]:
        """Validate that startup-specific terms are grounded in source."""
        issues = []

        for term in self.startup_terms:
            term_lower = term.lower()
            if term_lower in response.lower():
                # Check if this term is explained or grounded in context
                if term_lower not in context.lower():
                    # For important terms like PMF, unicorn, etc., this is an issue
                    if term in ["product-market fit", "PMF", "unicorn", "valuation"]:
                        issues.append(f"Business term '{term}' used but not grounded in source context")

        return issues

    def _validate_response_structure(self, response: str, query: str) -> List[str]:
        """Validate that response structure matches what was asked."""
        issues = []

        # Check for list requests
        if any(word in query.lower() for word in ["list", "enumerate", "name", "what are"]):
            # Verify response has enumeration
            has_enumeration = any(
                pattern in response
                for pattern in ["1.", "2.", "3.", "-", "•", "first", "second", "third"]
            )
            if not has_enumeration and len(response.split()) > 50:
                issues.append("Query requests a list but response lacks structured enumeration")

        # Check for definition requests
        if any(word in query.lower() for word in ["what is", "define", "explain", "meaning"]):
            word_count = len(response.split())
            if word_count < 10:
                issues.append("Query requests definition but response appears too brief")

        # Check for metric/comparison requests
        if any(word in query.lower() for word in ["how many", "how much", "compare", "versus"]):
            has_numbers = bool(re.search(r'\d+', response))
            has_comparison = "vs" in response or "compared to" in response.lower()

            if not has_numbers and "not available" not in response.lower():
                issues.append("Query requests metrics but response lacks numerical data")

            if "compare" in query.lower() and not has_comparison and len(response.split()) > 30:
                issues.append("Query requests comparison but response doesn't compare elements")

        return issues

    def should_reject_response(
        self,
        response: str,
        context: str,
        query: str
    ) -> Tuple[bool, str]:
        """
        Determine if a response should be rejected due to hallucination risk.

        Returns a tuple of (should_reject: bool, reason: str)

        Rejection criteria:
        - Confidence score below 30%
        - Critical issues found (ungrounded metrics or companies)
        """
        validation = self.validate_response(response, context, query)

        # Reject if confidence is very low
        if validation.confidence < 0.3:
            return True, (
                f"Response rejected due to low confidence ({validation.confidence:.2f}). "
                f"Issues: {', '.join(validation.issues[:3])}"
            )

        # Reject if critical issues found
        critical_issues = [
            issue for issue in validation.issues
            if "not found in context" in issue or "not available in context" in issue
        ]
        if critical_issues:
            return True, f"Response rejected due to ungrounded claims: {critical_issues[0]}"

        return False, "Response accepted"

    def get_safe_response(self, query: str, context: str) -> str:
        """
        Generate a safe fallback response when original is rejected.

        This ensures the user always gets an honest answer rather than
        a hallucinated one. The response acknowledges what's not available.

        Args:
            query: The original founder query
            context: The available context

        Returns:
            Safe, truthful response
        """
        if not context or len(context.strip()) < 50:
            return (
                "This information is not available in the startup knowledge base. "
                "Please check if the relevant case study has been uploaded to the system."
            )

        # Check if query asks about a specific company
        for company in self.known_companies:
            if company.lower() in query.lower():
                return (
                    f"The {company} case study is not available in the current knowledge base. "
                    f"Please check if the case study file has been uploaded."
                )

        # Generic fallback
        return (
            "The requested information is not available in the startup knowledge base. "
            "Please check that the relevant case study has been uploaded and try rephrasing your question."
        )


# Global instance for easy importing
hallucination_validator = HallucinationValidator()