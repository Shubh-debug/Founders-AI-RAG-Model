"""
Startup analysis tools for metric extraction and sector classification.

Provides pattern-based startup metric extraction, sector classification using
machine learning with keyword fallback, and comprehensive founder response analysis.
"""

import re
import logging
from typing import List, Dict, Any, Set
from .startup_classifier import startup_classifier

logger = logging.getLogger(__name__)


class StartupMetricsExtractor:
    """
    Extracts startup metrics from text using pattern matching.
    
    Provides methods to identify and extract various types of startup metrics
    including funding data, growth metrics, and business indicators.
    """
    
    # Startup metric patterns for different types of business indicators
    METRIC_PATTERNS = [
        # Funding amounts (e.g., "$10M" or "$10 million")
        r"\$[\d,]+(?:\.\d+)?(?:\s*(?:M|B|K|million|billion|thousand))?",
        
        # ARR/MRR patterns (e.g., "ARR $10M" or "$10M ARR")
        r"(?:ARR|MRR|GMV|AUM)\s*[:\-]?\s*\$?[\d,]+(?:\.\d+)?(?:\s*(?:M|B|K|million|billion))?",
        
        # User metrics (e.g., "500K MAU" or "1M DAU")
        r"(?:MAU|DAU|WAU|PAU|TAU)\s*[:\-]?\s*[\d\.]+\s*(?:K|M|B)?",
        
        # Growth percentages (e.g., "200% YoY" or "50% growth")
        r"[\d\.]+\s*%\s*(?:growth|YoY|MoM|QoQ|increase|decrease|CAGR)",
        
        # Series funding (e.g., "Series A", "Series B+")
        r"Series\s+[A-K]\+?",
        
        # Customer metrics (e.g., "500K users" or "1M customers")
        r"[\d\.]+\s*(?:K|M|B)\s+(?:users|customers|downloads|transactions)",
        
        # Valuation (e.g., "$1B valuation" or "Unicorn")
        r"(?:\$[\d,]+(?:\.\d+)?\s*(?:M|B|million|billion)\s+valuation|unicorn)",
        
        # Retention/Churn (e.g., "85% retention" or "5% churn")
        r"(?:retention|churn|repeat|conversion)\s*[:\-]?\s*[\d\.]+\s*%",
    ]
    
    @classmethod
    def extract_metrics(cls, text: str) -> List[str]:
        """
        Extract startup metrics from the given text.
        
        Args:
            text: Text to extract metrics from
            
        Returns:
            List[str]: List of unique startup metrics found in the text
        """
        if not text or not text.strip():
            return []
        
        metrics: List[str] = []
        for pattern in cls.METRIC_PATTERNS:
            try:
                matches = re.findall(pattern, text, re.IGNORECASE)
                metrics.extend(matches)
            except re.error as e:
                logger.warning(f"Invalid regex pattern {pattern}: {e}")
                continue
        
        return cls._remove_duplicates(metrics)
    
    @staticmethod
    def _remove_duplicates(metrics: List[str]) -> List[str]:
        """
        Remove duplicate metrics while preserving order.
        
        Args:
            metrics: List of metrics that may contain duplicates
            
        Returns:
            List[str]: List of unique metrics
        """
        seen: Set[str] = set()
        unique_metrics: List[str] = []
        for metric in metrics:
            if metric not in seen:
                seen.add(metric)
                unique_metrics.append(metric)
        return unique_metrics


class StartupSectorClassifier:
    """Classifies startup text into sectors using ML models with keyword fallback."""
    
    # Keywords for different startup sectors
    FINTECH_KEYWORDS = [
        "fintech", "payment", "digital banking", "lending", "insurance",
        "wealth", "investment", "trading", "banking", "transaction",
        "card", "wallet", "remittance", "neobank", "lending", "credit"
    ]
    
    CRYPTO_KEYWORDS = [
        "crypto", "blockchain", "bitcoin", "ethereum", "defi", "web3",
        "nft", "token", "cryptocurrency", "dapp", "smart contract",
        "exchange", "decentralized", "digital currency", "dex", "web3"
    ]
    
    AI_KEYWORDS = [
        "ai", "machine learning", "deep learning", "neural network",
        "nlp", "computer vision", "artificial intelligence", "data science",
        "algorithm", "model", "prediction", "analytics", "automation", "gpt"
    ]
    
    ECOMMERCE_KEYWORDS = [
        "ecommerce", "marketplace", "retail", "shopping", "logistics",
        "delivery", "seller", "buyer", "product", "order", "inventory",
        "supplier", "storefront", "checkout", "store", "fulfillment"
    ]
    
    SOCIAL_KEYWORDS = [
        "social", "content", "creator", "community", "network", "sharing",
        "platform", "engagement", "followers", "viral", "trending",
        "streaming", "messaging", "chat", "video", "influencer"
    ]
    
    @classmethod
    def classify_text(cls, text: str) -> str:
        """
        Classify startup text into sectors using ML model with keyword fallback.
        
        Args:
            text: Text to classify
            
        Returns:
            str: Sector classification (Fintech, Crypto, AI, E-commerce, Social, Other)
        """
        if not text or not text.strip():
            return "Other"
        
        try:
            # Try ML-based classification first
            result = startup_classifier.classify(text)
            return result["category"]
        except Exception as e:
            logger.warning(f"ML classification failed, using keyword fallback: {e}")
            return cls._classify_by_keywords(text)
    
    @classmethod
    def _classify_by_keywords(cls, text: str) -> str:
        """
        Classify text using keyword matching as fallback.
        
        Args:
            text: Text to classify
            
        Returns:
            str: Sector based on keyword matching
        """
        text_lower = text.lower()
        
        if cls._contains_any_keyword(text_lower, cls.FINTECH_KEYWORDS):
            return "Fintech"
        
        if cls._contains_any_keyword(text_lower, cls.CRYPTO_KEYWORDS):
            return "Crypto"
        
        if cls._contains_any_keyword(text_lower, cls.AI_KEYWORDS):
            return "AI"
        
        if cls._contains_any_keyword(text_lower, cls.ECOMMERCE_KEYWORDS):
            return "E-commerce"
        
        if cls._contains_any_keyword(text_lower, cls.SOCIAL_KEYWORDS):
            return "Social"
        
        return "Other"
    
    @staticmethod
    def _contains_any_keyword(text: str, keywords: List[str]) -> bool:
        """
        Check if text contains any of the specified keywords.
        
        Args:
            text: Text to search in
            keywords: List of keywords to search for
            
        Returns:
            bool: True if any keyword is found, False otherwise
        """
        return any(keyword in text for keyword in keywords)


class StartupResponseAnalyzer:
    """Analyzes founder responses for metrics and sector classification."""
    
    def __init__(self):
        self.metrics_extractor = StartupMetricsExtractor()
        self.sector_classifier = StartupSectorClassifier()
    
    def analyze_response(
        self,
        query: str,
        response_text: str,
        context_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze a startup response for metrics and sector classification.
        
        Args:
            query: Original founder query
            response_text: Generated response text
            context_documents: Context documents (case studies) used for the response
            
        Returns:
            Dict[str, Any]: Analysis results including metrics and sector
        """
        try:
            # Extract metrics from response
            metrics = self.metrics_extractor.extract_metrics(response_text)
            
            # Combine query and context for sector classification
            combined_text = self._combine_text_for_classification(
                query, response_text, context_documents
            )
            
            # Classify the combined text
            sector = self.sector_classifier.classify_text(combined_text)
            
            return {
                "metrics": metrics,
                "sector": sector
            }
        
        except Exception as e:
            logger.error(f"Error analyzing startup response: {e}")
            return {
                "metrics": [],
                "sector": "Other"
            }
    
    def _combine_text_for_classification(
        self,
        query: str,
        response_text: str,
        context_documents: List[Dict[str, Any]]
    ) -> str:
        """
        Combine query, response, and context for sector classification.
        
        Args:
            query: Original query
            response_text: Response text
            context_documents: Context case studies
            
        Returns:
            str: Combined text for classification
        """
        text_parts = [query, response_text]
        
        # Add content from context documents (case studies)
        for doc in context_documents:
            content = doc.get("content", "")
            if content:
                text_parts.append(content)
        
        return " ".join(text_parts)


# Global instances for backward compatibility
startup_metrics_extractor = StartupMetricsExtractor()
startup_sector_classifier = StartupSectorClassifier()
startup_response_analyzer = StartupResponseAnalyzer()


# Backward compatibility functions
def extract_startup_citations(text: str) -> List[str]:
    """Extract startup metrics from text (backward compatibility)."""
    return startup_metrics_extractor.extract_metrics(text)


def classify_startup_text(text: str) -> str:
    """Classify startup text into sectors (backward compatibility)."""
    return startup_sector_classifier.classify_text(text)


def analyze_startup_response(
    query: str,
    response_text: str,
    context_docs: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Analyze startup response for metrics and sector (backward compatibility)."""
    return startup_response_analyzer.analyze_response(query, response_text, context_docs)