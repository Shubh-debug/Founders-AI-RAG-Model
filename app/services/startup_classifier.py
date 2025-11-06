"""
Machine learning-based startup sector classification.

Uses scikit-learn with TF-IDF vectorization and Random Forest classification
to categorize startup text into Fintech, Crypto, AI, E-commerce, Social, or Other sectors.
"""

import pickle
import numpy as np
from typing import Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging

logger = logging.getLogger(__name__)


class StartupSectorClassifier:
    """Machine learning classifier for startup sector categorization."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        self.is_trained = False
        self.categories = {
            0: "Fintech",
            1: "Crypto",
            2: "AI",
            3: "E-commerce",
            4: "Social",
            5: "Other"
        }
    
    def _create_training_data(self):
        """Create training data from startup case studies or fallback data."""
        import asyncio
        from pathlib import Path
        import sys
        
        sys.path.append(str(Path(__file__).parent.parent.parent))
        
        try:
            from core.database import db_manager
            
            async def get_document_training_data():
                try:
                    await db_manager.initialize()
                    async with db_manager.get_connection() as conn:
                        result = await conn.fetch("""
                            SELECT content, source, title
                            FROM startup_documents
                            WHERE content IS NOT NULL
                            AND LENGTH(content) > 100
                            LIMIT 50
                        """)
                        
                        if result:
                            texts = []
                            labels = []
                            for row in result:
                                content = row['content']
                                source = row['source'] or ''
                                title = row['title'] or ''
                                label = self._classify_document_content(content, source, title)
                                texts.append(content[:500])
                                labels.append(label)
                            return {"texts": texts, "labels": labels}
                
                except Exception as e:
                    logger.warning(f"Could not get training data from database: {e}")
                    return None
            
            # Handle async context - can't run async code if already in event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    training_data = None
                else:
                    training_data = asyncio.run(get_document_training_data())
            except Exception as e:
                logger.warning(f"Could not get training data from database: {e}")
                training_data = None
            
            if not training_data or len(training_data["texts"]) < 10:
                logger.info("Using fallback training data - no sufficient startup documents in database")
                training_data = self._get_fallback_training_data()
            
            return training_data
        
        except Exception as e:
            logger.warning(f"Error creating training data: {e}, using fallback")
            return self._get_fallback_training_data()
    
    def _classify_document_content(self, content: str, source: str, title: str) -> int:
        """Classify startup document into sector category."""
        content_lower = content.lower()
        source_lower = source.lower()
        title_lower = title.lower()
        
        fintech_keywords = [
            'fintech', 'payment', 'digital banking', 'lending', 'insurance',
            'wealth management', 'investment', 'trading', 'banking', 'financial',
            'transaction', 'payment gateway', 'money transfer', 'credit'
        ]
        
        crypto_keywords = [
            'crypto', 'blockchain', 'bitcoin', 'ethereum', 'defi', 'web3',
            'nft', 'token', 'cryptocurrency', 'dapp', 'smart contract',
            'exchange', 'decentralized', 'digital currency', 'blockchain'
        ]
        
        ai_keywords = [
            'ai', 'machine learning', 'deep learning', 'neural network',
            'nlp', 'computer vision', 'artificial intelligence', 'data science',
            'algorithm', 'model', 'prediction', 'analytics', 'automation'
        ]
        
        ecommerce_keywords = [
            'ecommerce', 'marketplace', 'retail', 'shopping', 'logistics',
            'delivery', 'seller', 'buyer', 'product', 'order', 'inventory',
            'supplier', 'storefront', 'checkout', 'store'
        ]
        
        social_keywords = [
            'social', 'content', 'creator', 'community', 'network', 'sharing',
            'platform', 'engagement', 'followers', 'viral', 'trending',
            'live streaming', 'messaging', 'chat', 'video'
        ]
        
        # Check fintech first
        if any(keyword in content_lower or keyword in source_lower or keyword in title_lower
               for keyword in fintech_keywords):
            return 0
        
        # Check crypto second
        if any(keyword in content_lower or keyword in source_lower or keyword in title_lower
               for keyword in crypto_keywords):
            return 1
        
        # Check AI third
        if any(keyword in content_lower or keyword in source_lower or keyword in title_lower
               for keyword in ai_keywords):
            return 2
        
        # Check e-commerce fourth
        if any(keyword in content_lower or keyword in source_lower or keyword in title_lower
               for keyword in ecommerce_keywords):
            return 3
        
        # Check social fifth
        if any(keyword in content_lower or keyword in source_lower or keyword in title_lower
               for keyword in social_keywords):
            return 4
        
        return 5  # Default to "Other" category
    
    def _get_fallback_training_data(self):
        """Fallback training data for startup sectors."""
        return {
            "texts": [
                "PolicyBazaar revolutionized insurance distribution through digital channels",
                "Fintech platform enabling seamless online payment transactions",
                "Digital banking solution for retail customers",
                "Lending platform for quick personal loans",
                "Investment platform with low-cost trading",
                "Wealth management and portfolio optimization services",
                "Money transfer and remittance platform",
                "Insurance aggregator marketplace",
                
                "CoinDCX cryptocurrency exchange platform",
                "Blockchain-based smart contracts",
                "Decentralized finance DeFi protocol",
                "Web3 infrastructure for dApps",
                "NFT marketplace for digital collectibles",
                "Cryptocurrency trading and wallet services",
                "Bitcoin and Ethereum exchange",
                "Decentralized autonomous organization DAO",
                
                "AI-powered computer vision for object detection",
                "Machine learning model for price prediction",
                "Natural language processing for text analysis",
                "Deep learning neural networks for image recognition",
                "Data science and analytics platform",
                "AI automation for business processes",
                "Predictive analytics for customer behavior",
                "Algorithm optimization for performance",
                
                "Groww investment and mutual fund platform",
                "E-commerce marketplace for multiple sellers",
                "Online retail with fast delivery logistics",
                "Inventory management and supplier network",
                "Shopping platform with personalized recommendations",
                "Order management and fulfillment system",
                "Seller tools for product management",
                "Checkout and payment integration",
                
                "ShareChat social media content platform",
                "Creator economy platform for influencers",
                "Community engagement and networking",
                "Live streaming video platform",
                "Viral content sharing and trending",
                "Messaging and chat application",
                "Social network for user connections",
                "Content moderation and community standards",
                
                "Fractal analytics and big data processing",
                "Good Glamm beauty and wellness brand",
                "Specialized startup focused on unique domain",
                "Innovative business model and strategy",
                "Growth trajectory and market expansion",
                "Customer acquisition and retention",
                "Revenue generation and profitability",
                "Organizational structure and team"
            ],
            "labels": [
                0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1,
                2, 2, 2, 2, 2, 2, 2, 2,
                3, 3, 3, 3, 3, 3, 3, 3,
                4, 4, 4, 4, 4, 4, 4, 4,
                5, 5, 5, 5, 5, 5, 5, 5
            ]
        }
    
    def train(self):
        """Train the sector classifier on startup documents."""
        try:
            training_data = self._create_training_data()
            X = self.vectorizer.fit_transform(training_data["texts"])
            y = np.array(training_data["labels"])
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            self.classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_test)
            
            logger.info("Classification Report:")
            logger.info(classification_report(y_test, y_pred,
                target_names=list(self.categories.values())))
            
            self.is_trained = True
            logger.info("Startup sector classifier trained successfully")
        
        except Exception as e:
            logger.error(f"Failed to train classifier: {e}")
            raise
    
    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify startup text into sector category.
        
        Args:
            text: Startup document text to classify
            
        Returns:
            Dict with category, confidence, and probabilities
        """
        if not self.is_trained:
            self.train()
        
        try:
            X = self.vectorizer.transform([text])
            prediction = self.classifier.predict(X)[0]
            probabilities = self.classifier.predict_proba(X)[0]
            confidence = float(np.max(probabilities))
            category = self.categories[prediction]
            
            return {
                "category": category,
                "confidence": confidence,
                "probabilities": {
                    self.categories[i]: float(prob)
                    for i, prob in enumerate(probabilities)
                }
            }
        
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {
                "category": "Other",
                "confidence": 0.0,
                "probabilities": {"Other": 1.0}
            }
    
    def save_model(self, filepath: str):
        """Save the trained model to file."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "categories": self.categories,
            "is_trained": self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a previously trained model from file."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.vectorizer = model_data["vectorizer"]
            self.classifier = model_data["classifier"]
            self.categories = model_data["categories"]
            self.is_trained = model_data["is_trained"]
            
            logger.info(f"Model loaded from {filepath}")
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


# Global instance
startup_classifier = StartupSectorClassifier()