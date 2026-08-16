"""
Microsoft DeBERTa v3 Transformer NLP Analyzer for Unified Social Media Threat Detector.

Uses microsoft/deberta-v3-base from Hugging Face for deep natural language understanding
of social media posts, bios, crypto phishing cues, and social engineering context.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Lazy singleton instance
_DEBERTA_ANALYZER_INSTANCE = None

class DeBERTaAnalyzer:
    """
    NLP Threat Analyzer powered by Microsoft DeBERTa v3 transformer architecture.
    """
    
    def __init__(self, model_name: str = "microsoft/deberta-v3-base"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._is_loaded = False
        
        # High risk scam triggers for fast neural scoring
        self.phishing_keywords = [
            'airdrop', 'giveaway', 'doubler', 'seed phrase', 'claim', 'metamask',
            'trustwallet', 'telegram', 'whatsapp', 'presale', 'whitelist', 'solana',
            'usdt', 'guaranteed profit', 'instant payout', 'free bitcoin', 'dm for collabs'
        ]

    def load_model(self) -> bool:
        """
        Lazily load DeBERTa tokenizer and model.
        """
        if self._is_loaded:
            return True

        try:
            logger.info(f"Loading Microsoft DeBERTa transformer model: {self.model_name}")
            from transformers import pipeline
            
            # Initialize zero-shot or text-classification pipeline
            self.pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=-1  # CPU inference
            )
            self._is_loaded = True
            logger.info("Microsoft DeBERTa transformer initialized successfully!")
            return True
        except Exception as e:
            logger.warning(f"Could not load Hugging Face DeBERTa model '{self.model_name}': {str(e)}")
            self._is_loaded = False
            return False

    def predict_phishing_score(self, text: str) -> float:
        """
        Evaluate a single post or bio string for social engineering / phishing threat probability.
        """
        if not text or len(text.strip()) == 0:
            return 0.0

        lower_text = text.lower()
        
        # Calculate fast keyword weight
        keyword_hits = sum(1 for kw in self.phishing_keywords if kw in lower_text)
        base_score = min(1.0, keyword_hits * 0.25)
        
        # Check regex patterns (wallet addresses, shorteners)
        if re.search(r'0x[a-fA-F0-9]{40}', text) or re.search(r'https?://(bit\.ly|t\.me|wa\.me)/\S*', text):
            base_score = max(base_score, 0.85)

        # Attempt Transformer Zero-Shot Inference if loaded
        if self._is_loaded and self.pipeline is not None:
            try:
                candidate_labels = ["legitimate communication", "phishing crypto scam", "spam promotion"]
                res = self.pipeline(text[:512], candidate_labels)
                labels = res.get('labels', [])
                scores = res.get('scores', [])
                
                label_score_map = dict(zip(labels, scores))
                phish_prob = label_score_map.get("phishing crypto scam", 0.0) + label_score_map.get("spam promotion", 0.0)
                return float(max(base_score, phish_prob))
            except Exception as e:
                logger.debug(f"DeBERTa pipeline inference bypass: {str(e)}")

        return float(base_score)

    def analyze_post_texts(self, post_texts: List[str]) -> Dict[str, float]:
        """
        Analyze a list of post texts and return aggregated DeBERTa threat metrics.
        """
        if not post_texts:
            return {
                'deberta_phishing_score': 0.0,
                'deberta_spam_confidence': 0.0,
                'deberta_high_risk_posts_count': 0
            }

        scores = [self.predict_phishing_score(t) for t in post_texts]
        max_score = max(scores) if scores else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0
        high_risk_count = sum(1 for s in scores if s > 0.6)

        return {
            'deberta_phishing_score': float(max_score),
            'deberta_spam_confidence': float(avg_score),
            'deberta_high_risk_posts_count': int(high_risk_count)
        }

def get_deberta_analyzer() -> DeBERTaAnalyzer:
    """
    Get singleton DeBERTa analyzer instance.
    """
    global _DEBERTA_ANALYZER_INSTANCE
    if _DEBERTA_ANALYZER_INSTANCE is None:
        _DEBERTA_ANALYZER_INSTANCE = DeBERTaAnalyzer()
    return _DEBERTA_ANALYZER_INSTANCE
