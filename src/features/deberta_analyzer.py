"""
DeBERTa Analyzer — Compatibility Shim

This module now delegates to the fine-tuned DistilBERT NLP classifier
(src/features/nlp_classifier.py) while preserving the original API
surface so no other code needs to change.

The old zero-shot DeBERTa approach required 900MB+ model download and
was impractical for real-time inference. The fine-tuned DistilBERT
is 260MB, loads in ~3 seconds, and has domain-specific accuracy.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# ─── Singleton ────────────────────────────────────────────────────────────────
_DEBERTA_ANALYZER_INSTANCE = None


class DeBERTaAnalyzer:
    """
    Backwards-compatible wrapper around the fine-tuned NLP classifier.
    Returns the same keys as the original DeBERTa implementation so
    feature_extractor.py works without changes.
    """

    def __init__(self):
        self._clf = None

    def _get_clf(self):
        if self._clf is None:
            from src.features.nlp_classifier import get_nlp_classifier
            self._clf = get_nlp_classifier()
        return self._clf

    def load_model(self) -> bool:
        """Lazy-load the underlying NLP classifier."""
        clf = self._get_clf()
        clf._load_model()
        return clf.is_model_loaded()

    def predict_phishing_score(self, text: str) -> float:
        """Return phishing/threat probability for a single text."""
        clf = self._get_clf()
        result = clf.classify_text(text)
        # Sum of all non-legitimate scores weighted by confidence
        scores = result.get("scores", {})
        threat_score = (
            scores.get("crypto_scam", 0.0) * 1.0
            + scores.get("phishing", 0.0) * 1.0
            + scores.get("mention_spam", 0.0) * 0.7
            + scores.get("social_engineering", 0.0) * 0.8
        )
        return round(min(1.0, threat_score), 4)

    def analyze_post_texts(self, post_texts: List[str]) -> Dict[str, Any]:
        """
        Analyse a list of post texts and return DeBERTa-compatible metrics.

        Returns the original keys PLUS new NLP keys so the feature
        vector gains richer signals without breaking existing features.
        """
        if not post_texts:
            return {
                "deberta_phishing_score": 0.0,
                "deberta_spam_confidence": 0.0,
                "deberta_high_risk_posts_count": 0,
                # New keys from fine-tuned model
                "nlp_phishing_score": 0.0,
                "nlp_spam_confidence": 0.0,
                "nlp_threat_class": 0,
                "nlp_high_risk_count": 0,
            }

        clf = self._get_clf()
        metrics = clf.analyze_posts(post_texts)

        logger.debug(
            "NLP Classifier (%s) — phishing_score=%.3f spam_confidence=%.3f threat_class=%s",
            clf.model_status(),
            metrics["nlp_phishing_score"],
            metrics["nlp_spam_confidence"],
            metrics["nlp_threat_class"],
        )

        return {
            # Backwards-compatible keys
            "deberta_phishing_score": metrics["nlp_phishing_score"],
            "deberta_spam_confidence": metrics["nlp_spam_confidence"],
            "deberta_high_risk_posts_count": metrics["nlp_high_risk_count"],
            # New enriched keys
            "nlp_phishing_score": metrics["nlp_phishing_score"],
            "nlp_spam_confidence": metrics["nlp_spam_confidence"],
            "nlp_threat_class": metrics["nlp_threat_class"],
            "nlp_high_risk_count": metrics["nlp_high_risk_count"],
        }


def get_deberta_analyzer() -> DeBERTaAnalyzer:
    """Return global singleton DeBERTa/NLP analyzer."""
    global _DEBERTA_ANALYZER_INSTANCE
    if _DEBERTA_ANALYZER_INSTANCE is None:
        _DEBERTA_ANALYZER_INSTANCE = DeBERTaAnalyzer()
    return _DEBERTA_ANALYZER_INSTANCE
