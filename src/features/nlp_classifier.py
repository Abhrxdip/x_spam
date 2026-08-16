"""
Social Engineering Classifier — Inference Module

Loads the fine-tuned DistilBERT model from models/nlp_classifier/
and classifies social media post text into 5 threat categories:

    0 - legitimate
    1 - crypto_scam
    2 - phishing
    3 - mention_spam
    4 - social_engineering

Usage:
    from src.features.nlp_classifier import get_nlp_classifier
    clf = get_nlp_classifier()
    result = clf.classify_text("Free Bitcoin! Send 0.1 ETH to claim 1 ETH back!")
    # {'label': 'crypto_scam', 'confidence': 0.97, 'scores': {...}}

    metrics = clf.analyze_posts(["text1", "text2", ...])
    # {'nlp_phishing_score': 0.87, 'nlp_spam_confidence': 0.72, ...}
"""

import os
import re
import logging
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

# ─── Paths ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL_DIR = os.path.join(_PROJECT_ROOT, "models", "nlp_classifier")

# ─── Label definitions ───────────────────────────────────────────────────────
LABEL2ID = {
    "legitimate": 0,
    "crypto_scam": 1,
    "phishing": 2,
    "mention_spam": 3,
    "social_engineering": 4,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ─── Threat labels that count as "threat" ────────────────────────────────────
THREAT_LABELS = {"crypto_scam", "phishing", "mention_spam", "social_engineering"}

# ─── Fast keyword heuristics (used when model not loaded) ────────────────────
_CRYPTO_KW = [
    "airdrop", "giveaway", "doubler", "seed phrase", "metamask", "trustwallet",
    "whitelist", "presale", "claim", "0x", "usdt", "free bitcoin", "free eth",
    "free solana", "free bnb", "free crypto", "send 0.", "doubling", "multiply"
]
_PHISHING_KW = [
    "account suspended", "verify now", "click here", "your account has been",
    "update your billing", "login immediately", "bit.ly", "tinyurl", "wa.me",
    "t.me", "limited access", "restore access", "confirm your", "password expire"
]
_MENTION_PAT = re.compile(r"(@\w+\s*){3,}")  # 3+ @mentions in a row
_SOCIAL_ENG_KW = [
    "dm me", "message me", "inbox me", "work from home", "earn $", "earn from home",
    "guaranteed return", "guaranteed profit", "passive income", "make money",
    "investment tip", "trading signal", "forex signal", "stranded", "send $",
    "pay you back", "western union", "sugar daddy", "dating", "lonely"
]

# ─── Singleton ────────────────────────────────────────────────────────────────
_NLP_CLASSIFIER_INSTANCE = None


class SocialEngineeringClassifier:
    """
    DistilBERT-based 5-class social engineering tweet classifier.

    Lazy-loads the fine-tuned model on first inference call.
    Falls back gracefully to fast keyword heuristics if model
    has not been trained yet (run scripts/finetune_nlp.py first).
    """

    def __init__(self, model_dir: str = _MODEL_DIR):
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None
        self._loaded = False
        self._model_available = os.path.exists(
            os.path.join(model_dir, "config.json")
        )
        if self._model_available:
            logger.info(
                "Fine-tuned NLP classifier found at %s — will lazy-load on first use.",
                model_dir,
            )
        else:
            logger.warning(
                "Fine-tuned NLP model NOT found at %s. "
                "Run 'python scripts/finetune_nlp.py' to train it. "
                "Using keyword heuristics as fallback.",
                model_dir,
            )

    # ── Model Loading ────────────────────────────────────────────────────────
    def _load_model(self) -> bool:
        if self._loaded:
            return True
        if not self._model_available:
            return False
        try:
            import torch
            from transformers import (
                DistilBertTokenizerFast,
                DistilBertForSequenceClassification,
            )
            logger.info("Loading fine-tuned DistilBERT from %s ...", self.model_dir)
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_dir)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.model_dir
            )
            self.model.eval()
            self._loaded = True
            logger.info("DistilBERT NLP classifier loaded successfully.")
            return True
        except Exception as e:
            logger.warning("Could not load NLP classifier model: %s", str(e))
            return False

    # ── Single text classification ───────────────────────────────────────────
    def classify_text(self, text: str) -> Dict[str, Any]:
        """
        Classify a single post/tweet text.

        Returns:
            {
              'label': 'crypto_scam',
              'label_id': 1,
              'confidence': 0.97,
              'scores': {'legitimate': 0.01, 'crypto_scam': 0.97, ...},
              'source': 'model'  # or 'heuristic'
            }
        """
        if not text or not text.strip():
            return self._default_result("legitimate", 0.5, "empty")

        # Try fine-tuned model first
        if self._load_model():
            return self._model_classify(text)
        # Fallback to keyword heuristics
        return self._heuristic_classify(text)

    def _model_classify(self, text: str) -> Dict[str, Any]:
        """Run DistilBERT inference."""
        try:
            import torch

            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=128,
                return_tensors="pt",
                padding=True,
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0].tolist()

            label_id = int(logits.argmax(dim=-1).item())
            label = ID2LABEL[label_id]
            scores = {ID2LABEL[i]: round(p, 4) for i, p in enumerate(probs)}

            return {
                "label": label,
                "label_id": label_id,
                "confidence": round(probs[label_id], 4),
                "scores": scores,
                "source": "model",
            }
        except Exception as e:
            logger.debug("Model inference error: %s — falling back to heuristic", e)
            return self._heuristic_classify(text)

    def _heuristic_classify(self, text: str) -> Dict[str, Any]:
        """Fast keyword-based fallback classifier."""
        lower = text.lower()

        # Scores for each class (0–1 range)
        s = {
            "legitimate": 0.7,
            "crypto_scam": 0.0,
            "phishing": 0.0,
            "mention_spam": 0.0,
            "social_engineering": 0.0,
        }

        # Crypto scam signals
        kw_hits = sum(1 for kw in _CRYPTO_KW if kw in lower)
        s["crypto_scam"] = min(1.0, kw_hits * 0.25)
        if re.search(r"0x[a-fA-F0-9]{20,}", text):
            s["crypto_scam"] = max(s["crypto_scam"], 0.90)

        # Phishing signals
        ph_hits = sum(1 for kw in _PHISHING_KW if kw in lower)
        s["phishing"] = min(1.0, ph_hits * 0.30)
        if re.search(r"https?://(bit\.ly|tinyurl|wa\.me|t\.me)/\S+", lower):
            s["phishing"] = max(s["phishing"], 0.75)

        # Mention spam signals
        if _MENTION_PAT.search(text):
            mention_count = len(re.findall(r"@\w+", text))
            s["mention_spam"] = min(1.0, mention_count * 0.15)

        # Social engineering signals
        se_hits = sum(1 for kw in _SOCIAL_ENG_KW if kw in lower)
        s["social_engineering"] = min(1.0, se_hits * 0.20)

        # Determine winner
        best_label = max(s, key=s.__getitem__)
        best_conf = s[best_label]

        # Downgrade 'legitimate' confidence if any threat score is substantial
        max_threat = max(s[l] for l in THREAT_LABELS)
        if max_threat > 0.3:
            s["legitimate"] = max(0.05, 0.7 - max_threat)

        return {
            "label": best_label,
            "label_id": LABEL2ID[best_label],
            "confidence": round(best_conf, 4),
            "scores": {k: round(v, 4) for k, v in s.items()},
            "source": "heuristic",
        }

    def _default_result(self, label: str, confidence: float, source: str) -> Dict[str, Any]:
        scores = {k: 0.0 for k in LABEL2ID}
        scores[label] = confidence
        return {
            "label": label,
            "label_id": LABEL2ID[label],
            "confidence": confidence,
            "scores": scores,
            "source": source,
        }

    # ── Batch analysis for feature extractor ────────────────────────────────
    def analyze_posts(self, post_texts: List[str]) -> Dict[str, float]:
        """
        Analyze a list of post texts and return aggregated threat metrics
        compatible with the feature extractor.

        Returns:
            {
              'nlp_phishing_score':    float  # max threat prob across all posts
              'nlp_spam_confidence':   float  # average threat prob
              'nlp_threat_class':      int    # most common threat class (0-4)
              'nlp_high_risk_count':   int    # posts with confidence > 0.7
              'nlp_source':            str    # 'model' or 'heuristic'
            }
        """
        if not post_texts:
            return {
                "nlp_phishing_score": 0.0,
                "nlp_spam_confidence": 0.0,
                "nlp_threat_class": 0,
                "nlp_high_risk_count": 0,
                "nlp_source": "none",
            }

        results = [self.classify_text(t) for t in post_texts[:20]]  # cap at 20 for speed

        # Aggregate threat scores (non-legitimate classes)
        threat_scores = []
        threat_class_votes = []
        high_risk_count = 0
        source = results[0]["source"] if results else "none"

        for r in results:
            if r["label"] in THREAT_LABELS:
                score = r["confidence"]
            else:
                # Even for 'legitimate', use max threat score
                score = max(r["scores"].get(l, 0.0) for l in THREAT_LABELS)
            threat_scores.append(score)

            if r["label"] in THREAT_LABELS:
                threat_class_votes.append(r["label_id"])

            if score > 0.65:
                high_risk_count += 1

        max_threat = max(threat_scores) if threat_scores else 0.0
        avg_threat = sum(threat_scores) / len(threat_scores) if threat_scores else 0.0

        # Most common threat class
        if threat_class_votes:
            from collections import Counter
            most_common = Counter(threat_class_votes).most_common(1)[0][0]
        else:
            most_common = 0

        return {
            "nlp_phishing_score": round(float(max_threat), 4),
            "nlp_spam_confidence": round(float(avg_threat), 4),
            "nlp_threat_class": int(most_common),
            "nlp_high_risk_count": int(high_risk_count),
            "nlp_source": source,
        }

    def is_model_loaded(self) -> bool:
        return self._loaded

    def model_status(self) -> str:
        if self._loaded:
            return "fine-tuned DistilBERT (active)"
        elif self._model_available:
            return "fine-tuned DistilBERT (not yet loaded)"
        return "keyword heuristics (model not trained)"


# ── Singleton accessor ────────────────────────────────────────────────────────
def get_nlp_classifier() -> SocialEngineeringClassifier:
    """Return the global singleton NLP classifier instance."""
    global _NLP_CLASSIFIER_INSTANCE
    if _NLP_CLASSIFIER_INSTANCE is None:
        _NLP_CLASSIFIER_INSTANCE = SocialEngineeringClassifier()
    return _NLP_CLASSIFIER_INSTANCE
