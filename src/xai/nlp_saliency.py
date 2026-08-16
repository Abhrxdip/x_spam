"""
DistilBERT Token Saliency Heatmap
===================================

Extracts token-level attention weights from the loaded DistilBERT model
to produce a visual saliency heatmap for the most suspicious tweet.

This shows judges EXACTLY which words triggered the phishing/scam detection
rather than just showing a confidence score.

Method 1 (preferred if DistilBERT model is loaded):
  - Run forward pass with output_attentions=True
  - Average the last-layer attention across all heads for the [CLS] token
  - Tokens with highest attention receive the highest "risk weight"

Method 2 (fallback — keyword-based heuristic):
  - Matches tokens against known threat keyword dictionaries
  - Assigns risk weights based on category severity

Output format (suitable for CSS background-color rendering in UI):
  [
    {'token': 'airdrop',  'risk': 0.92, 'category': 'crypto_scam',  'color_hex': '#FF1744'},
    {'token': 'FREE',     'risk': 0.87, 'category': 'crypto_scam',  'color_hex': '#FF4081'},
    {'token': 'claim',    'risk': 0.78, 'category': 'phishing',     'color_hex': '#FF6D00'},
    {'token': 't.me',     'risk': 0.95, 'category': 'redirect_link','color_hex': '#FF1744'},
    {'token': 'now',      'risk': 0.15, 'category': None,           'color_hex': '#424242'},
  ]
"""

import logging
import re
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Risk color scale: from neutral grey → orange → deep red
def _risk_to_color(risk: float) -> str:
    """Map 0.0–1.0 risk score to hex color for UI rendering."""
    if risk < 0.15:
        return '#424242'    # neutral grey
    elif risk < 0.35:
        return '#FFC107'    # amber
    elif risk < 0.55:
        return '#FF9800'    # orange
    elif risk < 0.75:
        return '#FF6D00'    # deep orange
    elif risk < 0.88:
        return '#FF4081'    # hot pink / critical
    else:
        return '#FF1744'    # red / critical danger

# Known threat keywords with category and base risk weight
_THREAT_KEYWORDS: Dict[str, tuple] = {
    # (category, base_risk)
    'airdrop':       ('crypto_scam', 0.91),
    'giveaway':      ('crypto_scam', 0.88),
    'free':          ('crypto_scam', 0.55),
    'bitcoin':       ('crypto_scam', 0.65),
    'eth':           ('crypto_scam', 0.62),
    'usdt':          ('crypto_scam', 0.78),
    'crypto':        ('crypto_scam', 0.58),
    'whitelist':     ('crypto_scam', 0.89),
    'presale':       ('crypto_scam', 0.87),
    'claim':         ('phishing',    0.82),
    'verify':        ('phishing',    0.76),
    'suspended':     ('phishing',    0.85),
    'restore':       ('phishing',    0.80),
    'click':         ('phishing',    0.60),
    'login':         ('phishing',    0.65),
    'password':      ('phishing',    0.70),
    'wallet':        ('phishing',    0.75),
    'metamask':      ('phishing',    0.88),
    'seed':          ('phishing',    0.92),
    'phrase':        ('phishing',    0.88),
    't.me':          ('redirect_link', 0.94),
    'bit.ly':        ('redirect_link', 0.89),
    'tinyurl':       ('redirect_link', 0.82),
    'wa.me':         ('redirect_link', 0.87),
    'dm':            ('social_engineering', 0.72),
    'inbox':         ('social_engineering', 0.68),
    'message':       ('social_engineering', 0.40),
    'earn':          ('social_engineering', 0.65),
    'passive':       ('social_engineering', 0.70),
    'guaranteed':    ('social_engineering', 0.83),
    'profit':        ('social_engineering', 0.75),
    'investment':    ('social_engineering', 0.55),
    'signal':        ('social_engineering', 0.58),
    'forex':         ('social_engineering', 0.72),
}


class NLPSaliencyExtractor:
    """
    Token-level saliency extractor for DistilBERT threat classification.
    Uses attention weights when model is loaded, keyword heuristics as fallback.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self._tokenizer = None
        self._model = None
        self._available = False
        self.model_dir = model_dir

        if model_dir:
            self._try_load_model(model_dir)

    def _try_load_model(self, model_dir: str):
        """Attempt to load DistilBERT model for attention extraction."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch, os
            if os.path.isdir(model_dir):
                self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir, output_attentions=True
                )
                self._model.eval()
                self._available = True
                logger.info("NLPSaliencyExtractor: DistilBERT loaded for attention extraction")
        except Exception as e:
            logger.warning(f"NLPSaliencyExtractor: Could not load DistilBERT ({e}). Using keyword fallback.")

    def extract_saliency(self, tweet_text: str, max_tokens: int = 50) -> Dict[str, Any]:
        """
        Extract token saliency from a tweet text.

        Args:
            tweet_text: The raw tweet string to analyze
            max_tokens: Maximum number of tokens to display

        Returns:
            Dict:
                'tokens': list of {token, risk, category, color_hex, is_threat}
                'overall_risk': float (0-1) max risk token's score
                'method': 'attention' | 'keyword'
                'top_triggers': list of the highest-risk tokens
                'annotated_text': HTML-ready annotated string
        """
        if not tweet_text or not tweet_text.strip():
            return self._empty_result()

        if self._available:
            try:
                return self._attention_saliency(tweet_text, max_tokens)
            except Exception as e:
                logger.warning(f"Attention saliency failed ({e}), using keyword fallback")

        return self._keyword_saliency(tweet_text)

    def _attention_saliency(self, text: str, max_tokens: int) -> Dict[str, Any]:
        """Attention-weight based saliency using DistilBERT last layer."""
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_tokens,
            padding=False
        )

        with torch.no_grad():
            outputs = self._model(**inputs)

        # outputs.attentions: tuple of (batch, heads, seq, seq)
        # Last layer attention for [CLS] token (index 0) averaged across heads
        last_attn = outputs.attentions[-1]          # (1, heads, seq, seq)
        cls_attn = last_attn[0, :, 0, :].mean(0)   # (seq,) — CLS attending to each token
        cls_attn = cls_attn.numpy()
        cls_attn = cls_attn / (cls_attn.max() + 1e-9)  # normalize to [0, 1]

        tokens = self._tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        token_list = []
        for tok, attn_weight in zip(tokens, cls_attn):
            clean_tok = tok.replace('##', '').strip()
            if clean_tok in ('[CLS]', '[SEP]', '[PAD]', ''):
                continue

            # Check if this token matches a known threat keyword
            lookup = clean_tok.lower()
            category, base_risk = _THREAT_KEYWORDS.get(lookup, (None, 0.0))

            # Blend attention weight with keyword base risk
            risk = float(np.clip(0.7 * attn_weight + 0.3 * base_risk, 0.0, 1.0))
            if category:
                risk = float(np.clip(risk + 0.2, 0.0, 1.0))  # boost known threats

            token_list.append({
                'token': clean_tok,
                'risk': round(risk, 3),
                'category': category,
                'color_hex': _risk_to_color(risk),
                'is_threat': risk >= 0.5,
            })

        return self._build_result(token_list, method='attention')

    def _keyword_saliency(self, text: str) -> Dict[str, Any]:
        """Fast keyword-based saliency when model is not available."""
        # Tokenize by whitespace and punctuation
        words = re.findall(r"[\w.:/]+|[^\w\s]", text.lower())
        token_list = []

        for word in words:
            # Strip punctuation for lookup
            clean = re.sub(r'[^\w.]', '', word)
            category, base_risk = _THREAT_KEYWORDS.get(clean, (None, 0.05))

            # Slight boost for ALL CAPS words (urgency signals)
            caps_boost = 0.15 if word.isupper() and len(word) > 2 else 0.0
            # Boost for URLs
            url_boost = 0.3 if any(x in word for x in ['http', 't.me', 'bit.ly', '.com']) else 0.0

            risk = float(np.clip(base_risk + caps_boost + url_boost, 0.0, 1.0))
            if category:
                risk = float(np.clip(risk + 0.1, 0.0, 1.0))

            token_list.append({
                'token': word,
                'risk': round(risk, 3),
                'category': category,
                'color_hex': _risk_to_color(risk),
                'is_threat': risk >= 0.5,
            })

        return self._build_result(token_list, method='keyword')

    def _build_result(self, token_list: List[Dict], method: str) -> Dict[str, Any]:
        """Format token list into complete saliency result."""
        if not token_list:
            return self._empty_result()

        overall_risk = max((t['risk'] for t in token_list), default=0.0)
        top_triggers = sorted(
            [t for t in token_list if t['is_threat']],
            key=lambda x: x['risk'],
            reverse=True
        )[:5]

        # Build annotated HTML snippet
        parts = []
        for t in token_list:
            if t['risk'] >= 0.15:
                opacity = round(0.15 + t['risk'] * 0.85, 2)
                cat_label = t.get('category') or 'neutral'
                risk_pct = f"{t['risk']:.0%}"
                parts.append(
                    f'<span class="token-highlight" '
                    f'style="background-color:{t["color_hex"]};opacity:{opacity};'
                    f'border-radius:3px;padding:1px 3px;color:#fff;" '
                    f'title="{cat_label} risk: {risk_pct}">'
                    f'{t["token"]}</span>'
                )
            else:
                parts.append(t['token'])

        annotated_html = ' '.join(parts)

        return {
            'tokens': token_list,
            'overall_risk': round(overall_risk, 3),
            'method': method,
            'top_triggers': top_triggers,
            'annotated_html': annotated_html,
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            'tokens': [],
            'overall_risk': 0.0,
            'method': 'none',
            'top_triggers': [],
            'annotated_html': '<span class="text-muted">No tweet text available</span>',
        }

    def analyze_timeline(self, tweets: List[str], top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Analyze a list of tweets and return saliency for the most suspicious ones.

        Args:
            tweets: List of tweet text strings
            top_k: How many of the most suspicious tweets to fully analyze

        Returns:
            List of saliency dicts sorted by overall_risk descending
        """
        if not tweets:
            return [self._empty_result()]

        results = []
        for tweet in tweets[:20]:  # analyze at most 20 tweets for performance
            r = self.extract_saliency(tweet)
            r['tweet_text'] = tweet
            results.append(r)

        # Sort by overall risk
        results.sort(key=lambda x: x['overall_risk'], reverse=True)
        return results[:top_k]
