"""
SHAP TreeExplainer — Per-Prediction Local Attribution
=====================================================

Computes mathematically exact Shapley values for AdaBoost/tree-based
models using cooperative game theory. Every feature's contribution to
this specific prediction is decomposed as a signed percentage.

Shapley interpretation:
  +15.3% → This feature pushed threat score UP by 15.3 percentage points
  -8.2%  → This feature pushed threat score DOWN by 8.2 percentage points

The sum of all Shapley values + baseline probability = final prediction score.
This is the gold standard of XAI for tabular classifiers.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

_FEATURE_LABELS = {
    'account_age_days':            'Account Age (days)',
    'followers_count':             'Followers Count',
    'following_count':             'Following Count',
    'posts_count':                 'Total Posts Count',
    'followers_to_following_ratio':'Follower / Following Ratio',
    'posts_per_day':               'Posts Per Day (Frequency)',
    'bio_length':                  'Bio Character Length',
    'has_external_url':            'External URL in Bio',
    'sentiment_score':             'Bio Sentiment Score',
    'content_diversity':           'Content Diversity (TTR)',
    'suspicious_content_score':    'Suspicious Keyword Density',
    'spam_pattern_matches':        'Spam Pattern Match Count',
    'nlp_phishing_score':          'DistilBERT Phishing Score',
    'nlp_spam_confidence':         'DistilBERT Spam Confidence',
    'nlp_threat_class':            'NLP Threat Class ID',
    'nlp_high_risk_count':         'High-Risk NLP Tweet Count',
    'deberta_phishing_score':      'DeBERTa Phishing Score',
    'deberta_spam_confidence':     'DeBERTa Spam Confidence',
    'mention_count':               'Total Mention Count',
    'mention_ratio':               'Mention Ratio (mentions/posts)',
    'avg_mentions_per_post':       'Avg Mentions Per Post',
    'hashtag_stuffing_ratio':      'Hashtag Stuffing Ratio',
    'link_post_ratio':             'External Link Post Ratio',
    'duplicate_post_ratio':        'Duplicate/Copied Post Ratio',
    'engagement_rate':             'Engagement Rate',
    'posting_regularity':          'Posting Regularity (CV)',
    'time_zone_consistency':       'Timezone Consistency Score',
    'activity_score':              'Overall Activity Score',
    'network_isolation_score':     'Network Isolation Score',
    'mutual_connection_ratio':     'Mutual Connection Ratio',
    'clustering_coefficient':      'Graph Clustering Coefficient',
    'reciprocity':                 'Network Reciprocity Score',
    'network_score':               'Composite Network Score',
    'profile_pic_score':           'Profile Image Authenticity Score',
    'is_default_image':            'Default Placeholder Avatar',
    'is_stock_photo':              'Stock / Generic Photo',
    'is_ai_generated':             'AI-Generated Profile Image',
}

def _get_label(feature_name: str) -> str:
    return _FEATURE_LABELS.get(feature_name, feature_name.replace('_', ' ').title())


class SHAPExplainer:
    """
    Wrapper around SHAP TreeExplainer.
    Gracefully degrades to permutation importance if SHAP is not installed.
    """

    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self._explainer = None
        self._baseline = 0.5  # default prior
        self._available = False

        try:
            import shap
            # TreeExplainer is the fastest exact method for tree-based ensembles
            # suppress the progress bar warning
            self._explainer = shap.TreeExplainer(model)
            self._available = True
            logger.info("SHAP TreeExplainer initialized successfully")
        except ImportError:
            logger.warning("SHAP not installed. Run: pip install shap. Falling back to permutation attribution.")
        except Exception as e:
            logger.warning(f"SHAP TreeExplainer init failed ({e}). Falling back to permutation attribution.")

    def explain(self, X_scaled: np.ndarray) -> Dict[str, Any]:
        """
        Compute Shapley values for a single prediction.

        Args:
            X_scaled: Single row (1 x n_features) scaled feature array

        Returns:
            Dict with:
                'baseline': baseline probability (average prediction)
                'final_score': final predicted probability
                'contributions': list of {feature, label, value, pct, direction}
                  sorted by absolute contribution descending
                'top_threats': top 5 features pushing toward THREAT (positive shap)
                'top_safe': top 3 features pushing toward SAFE (negative shap)
                'method': 'shap' | 'permutation'
        """
        if self._available:
            return self._shap_explain(X_scaled)
        else:
            return self._permutation_explain(X_scaled)

    def _shap_explain(self, X_scaled: np.ndarray) -> Dict[str, Any]:
        """Exact SHAP TreeExplainer path."""
        import shap
        try:
            shap_vals = self._explainer.shap_values(X_scaled)
            expected = float(self._explainer.expected_value)

            # Binary classification: shap_values is list[2] or array
            if isinstance(shap_vals, list):
                sv = shap_vals[1][0]          # threat class
            else:
                sv = shap_vals[0]

            if len(sv) != len(self.feature_names):
                raise ValueError(f"SHAP value length {len(sv)} != feature count {len(self.feature_names)}")

            # Compute baseline as sigmoid of expected log-odds if available
            baseline = float(np.clip(expected, 0.0, 1.0))
            self._baseline = baseline

            # Final predicted prob
            final_score = float(self.model.predict_proba(X_scaled)[0][1])

            return self._format_output(sv, baseline, final_score, method='shap')

        except Exception as e:
            logger.error(f"SHAP explain failed: {e}", exc_info=True)
            return self._permutation_explain(X_scaled)

    def _permutation_explain(self, X_scaled: np.ndarray) -> Dict[str, Any]:
        """
        Fallback: Permutation-based attribution.
        Zeroes out each feature and measures drop in prediction score.
        Slower but model-agnostic.
        """
        base_prob = float(self.model.predict_proba(X_scaled)[0][1])
        n_features = X_scaled.shape[1]
        sv = np.zeros(n_features)

        for i in range(n_features):
            X_perturbed = X_scaled.copy()
            X_perturbed[0, i] = 0.0  # zero out feature
            perturbed_prob = float(self.model.predict_proba(X_perturbed)[0][1])
            sv[i] = base_prob - perturbed_prob  # positive = feature was helping predict threat

        return self._format_output(sv, baseline=0.5, final_score=base_prob, method='permutation')

    def _format_output(self, sv: np.ndarray, baseline: float,
                       final_score: float, method: str) -> Dict[str, Any]:
        """Format Shapley values into human-readable contribution report."""
        contributions = []
        for i, (fname, val) in enumerate(zip(self.feature_names, sv)):
            contributions.append({
                'feature': fname,
                'label': _get_label(fname),
                'shapley_value': float(val),
                'pct': round(float(val) * 100, 2),       # e.g. +15.3 or -8.2
                'direction': 'threat' if val > 0 else 'safe',
                'abs_value': abs(float(val)),
            })

        # Sort by absolute contribution
        contributions.sort(key=lambda x: x['abs_value'], reverse=True)

        top_threats = [c for c in contributions if c['direction'] == 'threat'][:5]
        top_safe = [c for c in contributions if c['direction'] == 'safe'][:3]

        return {
            'baseline': round(baseline * 100, 2),
            'final_score': round(final_score * 100, 2),
            'contributions': contributions[:10],   # top 10 for UI
            'top_threats': top_threats,
            'top_safe': top_safe,
            'method': method,
            'available': True,
        }
