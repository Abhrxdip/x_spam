"""
XAI Engine — Master Explainability Orchestrator
=================================================

Coordinates all 4 XAI layers for a single prediction:
  1. SHAP TreeExplainer    → per-feature signed Shapley contributions
  2. LIME Verifier         → model-agnostic cross-verification
  3. Counterfactual Engine → minimum interventions to reach SAFE status
  4. NLP Token Saliency    → DistilBERT attention heatmap on top tweet

Called from detector.py:
    xai_report = self.xai_engine.explain(X_scaled, feature_df, raw_features, tweets)

The returned XAIReport dict is attached to the analysis result and
rendered in the forensic dossier UI panel in results.html.
"""

import logging
import os
import numpy as np
from typing import List, Dict, Any, Optional

from src.xai.shap_explainer import SHAPExplainer
from src.xai.lime_explainer import LIMEVerifier
from src.xai.counterfactual import CounterfactualEngine
from src.xai.nlp_saliency import NLPSaliencyExtractor

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_NLP_MODEL_DIR = os.path.join(_PROJECT_ROOT, 'models', 'nlp_classifier')


class XAIEngine:
    """
    Master XAI orchestrator. Initialize once at server startup,
    then call explain() for each profile scan.
    """

    def __init__(self, model, scaler, feature_names: List[str]):
        """
        Args:
            model:         Trained sklearn AdaBoost / tree model
            scaler:        Fitted StandardScaler from training
            feature_names: List of feature column names (in order)
        """
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names

        logger.info("Initializing XAI Engine — loading SHAP, LIME, Counterfactual, NLP Saliency...")

        self.shap_explainer   = SHAPExplainer(model, feature_names)
        self.lime_verifier    = LIMEVerifier(model, scaler, feature_names)
        self.cf_engine        = CounterfactualEngine(model, scaler, feature_names)
        self.nlp_saliency     = NLPSaliencyExtractor(
            model_dir=_NLP_MODEL_DIR if os.path.isdir(_NLP_MODEL_DIR) else None
        )

        logger.info("XAI Engine initialized successfully")

    def explain(
        self,
        X_scaled: np.ndarray,
        raw_features: Dict[str, Any],
        tweets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete XAI forensic report for one prediction.

        Args:
            X_scaled:     Scaled feature array (1 x n_features)
            raw_features: Original unscaled feature dict from feature_extractor
            tweets:       List of raw tweet text strings (for NLP saliency)

        Returns:
            XAIReport dict:
            {
                'shap':            { baseline, final_score, top_threats, top_safe, contributions, method }
                'lime':            { available, top_features, score, method }
                'counterfactual':  { original_score, projected_score, interventions, summary }
                'nlp_saliency':    { tokens, top_triggers, annotated_html, overall_risk }
                'consensus':       { agreement_pct, consensus_level, description }
                'xai_available':   bool
            }
        """
        report = {}

        # ── 1. SHAP ───────────────────────────────────────────────────────────
        try:
            report['shap'] = self.shap_explainer.explain(X_scaled)
        except Exception as e:
            logger.error(f"XAI SHAP failed: {e}")
            report['shap'] = {'available': False, 'error': str(e), 'contributions': [],
                              'top_threats': [], 'top_safe': [], 'method': 'error',
                              'baseline': 50.0, 'final_score': 50.0}

        # ── 2. LIME ───────────────────────────────────────────────────────────
        try:
            report['lime'] = self.lime_verifier.verify(X_scaled, num_samples=300)
        except Exception as e:
            logger.error(f"XAI LIME failed: {e}")
            report['lime'] = {'available': False, 'top_features': [], 'method': 'error'}

        # ── 3. SHAP–LIME Consensus ────────────────────────────────────────────
        try:
            shap_top_names = [c['feature'] for c in report['shap'].get('top_threats', [])]
            lime_conditions = [f['condition'] for f in report['lime'].get('top_features', [])]
            report['consensus'] = self.lime_verifier.compute_consensus(shap_top_names, lime_conditions)
        except Exception as e:
            logger.warning(f"Consensus computation failed: {e}")
            report['consensus'] = {
                'agreement_pct': 0,
                'consensus_level': 'UNKNOWN',
                'consensus_color': 'secondary',
                'description': 'Consensus unavailable'
            }

        # ── 4. Counterfactual ─────────────────────────────────────────────────
        try:
            report['counterfactual'] = self.cf_engine.generate(X_scaled, raw_features)
        except Exception as e:
            logger.error(f"XAI Counterfactual failed: {e}")
            report['counterfactual'] = {
                'original_score': 0,
                'projected_score': 0,
                'interventions': [],
                'summary': 'Counterfactual analysis unavailable',
                'is_achievable': False,
                'already_safe': False
            }

        # ── 5. NLP Token Saliency ─────────────────────────────────────────────
        try:
            tweet_list = tweets or []
            if tweet_list:
                saliency_results = self.nlp_saliency.analyze_timeline(tweet_list, top_k=1)
                report['nlp_saliency'] = saliency_results[0] if saliency_results else self.nlp_saliency._empty_result()
            else:
                report['nlp_saliency'] = self.nlp_saliency._empty_result()
        except Exception as e:
            logger.error(f"XAI NLP Saliency failed: {e}")
            report['nlp_saliency'] = {
                'tokens': [], 'overall_risk': 0.0,
                'top_triggers': [], 'annotated_html': '',
                'method': 'error'
            }

        report['xai_available'] = True
        return report
