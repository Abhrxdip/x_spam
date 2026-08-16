"""
LIME Cross-Verifier — Model-Agnostic Local Surrogate
======================================================

LIME (Local Interpretable Model-agnostic Explanations) independently
verifies SHAP by building a local linear approximation around the
prediction point in a perturbed neighbourhood of 500 samples.

When SHAP and LIME agree on the top features → XAI Consensus HIGH.
When they disagree → the model may have complex non-linear interactions
worth flagging in the forensic report.

Formula:
    L(x) = argmin_{g ∈ G} L(f, g, πx) + Ω(g)
    where f = AdaBoost, g = ridge regression, πx = locality kernel
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class LIMEVerifier:
    """
    LIME-based cross-verification of SHAP attributions.
    Degrades gracefully if `lime` package is not installed.
    """

    def __init__(self, model, scaler, feature_names: List[str]):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self._available = False
        self._explainer = None

        try:
            from lime.lime_tabular import LimeTabularExplainer
            import numpy as np

            # LIME background reference distribution
            dummy_train = np.random.normal(0, 1, (100, len(feature_names)))

            self._explainer = LimeTabularExplainer(
                training_data=dummy_train,
                feature_names=feature_names,
                class_names=['legitimate', 'threat'],
                mode='classification',
                discretize_continuous=False,
                random_state=42
            )
            self._available = True
            logger.info("LIME LimeTabularExplainer initialized successfully")
        except ImportError:
            logger.warning("LIME not installed. Run: pip install lime. LIME cross-verification disabled.")
        except Exception as e:
            logger.warning(f"LIME init failed: {e}")

    def verify(self, X_scaled: np.ndarray, num_samples: int = 500) -> Dict[str, Any]:
        """
        Run LIME local surrogate and return top feature attributions.

        Args:
            X_scaled: Single row (1 x n_features) scaled feature array
            num_samples: Number of neighbourhood samples for surrogate fitting

        Returns:
            Dict with:
                'available': bool
                'top_features': list of {feature, label, weight, direction}
                'intercept': linear intercept of surrogate
                'score': local surrogate fidelity score (R²)
                'consensus_features': features agreed upon by both SHAP and LIME
        """
        if not self._available:
            return {'available': False, 'top_features': [], 'method': 'disabled'}

        try:
            def _predict_fn(X):
                """LIME needs a raw predict_proba function."""
                return self.model.predict_proba(X)

            explanation = self._explainer.explain_instance(
                data_row=X_scaled[0],
                predict_fn=_predict_fn,
                num_features=10,
                num_samples=num_samples,
                labels=(1,)  # explain threat class
            )

            # Extract LIME weights for threat class
            lime_list = explanation.as_list(label=1)
            intercept = float(explanation.intercept[1])
            score = float(explanation.score)

            top_features = []
            for condition, weight in lime_list:
                top_features.append({
                    'condition': condition,      # e.g. "link_post_ratio > 0.50"
                    'weight': round(float(weight), 4),
                    'direction': 'threat' if weight > 0 else 'safe',
                    'abs_weight': abs(float(weight)),
                })

            top_features.sort(key=lambda x: x['abs_weight'], reverse=True)

            return {
                'available': True,
                'top_features': top_features,
                'intercept': round(intercept, 4),
                'score': round(score, 4),
                'method': 'lime',
            }

        except Exception as e:
            logger.error(f"LIME verify failed: {e}", exc_info=True)
            return {'available': False, 'top_features': [], 'method': 'lime_error', 'error': str(e)}

    def compute_consensus(self, shap_top: List[str], lime_top: List[str]) -> Dict[str, Any]:
        """
        Compute agreement between SHAP and LIME top features.

        Args:
            shap_top: List of top feature names from SHAP
            lime_top: List of feature condition strings from LIME

        Returns:
            Consensus dict with agreement score and shared features
        """
        # Extract raw feature names from LIME conditions (e.g. "link_post_ratio > 0.50")
        lime_feature_names = set()
        for cond in lime_top:
            for fname in self.feature_names:
                if fname in cond:
                    lime_feature_names.add(fname)
                    break

        shap_set = set(shap_top)
        agreed = shap_set & lime_feature_names

        if len(shap_set) == 0:
            agreement_pct = 0.0
        else:
            agreement_pct = len(agreed) / len(shap_set) * 100

        if agreement_pct >= 60:
            consensus_level = 'HIGH'
            consensus_color = 'success'
        elif agreement_pct >= 30:
            consensus_level = 'MODERATE'
            consensus_color = 'warning'
        else:
            consensus_level = 'LOW'
            consensus_color = 'danger'

        return {
            'agreement_pct': round(agreement_pct, 1),
            'consensus_level': consensus_level,
            'consensus_color': consensus_color,
            'agreed_features': list(agreed),
            'description': (
                f"SHAP & LIME agree on {len(agreed)} of {len(shap_set)} top features "
                f"({agreement_pct:.0f}% XAI consensus)."
            )
        }
