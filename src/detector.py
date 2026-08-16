"""
Unified Social Media Threat Detector

Combines the best of both projects:
- fake-profile-detector: Flask web app, multi-analyzer architecture, explainable results
- spam_identifier: Advanced ML models, feature engineering, spam detection

This is the main detection class that orchestrates the entire process.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
import joblib
from datetime import datetime

from src.features.feature_extractor import UnifiedFeatureExtractor
from src.models.train_model import ModelTrainer

# Configure logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class UnifiedThreatDetector:
    """
    Main class for detecting threats (fake profiles, spam, scams) on social media platforms.
    
    This class orchestrates the entire detection process by:
    1. Extracting features from profile data
    2. Running the ML model to predict threat probability
    3. Generating detailed reports with confidence scores and explanations
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the detector with pre-trained models and feature extractor.

        Args:
            model_path: Path to the pre-trained ML model (defaults to models/threat_detector_model.pkl in project root)
        """
        logger.info("Initializing UnifiedThreatDetector")

        # Resolve model path relative to project root
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, 'models', 'threat_detector_model.pkl')
        elif not os.path.isabs(model_path):
            model_path = os.path.join(PROJECT_ROOT, model_path)

        # Load the ML model if it exists
        self.model_trainer = ModelTrainer()
        self.model = None
        self.model_name = None

        try:
            bundle = self.model_trainer.load_model(model_path)
            self.model = bundle['model']
            self.model_name = bundle['model_name']
            logger.info(f"Loaded {self.model_name} model from {model_path}")
        except FileNotFoundError:
            logger.warning(f"Model not found at {model_path}, will use heuristic detection")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)

        # Initialize the feature extractor
        self.feature_extractor = UnifiedFeatureExtractor()

        # Threat thresholds
        self.threat_threshold = 0.7
        self.suspicious_threshold = 0.4

        logger.info("UnifiedThreatDetector initialized successfully")
    
    def analyze_profile(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single profile and determine if it's a threat.
        
        Args:
            profile_data: Dictionary containing profile information
            
        Returns:
            Dictionary with analysis results including:
            - is_threat: Boolean indicating if profile is likely a threat
            - threat_type: Type of threat (fake, spam, scam, bot, etc.)
            - probability: Confidence score (0-1)
            - indicators: List of suspicious indicators found
            - feature_importance: Which features contributed most to the decision
            - recommendations: Suggested actions
        """
        logger.info(f"Analyzing profile: {profile_data.get('username', 'Unknown')}")
        
        try:
            # Extract features from the profile data
            features = self.feature_extractor.extract_features(profile_data)
            
            # Prepare features for the model
            feature_vector = self._prepare_feature_vector(features)
            
            # Make prediction using the model or fallback to heuristic
            if self.model:
                probability = self._predict_with_model(feature_vector)
                is_threat = probability >= self.threat_threshold
                feature_importance = self._get_feature_importance(feature_vector)
            else:
                # Fallback to heuristic analysis if no model is available
                probability, is_threat, feature_importance = self._heuristic_detection(features)
            
            # Determine threat type
            threat_type = self._classify_threat_type(features, probability)
            
            # Generate list of suspicious indicators
            indicators = self._identify_suspicious_indicators(features, feature_importance)
            
            # Generate recommendations based on the analysis
            recommendations = self._generate_recommendations(is_threat, probability, threat_type, indicators)
            
            # Prepare the final result
            sanitized_features = {}
            for k, v in features.items():
                if isinstance(v, (np.floating, float)):
                    sanitized_features[k] = float(v)
                elif isinstance(v, (np.integer, int, np.bool_, bool)):
                    sanitized_features[k] = int(v) if isinstance(v, (np.integer, int)) else bool(v)
                else:
                    sanitized_features[k] = str(v)

            result = {
                'is_threat': is_threat,
                'threat_type': threat_type,
                'probability': float(probability),
                'indicators': indicators,
                'features': sanitized_features,
                'feature_importance': feature_importance,
                'recommendations': recommendations,
                'profile_data': {
                    'username': profile_data.get('username', 'Unknown'),
                    'platform': profile_data.get('platform', 'Unknown'),
                    'followers_count': profile_data.get('followers_count', 0),
                    'following_count': profile_data.get('following_count', 0),
                    'posts_count': profile_data.get('posts_count', 0),
                    'verified': profile_data.get('verified', False),
                    'bio': profile_data.get('bio', '')
                },
                'analysis_timestamp': datetime.now().isoformat(),
                'model_used': self.model_name or 'HistGradientBoosting + DistilBERT'
            }
            
            logger.info(f"Analysis complete: threat={is_threat}, type={threat_type}, prob={probability:.3f}")
            return result
        
        except Exception as e:
            logger.error(f"Error analyzing profile: {str(e)}", exc_info=True)
            return {
                'is_threat': False,
                'threat_type': 'error',
                'probability': 0.0,
                'indicators': [{'type': 'error', 'description': f'Analysis failed: {str(e)}'}],
                'feature_importance': {},
                'recommendations': ['Please try again or contact support'],
                'profile_data': profile_data,
                'analysis_timestamp': datetime.now().isoformat(),
                'model_used': 'error'
            }
    
    def batch_analyze(self, profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze multiple profiles in batch.
        
        Args:
            profiles: List of profile data dictionaries
            
        Returns:
            List of analysis results
        """
        logger.info(f"Starting batch analysis of {len(profiles)} profiles")
        
        results = []
        for i, profile in enumerate(profiles):
            try:
                result = self.analyze_profile(profile)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing profile {i}: {str(e)}")
                results.append({
                    'is_threat': False,
                    'threat_type': 'error',
                    'probability': 0.0,
                    'indicators': [{'type': 'error', 'description': str(e)}],
                    'feature_importance': {},
                    'recommendations': ['Analysis failed'],
                    'profile_data': profile,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'model_used': 'error'
                })
        
        logger.info(f"Batch analysis complete: {sum(1 for r in results if r['is_threat'])} threats detected")
        return results
    
    def batch_analyze_from_file(self, filepath: str, platform: str = 'twitter') -> List[Dict[str, Any]]:
        """
        Analyze profiles from a batch file.
        
        Args:
            filepath: Path to the batch file (CSV or JSON)
            platform: Default platform for profiles
            
        Returns:
            List of analysis results
        """
        from src.utils.data_processor import process_batch_file
        
        logger.info(f"Processing batch file: {filepath}")
        
        # Process the batch file to get profile data
        profiles = process_batch_file(filepath, platform)
        
        # Analyze each profile
        return self.batch_analyze(profiles)
    
    def _prepare_feature_vector(self, features: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare feature vector for model input as DataFrame with feature names.
        
        Args:
            features: Dictionary of extracted features
            
        Returns:
            DataFrame containing single feature row
        """
        # Get expected feature names from model trainer
        expected_features = self.model_trainer.feature_names
        
        if not expected_features:
            # Fallback: use all numeric features
            numeric_features = {k: v for k, v in features.items() 
                              if isinstance(v, (int, float, np.number))}
            return pd.DataFrame([numeric_features])
        
        # Create feature vector in correct order
        feature_dict = {}
        for feature_name in expected_features:
            value = features.get(feature_name, 0)
            # Handle categorical features
            if feature_name in self.model_trainer.categorical_features:
                if feature_name in self.model_trainer.label_encoders:
                    le = self.model_trainer.label_encoders[feature_name]
                    try:
                        value = le.transform([str(value)])[0]
                    except Exception:
                        value = 0
            feature_dict[feature_name] = value
        
        return pd.DataFrame([feature_dict], columns=expected_features)
    
    def _predict_with_model(self, feature_df: pd.DataFrame) -> float:
        """
        Make prediction using the trained model.
        
        Args:
            feature_df: Preprocessed feature DataFrame
            
        Returns:
            Threat probability (0-1)
        """
        # Scale features
        feature_vector_scaled = self.model_trainer.scaler.transform(feature_df)
        
        # Get probability
        probability = self.model.predict_proba(feature_vector_scaled)[0][1]
        
        return float(probability)
    
    def _get_feature_importance(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """
        Get feature importance for the prediction.
        
        Args:
            feature_vector: Feature vector
            
        Returns:
            Dictionary of feature importance scores
        """
        importance = self.model_trainer.get_feature_importance()
        
        # Normalize to sum to 1
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return importance
    
    def _heuristic_detection(self, features: Dict[str, Any]) -> Tuple[float, bool, Dict[str, float]]:
        """
        Fallback heuristic detection when no model is available.
        
        Args:
            features: Extracted features
            
        Returns:
            Tuple of (probability, is_threat, feature_importance)
        """
        logger.info("Using heuristic detection")
        
        # Calculate threat score based on key indicators
        score = 0.0
        weights = {
            'account_age_days': 0.15,
            'followers_to_following_ratio': 0.15,
            'suspicious_content_score': 0.20,
            'is_default_image': 0.10,
            'is_ai_generated': 0.10,
            'network_isolation_score': 0.10,
            'spam_pattern_matches': 0.10,
            'profile_pic_score': 0.10
        }
        
        # Account age (newer = more suspicious)
        age = features.get('account_age_days', 365)
        age_score = max(0, 1 - age / 365)  # 1 for new, 0 for 1+ year
        score += weights['account_age_days'] * age_score
        
        # Follower ratio (very low = bot-like)
        ratio = features.get('followers_to_following_ratio', 1)
        ratio_score = 1 if ratio < 0.1 else (0.5 if ratio < 0.5 else 0)
        score += weights['followers_to_following_ratio'] * ratio_score
        
        # Suspicious content
        score += weights['suspicious_content_score'] * features.get('suspicious_content_score', 0)
        
        # Default image
        score += weights['is_default_image'] * features.get('is_default_image', 0)
        
        # AI generated
        score += weights['is_ai_generated'] * features.get('is_ai_generated', 0)
        
        # Network isolation
        score += weights['network_isolation_score'] * features.get('network_isolation_score', 0.5)
        
        # Spam patterns
        spam_matches = features.get('spam_pattern_matches', 0)
        spam_score = min(1.0, spam_matches / 5)
        score += weights['spam_pattern_matches'] * spam_score
        
        # Profile pic score (low = suspicious)
        pic_score = features.get('profile_pic_score', 0.5)
        score += weights['profile_pic_score'] * (1 - pic_score)
        
        # Normalize
        probability = np.clip(score, 0, 1)
        is_threat = probability >= self.threat_threshold
        
        # Feature importance for heuristic
        feature_importance = {
            'account_age': weights['account_age_days'] * age_score,
            'follower_ratio': weights['followers_to_following_ratio'] * ratio_score,
            'suspicious_content': weights['suspicious_content_score'] * features.get('suspicious_content_score', 0),
            'default_image': weights['is_default_image'] * features.get('is_default_image', 0),
            'ai_generated': weights['is_ai_generated'] * features.get('is_ai_generated', 0),
            'network_isolation': weights['network_isolation_score'] * features.get('network_isolation_score', 0.5),
            'spam_patterns': weights['spam_pattern_matches'] * spam_score,
            'profile_pic': weights['profile_pic_score'] * (1 - pic_score)
        }
        
        return probability, is_threat, feature_importance
    
    def _classify_threat_type(self, features: Dict[str, Any], probability: float) -> str:
        """
        Classify the type of threat based on features.
        
        Args:
            features: Extracted features
            probability: Threat probability
            
        Returns:
            Threat type string
        """
        if probability < self.suspicious_threshold:
            return 'legitimate'
        
        # Check for specific threat types
        if features.get('is_ai_generated', 0) == 1 or features.get('is_default_image', 0) == 1:
            if features.get('network_isolation_score', 0) > 0.7:
                return 'fake_profile'
        
        if features.get('spam_pattern_matches', 0) > 3:
            return 'spam'
        
        if features.get('suspicious_content_score', 0) > 0.5:
            if 'crypto' in str(features).lower() or 'bitcoin' in str(features).lower():
                return 'scam'
            return 'spam'
        
        if features.get('Account.Type') == 'bot' or features.get('account_type') == 'bot':
            return 'bot'
        
        if features.get('followers_to_following_ratio', 1) < 0.1 and features.get('posts_per_day', 0) > 10:
            return 'bot'
        
        if probability >= self.threat_threshold:
            return 'suspicious'
        
        return 'suspicious'
    
    def _identify_suspicious_indicators(self, features: Dict[str, Any], 
                                        feature_importance: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Identify specific suspicious indicators from features.
        
        Args:
            features: Extracted features
            feature_importance: Feature importance scores
            
        Returns:
            List of indicator dictionaries
        """
        indicators = []
        
        # Account age
        age = features.get('account_age_days', 365)
        if age < 30:
            indicators.append({
                'type': 'new_account',
                'severity': 'high' if age < 7 else 'medium',
                'description': f'Account is only {age} days old',
                'value': age
            })
        
        # Follower ratio
        ratio = features.get('followers_to_following_ratio', 1)
        if ratio < 0.1:
            # Guard: ratio can be exactly 0 when followers_count is 0
            if ratio > 0:
                imbalance_desc = f'Following {int(1/ratio)}x more accounts than followers'
            else:
                imbalance_desc = 'Account has no followers despite active following'
            indicators.append({
                'type': 'follower_imbalance',
                'severity': 'high',
                'description': imbalance_desc,
                'value': ratio
            })
        elif ratio < 0.5:
            indicators.append({
                'type': 'follower_imbalance',
                'severity': 'medium',
                'description': 'Following significantly more accounts than followers',
                'value': ratio
            })
        
        # Suspicious content
        sus_score = features.get('suspicious_content_score', 0)
        if sus_score > 0.3:
            indicators.append({
                'type': 'suspicious_content',
                'severity': 'high' if sus_score > 0.6 else 'medium',
                'description': f'Content contains suspicious keywords and patterns',
                'value': sus_score
            })
        
        # Spam patterns
        spam_matches = features.get('spam_pattern_matches', 0)
        if spam_matches > 0:
            indicators.append({
                'type': 'spam_patterns',
                'severity': 'high' if spam_matches > 3 else 'medium',
                'description': f'Found {spam_matches} spam-like patterns in content',
                'value': spam_matches
            })
        
        # Default image
        if features.get('is_default_image', 0) == 1:
            indicators.append({
                'type': 'default_profile_image',
                'severity': 'medium',
                'description': 'Using default or placeholder profile image',
                'value': 1
            })
        
        # AI generated image
        if features.get('is_ai_generated', 0) == 1:
            indicators.append({
                'type': 'ai_generated_image',
                'severity': 'high',
                'description': 'Profile image appears to be AI-generated',
                'value': 1
            })
        
        # Network isolation
        net_iso = features.get('network_isolation_score', 0.5)
        if net_iso > 0.7:
            indicators.append({
                'type': 'network_isolation',
                'severity': 'medium',
                'description': 'Account shows signs of network isolation',
                'value': net_iso
            })
        
        # Low engagement
        engagement = features.get('engagement_rate', 0)
        if engagement < 0.01 and features.get('posts_count', 0) > 10:
            indicators.append({
                'type': 'low_engagement',
                'severity': 'medium',
                'description': 'Very low engagement rate despite posting activity',
                'value': engagement
            })
        
        # High posting frequency
        posts_per_day = features.get('posts_per_day', 0)
        if posts_per_day > 50:
            indicators.append({
                'type': 'excessive_posting',
                'severity': 'high',
                'description': f'Posting {posts_per_day:.1f} times per day on average',
                'value': posts_per_day
            })
        
        # Mention Spam Attack
        avg_mentions = features.get('avg_mentions_per_post', 0)
        mention_ratio = features.get('mention_ratio', 0)
        if avg_mentions > 2.0 or mention_ratio > 0.5:
            indicators.append({
                'type': 'mention_spam',
                'severity': 'high',
                'description': f'Frequent @username tagging in posts (avg {avg_mentions:.1f} mentions/post)',
                'value': avg_mentions
            })

        # Phishing Link Campaign in Posts
        link_ratio = features.get('link_post_ratio', 0)
        if link_ratio > 0.5:
            indicators.append({
                'type': 'link_spam_campaign',
                'severity': 'high',
                'description': f'High percentage of posts ({link_ratio*100:.0f}%) containing external links',
                'value': link_ratio
            })

        # Repetitive Duplicate Postings
        dup_ratio = features.get('duplicate_post_ratio', 0)
        if dup_ratio > 0.3:
            indicators.append({
                'type': 'repetitive_posting',
                'severity': 'medium',
                'description': f'Repetitive or copy-pasted post text ({dup_ratio*100:.0f}% duplicates)',
                'value': dup_ratio
            })

        # Hashtag Stuffing
        hashtag_ratio = features.get('hashtag_stuffing_ratio', 0)
        if hashtag_ratio > 0.4:
            indicators.append({
                'type': 'hashtag_stuffing',
                'severity': 'medium',
                'description': 'Overuse of excessive hashtags per post (4+ hashtags)',
                'value': hashtag_ratio
            })

        # Unverified account with high activity
        if features.get('Twitter.Verified') == 'no' and features.get('followers_count', 0) > 10000:
            indicators.append({
                'type': 'unverified_high_followers',
                'severity': 'low',
                'description': 'High follower count without verification',
                'value': features.get('followers_count', 0)
            })

        # Fine-tuned NLP Classifier Threat Detection
        nlp_score = features.get('nlp_phishing_score', 0.0)
        nlp_class = features.get('nlp_threat_class', 0)
        nlp_high  = features.get('nlp_high_risk_count', 0)
        _NLP_CLASS_NAMES = {
            1: 'Crypto Scam',
            2: 'Phishing',
            3: 'Mention Spam',
            4: 'Social Engineering',
        }
        if nlp_score > 0.55 and nlp_class in _NLP_CLASS_NAMES:
            indicators.append({
                'type': 'nlp_threat_detected',
                'severity': 'high' if nlp_score > 0.75 else 'medium',
                'description': (
                    f'AI language model classified posts as {_NLP_CLASS_NAMES[nlp_class]} '
                    f'(confidence {nlp_score:.0%}, {nlp_high} high-risk posts)'
                ),
                'value': nlp_score,
            })
        
        # If account has clean metrics, add positive authenticity signals
        if not indicators or all(ind['severity'] == 'low' for ind in indicators):
            if age > 180:
                indicators.append({
                    'type': 'established_account',
                    'severity': 'low',
                    'severity_class': 'success',
                    'type_label': 'Established Lifespan',
                    'description': f'Account has an established history of {age} days',
                    'value': age
                })
            if ratio >= 0.5:
                indicators.append({
                    'type': 'balanced_network',
                    'severity': 'low',
                    'severity_class': 'success',
                    'type_label': 'Organic Network',
                    'description': 'Balanced follower-to-following network density',
                    'value': ratio
                })
            if nlp_score < 0.3:
                indicators.append({
                    'type': 'clean_nlp_intent',
                    'severity': 'low',
                    'severity_class': 'success',
                    'type_label': 'Clean Content',
                    'description': 'No malicious phishing or social engineering patterns in text',
                    'value': nlp_score
                })

        # Ensure all indicators have severity_class and type_label
        for ind in indicators:
            if 'severity_class' not in ind:
                sev = ind.get('severity', 'low')
                if sev == 'high':
                    ind['severity_class'] = 'danger'
                elif sev == 'medium':
                    ind['severity_class'] = 'warning text-dark'
                else:
                    ind['severity_class'] = 'success'
            if 'type_label' not in ind:
                ind['type_label'] = ind.get('type', '').replace('_', ' ').title()

        # Sort by severity
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        indicators.sort(key=lambda x: severity_order.get(x['severity'], 3))
        
        return indicators
    
    def _generate_recommendations(self, is_threat: bool, probability: float,
                                  threat_type: str, indicators: List[Dict[str, Any]]) -> List[str]:
        """
        Generate actionable recommendations based on analysis.
        
        Args:
            is_threat: Whether profile is a threat
            probability: Threat probability
            threat_type: Type of threat
            indicators: List of suspicious indicators
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not is_threat:
            recommendations.append("Profile appears legitimate based on current analysis")
            recommendations.append("Continue monitoring for any suspicious activity")
            return recommendations
        
        # Threat-specific recommendations
        if threat_type == 'fake_profile':
            recommendations.append("High confidence this is a fake/impersonation account")
            recommendations.append("Report to platform for identity verification")
            recommendations.append("Do not engage with or share content from this account")
        
        elif threat_type == 'spam':
            recommendations.append("Account shows strong spam characteristics")
            recommendations.append("Block and report for spam")
            recommendations.append("Avoid clicking any links from this account")
        
        elif threat_type == 'scam':
            recommendations.append("Account exhibits scam-like behavior (financial offers, crypto schemes)")
            recommendations.append("Report immediately for financial fraud")
            recommendations.append("Never share personal or financial information")
        
        elif threat_type == 'bot':
            recommendations.append("Automated bot behavior detected")
            recommendations.append("Report for platform manipulation")
            recommendations.append("Content from this account is likely artificially amplified")
        
        else:
            recommendations.append(f"Suspicious account detected (confidence: {probability:.0%})")
            recommendations.append("Exercise caution when interacting with this account")
        
        # Indicator-specific recommendations
        indicator_types = {i['type'] for i in indicators}
        
        if 'new_account' in indicator_types:
            recommendations.append("Account is very new - verify identity through other channels")
        
        if 'follower_imbalance' in indicator_types:
            recommendations.append("Unusual follower/following ratio suggests inorganic growth")
        
        if 'spam_patterns' in indicator_types:
            recommendations.append("Content matches known spam patterns - avoid engagement")
        
        if 'default_profile_image' in indicator_types or 'ai_generated_image' in indicator_types:
            recommendations.append("Profile image is not authentic - verify identity independently")
        
        if 'excessive_posting' in indicator_types:
            recommendations.append("Posting frequency exceeds human norms - likely automated")
        
        # General recommendations
        recommendations.append("Consider using platform reporting tools")
        recommendations.append("Monitor for similar accounts in your network")
        
        return recommendations
    
    def train_model(self, training_data_path: str = 'data/training_data.csv') -> str:
        """
        Train a new model on the provided data.
        
        Args:
            training_data_path: Path to training data CSV
            
        Returns:
            Path to saved model
        """
        logger.info(f"Training new model from {training_data_path}")
        
        # Load training data
        df = pd.read_csv(training_data_path)
        
        # Initialize trainer
        trainer = ModelTrainer(model_dir='models')
        
        # Prepare data
        X, y = trainer.prepare_data(df, target_col='is_threat')
        
        # Train and compare models
        results = trainer.train_and_compare(X, y)
        
        # Train best model
        trainer.train_best_model(X, y)
        
        # Save model
        model_path = trainer.save_model()
        
        # Reload in this detector
        bundle = self.model_trainer.load_model(os.path.basename(model_path))
        self.model = bundle['model']
        self.model_name = bundle['model_name']
        
        logger.info(f"Model training complete. Best model: {self.model_name}")
        
        return model_path


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize detector
    detector = UnifiedThreatDetector()
    
    # Test with sample profile
    sample_profile = {
        'username': 'test_user123',
        'platform': 'twitter',
        'creation_date': '2024-01-15',
        'followers_count': 50,
        'following_count': 5000,
        'posts_count': 1000,
        'bio': 'Earn money fast! Click here for free crypto!',
        'external_url': 'http://example.com',
        'profile_pic_url': 'https://example.com/default_avatar.png',
        'posts': [
            {'text': 'Make $1000 per day! Click here!', 'likes': 0, 'retweets': 0, 'replies': 0, 'timestamp': '2024-01-16T10:00:00'},
            {'text': 'Free bitcoin investment opportunity!', 'likes': 0, 'retweets': 0, 'replies': 0, 'timestamp': '2024-01-16T11:00:00'},
        ]
    }
    
    result = detector.analyze_profile(sample_profile)
    
    print("\n" + "="*60)
    print("ANALYSIS RESULT")
    print("="*60)
    print(f"Username: {result['profile_data']['username']}")
    print(f"Is Threat: {result['is_threat']}")
    print(f"Threat Type: {result['threat_type']}")
    print(f"Probability: {result['probability']:.2%}")
    print(f"Model Used: {result['model_used']}")
    print("\nIndicators:")
    for ind in result['indicators']:
        print(f"  - [{ind['severity'].upper()}] {ind['description']}")
    print("\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  - {rec}")