"""
Unified Feature Extractor for Social Media Threat Detection

Combines feature engineering from both projects:
- fake-profile-detector: Account metrics, activity, content, network, image features
- spam_identifier: Sentiment, country, account type, word frequency, link features
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np
import re

# Configure logging
logger = logging.getLogger(__name__)

class UnifiedFeatureExtractor:
    """
    Extracts comprehensive features from social media profile data
    for use in machine learning models.
    """
    
    def __init__(self):
        """Initialize the UnifiedFeatureExtractor."""
        logger.info("Initializing UnifiedFeatureExtractor")
        self.suspicious_keywords = self._load_suspicious_keywords()
        self.spam_patterns = self._load_spam_patterns()
        logger.info("UnifiedFeatureExtractor initialized successfully")
    
    def extract_features(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all features from profile data.

        Args:
            profile_data: Dictionary containing profile information

        Returns:
            Dictionary with extracted features
        """
        logger.info(f"Extracting features for profile: {profile_data.get('username', 'Unknown')}")

        features = {}

        # Extract features from both projects - order matters for consistent feature names
        features.update(self._extract_account_metrics(profile_data))
        features.update(self._extract_content_features(profile_data))
        features.update(self._extract_activity_features(profile_data))
        features.update(self._extract_network_features(profile_data))
        features.update(self._extract_image_features(profile_data))
        features.update(self._extract_spam_identifier_features(profile_data))

        logger.info(f"Feature extraction complete for {profile_data.get('username', 'Unknown')}")
        return features
    
    def _load_suspicious_keywords(self) -> List[str]:
        """Load comprehensive suspicious keywords for content analysis."""
        return [
            'free', 'money', 'earn', 'cash', 'prize', 'winner', 'click', 'subscribe',
            'follow', 'dm', 'investment', 'crypto', 'bitcoin', 'ethereum', 'solana', 'usdt',
            'airdrop', 'giveaway', 'doubler', 'presale', 'whitelist', 'claim', 'wallet',
            'seed', 'phrase', 'metamask', 'trustwallet', 'telegram', 'whatsapp', 'dating',
            'hot', 'singles', 'weight', 'diet', 'miracle', 'cure', 'rich', 'profit', 'bonus',
            'offer', 'limited', 'urgent', 'act', 'now', 'verify', 'account', 'suspended',
            'locked', 'guaranteed', 'passive', 'income', 'payout', 'doubling', 'trade'
        ]
    
    def _load_spam_patterns(self) -> List[str]:
        """Load regex patterns for modern social media spam and scam detection."""
        return [
            r'(earn|make|win)(\s+)?\$\d+(\s+)?(per|a)?(\s+)?(day|week|month|hour|daily)?',
            r'free\s+(money|crypto|bitcoin|ethereum|nft|airdrop|tokens)',
            r'work\s+from\s+home',
            r'(click|tap|claim)\s+(here|now|link|below)',
            r'(check|see)(\s+)?(my|this)(\s+)?(profile|bio|link)',
            r'follow(\s+)?(me|back)',
            r'(dm|message|inbox)(\s+)?(me|us)',
            r'hot\s+(singles|girls|guys)',
            r'(bitcoin|crypto|forex|option)(\s+)?(investment|trading|doubler|giveaway)',
            r'get\s+rich\s+(quick|fast)',
            r'guaranteed\s+(profit|returns|payout)',
            r'0x[a-fA-F0-9]{40}',  # Ethereum/EVM Wallet Address
            r'https?://(bit\.ly|tinyurl\.com|t\.co|t\.me|wa\.me|cutt\.ly|rebrand\.ly)/\S*',
            r'[A-Za-z0-9_.+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9-.]+',
            r'https?://[A-Za-z0-9.-]+\.[A-Za-z]{2,}(/\S*)?'
        ]
    
    def _extract_account_metrics(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic account metrics."""
        features = {}
        
        # Account age
        creation_date_str = profile_data.get('creation_date')
        account_age_days = 365  # Default
        
        if creation_date_str:
            try:
                # Also handle Twitter's raw date: "Tue Jun 02 20:12:29 +0000 2009"
                for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d",
                            "%a %b %d %H:%M:%S +0000 %Y"]:
                    try:
                        creation_date = datetime.strptime(creation_date_str, fmt)
                        account_age_days = (datetime.now() - creation_date).days
                        break
                    except ValueError:
                        continue
            except Exception as e:
                logger.warning(f"Error parsing creation date: {str(e)}")
        
        features['account_age_days'] = max(0, account_age_days)
        
        # Follower/following counts
        followers_count = profile_data.get('followers_count', 0)
        following_count = profile_data.get('following_count', 0)
        posts_count = profile_data.get('posts_count', profile_data.get('post_count', 0))
        
        features['followers_count'] = followers_count
        features['following_count'] = following_count
        features['posts_count'] = posts_count
        
        # Derived ratios
        features['followers_to_following_ratio'] = (
            followers_count / following_count if following_count > 0 else 0
        )
        
        features['posts_per_day'] = (
            posts_count / account_age_days if account_age_days > 0 else 0
        )
        
        return features
    
    def _extract_content_features(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content-related features."""
        features = {}
        
        # Bio features
        bio = profile_data.get('bio', '')
        features['bio_length'] = len(bio) if bio else 0
        features['has_external_url'] = 1 if profile_data.get('external_url') else 0
        
        # Posts analysis
        posts = profile_data.get('posts', [])
        if posts:
            post_texts = []
            for post in posts:
                if isinstance(post, dict):
                    post_texts.append(post.get('text', ''))
                elif isinstance(post, str):
                    post_texts.append(post)
            
            # Sentiment analysis (simple)
            features['sentiment_score'] = self._calculate_sentiment(post_texts)
            
            # Content diversity
            features['content_diversity'] = self._calculate_diversity(post_texts)
            
            # Suspicious content score
            features['suspicious_content_score'] = self._calculate_suspicious_score(post_texts)
            
            # Spam pattern matches
            features['spam_pattern_matches'] = self._count_spam_patterns(post_texts)

            # Mention Spam Analysis (@username tagging)
            total_mentions = sum(len(re.findall(r'@[A-Za-z0-9_]+', t)) for t in post_texts)
            posts_with_mentions = sum(1 for t in post_texts if re.search(r'@[A-Za-z0-9_]+', t))
            features['mention_count'] = total_mentions
            features['mention_ratio'] = posts_with_mentions / len(post_texts) if post_texts else 0
            features['avg_mentions_per_post'] = total_mentions / len(post_texts) if post_texts else 0

            # Hashtag Stuffing Analysis (#hashtag)
            posts_with_many_hashtags = sum(1 for t in post_texts if len(re.findall(r'#[A-Za-z0-9_]+', t)) >= 4)
            features['hashtag_stuffing_ratio'] = posts_with_many_hashtags / len(post_texts) if post_texts else 0

            # External Links in Postings
            posts_with_links = sum(1 for t in post_texts if re.search(r'https?://\S+', t))
            features['link_post_ratio'] = posts_with_links / len(post_texts) if post_texts else 0

            # Duplicate / Repetitive Post Ratio
            unique_posts = len(set(post_texts))
            features['duplicate_post_ratio'] = 1.0 - (unique_posts / len(post_texts)) if post_texts else 0

            # ── Fine-tuned DistilBERT NLP Classifier ──────────────────────────
            # Replaces old zero-shot DeBERTa with domain-specific fine-tuned model.
            # Falls back to keyword heuristics if model not yet trained.
            # Run: python scripts/finetune_nlp.py  (one-time, ~20 min on CPU)
            try:
                from src.features.deberta_analyzer import get_deberta_analyzer
                nlp_analyzer = get_deberta_analyzer()
                nlp_metrics = nlp_analyzer.analyze_post_texts(post_texts)
                features['deberta_phishing_score'] = nlp_metrics['deberta_phishing_score']
                features['deberta_spam_confidence'] = nlp_metrics['deberta_spam_confidence']
                features['nlp_phishing_score']      = nlp_metrics.get('nlp_phishing_score', 0.0)
                features['nlp_spam_confidence']     = nlp_metrics.get('nlp_spam_confidence', 0.0)
                features['nlp_threat_class']        = nlp_metrics.get('nlp_threat_class', 0)
                features['nlp_high_risk_count']     = nlp_metrics.get('nlp_high_risk_count', 0)
            except Exception as e:
                logger.debug(f"NLP classifier fallback: {str(e)}")
                features['deberta_phishing_score'] = 0.0
                features['deberta_spam_confidence'] = 0.0
                features['nlp_phishing_score']     = 0.0
                features['nlp_spam_confidence']    = 0.0
                features['nlp_threat_class']       = 0
                features['nlp_high_risk_count']    = 0
        else:
            features['sentiment_score'] = 0.5
            features['content_diversity'] = 1.0
            features['suspicious_content_score'] = 0.0
            features['spam_pattern_matches'] = 0
            features['mention_count'] = 0
            features['mention_ratio'] = 0.0
            features['avg_mentions_per_post'] = 0.0
            features['hashtag_stuffing_ratio'] = 0.0
            features['link_post_ratio'] = 0.0
            features['duplicate_post_ratio'] = 0.0
            features['deberta_phishing_score'] = 0.0
            features['deberta_spam_confidence'] = 0.0
            features['nlp_phishing_score']     = 0.0
            features['nlp_spam_confidence']    = 0.0
            features['nlp_threat_class']       = 0
            features['nlp_high_risk_count']    = 0
        
        return features
    
    def _extract_activity_features(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract activity-related features."""
        features = {}
        
        posts = profile_data.get('posts', [])
        if posts:
            # Calculate posting frequency
            features['posts_per_day'] = self._extract_account_metrics(profile_data).get('posts_per_day', 0)
            
            # Engagement rate
            total_engagement = sum(
                post.get('likes', 0) + post.get('retweets', 0) + post.get('replies', 0)
                for post in posts if isinstance(post, dict)
            )
            followers = profile_data.get('followers_count', 1)
            features['engagement_rate'] = min(1.0, total_engagement / (len(posts) * max(followers, 1)))
            
            # Posting regularity
            features['posting_regularity'] = self._calculate_regularity(posts)
            
            # Activity score
            features['activity_score'] = np.clip(
                0.5 + (features['posts_per_day'] - 2) * 0.1, 0, 1
            )
            
            # Time zone consistency
            features['time_zone_consistency'] = self._calculate_timezone_consistency(posts)
        else:
            features['engagement_rate'] = 0
            features['posting_regularity'] = 0.5
            features['activity_score'] = 0.5
            features['time_zone_consistency'] = 0.5
        
        return features
    
    def _extract_network_features(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract network-related features."""
        features = {}
        
        followers_count = profile_data.get('followers_count', 0)
        following_count = profile_data.get('following_count', 0)
        
        # Network isolation score
        if followers_count == 0 and following_count == 0:
            features['network_isolation_score'] = 1.0
        elif followers_count == 0:
            features['network_isolation_score'] = 0.8
        elif following_count == 0:
            features['network_isolation_score'] = 0.3
        else:
            ratio = followers_count / following_count
            # Very high ratio (celebrity) or very low ratio (bot) both suspicious
            if ratio > 100:
                features['network_isolation_score'] = 0.4
            elif ratio < 0.1:
                features['network_isolation_score'] = 0.9
            else:
                features['network_isolation_score'] = 0.5
        
        # Mutual connection ratio
        followers = profile_data.get('followers', [])
        following = profile_data.get('following', [])
        if followers and following:
            mutual = len(set(followers) & set(following))
            features['mutual_connection_ratio'] = mutual / len(following)
        else:
            features['mutual_connection_ratio'] = 0.5
        
        # Clustering coefficient (simplified)
        features['clustering_coefficient'] = 0.5
        
        # Reciprocity
        if followers and following:
            followers_set = set(followers)
            following_set = set(following)
            reciprocal = len(followers_set & following_set)
            features['reciprocity'] = reciprocal / len(following_set) if following_set else 0.5
        else:
            features['reciprocity'] = 0.5
        
        # Overall network score
        features['network_score'] = (
            0.4 * features['network_isolation_score'] +
            0.3 * (1 - features['mutual_connection_ratio']) +
            0.3 * (1 - features['reciprocity'])
        )
        
        return features
    
    def _extract_image_features(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract image-related features."""
        features = {}
        
        profile_pic_url = profile_data.get('profile_pic_url')
        
        if not profile_pic_url:
            features['profile_pic_score'] = 0.5
            features['is_default_image'] = 1
            features['is_stock_photo'] = 0
            features['is_ai_generated'] = 0
        else:
            # Check for default image patterns
            default_patterns = ['default', 'placeholder', 'blank', 'avatar', 'profile_photo']
            is_default = any(p in profile_pic_url.lower() for p in default_patterns)
            
            features['is_default_image'] = 1 if is_default else 0
            features['is_stock_photo'] = 0  # Would need image analysis
            features['is_ai_generated'] = 0  # Would need AI detection
            
            # Score: default images are more suspicious
            features['profile_pic_score'] = 0.3 if is_default else 0.7
        
        return features
    
    def _extract_spam_identifier_features(self, profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features from spam_identifier project.
        These include sentiment, country, account type, word frequencies, links.
        """
        features = {}
        
        # Categorical features (from spam_identifier)
        features['Sentiment'] = profile_data.get('sentiment_label', 'neutral')
        features['Country'] = profile_data.get('country', 'Unknown')
        features['Account.Type'] = profile_data.get('account_type', 'individual')
        features['Gender'] = profile_data.get('gender', 'unknown')
        features['Thread.Entry.Type'] = profile_data.get('thread_entry_type', 'original')
        features['Twitter.Verified'] = 'yes' if profile_data.get('verified', False) else 'no'
        
        # Word frequency features (from spam_identifier)
        text_content = ' '.join([
            profile_data.get('bio', ''),
            ' '.join([p.get('text', '') if isinstance(p, dict) else str(p) 
                     for p in profile_data.get('posts', [])])
        ]).lower()
        
        features['word_sex'] = text_content.count('sex') + text_content.count('sexy')
        features['word_good'] = text_content.count('good') + text_content.count('great')
        features['word_woman'] = text_content.count('woman') + text_content.count('girl')
        features['word_new'] = text_content.count('new')
        features['word_like'] = text_content.count('like')
        features['name_2_w'] = 1 if ' ' in profile_data.get('username', '') else 0
        
        # Link features (from spam_identifier)
        features['links_twitter'] = 1 if 'twitter.com' in text_content or 't.co' in text_content else 0
        features['links_youtube'] = 1 if 'youtube.com' in text_content or 'youtu.be' in text_content else 0
        features['links_facebook'] = 1 if 'facebook.com' in text_content or 'fb.me' in text_content else 0
        features['links_instagram'] = 1 if 'instagram.com' in text_content or 'instagr.am' in text_content else 0
        features['links_other'] = 1 if re.search(r'https?://(?!.*(twitter|youtube|facebook|instagram))', text_content) else 0
        
        return features
    
    def _calculate_sentiment(self, texts: List[str]) -> float:
        """Calculate simple sentiment score (-1 to 1)."""
        if not texts:
            return 0.5
        
        positive_words = ['good', 'great', 'love', 'amazing', 'excellent', 'happy', 'best', 'awesome']
        negative_words = ['bad', 'terrible', 'hate', 'awful', 'worst', 'sad', 'angry', 'scam']
        
        total_score = 0
        total_words = 0
        
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            for word in words:
                if word in positive_words:
                    total_score += 1
                elif word in negative_words:
                    total_score -= 1
                total_words += 1
        
        if total_words == 0:
            return 0.5
        
        return np.clip(0.5 + (total_score / total_words), 0, 1)
    
    def _calculate_diversity(self, texts: List[str]) -> float:
        """Calculate content diversity (0 to 1)."""
        if not texts:
            return 1.0
        
        all_words = []
        for text in texts:
            all_words.extend(re.findall(r'\b\w+\b', text.lower()))
        
        if not all_words:
            return 1.0
        
        unique_words = set(all_words)
        return len(unique_words) / len(all_words)
    
    def _calculate_suspicious_score(self, texts: List[str]) -> float:
        """Calculate suspicious content score (0 to 1)."""
        if not texts:
            return 0.0
        
        total_words = 0
        suspicious_count = 0
        
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            total_words += len(words)
            for word in words:
                if word in self.suspicious_keywords:
                    suspicious_count += 1
        
        if total_words == 0:
            return 0.0
        
        return min(1.0, suspicious_count / total_words * 10)
    
    def _count_spam_patterns(self, texts: List[str]) -> int:
        """Count spam pattern matches in texts."""
        if not texts:
            return 0
        
        count = 0
        combined_text = ' '.join(texts)
        
        for pattern in self.spam_patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            count += len(matches)
        
        return count
    
    def _calculate_regularity(self, posts: List[Any]) -> float:
        """Calculate posting regularity (0 to 1)."""
        if len(posts) < 2:
            return 0.5
        
        timestamps = []
        # Twitter live API returns timestamps like "Thu Aug 14 12:00:00 +0000 2025"
        # Synthetic/fallback data uses ISO format "2025-08-14T12:00:00"
        _TS_FORMATS = [
            "%Y-%m-%dT%H:%M:%S",
            "%a %b %d %H:%M:%S +0000 %Y",
            "%Y-%m-%d %H:%M:%S",
        ]
        for post in posts:
            if isinstance(post, dict) and 'timestamp' in post:
                for fmt in _TS_FORMATS:
                    try:
                        ts = datetime.strptime(post['timestamp'], fmt)
                        timestamps.append(ts)
                        break
                    except (ValueError, TypeError):
                        continue
        
        if len(timestamps) < 2:
            return 0.5
        
        timestamps.sort()
        intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                    for i in range(len(timestamps)-1)]
        
        if not intervals:
            return 0.5
        
        # Calculate coefficient of variation
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        if mean_interval == 0:
            return 0.5
        
        cv = std_interval / mean_interval
        
        # Lower CV means more regular (more suspicious for bots)
        regularity = np.clip(1 - cv, 0, 1)
        return regularity
    
    def _calculate_timezone_consistency(self, posts: List[Any]) -> float:
        """Calculate time zone consistency (0 to 1)."""
        if len(posts) < 3:
            return 0.5
        
        hours = []
        _TS_FORMATS = [
            "%Y-%m-%dT%H:%M:%S",
            "%a %b %d %H:%M:%S +0000 %Y",
            "%Y-%m-%d %H:%M:%S",
        ]
        for post in posts:
            if isinstance(post, dict) and 'timestamp' in post:
                for fmt in _TS_FORMATS:
                    try:
                        ts = datetime.strptime(post['timestamp'], fmt)
                        hours.append(ts.hour)
                        break
                    except (ValueError, TypeError):
                        continue
        
        if len(hours) < 3:
            return 0.5
        
        # Calculate variance in posting hours
        hour_std = np.std(hours)
        
        # Lower variance means more consistent (could be bot)
        consistency = np.clip(1 - (hour_std / 12), 0, 1)
        return consistency
