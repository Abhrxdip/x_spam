"""
Data Processor for Unified Threat Detector

Handles processing of profile URLs, usernames, and batch files.
"""

import os
import re
import random
import logging
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urlparse
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

def process_profile_url(profile_input: str, platform: str) -> Dict[str, Any]:
    """
    Process a profile URL or username and extract profile data.
    
    In a real implementation, this would:
    1. Determine if input is a URL or username
    2. Extract the username if it's a URL
    3. Call the appropriate API to get profile data
    
    For demonstration purposes, this function simulates API calls
    with realistic but synthetic data.
    
    Args:
        profile_input: URL or username of the profile
        platform: Social media platform (twitter, instagram, facebook)
        
    Returns:
        Dictionary with profile data
    """
    logger.info(f"Processing profile: {profile_input} on platform: {platform}")
    
    # Normalize platform name
    platform = platform.lower().strip()
    
    # Extract username from URL if needed
    username = extract_username_from_input(profile_input, platform)
    
    if not username:
        raise ValueError(f"Could not extract a valid username from '{profile_input}'")
    
    # Get profile data based on platform
    if platform == 'twitter' or platform == 'x':
        profile_data = get_twitter_profile_data(username)
    elif platform == 'instagram':
        profile_data = get_instagram_profile_data(username)
    elif platform == 'facebook':
        profile_data = get_facebook_profile_data(username)
    else:
        raise ValueError(f"Unsupported platform: {platform}")
    
    # Add original input and platform to profile data
    profile_data['original_input'] = profile_input
    profile_data['platform'] = platform
    profile_data['username'] = username
    
    # Add URL if not already present
    if 'url' not in profile_data:
        profile_data['url'] = generate_profile_url(username, platform)
    
    logger.info(f"Successfully processed profile for {username} on {platform}")
    
    return profile_data

def extract_username_from_input(profile_input: str, platform: str) -> Optional[str]:
    """
    Extract username from a profile URL or direct username input.
    Handles handles starting with '@', URLs, and raw handles.
    """
    if not profile_input:
        return None

    cleaned = profile_input.strip()
    if cleaned.startswith('@'):
        cleaned = cleaned[1:].strip()

    # Check if it's already a valid handle (alphanumeric, underscores, dots)
    if re.match(r'^[A-Za-z0-9_.]+$', cleaned):
        return cleaned

    # Try to extract from URL
    try:
        url_to_parse = cleaned if re.match(r'^https?://', cleaned, re.I) else 'https://' + cleaned
        parsed = urlparse(url_to_parse)
        path_segments = [s for s in parsed.path.split('/') if s]

        if path_segments:
            first_seg = path_segments[0].strip().lstrip('@')
            if first_seg.lower() in ['user', 'profile', 'accounts', 'in'] and len(path_segments) > 1:
                first_seg = path_segments[1].strip().lstrip('@')

            first_seg = re.sub(r'[\?#].*$', '', first_seg)
            if re.match(r'^[A-Za-z0-9_.]+$', first_seg):
                return first_seg
    except Exception as e:
        logger.warning(f"Error extracting username from URL: {str(e)}")

    # Fallback match for handle substring
    match = re.search(r'([A-Za-z0-9_]{3,30})', profile_input)
    if match:
        return match.group(1)

    return None

def generate_profile_url(username: str, platform: str) -> str:
    """Generate a profile URL from username and platform."""
    urls = {
        'twitter': f'https://twitter.com/{username}',
        'x': f'https://x.com/{username}',
        'instagram': f'https://instagram.com/{username}',
        'facebook': f'https://facebook.com/{username}'
    }
    return urls.get(platform.lower(), f'https://{platform}.com/{username}')

def _get_seeded_rng(username: str):
    """Create a deterministic random number generator for a username."""
    import hashlib
    seed_val = int(hashlib.md5(username.lower().strip().encode('utf-8')).hexdigest()[:8], 16)
    return random.Random(seed_val)

def fetch_official_x_api_v2_profile(username: str) -> Optional[Dict[str, Any]]:
    """
    Fetch user profile metadata & tweets using official X API v2 Bearer Token.
    Reads token safely from environment variable X_BEARER_TOKEN or .env file.
    """
    import urllib.request
    import json
    from dotenv import load_dotenv
    load_dotenv()
    
    bearer_token = os.getenv('X_BEARER_TOKEN')
    if not bearer_token:
        return None

    try:
        url = f"https://api.twitter.com/2/users/by/username/{username}?user.fields=created_at,description,public_metrics,verified,profile_image_url"
        req = urllib.request.Request(
            url, 
            headers={'Authorization': f'Bearer {bearer_token}'}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            user_info = data.get('data', {})
            if user_info:
                user_id = user_info.get('id')
                metrics = user_info.get('public_metrics', {})
                
                # Fetch recent tweets if user ID exists
                posts = []
                if user_id:
                    try:
                        tweets_url = f"https://api.twitter.com/2/users/{user_id}/tweets?tweet.fields=created_at,public_metrics&max_results=10"
                        t_req = urllib.request.Request(tweets_url, headers={'Authorization': f'Bearer {bearer_token}'})
                        with urllib.request.urlopen(t_req, timeout=5) as t_resp:
                            t_data = json.loads(t_resp.read().decode('utf-8'))
                            for t in t_data.get('data', []):
                                p_metrics = t.get('public_metrics', {})
                                posts.append({
                                    'text': t.get('text', ''),
                                    'likes': p_metrics.get('like_count', 0),
                                    'retweets': p_metrics.get('retweet_count', 0),
                                    'replies': p_metrics.get('reply_count', 0),
                                    'timestamp': t.get('created_at', '')
                                })
                    except Exception as te:
                        logger.info(f"X API v2 tweets fetch note: {str(te)}")

                logger.info(f"Successfully fetched live X API v2 profile for '{username}'")
                return {
                    'username': user_info.get('username', username),
                    'display_name': user_info.get('name', username),
                    'bio': user_info.get('description', f"Official X handle @{username}"),
                    'followers_count': metrics.get('followers_count', 0),
                    'following_count': metrics.get('following_count', 0),
                    'posts_count': metrics.get('tweet_count', 0),
                    'profile_pic_url': user_info.get('profile_image_url', ''),
                    'verified': user_info.get('verified', False),
                    'creation_date': user_info.get('created_at', ''),
                    'external_url': f"https://x.com/{username}",
                    'posts': posts
                }
    except Exception as e:
        logger.info(f"Official X API v2 fetch note for '{username}': {str(e)}")

    return None

def _get_x_guest_token() -> Optional[str]:
    """
    Obtain a short-lived guest token from Twitter/X.
    This is the same unauthenticated flow that x.com uses in the browser.
    No API key required.
    """
    import urllib.request, json
    # Public app-only bearer token used by x.com itself (read-only, unauthenticated)
    _BEARER = (
        "AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs"
        "%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"
    )
    try:
        req = urllib.request.Request(
            "https://api.twitter.com/1.1/guest/activate.json",
            data=b"",
            method="POST",
            headers={
                "Authorization": f"Bearer {_BEARER}",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                "Origin": "https://x.com",
                "Referer": "https://x.com/",
            },
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read()).get("guest_token")
    except Exception as e:
        logger.info(f"Guest token fetch error: {e}")
        return None


def fetch_live_twitter_profile(username: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a real live public X/Twitter profile using Twitter's internal
    GraphQL API (the same endpoint x.com uses in the browser).

    Strategy 1: Official X API v2 Bearer Token (if configured in .env).
    Strategy 2: Guest-token + internal GraphQL — no key needed.
    """
    import urllib.request, json, urllib.parse

    # ── Strategy 1: Official API v2 (needs paid plan or elevated access) ─────
    v2_data = fetch_official_x_api_v2_profile(username)
    if v2_data:
        return v2_data

    # ── Strategy 2: Guest-token + internal GraphQL ────────────────────────────
    _BEARER = (
        "AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs"
        "%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA"
    )
    _COMMON_HEADERS = {
        "Authorization": f"Bearer {_BEARER}",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Referer": "https://x.com/",
        "Origin": "https://x.com",
        "x-twitter-active-user": "yes",
        "x-twitter-client-language": "en",
    }

    try:
        guest_token = _get_x_guest_token()
        if not guest_token:
            raise RuntimeError("Could not obtain guest token")

        headers = {**_COMMON_HEADERS, "X-Guest-Token": guest_token}

        # ── 2a. Fetch user profile via UserByScreenName GraphQL ───────────────
        variables = json.dumps({
            "screen_name": username,
            "withSafetyModeUserFields": True,
        })
        features = json.dumps({
            "hidden_profile_subscriptions_enabled": True,
            "rweb_tipjar_consumption_enabled": True,
            "responsive_web_graphql_exclude_directive_enabled": True,
            "verified_phone_label_enabled": False,
            "subscriptions_verification_info_is_identity_verified_enabled": True,
            "subscriptions_verification_info_verified_since_enabled": True,
            "highlights_tweets_tab_ui_enabled": True,
            "responsive_web_twitter_article_notes_tab_enabled": True,
            "creator_subscriptions_tweet_preview_api_enabled": True,
            "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
            "responsive_web_graphql_timeline_navigation_enabled": True,
        })
        params = urllib.parse.urlencode({"variables": variables, "features": features})
        profile_url = (
            f"https://twitter.com/i/api/graphql/"
            f"NimuplG1OB7Fd2btCLdBOw/UserByScreenName?{params}"
        )

        req = urllib.request.Request(profile_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())

        user_result = (
            data.get("data", {}).get("user", {}).get("result", {})
        )
        legacy = user_result.get("legacy", {})

        if not legacy:
            logger.info(f"No legacy data returned for '{username}'")
            return None

        user_id = user_result.get("rest_id", "")
        followers  = legacy.get("followers_count", 0)
        following  = legacy.get("friends_count", 0)
        tweet_count = legacy.get("statuses_count", 0)
        display_name = legacy.get("name", username)
        bio = legacy.get("description", "")
        verified = legacy.get("verified", False) or bool(
            user_result.get("is_blue_verified")
        )
        created_at = legacy.get("created_at", "")
        profile_pic = legacy.get("profile_image_url_https", "").replace(
            "_normal", "_400x400"
        )

        # ── 2b. Fetch recent tweets via UserTweets GraphQL ────────────────────
        posts_data = []
        if user_id:
            try:
                t_vars = json.dumps({
                    "userId": user_id,
                    "count": 10,
                    "includePromotedContent": False,
                    "withQuickPromoteEligibilityTweetFields": False,
                    "withVoice": True,
                    "withV2Timeline": True,
                })
                t_features = json.dumps({
                    "rweb_lists_timeline_redesign_enabled": True,
                    "responsive_web_graphql_exclude_directive_enabled": True,
                    "verified_phone_label_enabled": False,
                    "creator_subscriptions_tweet_preview_api_enabled": True,
                    "responsive_web_graphql_timeline_navigation_enabled": True,
                    "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
                    "tweetypie_unmention_optimization_enabled": True,
                    "responsive_web_edit_tweet_api_enabled": True,
                    "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True,
                    "view_counts_everywhere_api_enabled": True,
                    "longform_notetweets_consumption_enabled": True,
                    "tweet_awards_web_tipping_enabled": False,
                    "freedom_of_speech_not_reach_fetch_enabled": True,
                    "standardized_nudges_misinfo": True,
                    "longform_notetweets_rich_text_read_enabled": True,
                    "responsive_web_enhance_cards_enabled": False,
                })
                t_params = urllib.parse.urlencode(
                    {"variables": t_vars, "features": t_features}
                )
                tweets_url = (
                    f"https://twitter.com/i/api/graphql/"
                    f"V7H0Ap3_Hh2FyS75OCDO3Q/UserTweets?{t_params}"
                )
                t_req = urllib.request.Request(tweets_url, headers=headers)
                with urllib.request.urlopen(t_req, timeout=10) as t_r:
                    t_data = json.loads(t_r.read())

                # Walk the timeline instructions to extract tweet entries
                instructions = (
                    t_data.get("data", {})
                    .get("user", {})
                    .get("result", {})
                    .get("timeline_v2", {})
                    .get("timeline", {})
                    .get("instructions", [])
                )
                for instr in instructions:
                    for entry in instr.get("entries", []):
                        content = entry.get("content", {})
                        item_content = content.get("itemContent", {})
                        tweet_result = (
                            item_content.get("tweet_results", {}).get("result", {})
                        )
                        t_legacy = tweet_result.get("legacy", {})
                        if t_legacy and t_legacy.get("full_text"):
                            metrics = t_legacy.get("public_metrics", {})
                            posts_data.append({
                                "text": t_legacy.get("full_text", ""),
                                "likes": t_legacy.get("favorite_count", 0),
                                "retweets": t_legacy.get("retweet_count", 0),
                                "replies": t_legacy.get("reply_count", 0),
                                "timestamp": t_legacy.get("created_at", ""),
                            })
            except Exception as te:
                logger.info(f"Tweet fetch note for '{username}': {te}")

        logger.info(
            f"Successfully fetched live X/Twitter profile for '{username}' "
            f"({followers:,} followers, {tweet_count:,} tweets)"
        )
        return {
            "username": legacy.get("screen_name", username),
            "display_name": display_name,
            "bio": bio,
            "followers_count": followers,
            "following_count": following,
            "posts_count": tweet_count,
            "profile_pic_url": profile_pic,
            "verified": verified,
            "creation_date": created_at,
            "external_url": f"https://x.com/{username}",
            "posts": posts_data,
            "sentiment_label": "neutral",
            "country": "US",
            "account_type": "individual",
            "gender": "unknown",
            "thread_entry_type": "original",
        }

    except Exception as e:
        logger.info(f"GraphQL fetch failed for '{username}': {e}")

    return None

def get_twitter_profile_data(username: str) -> Dict[str, Any]:
    """
    Get Twitter/X profile data.
    First attempts live public X profile data ingestion.
    Fallback to deterministic seeding for demo presets or offline mode.
    """
    lower_un = username.lower().strip()

    # Predefined Handles for Instant Testing (Exact Matches Only)
    if lower_un in ['mention_spammer_bot', 'tag_spammer_bot', 'crypto_scam_bot', 'scam_bot', 'test_scam_user', 'botnet_account_99', 'legit_tech_dev', 'legit_user']:
        pass # proceed to demo preset logic below
    else:
        # Attempt live X profile extraction first for real user handles
        live_data = fetch_live_twitter_profile(username)
        if live_data:
            return live_data

    rng = _get_seeded_rng(username)

    # Predefined Handles for Instant Testing (Exact Matches Only)
    if lower_un in ['mention_spammer_bot', 'tag_spammer_bot']:
        account_age = rng.randint(5, 30)
        followers = rng.randint(10, 100)
        following = rng.randint(4000, 8500)
        posts = rng.randint(300, 1500)
        verified = False
        is_default = True
        account_type = 'bot'
        bio = "Tagging winner lists! DM for free crypto rewards and instant payouts!"
        post_texts = [
            "Hey @alice @bob @charlie @david @eve You won $500! Claim here http://bit.ly/claim-win",
            "@user10 @user22 @user44 Congratulations! Check your prize at http://t.me/claim_bot",
            "Urgent notification for @user99 @user88 @user77 Claim $1000 USDT right now http://bit.ly/crypto-prize"
        ]
        sentiment_label = 'positive'
    elif lower_un in ['crypto_scam_bot', 'scam_bot', 'test_scam_user', 'botnet_account_99']:
        account_age = rng.randint(2, 20)
        followers = rng.randint(5, 80)
        following = rng.randint(3000, 9500)
        posts = rng.randint(100, 2500)
        verified = False
        is_default = True
        account_type = 'bot'
        bio = "Earn $5000/day guaranteed! Free Bitcoin & Ethereum giveaway. Click link below to claim!"
        post_texts = [
            "Make $1000 per day! Click here http://bit.ly/free-crypto-claim",
            "Free bitcoin investment opportunity! DM me now!",
            "Instant payout guaranteed. DM for details!",
            "Act fast! Limited time bonus giveaway!"
        ]
        sentiment_label = 'positive'
    elif lower_un in ['legit_tech_dev', 'legit_user', 'official_dev']:
        account_age = rng.randint(730, 2500)
        followers = rng.randint(5000, 80000)
        following = rng.randint(200, 1500)
        posts = rng.randint(1500, 12000)
        verified = True
        is_default = False
        account_type = 'individual'
        bio = "Senior Software Engineer | Open source contributor | Building AI security tools"
        post_texts = [
            "Just released a new open source library for ML model evaluation!",
            "Great discussion on system architecture at today's conference.",
            "Exploring recent advancements in deep learning models."
        ]
        sentiment_label = 'positive'
    else:
        # Generic Handle: Deterministic, realistic metrics derived from username hash
        account_age = rng.randint(180, 2800)
        followers = rng.randint(250, 25000)
        following = rng.randint(100, 1800)
        posts = rng.randint(100, 6000)
        verified = rng.random() > 0.85
        is_default = False
        account_type = 'individual'
        bio = f"Official account of {username.replace('_', ' ').title()} | Sharing updates and insights."
        post_texts = [
            f"Excited to share our latest project updates today! #{username}",
            "Great catching up with colleagues and discussing technology trends.",
            "Thanks everyone for the support and great feedback on recent posts!"
        ]
        sentiment_label = 'positive'

    creation_date = (datetime.now() - timedelta(days=account_age)).strftime("%Y-%m-%d")

    num_posts = min(len(post_texts), posts)
    posts_data = []
    for i in range(num_posts):
        posts_data.append({
            'text': post_texts[i % len(post_texts)],
            'likes': rng.randint(0, 500),
            'retweets': rng.randint(0, 50),
            'replies': rng.randint(0, 20),
            'timestamp': (datetime.now() - timedelta(days=rng.randint(0, 30),
                                                        hours=rng.randint(0, 23),
                                                        minutes=rng.randint(0, 59))).strftime("%Y-%m-%dT%H:%M:%S")
        })

    profile_pic = 'https://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png' if is_default else f'https://pbs.twimg.com/profile_images/{rng.randint(1000000, 9999999)}/photo.jpg'

    return {
        'username': username,
        'display_name': username.replace('_', ' ').title(),
        'bio': bio,
        'external_url': f'https://linktr.ee/{username}' if rng.random() > 0.4 else None,
        'profile_pic_url': profile_pic,
        'creation_date': creation_date,
        'followers_count': followers,
        'following_count': following,
        'posts_count': posts,
        'verified': verified,
        'location': rng.choice(['New York', 'London', 'Tokyo', 'San Francisco', 'India']),
        'posts': posts_data,
        'sentiment_label': sentiment_label,
        'country': rng.choice(['US', 'UK', 'India', 'China', 'Russia', 'Other']),
        'account_type': account_type,
        'gender': rng.choice(['male', 'female', 'unknown']),
        'thread_entry_type': rng.choice(['original', 'reply', 'retweet'])
    }

def get_instagram_profile_data(username: str) -> Dict[str, Any]:
    """Get Instagram profile data (deterministic synthetic for demo)."""
    rng = _get_seeded_rng(username)
    account_age = rng.randint(30, 2500)
    creation_date = (datetime.now() - timedelta(days=account_age)).strftime("%Y-%m-%d")
    
    return {
        'username': username,
        'display_name': username.replace('_', ' ').title(),
        'bio': "Official Instagram account",
        'external_url': f'https://linktr.ee/{username}',
        'profile_pic_url': f'https://instagram.cdn.com/{username}.jpg',
        'creation_date': creation_date,
        'followers_count': rng.randint(100, 50000),
        'following_count': rng.randint(50, 2000),
        'posts_count': rng.randint(10, 500),
        'verified': rng.random() > 0.9,
        'posts': [],
        'sentiment_label': 'positive',
        'country': 'US',
        'account_type': 'individual',
        'gender': 'unknown',
        'thread_entry_type': 'original'
    }

def get_facebook_profile_data(username: str) -> Dict[str, Any]:
    """Get Facebook profile data (deterministic synthetic for demo)."""
    rng = _get_seeded_rng(username)
    account_age = rng.randint(30, 2500)
    creation_date = (datetime.now() - timedelta(days=account_age)).strftime("%Y-%m-%d")
    
    return {
        'username': username,
        'display_name': username.replace('_', ' ').title(),
        'bio': "Official Facebook profile",
        'external_url': None,
        'profile_pic_url': f'https://facebook.cdn.com/{username}.jpg',
        'creation_date': creation_date,
        'followers_count': rng.randint(100, 20000),
        'following_count': rng.randint(50, 1000),
        'posts_count': rng.randint(10, 300),
        'verified': False,
        'posts': [],
        'sentiment_label': 'neutral',
        'country': 'US',
        'account_type': 'individual',
        'gender': 'unknown',
        'thread_entry_type': 'original'
    }

def process_batch_file(filepath: str, platform: str) -> List[Dict[str, Any]]:
    """
    Process a batch file (CSV or JSON) containing multiple profiles.
    
    Args:
        filepath: Path to the batch file
        platform: Default platform for profiles
        
    Returns:
        List of profile data dictionaries
    """
    logger.info(f"Processing batch file: {filepath}")
    
    profiles = []
    
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
        
        # Expected columns: username, platform (optional)
        for _, row in df.iterrows():
            username = row.get('username') or row.get('profile_url') or row.get('url')
            if username:
                profile_platform = row.get('platform', platform)
                try:
                    profile_data = process_profile_url(str(username), profile_platform)
                    profiles.append(profile_data)
                except Exception as e:
                    logger.warning(f"Failed to process {username}: {str(e)}")
    
    elif filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    username = item.get('username') or item.get('profile_url')
                    profile_platform = item.get('platform', platform)
                else:
                    username = item
                    profile_platform = platform
                
                if username:
                    try:
                        profile_data = process_profile_url(str(username), profile_platform)
                        profiles.append(profile_data)
                    except Exception as e:
                        logger.warning(f"Failed to process {username}: {str(e)}")
    
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
    
    logger.info(f"Processed {len(profiles)} profiles from batch file")
    return profiles