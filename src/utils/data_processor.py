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

def fetch_live_twitter_profile(username: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to fetch real live public X/Twitter profile metrics and tweets using public syndication.
    """
    import urllib.request
    import json
    
    try:
        url = f"https://syndication.twitter.com/srv/timeline-profile/screen-name/{username}"
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            html = resp.read().decode('utf-8')

        if '__NEXT_DATA__' in html:
            json_str = html.split('<script id="__NEXT_DATA__" type="application/json">')[1].split('</script>')[0]
            data = json.loads(json_str)
            
            timeline_entries = data.get('props', {}).get('pageProps', {}).get('timeline', {}).get('entries', [])
            user_data = None
            posts_data = []

            for entry in timeline_entries:
                tweet = entry.get('content', {}).get('tweet', {})
                if tweet:
                    if not user_data:
                        user_info = tweet.get('user', {})
                        user_data = {
                            'username': user_info.get('screen_name', username),
                            'display_name': user_info.get('name', username),
                            'followers_count': user_info.get('followers_count', 0),
                            'following_count': user_info.get('friends_count', 0),
                            'posts_count': user_info.get('statuses_count', 0),
                            'profile_pic_url': user_info.get('profile_image_url_https', ''),
                            'verified': user_info.get('verified', False),
                            'creation_date': user_info.get('created_at', '')
                        }
                    posts_data.append({
                        'text': tweet.get('text', ''),
                        'likes': tweet.get('favorite_count', 0),
                        'retweets': tweet.get('retweet_count', 0),
                        'replies': 0,
                        'timestamp': tweet.get('created_at', '')
                    })

            if user_data:
                user_data['posts'] = posts_data
                user_data['bio'] = f"Official Twitter handle @{username}"
                user_data['external_url'] = f"https://x.com/{username}"
                user_data['sentiment_label'] = 'neutral'
                user_data['country'] = 'US'
                user_data['account_type'] = 'individual'
                user_data['gender'] = 'unknown'
                user_data['thread_entry_type'] = 'original'
                logger.info(f"Successfully fetched live X/Twitter profile for '{username}'")
                return user_data
    except Exception as e:
        logger.info(f"Live Twitter fetch bypass for '{username}': {str(e)}")
    
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