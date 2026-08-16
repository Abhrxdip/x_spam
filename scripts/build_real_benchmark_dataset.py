"""
Build Real Academic Benchmark Dataset and Train 13 ML Models.

Ingests real profiles from:
- Botwiki-2019 (Verified Real Bots)
- Cresci-RTbust-2019 (Real Retweet Botnets & Humans)
- Verified-2019 (Real Verified Humans)
- TwiBot-20 (Academic Benchmark Profiles)

Extracts authentic 44-dimensional feature vectors and trains all 13 ML models.
"""

import os
import sys
import tarfile
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.features.feature_extractor import UnifiedFeatureExtractor
from src.models.train_model import ModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_real_profiles() -> List[Dict[str, Any]]:
    """Load and label all real profiles from data archives."""
    all_profiles = []
    
    # 1. Load Botwiki-2019 (Bots)
    botwiki_path = os.path.join(PROJECT_ROOT, 'data', 'botwiki-2019.tar.gz')
    if os.path.exists(botwiki_path):
        logger.info("Loading Botwiki-2019 dataset...")
        with tarfile.open(botwiki_path, 'r:gz') as tar:
            f = tar.extractfile('botwiki-2019_tweets.json')
            data = json.load(f)
            for item in data:
                u = item.get('user', {})
                if u:
                    all_profiles.append({
                        'user': u,
                        'tweets': [t.get('text', '') for t in item.get('tweets', [])] if 'tweets' in item else [],
                        'is_threat': 1,
                        'source': 'botwiki-2019'
                    })
        logger.info(f"Loaded {len(data)} profiles from Botwiki-2019 (Label: Bot)")

    # 2. Load Cresci-RTbust-2019 (Mixed: Bots & Humans)
    cresci_path = os.path.join(PROJECT_ROOT, 'data', 'cresci-rtbust-2019.tar.gz')
    if os.path.exists(cresci_path):
        logger.info("Loading Cresci-RTbust-2019 dataset...")
        cresci_labels = {}
        with tarfile.open(cresci_path, 'r:gz') as tar:
            tsv_f = tar.extractfile('cresci-rtbust-2019.tsv')
            tsv_df = pd.read_csv(tsv_f, sep='\t', header=None, names=['id', 'label'])
            for _, row in tsv_df.iterrows():
                cresci_labels[str(row['id'])] = 1 if 'bot' in str(row['label']).lower() else 0
            
            json_f = tar.extractfile('cresci-rtbust-2019_tweets.json')
            data = json.load(json_f)
            cresci_count = 0
            for item in data:
                u = item.get('user', {})
                if u:
                    uid = str(u.get('id', ''))
                    label = cresci_labels.get(uid, 1 if 'bot' in str(u.get('description', '')).lower() else 0)
                    all_profiles.append({
                        'user': u,
                        'tweets': [t.get('text', '') for t in item.get('tweets', [])] if 'tweets' in item else [],
                        'is_threat': label,
                        'source': 'cresci-rtbust-2019'
                    })
                    cresci_count += 1
        logger.info(f"Loaded {cresci_count} profiles from Cresci-RTbust-2019")

    # 3. Load Verified-2019 (Verified Humans)
    verified_path = os.path.join(PROJECT_ROOT, 'data', 'verified-2019.tar.gz')
    if os.path.exists(verified_path):
        logger.info("Loading Verified-2019 dataset...")
        with tarfile.open(verified_path, 'r:gz') as tar:
            json_f = tar.extractfile('verified-2019_tweets.json')
            data = json.load(json_f)
            for item in data:
                u = item.get('user', {})
                if u:
                    all_profiles.append({
                        'user': u,
                        'tweets': [t.get('text', '') for t in item.get('tweets', [])] if 'tweets' in item else [],
                        'is_threat': 0,
                        'source': 'verified-2019'
                    })
        logger.info(f"Loaded {len(data)} profiles from Verified-2019 (Label: Human)")

    # 4. Load TwiBot-20
    twibot_path = os.path.join(PROJECT_ROOT, 'data', 'twibot-20.json')
    if os.path.exists(twibot_path):
        logger.info("Loading TwiBot-20 dataset...")
        with open(twibot_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                p = item.get('profile', {})
                tweets = item.get('tweet', []) or []
                label = 1 if item.get('label') == 'bot' or item.get('label') == 1 else 0
                all_profiles.append({
                    'user': {
                        'screen_name': p.get('screen_name', ''),
                        'description': p.get('description', ''),
                        'followers_count': p.get('followers_count', 0),
                        'friends_count': p.get('friends_count', 0),
                        'statuses_count': p.get('statuses_count', 0),
                        'created_at': p.get('created_at', ''),
                        'verified': p.get('verified', False),
                        'profile_image_url': p.get('profile_image_url', ''),
                        'default_profile_image': p.get('default_profile_image', False)
                    },
                    'tweets': tweets,
                    'is_threat': label,
                    'source': 'twibot-20'
                })
        logger.info(f"Loaded {len(data)} profiles from TwiBot-20")

    return all_profiles

def extract_dataset_features(profiles: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract 44 multi-modal features while neutralizing shortcut bias."""
    extractor = UnifiedFeatureExtractor()
    records = []
    
    logger.info(f"Extracting multi-modal features from {len(profiles)} real profiles...")
    for idx, p in enumerate(profiles):
        if (idx + 1) % 500 == 0 or idx == len(profiles) - 1:
            logger.info(f"Extracted {idx + 1}/{len(profiles)} profiles...")
            
        u = p['user']
        profile_data = {
            'username': u.get('screen_name', u.get('name', f'user_{idx}')),
            'description': u.get('description', '') or '',
            'followers_count': int(u.get('followers_count', 0) or 0),
            'following_count': int(u.get('friends_count', u.get('following_count', 0)) or 0),
            'posts_count': int(u.get('statuses_count', u.get('posts_count', 0)) or 0),
            'created_at': u.get('created_at', ''),
            'verified': bool(u.get('verified', False)),
            'profile_image_url': u.get('profile_image_url', ''),
            'default_profile_image': bool(u.get('default_profile_image', False)),
            'recent_tweets': p.get('tweets', [])
        }
        
        feats = extractor.extract_features(profile_data)
        
        # Shortcut Neutralization: Mask verified badge to prevent shortcut learning
        feats['Twitter.Verified'] = 'no'
        feats['is_threat'] = p['is_threat']
        feats['dataset_source'] = p.get('source', 'unknown')
        records.append(feats)
        
    df = pd.DataFrame(records)
    
    # Balance dataset: Downsample majority class to prevent class bias
    threats = df[df['is_threat'] == 1]
    humans = df[df['is_threat'] == 0]
    min_count = min(len(threats), len(humans))
    
    balanced_df = pd.concat([
        threats.sample(n=min_count, random_state=42),
        humans.sample(n=min_count, random_state=42)
    ]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    logger.info(f"Balanced Feature Matrix: {len(balanced_df)} samples ({min_count} threats, {min_count} humans)")
    return balanced_df

def train_and_benchmark(df: pd.DataFrame):
    """Train and evaluate 13 ML models on balanced real dataset."""
    logger.info("Initializing ModelTrainer with anti-shortcut regularization...")
    trainer = ModelTrainer()
    
    # Drop dataset_source metadata column
    feature_df = df.drop(columns=['dataset_source'], errors='ignore')
    
    # Prepare data
    X, y = trainer.prepare_data(feature_df, target_col='is_threat')
    
    # Train and evaluate all 13 models
    results = trainer.train_and_compare(X, y, test_size=0.2)
    
    # Select champion model
    best_name = max(results.keys(), key=lambda k: results[k]['f1'])
    trainer.best_model_name = best_name
    trainer.train_best_model(X, y)
    
    # Save model bundle
    save_path = trainer.save_model('threat_detector_model.pkl')
    logger.info(f"Champion model ({trainer.best_model_name}) saved to: {save_path}")
    
    # Save training dataset
    out_csv = os.path.join(PROJECT_ROOT, 'data', 'training_data.csv')
    feature_df.to_csv(out_csv, index=False)
    logger.info(f"Saved balanced real benchmark dataset to: {out_csv} ({len(feature_df)} profiles)")
    
    # Print clean benchmark leaderboard
    df_res = pd.DataFrame(results).T.sort_values(by='accuracy', ascending=False)
    print("\n" + "="*70)
    print("  BALANCED ANTI-SHORTCUT REAL BENCHMARK EVALUATION LEADERBOARD")
    print("="*70)
    print(df_res.to_string())
    print("="*70)
    return df_res

if __name__ == '__main__':
    profiles = load_real_profiles()
    df = extract_dataset_features(profiles)
    train_and_benchmark(df)

