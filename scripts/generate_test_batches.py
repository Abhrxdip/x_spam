"""
Generate ready-to-upload batch test files (CSV and JSON) from existing datasets
(TwiBot-20, TwiBot-22, bot_detection_data.csv) for batch threat analysis testing.
"""

import os
import json
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'test_batches')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_twibot20_batch():
    src_path = os.path.join(PROJECT_ROOT, 'data', 'twibot-20.json')
    if os.path.exists(src_path):
        with open(src_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Take 25 diverse samples
        sample = data[:25]
        dst_path = os.path.join(OUTPUT_DIR, 'twibot20_batch_25_profiles.json')
        with open(dst_path, 'w', encoding='utf-8') as f:
            json.dump(sample, f, indent=2)
        print(f"Created: {dst_path} ({len(sample)} profiles)")

def create_twibot22_batch():
    src_path = os.path.join(PROJECT_ROOT, 'data', 'twibot-22.json')
    if os.path.exists(src_path):
        with open(src_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        dst_path = os.path.join(OUTPUT_DIR, 'twibot22_batch_profiles.json')
        with open(dst_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Created: {dst_path} ({len(data)} profiles)")

def create_bot_detection_csv_batch():
    src_path = os.path.join(PROJECT_ROOT, 'data', 'bot_detection_data.csv')
    if os.path.exists(src_path):
        df = pd.read_csv(src_path)
        # Take 25 bots and 25 humans
        bots = df[df['Bot Label'] == 1].head(25)
        humans = df[df['Bot Label'] == 0].head(25)
        sample = pd.concat([bots, humans]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        dst_path = os.path.join(OUTPUT_DIR, 'bot_detection_50_profiles.csv')
        sample.to_csv(dst_path, index=False)
        print(f"Created: {dst_path} ({len(sample)} profiles: 25 bots, 25 humans)")

def create_mixed_security_csv():
    """Create a clean, compact CSV with clear ground truth and standard columns."""
    profiles = [
        # Verified / Authentic Users
        {"username": "elonmusk", "followers_count": 195000000, "following_count": 720, "posts_count": 48000, "bio": "Owner of X, CEO Tesla, SpaceX. Free speech advocate.", "verified": True, "location": "Austin, TX"},
        {"username": "sundarpichai", "followers_count": 5400000, "following_count": 310, "posts_count": 3100, "bio": "CEO of Alphabet and Google. Technology, AI, and innovation.", "verified": True, "location": "Mountain View, CA"},
        {"username": "sama", "followers_count": 3200000, "following_count": 640, "posts_count": 6200, "bio": "CEO @ OpenAI. Working on safe AGI.", "verified": True, "location": "San Francisco, CA"},
        {"username": "ylecun", "followers_count": 890000, "following_count": 450, "posts_count": 14500, "bio": "Chief AI Scientist at Meta, Professor at NYU. Turing Award Laureate.", "verified": True, "location": "New York, NY"},
        {"username": "DrSarahJenkins", "followers_count": 14200, "following_count": 520, "posts_count": 2800, "bio": "AI Researcher @ Stanford. Computational Social Systems.", "verified": True, "location": "Palo Alto, CA"},
        {"username": "DavidMillerTech", "followers_count": 8600, "following_count": 920, "posts_count": 1420, "bio": "Senior DevOps Engineer | Cloud Architecture | Python & Rust enthusiast.", "verified": False, "location": "Seattle, WA"},
        {"username": "EmmaWatson", "followers_count": 28000000, "following_count": 180, "posts_count": 1200, "bio": "Actor, activist, UN Women Goodwill Ambassador.", "verified": True, "location": "London, UK"},
        {"username": "TechCrunch", "followers_count": 10300000, "following_count": 910, "posts_count": 185000, "bio": "Startup and technology news. Delivering what's next in tech.", "verified": True, "location": "San Francisco, CA"},
        {"username": "BBCBreaking", "followers_count": 52000000, "following_count": 3, "posts_count": 94000, "bio": "Breaking news alerts and updates from the BBC.", "verified": True, "location": "London, UK"},
        {"username": "OpenAI", "followers_count": 3800000, "following_count": 12, "posts_count": 1850, "bio": "Research and deployment company building safe AGI.", "verified": True, "location": "San Francisco, CA"},

        # Malicious / Scams / Bot Profiles
        {"username": "sol_airdrop_fast22", "followers_count": 12, "following_count": 4850, "posts_count": 18900, "bio": "OFFICIAL SOLANA AIRDROP 2024! Claim 5.0 SOL reward instantly! Connect wallet -> t.me/sol_claim_gift", "verified": False, "location": "Metaverse"},
        {"username": "crypto_pump_signal99", "followers_count": 45, "following_count": 7200, "posts_count": 34000, "bio": "1000x GEM SIGNALS! Free VIP Telegram access: t.me/cryptopump999. DM for paid promo.", "verified": False, "location": "Unknown"},
        {"username": "eth_support_desk_help", "followers_count": 3, "following_count": 4900, "posts_count": 8500, "bio": "Official MetaMask & Ethereum Customer Support. Send DM to restore lost seed phrase!", "verified": False, "location": "Web3 Support"},
        {"username": "elon_giveaway_sol24", "followers_count": 18, "following_count": 6500, "posts_count": 22000, "bio": "Elon Musk Official Double Your Crypto Event! Send 0.1 BTC get 0.2 BTC back instantly at muskgiveaway.live", "verified": False, "location": "Texas"},
        {"username": "bot_follower_booster_4u", "followers_count": 210, "following_count": 8900, "posts_count": 45000, "bio": "Get 10k Real Followers in 5 Minutes! Cheap SMM panel at followboost.xyz $1 per 1000!", "verified": False, "location": "Worldwide"},
        {"username": "binance_vip_airdrop01", "followers_count": 8, "following_count": 5100, "posts_count": 12400, "bio": "Binance 7th Anniversary 1,000,000 USDT reward pool! Connect Web3 wallet -> binance-airdrop-gift.co", "verified": False, "location": "Global"},
        {"username": "urgent_bank_security", "followers_count": 2, "following_count": 3800, "posts_count": 9200, "bio": "Security Alert: Account suspended due to unauthorized login. Verify details here: verify-bank-id.net", "verified": False, "location": "Security Dept"},
        {"username": "free_robux_generator_2026", "followers_count": 34, "following_count": 4500, "posts_count": 19000, "bio": "GET 50,000 ROBUX FREE NO HUMAN VERIFICATION! Click link: robux-claim-free.gg", "verified": False, "location": "Roblox World"},
        {"username": "passive_income_guru88", "followers_count": 85, "following_count": 7800, "posts_count": 29000, "bio": "Earn $500/day working from home on WhatsApp! Zero experience needed, DM me to start now!", "verified": False, "location": "Dubai"},
        {"username": "telegram_forex_whale", "followers_count": 64, "following_count": 6200, "posts_count": 17800, "bio": "Guaranteed 98% win-rate Forex signals! Copy trades automatically at t.me/forex_vip_whale", "verified": False, "location": "Wall St"}
    ]
    df = pd.DataFrame(profiles)
    dst_path = os.path.join(OUTPUT_DIR, 'mixed_security_20_profiles.csv')
    df.to_csv(dst_path, index=False)
    print(f"Created: {dst_path} ({len(df)} profiles: 10 authentic, 10 malicious)")

def create_usernames_only_csv():
    """Create a simple list of handles."""
    handles = [
        "elonmusk", "sama", "sundarpichai", "TechCrunch", "BBCBreaking",
        "sol_airdrop_fast22", "crypto_pump_signal99", "eth_support_desk_help",
        "elon_giveaway_sol24", "bot_follower_booster_4u", "OpenAI",
        "free_robux_generator_2026", "passive_income_guru88", "telegram_forex_whale"
    ]
    df = pd.DataFrame({'username': handles, 'platform': 'twitter'})
    dst_path = os.path.join(OUTPUT_DIR, 'handles_only_batch.csv')
    df.to_csv(dst_path, index=False)
    print(f"Created: {dst_path} ({len(df)} handles)")

if __name__ == '__main__':
    create_twibot20_batch()
    create_twibot22_batch()
    create_bot_detection_csv_batch()
    create_mixed_security_csv()
    create_usernames_only_csv()
    print(f"\nAll test batch files generated in: {OUTPUT_DIR}")
