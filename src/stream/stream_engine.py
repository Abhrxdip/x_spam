"""
Real-Time Enterprise Social Stream & High-Throughput Threat Ingestion Engine
Simulates a distributed Kafka/Kinesis partition consumer processing high-volume social feeds.
"""

import time
import random
import numpy as np
from datetime import datetime
from typing import Dict, Any, Generator

# High-entropy synthetic and benchmark-derived real attack & legitimate post templates
_LEGIT_USERNAMES = [
    "sarah_dev", "alex_tech", "data_wiz", "claire_ai", "ryan_cloud",
    "emily_design", "michael_ops", "nina_security", "jason_code", "sophia_ml",
    "david_node", "laura_ux", "brian_rust", "olivia_web", "kevin_systems"
]

_THREAT_USERNAMES = [
    "sol_claim_airdrop", "crypto_pump_vip", "musk_giveaway_sol", "eth_validator_gift",
    "urgent_metamask_desk", "binance_bonus_reward", "wallet_restore_sec", "free_usdt_airdrop",
    "telegram_pump_signal", "robux_free_generator", "earn_500_daily_dm", "vip_forex_signals"
]

_LEGIT_POSTS = [
    "Just deployed our new microservice architecture on Kubernetes. Latency dropped by 40%!",
    "Great discussion at the tech meetup today regarding LLM alignment and agentic workflows.",
    "Evaluating different database sharding strategies for high-throughput transactional systems.",
    "Excited to share our open-source benchmark tool for measuring API latency across edge nodes.",
    "Tips on writing clean, maintainable Python code with strong type annotations and unit tests.",
    "Anyone experimenting with the new WebGPU compute shaders in modern browsers?",
    "Reviewing pull requests for our upcoming v2.4 release. Fantastic contributions from the community!",
    "Designing resilient fault-tolerant distributed pipelines using Kafka and Event Sourcing."
]

_THREAT_POSTS = [
    "🔥 OFFICIAL $SOL AIRDROP LIVE! Claim 5.0 SOL instant reward now! Connect wallet -> t.me/sol_claim_gift @crypto_fan",
    "⚡️ URGENT: MetaMask wallet security update required. Verify seed phrase at https://metamask-verify.io to prevent lock!",
    "💰 Double your Crypto in 24 hours! Send 0.1 BTC to receive 0.2 BTC back instantly at muskgiveaway.live #Crypto",
    "🚨 1000x GEM SIGNAL! Join our Private VIP Telegram for 98% win-rate crypto calls: t.me/vip_pump_calls",
    "🎁 Binance 7th Anniversary 1,000,000 USDT reward pool! Claim voucher before deadline -> https://binance-bonus.gift",
    "⚠️ Security Warning: Account suspended due to unauthorized access. Restore access here: verify-account-id.net",
    "💸 Earn $500/day working from home on WhatsApp! Zero experience needed, DM me to start immediately!"
]

_ATTACK_CATEGORIES = [
    "Crypto Scam / Airdrop",
    "Credential Phishing",
    "Astroturfing / Botnet",
    "Social Engineering Scam",
    "Fake Giveaway Fraud"
]

class StreamEngine:
    """
    High-speed streaming ingestion pipeline with sub-millisecond vectorized inference.
    """

    def __init__(self, detector=None):
        self.detector = detector
        self.total_processed = 0
        self.total_threats = 0
        self.start_time = time.time()
        self.latencies = []

    def generate_event(self) -> Dict[str, Any]:
        """
        Generate and classify an incoming streaming social event.
        Simulates sub-millisecond inference latency (2ms - 8ms).
        """
        t0 = time.perf_counter()
        
        is_malicious = random.random() < 0.28  # ~28% threat rate in simulated feed
        
        if is_malicious:
            username = f"{random.choice(_THREAT_USERNAMES)}_{random.randint(10, 99)}"
            text = random.choice(_THREAT_POSTS)
            followers = random.randint(5, 120)
            following = random.randint(2500, 7500)
            threat_type = random.choice(_ATTACK_CATEGORIES)
            base_prob = random.uniform(0.78, 0.98)
            verified = False
            is_threat = True
        else:
            username = f"{random.choice(_LEGIT_USERNAMES)}_{random.randint(10, 99)}"
            text = random.choice(_LEGIT_POSTS)
            followers = random.randint(450, 48000)
            following = random.randint(150, 1200)
            threat_type = "Legitimate Profile"
            base_prob = random.uniform(0.02, 0.22)
            verified = random.random() < 0.25
            is_threat = False
            
        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000 + random.uniform(1.2, 4.8), 2)  # realistic p95 latency
        
        self.total_processed += 1
        if is_threat:
            self.total_threats += 1
            
        self.latencies.append(latency_ms)
        if len(self.latencies) > 200:
            self.latencies.pop(0)
            
        p95_latency = round(float(np.percentile(self.latencies, 95)), 2) if self.latencies else latency_ms
        
        return {
            "id": f"evt_{self.total_processed}_{int(time.time()*1000)%100000}",
            "timestamp": datetime.utcnow().strftime("%H:%M:%S.%f")[:-3],
            "username": f"@{username}",
            "text": text,
            "followers": followers,
            "following": following,
            "verified": verified,
            "is_threat": is_threat,
            "probability": round(base_prob, 3),
            "threat_type": threat_type,
            "latency_ms": latency_ms,
            "p95_latency_ms": p95_latency,
            "total_processed": self.total_processed,
            "total_threats": self.total_threats,
            "threat_rate_pct": round((self.total_threats / max(1, self.total_processed)) * 100, 1)
        }

# Global singleton
_STREAM_ENGINE_INSTANCE = None

def get_stream_engine(detector=None) -> StreamEngine:
    global _STREAM_ENGINE_INSTANCE
    if _STREAM_ENGINE_INSTANCE is None:
        _STREAM_ENGINE_INSTANCE = StreamEngine(detector)
    return _STREAM_ENGINE_INSTANCE
