"""Quick integration test for the fine-tuned NLP classifier."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.features.nlp_classifier import get_nlp_classifier

clf = get_nlp_classifier()

tests = [
    ("FREE AIRDROP! Claim 500 USDT now -> bit.ly/claim Send your wallet address!", "crypto_scam"),
    ("WARNING: Your account is SUSPENDED. Verify now: bit.ly/twitter-verify-2024", "phishing"),
    ("@u1 @u2 @u3 @u4 @u5 CONGRATULATIONS! You won $500! DM to claim!", "mention_spam"),
    ("DM me for investment tips! I turned $500 into $25,000 in 3 months!", "social_engineering"),
    ("Just shipped a new feature today. Really proud of the team's hard work!", "legitimate"),
]

print("=" * 60)
print("  NLP Classifier Integration Test")
print("=" * 60)
all_pass = True
for text, expected in tests:
    result = clf.classify_text(text)
    label, conf, source = result["label"], result["confidence"], result["source"]
    status = "[PASS]" if label == expected else "[FAIL]"
    if label != expected:
        all_pass = False
    print(f"{status} [{source}] -> {label} ({conf:.0%})")
    print(f"       Text: {text[:65]}...")
    print()

metrics = clf.analyze_posts([t for t, _ in tests])
print("Aggregate metrics:", metrics)
print()
print("=" * 60)
print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
print("Model status:", clf.model_status())
print("=" * 60)
