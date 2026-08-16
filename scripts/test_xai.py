import os
import sys
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.detector import UnifiedThreatDetector

logging.basicConfig(level=logging.INFO)

detector = UnifiedThreatDetector()
sample_profile = {
    'username': 'cryptopump_bot22',
    'description': 'Claim free airdrop! 100x profit guaranteed dm me http://t.me/scampump',
    'followers_count': 12,
    'following_count': 1800,
    'posts_count': 450,
    'created_at': '2024-07-01T00:00:00Z',
    'verified': False,
    'recent_tweets': [
        {'text': '🔥 FREE AIRDROP! Claim your 5000 USDT bonus right now at http://t.me/scampump dm me fast!'},
        {'text': 'Guaranteed 10x trading signals join VIP telegram channel today'}
    ]
}

res = detector.analyze_profile(sample_profile)
print('='*60)
print('ANALYSIS RESULT:')
print('Is Threat:', res['is_threat'])
print('Probability:', f"{res['probability']:.1%}")
print('Threat Type:', res['threat_type'])
print('XAI Available:', bool(res.get('xai_report')))

if res.get('xai_report'):
    xai = res['xai_report']
    print('\n[1] SHAP Attribution:')
    print('    Method:', xai.get('shap', {}).get('method'))
    print('    Baseline:', xai.get('shap', {}).get('baseline'), '%')
    print('    Final Score:', xai.get('shap', {}).get('final_score'), '%')
    print('    Top Threats:', [t['label'] + ' (+' + str(t['pct']) + '%)' for t in xai.get('shap', {}).get('top_threats', [])])
    print('    Top Safe Anchors:', [t['label'] + ' (' + str(t['pct']) + '%)' for t in xai.get('shap', {}).get('top_safe', [])])

    print('\n[2] Counterfactual Remediation:')
    print('    Current Score:', xai.get('counterfactual', {}).get('original_score'), '%')
    print('    Projected Score:', xai.get('counterfactual', {}).get('projected_score'), '%')
    print('    Summary:', xai.get('counterfactual', {}).get('summary'))
    print('    Interventions:')
    for step in xai.get('counterfactual', {}).get('interventions', []):
        print(f"      - {step['description']} (drops score by {step.get('actual_reduction', step.get('score_reduction'))}%)")

    print('\n[3] NLP Token Saliency:')
    print('    Method:', xai.get('nlp_saliency', {}).get('method'))
    print('    Overall Risk:', xai.get('nlp_saliency', {}).get('overall_risk'))
    print('    Top Trigger Tokens:', [t['token'] for t in xai.get('nlp_saliency', {}).get('top_triggers', [])])

    print('\n[4] XAI Consensus:')
    print('    Level:', xai.get('consensus', {}).get('consensus_level'))
    print('    Agreement:', xai.get('consensus', {}).get('agreement_pct'), '%')
    print('    Description:', xai.get('consensus', {}).get('description'))
print('='*60)
