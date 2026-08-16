"""
Visualization utilities for the Unified Threat Detector
"""

import logging
from typing import Dict, Any, List
import json

# Configure logging
logger = logging.getLogger(__name__)

def generate_report(result: Dict[str, Any], profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a comprehensive report for the analysis result.
    
    Args:
        result: Analysis result from detector
        profile_data: Original profile data
        
    Returns:
        Report data dictionary for rendering
    """
    logger.info(f"Generating report for {profile_data.get('username', 'Unknown')}")
    
    # Determine risk level
    probability = result.get('probability', 0)
    if probability >= 0.7:
        risk_level = 'high'
        risk_color = 'danger'
    elif probability >= 0.4:
        risk_level = 'medium'
        risk_color = 'warning'
    else:
        risk_level = 'low'
        risk_color = 'success'
    
    # Format indicators for display
    indicators = result.get('indicators', [])
    formatted_indicators = []
    for ind in indicators:
        formatted_indicators.append({
            'type': ind.get('type', 'unknown'),
            'severity': ind.get('severity', 'low'),
            'description': ind.get('description', ''),
            'value': ind.get('value', ''),
            'severity_class': {
                'high': 'danger',
                'medium': 'warning',
                'low': 'info'
            }.get(ind.get('severity', 'low'), 'secondary')
        })
    
    # Format feature importance for display
    feature_importance = result.get('feature_importance', {})
    formatted_features = []
    for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        formatted_features.append({
            'name': feature.replace('_', ' ').title(),
            'importance': importance,
            'percentage': f"{importance * 100:.1f}%"
        })
    
    # Prepare profile summary
    profile_summary = {
        'username': profile_data.get('username', 'Unknown'),
        'platform': profile_data.get('platform', 'Unknown'),
        'display_name': profile_data.get('display_name', ''),
        'bio': profile_data.get('bio', ''),
        'followers_count': profile_data.get('followers_count', 0),
        'following_count': profile_data.get('following_count', 0),
        'posts_count': profile_data.get('posts_count', 0),
        'verified': profile_data.get('verified', False),
        'creation_date': profile_data.get('creation_date', 'Unknown'),
        'profile_pic_url': profile_data.get('profile_pic_url', ''),
        'url': profile_data.get('url', '')
    }
    
    # Threat type display
    threat_type = result.get('threat_type', 'unknown')
    threat_type_display = {
        'fake_profile': 'Fake Profile / Impersonation',
        'spam': 'Spam Account',
        'scam': 'Scam / Fraud',
        'bot': 'Automated Bot',
        'suspicious': 'Suspicious Activity',
        'legitimate': 'Legitimate Account',
        'error': 'Analysis Error'
    }.get(threat_type, threat_type.replace('_', ' ').title())
    
    report = {
        'profile': profile_summary,
        'analysis': {
            'is_threat': result.get('is_threat', False),
            'threat_type': threat_type,
            'threat_type_display': threat_type_display,
            'probability': probability,
            'probability_percent': f"{probability * 100:.1f}%",
            'risk_level': risk_level,
            'risk_color': risk_color,
            'model_used': result.get('model_used', 'unknown'),
            'timestamp': result.get('analysis_timestamp', '')
        },
        'indicators': formatted_indicators,
        'feature_importance': formatted_features,
        'recommendations': result.get('recommendations', []),
        'summary': generate_summary(result, profile_data),
        'xai_report': result.get('xai_report', {}),
    }
    
    return report

def generate_summary(result: Dict[str, Any], profile_data: Dict[str, Any]) -> str:
    """Generate a human-readable summary of the analysis."""
    probability = result.get('probability', 0)
    threat_type = result.get('threat_type', 'unknown')
    indicators = result.get('indicators', [])
    
    username = profile_data.get('username', 'Unknown')
    
    if probability < 0.4:
        return f"@{username} appears to be a legitimate account with no significant risk indicators detected."
    
    summary_parts = [f"@{username} shows signs of {threat_type.replace('_', ' ')}"]
    
    if probability >= 0.7:
        summary_parts.append(f"with high confidence ({probability:.0%}).")
    else:
        summary_parts.append(f"with moderate confidence ({probability:.0%}).")
    
    # Add key indicators
    high_severity = [i for i in indicators if i.get('severity') == 'high']
    if high_severity:
        summary_parts.append(f"Key concerns: {', '.join([i['description'] for i in high_severity[:3]])}.")
    
    return ' '.join(summary_parts)

def generate_batch_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a summary report for batch analysis.
    
    Args:
        results: List of analysis results
        
    Returns:
        Batch report dictionary
    """
    total = len(results)
    threats = sum(1 for r in results if r.get('is_threat', False))
    
    # Count by threat type
    threat_types = {}
    for r in results:
        t = r.get('threat_type', 'unknown')
        threat_types[t] = threat_types.get(t, 0) + 1
    
    # Count by risk level
    risk_levels = {'high': 0, 'medium': 0, 'low': 0}
    for r in results:
        prob = r.get('probability', 0)
        if prob >= 0.7:
            risk_levels['high'] += 1
        elif prob >= 0.4:
            risk_levels['medium'] += 1
        else:
            risk_levels['low'] += 1
    
    # Top indicators across all profiles
    all_indicators = []
    for r in results:
        all_indicators.extend(r.get('indicators', []))
    
    indicator_counts = {}
    for ind in all_indicators:
        t = ind.get('type', 'unknown')
        indicator_counts[t] = indicator_counts.get(t, 0) + 1
    
    top_indicators = sorted(indicator_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        'total_analyzed': total,
        'threats_detected': threats,
        'threat_rate': f"{threats/total*100:.1f}%" if total > 0 else "0%",
        'threat_types': threat_types,
        'risk_levels': risk_levels,
        'top_indicators': top_indicators,
        'results': results
    }