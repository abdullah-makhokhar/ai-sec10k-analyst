"""
Results Formatter for SEC 10-K Financial Analysis
Aggregates and formats analysis results for presentation
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultsFormattingError(Exception):
    """Custom exception for results formatting errors"""
    pass

def format_analysis_results(all_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format and structure all analysis results for presentation
    
    Args:
        all_data: Dictionary containing all analysis components
        
    Returns:
        Formatted results dictionary
    """
    try:
        logger.info("Starting results formatting")
        
        formatted_results = {
            'metadata': _format_metadata(all_data),
            'company_info': _format_company_info(all_data),
            'financial_metrics': _format_financial_metrics(all_data),
            'risk_analysis': _format_risk_analysis(all_data),
            'sentiment_analysis': _format_sentiment_analysis(all_data),
            'executive_summary': _generate_executive_summary(all_data),
            'data_quality': _assess_data_quality(all_data),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        logger.info("Successfully formatted analysis results")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error formatting analysis results: {str(e)}")
        raise ResultsFormattingError(f"Failed to format results: {str(e)}")

def _format_metadata(all_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format metadata information"""
    filing_data = all_data.get('filing_data', {})
    parsed_data = all_data.get('parsed_data', {})
    
    return {
        'ticker': filing_data.get('ticker', 'Unknown'),
        'company_name': filing_data.get('company_name', 'Unknown Company'),
        'filing_date': filing_data.get('filing_date', 'Unknown'),
        'form_type': filing_data.get('form_type', '10-K'),
        'cik': filing_data.get('cik', 'Unknown'),
        'accession_number': filing_data.get('accession_number', 'Unknown'),
        'filing_size_mb': round(filing_data.get('filing_size', 0) / (1024 * 1024), 2),
        'sections_extracted': parsed_data.get('metadata', {}).get('sections_found', []),
        'analysis_completeness': _calculate_analysis_completeness(all_data)
    }

def _format_company_info(all_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format company information"""
    filing_data = all_data.get('filing_data', {})
    financial_metrics = all_data.get('financial_metrics', {})
    
    company_info = {
        'basic_info': {
            'ticker': filing_data.get('ticker', 'Unknown'),
            'company_name': filing_data.get('company_name', 'Unknown Company'),
            'cik': filing_data.get('cik', 'Unknown'),
            'filing_date': filing_data.get('filing_date', 'Unknown')
        },
        'financial_profile': {
            'revenue_current': financial_metrics.get('revenue_current'),
            'company_size': _classify_company_size(financial_metrics.get('revenue_current')),
            'profitability': _assess_profitability(financial_metrics)
        }
    }
    
    return company_info

def _format_financial_metrics(all_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format financial metrics with additional calculations"""
    financial_metrics = all_data.get('financial_metrics', {})
    
    formatted_metrics = {
        'revenue': {
            'current': financial_metrics.get('revenue_current'),
            'previous': financial_metrics.get('revenue_previous'),
            'growth_rate': _calculate_growth_rate(
                financial_metrics.get('revenue_current'),
                financial_metrics.get('revenue_previous')
            ),
            'trend': _assess_trend(
                financial_metrics.get('revenue_current'),
                financial_metrics.get('revenue_previous')
            )
        },
        'profitability': {
            'net_income_current': financial_metrics.get('net_income_current'),
            'net_income_previous': financial_metrics.get('net_income_previous'),
            'profit_margin': financial_metrics.get('profit_margin'),
            'income_growth_rate': _calculate_growth_rate(
                financial_metrics.get('net_income_current'),
                financial_metrics.get('net_income_previous')
            ),
            'profitability_assessment': _assess_profitability(financial_metrics)
        },
        'financial_position': {
            'total_assets': financial_metrics.get('total_assets'),
            'total_liabilities': financial_metrics.get('total_liabilities'),
            'equity_estimated': _calculate_equity(financial_metrics),
            'debt_to_equity': financial_metrics.get('debt_to_equity'),
            'financial_leverage': _assess_leverage(financial_metrics.get('debt_to_equity'))
        },
        'cash_flow': {
            'operating_cash_flow': financial_metrics.get('operating_cash_flow'),
            'cash_flow_strength': _assess_cash_flow_strength(
                financial_metrics.get('operating_cash_flow'),
                financial_metrics.get('revenue_current')
            )
        }
    }
    
    return formatted_metrics

def _format_risk_analysis(all_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format risk analysis results"""
    risk_analysis = all_data.get('risk_analysis', {})
    top_risks = risk_analysis.get('top_risks', [])
    
    formatted_risks = {
        'risk_summary': {
            'total_risks_identified': len(top_risks),
            'average_severity': _calculate_average_severity(top_risks),
            'risk_categories': _extract_risk_categories(top_risks),
            'overall_risk_level': _assess_overall_risk_level(top_risks)
        },
        'top_risks': [
            {
                'rank': i + 1,
                'category': risk.get('category', 'Unknown'),
                'description': risk.get('description', 'No description available'),
                'severity_score': risk.get('severity_score', 3),
                'severity_level': _get_severity_level(risk.get('severity_score', 3))
            }
            for i, risk in enumerate(top_risks[:5])
        ],
        'risk_distribution': _analyze_risk_distribution(top_risks)
    }
    
    return formatted_risks

def _format_sentiment_analysis(all_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format sentiment analysis results"""
    sentiment_analysis = all_data.get('sentiment_analysis', {})
    
    formatted_sentiment = {
        'overall_sentiment': {
            'score': sentiment_analysis.get('sentiment_score', 0.0),
            'category': sentiment_analysis.get('sentiment_category', 'neutral'),
            'confidence': sentiment_analysis.get('confidence_score', 0.0),
            'interpretation': _interpret_sentiment_score(
                sentiment_analysis.get('sentiment_score', 0.0)
            )
        },
        'sentiment_indicators': {
            'positive_signals': sentiment_analysis.get('key_positive_phrases', []),
            'negative_signals': sentiment_analysis.get('key_negative_phrases', []),
            'key_themes': sentiment_analysis.get('key_themes', []),
            'sentiment_balance': _calculate_sentiment_balance(sentiment_analysis)
        },
        'management_outlook': {
            'optimism_level': _assess_optimism_level(sentiment_analysis.get('sentiment_score', 0.0)),
            'confidence_level': _assess_confidence_level(sentiment_analysis.get('confidence_score', 0.0))
        }
    }
    
    return formatted_sentiment

def _generate_executive_summary(all_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive executive summary"""
    try:
        filing_data = all_data.get('filing_data', {})
        financial_metrics = all_data.get('financial_metrics', {})
        risk_analysis = all_data.get('risk_analysis', {})
        sentiment_analysis = all_data.get('sentiment_analysis', {})
        
        # Key highlights and concerns
        highlights = []
        concerns = []
        
        # Financial analysis
        revenue_current = financial_metrics.get('revenue_current')
        revenue_previous = financial_metrics.get('revenue_previous')
        profit_margin = financial_metrics.get('profit_margin')
        
        if revenue_current and revenue_previous:
            growth_rate = _calculate_growth_rate(revenue_current, revenue_previous)
            if growth_rate and growth_rate > 10:
                highlights.append(f"Strong revenue growth of {growth_rate:.1f}%")
            elif growth_rate and growth_rate < -5:
                concerns.append(f"Revenue decline of {abs(growth_rate):.1f}%")
        
        if profit_margin:
            if profit_margin > 15:
                highlights.append(f"Strong profit margin of {profit_margin:.1f}%")
            elif profit_margin < 5:
                concerns.append(f"Low profit margin of {profit_margin:.1f}%")
        
        # Risk assessment
        top_risks = risk_analysis.get('top_risks', [])
        high_severity_risks = [r for r in top_risks if r.get('severity_score', 0) >= 4]
        if high_severity_risks:
            concerns.append(f"{len(high_severity_risks)} high-severity risk factors identified")
        
        # Sentiment assessment
        sentiment_score = sentiment_analysis.get('sentiment_score', 0.0)
        if sentiment_score > 0.3:
            highlights.append("Positive management outlook and tone")
        elif sentiment_score < -0.3:
            concerns.append("Negative management sentiment detected")
        
        summary_text = _create_summary_narrative(
            filing_data, financial_metrics, risk_analysis, sentiment_analysis,
            highlights, concerns
        )
        
        executive_summary = {
            'summary_text': summary_text,
            'key_highlights': highlights[:5],
            'key_concerns': concerns[:5],
            'investment_considerations': _generate_investment_considerations(all_data),
            'overall_assessment': _generate_overall_assessment(highlights, concerns),
            'recommendation_level': _generate_recommendation_level(all_data)
        }
        
        return executive_summary
        
    except Exception as e:
        logger.error(f"Error generating executive summary: {str(e)}")
        return {
            'summary_text': "Executive summary could not be generated due to data processing errors.",
            'key_highlights': [],
            'key_concerns': ["Analysis incomplete due to processing errors"],
            'investment_considerations': [],
            'overall_assessment': 'Indeterminate',
            'recommendation_level': 'Neutral'
        }

def _create_summary_narrative(filing_data: Dict[str, Any], 
                             financial_metrics: Dict[str, Any],
                             risk_analysis: Dict[str, Any],
                             sentiment_analysis: Dict[str, Any],
                             highlights: List[str], concerns: List[str]) -> str:
    """Create narrative executive summary text"""
    company_name = filing_data.get('company_name', 'The company')
    filing_date = filing_data.get('filing_date', 'recent period')
    
    # Build narrative
    intro = f"{company_name} filed its 10-K for the period ending {filing_date}. "
    
    # Financial performance summary
    revenue_current = financial_metrics.get('revenue_current')
    net_income_current = financial_metrics.get('net_income_current')
    
    financial_summary = ""
    if revenue_current:
        financial_summary += f"The company reported revenue of ${revenue_current:,.0f} million"
        if net_income_current:
            if net_income_current > 0:
                financial_summary += f" with net income of ${net_income_current:,.0f} million. "
            else:
                financial_summary += f" but recorded a net loss of ${abs(net_income_current):,.0f} million. "
        else:
            financial_summary += ". "
    
    # Risk and sentiment overview
    risk_summary = ""
    top_risks = risk_analysis.get('top_risks', [])
    if top_risks:
        primary_risk = top_risks[0].get('category', 'business risks')
        risk_summary = f"Key risks include {primary_risk.lower()} and other operational challenges. "
    
    sentiment_summary = ""
    sentiment_score = sentiment_analysis.get('sentiment_score', 0.0)
    if sentiment_score > 0.2:
        sentiment_summary = "Management expresses optimism about future prospects. "
    elif sentiment_score < -0.2:
        sentiment_summary = "Management tone reflects caution and uncertainty. "
    else:
        sentiment_summary = "Management maintains a balanced perspective on business outlook. "
    
    # Conclusion
    if len(highlights) > len(concerns):
        conclusion = "Overall, the filing presents a generally positive business picture."
    elif len(concerns) > len(highlights):
        conclusion = "The filing reveals several areas of concern that warrant attention."
    else:
        conclusion = "The filing presents a balanced view with both positive aspects and challenges."
    
    return intro + financial_summary + risk_summary + sentiment_summary + conclusion

def _assess_data_quality(all_data: Dict[str, Any]) -> Dict[str, Any]:
    """Assess overall data quality and completeness"""
    quality_metrics = {
        'parsing_quality': 0,
        'financial_data_completeness': 0,
        'risk_analysis_quality': 0,
        'sentiment_analysis_quality': 0,
        'overall_quality_score': 0
    }
    
    try:
        # Parsing quality
        parsed_data = all_data.get('parsed_data', {})
        sections_found = len(parsed_data.get('metadata', {}).get('sections_found', []))
        quality_metrics['parsing_quality'] = min(100, (sections_found / 5) * 100)
        
        # Financial data completeness
        financial_metrics = all_data.get('financial_metrics', {})
        required_fields = ['revenue_current', 'net_income_current', 'total_assets']
        completed_fields = sum(1 for field in required_fields if financial_metrics.get(field) is not None)
        quality_metrics['financial_data_completeness'] = (completed_fields / len(required_fields)) * 100
        
        # Risk analysis quality
        risk_analysis = all_data.get('risk_analysis', {})
        risks_found = len(risk_analysis.get('top_risks', []))
        quality_metrics['risk_analysis_quality'] = min(100, (risks_found / 5) * 100)
        
        # Sentiment analysis quality
        sentiment_analysis = all_data.get('sentiment_analysis', {})
        confidence_score = sentiment_analysis.get('confidence_score', 0.0)
        quality_metrics['sentiment_analysis_quality'] = confidence_score * 100
        
        # Overall quality score (weighted average)
        weights = [0.2, 0.4, 0.2, 0.2]
        scores = [
            quality_metrics['parsing_quality'],
            quality_metrics['financial_data_completeness'],
            quality_metrics['risk_analysis_quality'],
            quality_metrics['sentiment_analysis_quality']
        ]
        
        quality_metrics['overall_quality_score'] = sum(w * s for w, s in zip(weights, scores))
        
    except Exception as e:
        logger.error(f"Error assessing data quality: {str(e)}")
    
    return quality_metrics

# Helper functions
def _calculate_analysis_completeness(all_data: Dict[str, Any]) -> float:
    """Calculate overall analysis completeness percentage"""
    total_components = 5
    completed_components = 0
    
    if all_data.get('filing_data'):
        completed_components += 1
    if all_data.get('parsed_data', {}).get('sections'):
        completed_components += 1
    if any(v is not None for v in all_data.get('financial_metrics', {}).values()):
        completed_components += 1
    if all_data.get('risk_analysis', {}).get('top_risks'):
        completed_components += 1
    if all_data.get('sentiment_analysis', {}).get('sentiment_score') is not None:
        completed_components += 1
    
    return (completed_components / total_components) * 100

def _calculate_growth_rate(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    """Calculate growth rate percentage"""
    if current is None or previous is None or previous == 0:
        return None
    return ((current - previous) / previous) * 100

def _assess_trend(current: Optional[float], previous: Optional[float]) -> str:
    """Assess trend direction"""
    growth_rate = _calculate_growth_rate(current, previous)
    if growth_rate is None:
        return "Unknown"
    elif growth_rate > 5:
        return "Growing"
    elif growth_rate < -5:
        return "Declining"
    else:
        return "Stable"

def _classify_company_size(revenue: Optional[float]) -> str:
    """Classify company size based on revenue"""
    if revenue is None:
        return "Unknown"
    elif revenue > 10000:
        return "Large Cap"
    elif revenue > 1000:
        return "Mid Cap"
    else:
        return "Small Cap"

def _assess_profitability(financial_metrics: Dict[str, Any]) -> str:
    """Assess profitability level"""
    profit_margin = financial_metrics.get('profit_margin')
    if profit_margin is None:
        return "Unknown"
    elif profit_margin > 15:
        return "Highly Profitable"
    elif profit_margin > 5:
        return "Profitable"
    elif profit_margin > 0:
        return "Marginally Profitable"
    else:
        return "Unprofitable"

def _assess_leverage(debt_to_equity: Optional[float]) -> str:
    """Assess financial leverage level"""
    if debt_to_equity is None:
        return "Unknown"
    elif debt_to_equity < 0.5:
        return "Conservative"
    elif debt_to_equity < 1.0:
        return "Moderate"
    else:
        return "High"

def _calculate_equity(financial_metrics: Dict[str, Any]) -> Optional[float]:
    """Calculate estimated equity"""
    assets = financial_metrics.get('total_assets')
    liabilities = financial_metrics.get('total_liabilities')
    if assets is not None and liabilities is not None:
        return assets - liabilities
    return None

def _assess_cash_flow_strength(operating_cf: Optional[float], revenue: Optional[float]) -> str:
    """Assess operating cash flow strength"""
    if operating_cf is None:
        return "Unknown"
    elif operating_cf <= 0:
        return "Weak"
    elif revenue is not None and revenue > 0:
        cf_margin = (operating_cf / revenue) * 100
        if cf_margin > 15:
            return "Strong"
        elif cf_margin > 8:
            return "Adequate"
        else:
            return "Moderate"
    else:
        return "Moderate"

def _calculate_average_severity(risks: List[Dict[str, Any]]) -> float:
    """Calculate average risk severity score"""
    if not risks:
        return 0.0
    scores = [risk.get('severity_score', 3) for risk in risks]
    return sum(scores) / len(scores)

def _extract_risk_categories(risks: List[Dict[str, Any]]) -> List[str]:
    """Extract unique risk categories"""
    categories = [risk.get('category', 'Unknown') for risk in risks]
    return list(set(categories))

def _assess_overall_risk_level(risks: List[Dict[str, Any]]) -> str:
    """Assess overall risk level"""
    if not risks:
        return "Unknown"
    
    avg_severity = _calculate_average_severity(risks)
    high_risk_count = sum(1 for risk in risks if risk.get('severity_score', 0) >= 4)
    
    if avg_severity >= 4 or high_risk_count >= 3:
        return "High"
    elif avg_severity >= 3 or high_risk_count >= 1:
        return "Moderate"
    else:
        return "Low"

def _get_severity_level(score: int) -> str:
    """Convert severity score to level"""
    levels = {1: "Low", 2: "Low-Medium", 3: "Medium", 4: "High", 5: "Very High"}
    return levels.get(score, "Unknown")

def _analyze_risk_distribution(risks: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze distribution of risks by category"""
    distribution = {}
    for risk in risks:
        category = risk.get('category', 'Unknown')
        distribution[category] = distribution.get(category, 0) + 1
    return distribution

def _interpret_sentiment_score(score: float) -> str:
    """Interpret sentiment score"""
    if score > 0.5:
        return "Very Positive"
    elif score > 0.1:
        return "Moderately Positive"
    elif score > -0.1:
        return "Neutral"
    elif score > -0.5:
        return "Moderately Negative"
    else:
        return "Very Negative"

def _calculate_sentiment_balance(sentiment_data: Dict[str, Any]) -> float:
    """Calculate sentiment balance ratio"""
    positive_count = len(sentiment_data.get('key_positive_phrases', []))
    negative_count = len(sentiment_data.get('key_negative_phrases', []))
    
    if positive_count + negative_count == 0:
        return 0.5
    
    return positive_count / (positive_count + negative_count)

def _assess_optimism_level(sentiment_score: float) -> str:
    """Assess management optimism level"""
    if sentiment_score > 0.3:
        return "High"
    elif sentiment_score > 0.0:
        return "Moderate"
    elif sentiment_score > -0.3:
        return "Low"
    else:
        return "Very Low"

def _assess_confidence_level(confidence_score: float) -> str:
    """Assess confidence level in sentiment analysis"""
    if confidence_score > 0.8:
        return "Very High"
    elif confidence_score > 0.6:
        return "High"
    elif confidence_score > 0.4:
        return "Moderate"
    else:
        return "Low"

def _generate_investment_considerations(all_data: Dict[str, Any]) -> List[str]:
    """Generate investment considerations"""
    considerations = []
    
    financial_metrics = all_data.get('financial_metrics', {})
    risk_analysis = all_data.get('risk_analysis', {})
    sentiment_analysis = all_data.get('sentiment_analysis', {})
    
    # Financial considerations
    revenue_growth = _calculate_growth_rate(
        financial_metrics.get('revenue_current'),
        financial_metrics.get('revenue_previous')
    )
    
    if revenue_growth and revenue_growth > 10:
        considerations.append("Strong revenue growth indicates business expansion")
    elif revenue_growth and revenue_growth < -10:
        considerations.append("Revenue decline raises concerns about sustainability")
    
    profit_margin = financial_metrics.get('profit_margin')
    if profit_margin and profit_margin < 5:
        considerations.append("Low profit margins may indicate operational challenges")
    
    # Risk considerations
    high_severity_risks = [r for r in risk_analysis.get('top_risks', []) 
                          if r.get('severity_score', 0) >= 4]
    if high_severity_risks:
        considerations.append(f"Monitor {len(high_severity_risks)} high-severity risk factors")
    
    # Sentiment considerations
    sentiment_score = sentiment_analysis.get('sentiment_score', 0.0)
    if sentiment_score < -0.3:
        considerations.append("Negative management sentiment may signal issues")
    elif sentiment_score > 0.3:
        considerations.append("Positive management outlook supports investment thesis")
    
    return considerations[:5]

def _generate_overall_assessment(highlights: List[str], concerns: List[str]) -> str:
    """Generate overall assessment"""
    highlight_count = len(highlights)
    concern_count = len(concerns)
    
    if highlight_count > concern_count + 1:
        return "Positive"
    elif concern_count > highlight_count + 1:
        return "Negative"
    else:
        return "Mixed"

def _generate_recommendation_level(all_data: Dict[str, Any]) -> str:
    """Generate high-level recommendation"""
    financial_metrics = all_data.get('financial_metrics', {})
    risk_analysis = all_data.get('risk_analysis', {})
    sentiment_analysis = all_data.get('sentiment_analysis', {})
    
    score = 0
    
    # Financial scoring
    profit_margin = financial_metrics.get('profit_margin')
    if profit_margin:
        if profit_margin > 15:
            score += 2
        elif profit_margin > 5:
            score += 1
        elif profit_margin < 0:
            score -= 2
    
    # Risk scoring
    high_risks = sum(1 for risk in risk_analysis.get('top_risks', []) 
                    if risk.get('severity_score', 0) >= 4)
    score -= high_risks
    
    # Sentiment scoring
    sentiment_score = sentiment_analysis.get('sentiment_score', 0.0)
    if sentiment_score > 0.3:
        score += 1
    elif sentiment_score < -0.3:
        score -= 1
    
    # Convert score to recommendation
    if score >= 3:
        return "Positive"
    elif score <= -3:
        return "Negative"
    else:
        return "Neutral"

def generate_executive_summary(analysis_data: Dict[str, Any]) -> str:
    """Generate executive summary text"""
    try:
        executive_summary = analysis_data.get('executive_summary', {})
        return executive_summary.get('summary_text', 'Executive summary not available.')
    except Exception as e:
        logger.error(f"Error generating executive summary: {str(e)}")
        return "Executive summary could not be generated."

def export_results_json(results: Dict[str, Any], filename: Optional[str] = None) -> str:
    """
    Export results to JSON file
    
    Args:
        results: Formatted analysis results
        filename: Optional filename (auto-generated if not provided)
        
    Returns:
        Filename of exported file
    """
    try:
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ticker = results.get('metadata', {}).get('ticker', 'unknown')
            filename = f"sec_analysis_{ticker}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Results exported to {filename}")
        return filename
        
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}")
        raise ResultsFormattingError(f"Failed to export results: {str(e)}")

def create_results_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a condensed summary of results for quick viewing
    
    Args:
        results: Full formatted analysis results
        
    Returns:
        Condensed summary dictionary
    """
    try:
        metadata = results.get('metadata', {})
        financial_metrics = results.get('financial_metrics', {})
        risk_analysis = results.get('risk_analysis', {})
        sentiment_analysis = results.get('sentiment_analysis', {})
        executive_summary = results.get('executive_summary', {})
        
        summary = {
            'company': {
                'ticker': metadata.get('ticker'),
                'name': metadata.get('company_name'),
                'filing_date': metadata.get('filing_date')
            },
            'financial_highlights': {
                'revenue': financial_metrics.get('revenue', {}).get('current'),
                'revenue_growth': financial_metrics.get('revenue', {}).get('growth_rate'),
                'profit_margin': financial_metrics.get('profitability', {}).get('profit_margin'),
                'financial_health': financial_metrics.get('financial_position', {}).get('financial_leverage')
            },
            'key_risks': [risk.get('category') for risk in risk_analysis.get('top_risks', [])[:3]],
            'management_sentiment': {
                'score': sentiment_analysis.get('overall_sentiment', {}).get('score'),
                'category': sentiment_analysis.get('overall_sentiment', {}).get('category'),
                'interpretation': sentiment_analysis.get('overall_sentiment', {}).get('interpretation')
            },
            'overall_assessment': executive_summary.get('overall_assessment'),
            'recommendation': executive_summary.get('recommendation_level'),
            'analysis_quality': results.get('data_quality', {}).get('overall_quality_score')
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error creating results summary: {str(e)}")
        return {'error': 'Failed to create summary'} 