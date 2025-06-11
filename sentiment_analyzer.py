"""
Sentiment Analysis Engine for SEC 10-K MD&A Sections
Analyzes management sentiment and tone from Management Discussion & Analysis sections
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from config import (
    GOOGLE_API_KEY, GEMINI_MODEL_PRIMARY, GEMINI_MODEL_FALLBACK,
    GEMINI_MAX_TOKENS, GEMINI_TEMPERATURE, SENTIMENT_THRESHOLDS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalysisError(Exception):
    """Custom exception for sentiment analysis errors"""
    pass

# Pydantic models for structured output
class SentimentAnalysis(BaseModel):
    """Structured sentiment analysis model"""
    sentiment_score: float = Field(description="Overall sentiment score from -1.0 (very negative) to 1.0 (very positive)")
    sentiment_category: str = Field(description="Sentiment category: positive, neutral, or negative")
    confidence_score: float = Field(description="Confidence in sentiment analysis from 0.0 to 1.0")
    key_positive_phrases: List[str] = Field(description="Key phrases indicating positive sentiment")
    key_negative_phrases: List[str] = Field(description="Key phrases indicating negative sentiment")
    key_themes: List[str] = Field(description="Main themes and topics discussed")

class SentimentComparison(BaseModel):
    """Year-over-year sentiment comparison model"""
    current_sentiment: float = Field(description="Current period sentiment score")
    previous_sentiment: float = Field(description="Previous period sentiment score")
    sentiment_change: float = Field(description="Change in sentiment (current - previous)")
    change_magnitude: str = Field(description="Magnitude of change: significant, moderate, or minimal")
    key_changes: List[str] = Field(description="Key areas where sentiment changed")

class SentimentAnalyzer:
    """LLM-powered sentiment analyzer for MD&A sections"""
    
    def __init__(self):
        self.setup_llm()
        self.setup_prompts()
    
    def setup_llm(self):
        """Initialize Google Gemini LLM with fallback"""
        try:
            if not GOOGLE_API_KEY:
                raise SentimentAnalysisError("Google API key not configured")
            
            # Try primary model first
            self.llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL_PRIMARY,
                temperature=GEMINI_TEMPERATURE,
                max_output_tokens=GEMINI_MAX_TOKENS
            )
            
            logger.info(f"Successfully initialized {GEMINI_MODEL_PRIMARY} for sentiment analysis")
            
        except Exception as e:
            logger.warning(f"Primary model {GEMINI_MODEL_PRIMARY} failed, trying fallback: {str(e)}")
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=GEMINI_MODEL_FALLBACK,
                    temperature=GEMINI_TEMPERATURE,
                    max_output_tokens=GEMINI_MAX_TOKENS
                )
                logger.info(f"Successfully initialized fallback model {GEMINI_MODEL_FALLBACK}")
            except Exception as fallback_error:
                raise SentimentAnalysisError(f"Failed to initialize LLM: {str(fallback_error)}")
    
    def setup_prompts(self):
        """Initialize prompt templates for sentiment analysis tasks"""
        
        # MD&A sentiment analysis prompt
        self.sentiment_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert financial sentiment analyst specializing in Management Discussion & Analysis (MD&A) sections of SEC 10-K filings.

Your task is to analyze the overall tone and sentiment expressed by management regarding the company's performance, outlook, and strategic direction.

SENTIMENT SCORING GUIDELINES:
- Score range: -1.0 (very negative) to +1.0 (very positive)
- 0.0 = neutral/balanced tone
- Consider: optimism vs pessimism, confidence vs uncertainty, growth vs decline language
- Focus on forward-looking statements and management's assessment of performance
- Weight recent performance discussions more heavily than historical context

ANALYSIS INSTRUCTIONS:
1. Analyze the overall tone and sentiment of management's discussion
2. Identify key positive and negative phrases/indicators
3. Extract main themes and topics discussed
4. Assign confidence score based on clarity and consistency of sentiment signals
5. Categorize overall sentiment as positive (>0.1), negative (<-0.1), or neutral

Return your response as valid JSON only, no additional text."""),
            
            ("human", """Analyze the sentiment of the following Management Discussion & Analysis section:

MD&A Section:
{mda_section}

Provide a comprehensive sentiment analysis including:
- sentiment_score: Float from -1.0 to 1.0
- sentiment_category: "positive", "neutral", or "negative"  
- confidence_score: Float from 0.0 to 1.0
- key_positive_phrases: List of positive indicators (max 5)
- key_negative_phrases: List of negative indicators (max 5)
- key_themes: Main topics/themes discussed (max 5)""")
        ])
        
        # Year-over-year comparison prompt
        self.comparison_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert analyst comparing year-over-year changes in management sentiment from MD&A sections.

Your task is to identify how management's tone and outlook has changed between two periods.

CHANGE MAGNITUDE GUIDELINES:
- "significant": Change > 0.3 or < -0.3
- "moderate": Change between 0.1-0.3 or -0.1 to -0.3  
- "minimal": Change between -0.1 and 0.1

INSTRUCTIONS:
1. Compare the sentiment and tone between current and previous MD&A sections
2. Calculate the change in sentiment (current - previous)
3. Identify specific areas where sentiment shifted
4. Assess the magnitude of change
5. Extract key differences in language, outlook, or topics

Return as valid JSON only, no additional text."""),
            
            ("human", """Compare the sentiment between these two MD&A sections:

CURRENT PERIOD MD&A:
{current_mda}

PREVIOUS PERIOD MD&A:
{previous_mda}

Provide year-over-year sentiment comparison:
- current_sentiment: Current period sentiment score (-1.0 to 1.0)
- previous_sentiment: Previous period sentiment score (-1.0 to 1.0)
- sentiment_change: Numerical change (current - previous)
- change_magnitude: "significant", "moderate", or "minimal"
- key_changes: List of key areas where sentiment changed (max 5)""")
        ])
    
    def analyze_management_sentiment(self, mda_section: str) -> Dict[str, Any]:
        """
        Analyze sentiment of Management Discussion & Analysis section
        
        Args:
            mda_section: Raw MD&A section text
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            if not mda_section or len(mda_section.strip()) < 200:
                raise SentimentAnalysisError("MD&A section is too short or empty")
            
            logger.info("Starting MD&A sentiment analysis")
            
            # Truncate if too long (keep first 40000 chars to stay within token limits)
            if len(mda_section) > 40000:
                mda_section = mda_section[:40000] + "..."
                logger.warning("Truncated MD&A section to fit token limits")
            
            # Create the chain with output parser
            parser = JsonOutputParser(pydantic_object=SentimentAnalysis)
            chain = self.sentiment_prompt | self.llm | parser
            
            # Analyze sentiment
            result = chain.invoke({"mda_section": mda_section})
            
            # Validate and post-process the result
            processed_result = self._process_sentiment_analysis(result)
            
            logger.info(f"Successfully analyzed sentiment: {processed_result.get('sentiment_category', 'unknown')} "
                       f"(score: {processed_result.get('sentiment_score', 0):.2f})")
            return processed_result
            
        except Exception as e:
            logger.error(f"Error analyzing MD&A sentiment: {str(e)}")
            # Return neutral sentiment on error
            return self._get_default_sentiment_result()
    
    def compare_year_over_year_tone(self, current_mda: str, previous_mda: str) -> Dict[str, Any]:
        """
        Compare sentiment changes between current and previous MD&A sections
        
        Args:
            current_mda: Current period MD&A section text
            previous_mda: Previous period MD&A section text
            
        Returns:
            Dictionary with year-over-year sentiment comparison
        """
        try:
            if (not current_mda or not previous_mda or 
                len(current_mda.strip()) < 200 or len(previous_mda.strip()) < 200):
                raise SentimentAnalysisError("MD&A sections are too short for comparison")
            
            logger.info("Starting year-over-year sentiment comparison")
            
            # Truncate if too long
            if len(current_mda) > 20000:
                current_mda = current_mda[:20000] + "..."
            if len(previous_mda) > 20000:
                previous_mda = previous_mda[:20000] + "..."
            
            # Create the chain with output parser
            parser = JsonOutputParser(pydantic_object=SentimentComparison)
            chain = self.comparison_prompt | self.llm | parser
            
            # Compare sentiment
            result = chain.invoke({
                "current_mda": current_mda,
                "previous_mda": previous_mda
            })
            
            # Validate and post-process the result
            processed_result = self._process_sentiment_comparison(result)
            
            logger.info(f"Successfully compared sentiment: "
                       f"{processed_result.get('change_magnitude', 'unknown')} change "
                       f"({processed_result.get('sentiment_change', 0):+.2f})")
            return processed_result
            
        except Exception as e:
            logger.error(f"Error comparing year-over-year sentiment: {str(e)}")
            # Return neutral comparison on error
            return self._get_default_comparison_result()
    
    def _process_sentiment_analysis(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate sentiment analysis results"""
        processed = {}
        
        # Validate sentiment score
        sentiment_score = raw_result.get('sentiment_score', 0.0)
        try:
            sentiment_score = float(sentiment_score)
            # Clamp between -1.0 and 1.0
            processed['sentiment_score'] = max(-1.0, min(1.0, sentiment_score))
        except (ValueError, TypeError):
            processed['sentiment_score'] = 0.0
        
        # Validate sentiment category
        sentiment_category = raw_result.get('sentiment_category', 'neutral')
        if isinstance(sentiment_category, str):
            sentiment_category = sentiment_category.lower().strip()
            if sentiment_category in ['positive', 'negative', 'neutral']:
                processed['sentiment_category'] = sentiment_category
            else:
                # Derive from score if category is invalid
                score = processed['sentiment_score']
                if score > SENTIMENT_THRESHOLDS['positive']:
                    processed['sentiment_category'] = 'positive'
                elif score < SENTIMENT_THRESHOLDS['negative']:
                    processed['sentiment_category'] = 'negative'
                else:
                    processed['sentiment_category'] = 'neutral'
        else:
            processed['sentiment_category'] = 'neutral'
        
        # Validate confidence score
        confidence_score = raw_result.get('confidence_score', 0.5)
        try:
            confidence_score = float(confidence_score)
            processed['confidence_score'] = max(0.0, min(1.0, confidence_score))
        except (ValueError, TypeError):
            processed['confidence_score'] = 0.5
        
        # Process phrase lists
        processed['key_positive_phrases'] = self._process_phrase_list(
            raw_result.get('key_positive_phrases', [])
        )
        processed['key_negative_phrases'] = self._process_phrase_list(
            raw_result.get('key_negative_phrases', [])
        )
        processed['key_themes'] = self._process_phrase_list(
            raw_result.get('key_themes', [])
        )
        
        return processed
    
    def _process_sentiment_comparison(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate sentiment comparison results"""
        processed = {}
        
        # Validate sentiment scores
        for field in ['current_sentiment', 'previous_sentiment']:
            value = raw_result.get(field, 0.0)
            try:
                value = float(value)
                processed[field] = max(-1.0, min(1.0, value))
            except (ValueError, TypeError):
                processed[field] = 0.0
        
        # Calculate sentiment change
        processed['sentiment_change'] = processed['current_sentiment'] - processed['previous_sentiment']
        
        # Validate change magnitude
        change_magnitude = raw_result.get('change_magnitude', 'minimal')
        abs_change = abs(processed['sentiment_change'])
        
        if abs_change >= 0.3:
            processed['change_magnitude'] = 'significant'
        elif abs_change >= 0.1:
            processed['change_magnitude'] = 'moderate'
        else:
            processed['change_magnitude'] = 'minimal'
        
        # Process key changes list
        processed['key_changes'] = self._process_phrase_list(
            raw_result.get('key_changes', [])
        )
        
        return processed
    
    def _process_phrase_list(self, phrase_list: List[str], max_items: int = 5) -> List[str]:
        """Process and validate phrase lists"""
        if not isinstance(phrase_list, list):
            return []
        
        processed_phrases = []
        for phrase in phrase_list[:max_items]:
            if isinstance(phrase, str) and phrase.strip():
                # Clean and truncate phrase
                clean_phrase = phrase.strip()[:200]  # Max 200 chars per phrase
                processed_phrases.append(clean_phrase)
        
        return processed_phrases
    
    def _get_default_sentiment_result(self) -> Dict[str, Any]:
        """Return default neutral sentiment result for error cases"""
        return {
            'sentiment_score': 0.0,
            'sentiment_category': 'neutral',
            'confidence_score': 0.0,
            'key_positive_phrases': [],
            'key_negative_phrases': [],
            'key_themes': []
        }
    
    def _get_default_comparison_result(self) -> Dict[str, Any]:
        """Return default comparison result for error cases"""
        return {
            'current_sentiment': 0.0,
            'previous_sentiment': 0.0,
            'sentiment_change': 0.0,
            'change_magnitude': 'minimal',
            'key_changes': []
        }

def analyze_management_sentiment(mda_section: str) -> Dict[str, Any]:
    """
    Main function to analyze sentiment of MD&A section
    
    Args:
        mda_section: Raw MD&A section text
        
    Returns:
        Dictionary with sentiment analysis results
    """
    analyzer = SentimentAnalyzer()
    return analyzer.analyze_management_sentiment(mda_section)

def compare_year_over_year_tone(current_mda: str, previous_mda: str) -> Dict[str, Any]:
    """
    Main function to compare year-over-year sentiment changes
    
    Args:
        current_mda: Current period MD&A section text
        previous_mda: Previous period MD&A section text
        
    Returns:
        Dictionary with sentiment comparison results
    """
    analyzer = SentimentAnalyzer()
    return analyzer.compare_year_over_year_tone(current_mda, previous_mda)

def extract_key_topics(mda_section: str) -> List[str]:
    """
    Extract key business topics and themes from MD&A section
    
    Args:
        mda_section: Raw MD&A section text
        
    Returns:
        List of key topics/themes
    """
    try:
        # Simple keyword-based topic extraction as fallback
        business_keywords = {
            'revenue': ['revenue', 'sales', 'income', 'earnings'],
            'operations': ['operations', 'business', 'operating', 'performance'],
            'market': ['market', 'competitive', 'industry', 'customers'],
            'financial': ['financial', 'cash', 'debt', 'capital', 'liquidity'],
            'growth': ['growth', 'expansion', 'development', 'investment'],
            'risk': ['risk', 'uncertainty', 'challenge', 'volatility'],
            'strategy': ['strategy', 'strategic', 'initiatives', 'plans'],
            'technology': ['technology', 'digital', 'innovation', 'platform']
        }
        
        mda_lower = mda_section.lower()
        topic_scores = {}
        
        for topic, keywords in business_keywords.items():
            score = sum(mda_lower.count(keyword) for keyword in keywords)
            if score > 0:
                topic_scores[topic] = score
        
        # Return top topics sorted by frequency
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, score in sorted_topics[:5]]
        
    except Exception as e:
        logger.error(f"Error extracting topics: {str(e)}")
        return []

def calculate_sentiment_metrics(sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate additional sentiment metrics and insights
    
    Args:
        sentiment_data: Sentiment analysis results
        
    Returns:
        Dictionary with calculated metrics
    """
    try:
        metrics = {}
        
        sentiment_score = sentiment_data.get('sentiment_score', 0.0)
        confidence_score = sentiment_data.get('confidence_score', 0.0)
        
        # Sentiment strength (absolute value)
        metrics['sentiment_strength'] = abs(sentiment_score)
        
        # Sentiment reliability (combination of score and confidence)
        metrics['sentiment_reliability'] = (metrics['sentiment_strength'] * confidence_score)
        
        # Sentiment interpretation
        if sentiment_score > 0.5:
            metrics['interpretation'] = 'Very Positive'
        elif sentiment_score > 0.1:
            metrics['interpretation'] = 'Moderately Positive'
        elif sentiment_score > -0.1:
            metrics['interpretation'] = 'Neutral'
        elif sentiment_score > -0.5:
            metrics['interpretation'] = 'Moderately Negative'
        else:
            metrics['interpretation'] = 'Very Negative'
        
        # Balance analysis
        positive_count = len(sentiment_data.get('key_positive_phrases', []))
        negative_count = len(sentiment_data.get('key_negative_phrases', []))
        
        if positive_count > 0 or negative_count > 0:
            metrics['sentiment_balance'] = positive_count / (positive_count + negative_count)
        else:
            metrics['sentiment_balance'] = 0.5  # Neutral when no phrases
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating sentiment metrics: {str(e)}")
        return {
            'sentiment_strength': 0.0,
            'sentiment_reliability': 0.0,
            'interpretation': 'Unknown',
            'sentiment_balance': 0.5
        }

def validate_sentiment_analysis(sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the quality of sentiment analysis results"""
    validation_result = {
        'valid': True,
        'quality_score': 0.0,
        'issues': [],
        'warnings': []
    }
    
    try:
        # Check required fields
        required_fields = ['sentiment_score', 'sentiment_category', 'confidence_score']
        missing_fields = [field for field in required_fields 
                         if field not in sentiment_data]
        
        if missing_fields:
            validation_result['issues'].append(f"Missing required fields: {missing_fields}")
            validation_result['valid'] = False
        
        # Validate sentiment score range
        sentiment_score = sentiment_data.get('sentiment_score')
        if sentiment_score is not None:
            if not isinstance(sentiment_score, (int, float)) or sentiment_score < -1.0 or sentiment_score > 1.0:
                validation_result['issues'].append("Sentiment score out of valid range (-1.0 to 1.0)")
                validation_result['valid'] = False
        
        # Validate confidence score
        confidence_score = sentiment_data.get('confidence_score')
        if confidence_score is not None:
            if not isinstance(confidence_score, (int, float)) or confidence_score < 0.0 or confidence_score > 1.0:
                validation_result['issues'].append("Confidence score out of valid range (0.0 to 1.0)")
                validation_result['valid'] = False
        
        # Check sentiment category consistency
        sentiment_category = sentiment_data.get('sentiment_category')
        if sentiment_score is not None and sentiment_category:
            if ((sentiment_score > 0.1 and sentiment_category != 'positive') or
                (sentiment_score < -0.1 and sentiment_category != 'negative') or
                (-0.1 <= sentiment_score <= 0.1 and sentiment_category != 'neutral')):
                validation_result['warnings'].append("Sentiment category may not match sentiment score")
        
        # Calculate quality score
        base_score = 100
        if confidence_score is not None and confidence_score < 0.5:
            base_score -= 20  # Low confidence penalty
        
        phrase_count = (len(sentiment_data.get('key_positive_phrases', [])) + 
                       len(sentiment_data.get('key_negative_phrases', [])))
        if phrase_count == 0:
            base_score -= 30  # No supporting phrases penalty
        
        # Deduct for issues and warnings
        base_score -= len(validation_result['issues']) * 25
        base_score -= len(validation_result['warnings']) * 10
        
        validation_result['quality_score'] = max(0, base_score)
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Error validating sentiment analysis: {str(e)}")
        return {
            'valid': False,
            'quality_score': 0.0,
            'issues': [f"Validation error: {str(e)}"],
            'warnings': []
        } 