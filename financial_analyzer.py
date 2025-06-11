"""
LLM-Powered Financial Analyzer for SEC 10-K Filings
Extracts structured financial metrics and risk factors using OpenAI GPT models
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from config import (
    GOOGLE_API_KEY, GEMINI_MODEL_PRIMARY, GEMINI_MODEL_FALLBACK,
    GEMINI_MAX_TOKENS, GEMINI_TEMPERATURE, REQUIRED_FINANCIAL_FIELDS,
    RISK_CATEGORIES
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialAnalysisError(Exception):
    """Custom exception for financial analysis errors"""
    pass

# Pydantic models for structured output
class FinancialMetrics(BaseModel):
    """Structured financial metrics model"""
    revenue_current: Optional[float] = Field(description="Current year total revenue/net sales in millions")
    revenue_previous: Optional[float] = Field(description="Previous year total revenue/net sales in millions")
    net_income_current: Optional[float] = Field(description="Current year net income in millions")
    net_income_previous: Optional[float] = Field(description="Previous year net income in millions")
    total_assets: Optional[float] = Field(description="Current year total assets in millions")
    total_liabilities: Optional[float] = Field(description="Current year total liabilities in millions")
    operating_cash_flow: Optional[float] = Field(description="Current year operating cash flow in millions")
    
    # Calculated ratios
    profit_margin: Optional[float] = Field(description="Net income / Revenue as percentage")
    debt_to_equity: Optional[float] = Field(description="Total liabilities / (Total assets - Total liabilities)")

class RiskFactor(BaseModel):
    """Individual risk factor model"""
    category: str = Field(description="Risk category classification")
    description: str = Field(description="Brief description of the risk factor")
    severity_score: int = Field(description="Risk severity score from 1-5 (5 being highest)")

class RiskAnalysis(BaseModel):
    """Structured risk analysis model"""
    top_risks: List[RiskFactor] = Field(description="Top 5 most significant risk factors")

class FinancialAnalyzer:
    """LLM-powered financial analyzer for SEC filings"""
    
    def __init__(self):
        self.setup_llm()
        self.setup_prompts()
    
    def setup_llm(self):
        """Initialize Google Gemini LLM with fallback"""
        try:
            if not GOOGLE_API_KEY:
                raise FinancialAnalysisError("Google API key not configured")
            
            # Try primary model first
            self.llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL_PRIMARY,
                temperature=GEMINI_TEMPERATURE,
                max_output_tokens=GEMINI_MAX_TOKENS
            )
            logger.info(f"Successfully initialized {GEMINI_MODEL_PRIMARY}")
            
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
                raise FinancialAnalysisError(f"Failed to initialize LLM: {str(fallback_error)}")
    
    def setup_prompts(self):
        """Initialize prompt templates for different analysis tasks"""
        
        # Enhanced financial metrics extraction prompt
        self.financial_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert financial analyst with deep expertise in SEC 10-K filing analysis and GAAP financial statements.

Your task is to extract precise financial metrics from SEC financial statements sections.

EXTRACTION GUIDELINES:
1. Focus on CONSOLIDATED financial statements (most authoritative)
2. Extract data for the TWO most recent fiscal years
3. Convert all amounts to MILLIONS of dollars (divide by 1,000 if in thousands)
4. Look for these specific statement types:
   - Consolidated Statements of Operations/Income (for revenue, net income)
   - Consolidated Balance Sheets (for assets, liabilities)
   - Consolidated Statements of Cash Flows (for operating cash flow)

REVENUE IDENTIFICATION:
- "Net Sales", "Total Net Sales", "Revenue", "Total Revenue"
- "Product Sales", "Service Revenue" (sum if separate)
- Look for line items clearly labeled as primary revenue

NET INCOME IDENTIFICATION:
- "Net Income", "Net Earnings", "Net Income (Loss)"
- Take the bottom-line figure after all expenses and taxes
- Use "(Loss)" or negative numbers appropriately

BALANCE SHEET IDENTIFICATION:
- "Total Assets", "Total Current Assets + Non-current Assets"
- "Total Liabilities", "Total Current Liabilities + Non-current Liabilities"

CASH FLOW IDENTIFICATION:
- "Cash flows from operating activities", "Net cash provided by operating activities"
- Look in the Cash Flow Statement section

VALIDATION RULES:
- Revenue should be positive and substantial (> $100M for large companies)
- Net income can be positive or negative
- Assets should equal liabilities + equity (basic accounting equation)
- If you see ranges or estimates, use the specific reported figures
- Only extract numbers you can confidently identify

Return ONLY valid JSON with numeric values in millions or null if not found."""),
            
            ("human", """Extract key financial metrics from this financial statements section:

{financial_section}

Required metrics (in millions USD):
- revenue_current: Most recent year total revenue/net sales
- revenue_previous: Prior year total revenue/net sales  
- net_income_current: Most recent year net income
- net_income_previous: Prior year net income
- total_assets: Most recent year total assets
- total_liabilities: Most recent year total liabilities
- operating_cash_flow: Most recent year operating cash flow

IMPORTANT: 
- Only include numbers you can definitively identify
- Convert thousands to millions (divide by 1,000)
- Use null for any metric not clearly found
- Ensure revenue and assets are substantial positive numbers
- Net income can be negative (losses)

Return as JSON only:""")
        ])
        
        # Risk factors analysis prompt
        self.risk_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are an expert risk analyst specializing in corporate risk assessment from SEC 10-K filings.

Your task is to analyze the Risk Factors section and identify the TOP 5 most significant risks.

RISK CATEGORIES to use:
{', '.join(RISK_CATEGORIES)}

INSTRUCTIONS:
1. Read the entire risk factors section carefully
2. Identify and rank the TOP 5 most significant risks
3. Classify each risk into one of the provided categories
4. Assign severity scores from 1-5 (5 = highest risk)
5. Provide concise but meaningful descriptions

Return as valid JSON only, no additional text."""),
            
            ("human", """Analyze the following Risk Factors section and identify the top 5 most significant risks.

Risk Factors Section:
{risk_section}

For each risk, provide:
- category: One of the predefined risk categories
- description: Concise description of the risk (2-3 sentences max)
- severity_score: Integer from 1-5 (5 = highest risk)

Return as JSON with this structure:
{{
  "top_risks": [
    {{
      "category": "Risk Category",
      "description": "Risk description",
      "severity_score": 3
    }}
  ]
}}""")
        ])
    
    def extract_financial_metrics(self, financial_statements: str) -> Dict[str, Any]:
        """
        Extract structured financial metrics from financial statements section
        
        Args:
            financial_statements: Raw financial statements section text
            
        Returns:
            Dictionary with extracted financial metrics
        """
        try:
            if not financial_statements or len(financial_statements.strip()) < 500:
                raise FinancialAnalysisError(f"Financial statements section is too short: {len(financial_statements)} characters")
            
            logger.info(f"Starting financial metrics extraction from {len(financial_statements)} characters")
            
            # Log preview of content for debugging
            preview = financial_statements[:500].replace('\n', ' ')
            logger.debug(f"Financial section preview: {preview}...")
            
            # Check if section contains key financial statement indicators
            financial_indicators = ['consolidated', 'statements', 'income', 'revenue', 'assets', 'liabilities']
            found_indicators = [ind for ind in financial_indicators if ind.lower() in financial_statements.lower()]
            logger.info(f"Found financial indicators: {found_indicators}")
            
            # Truncate if too long (keep first 20000 chars for better coverage)
            if len(financial_statements) > 20000:
                financial_statements = financial_statements[:20000] + "..."
                logger.warning("Truncated financial statements to fit token limits")
            
            # Create the chain with output parser
            parser = JsonOutputParser(pydantic_object=FinancialMetrics)
            chain = self.financial_prompt | self.llm | parser
            
            # Extract metrics
            logger.info("Invoking LLM for financial metrics extraction...")
            result = chain.invoke({"financial_section": financial_statements})
            
            # Validate and post-process the result
            processed_result = self._process_financial_metrics(result)
            
            # Log extraction results
            extracted_fields = [k for k, v in processed_result.items() if v is not None]
            logger.info(f"Successfully extracted financial metrics: {extracted_fields}")
            
            if not extracted_fields:
                logger.warning("No financial metrics were successfully extracted")
            
            return processed_result
            
        except Exception as e:
            logger.error(f"Error extracting financial metrics: {str(e)}")
            # Return empty structure on error
            return {field: None for field in REQUIRED_FINANCIAL_FIELDS}
    
    def analyze_risk_factors(self, risk_section: str) -> Dict[str, Any]:
        """
        Analyze and categorize risk factors from Risk Factors section
        
        Args:
            risk_section: Raw risk factors section text
            
        Returns:
            Dictionary with analyzed risk factors
        """
        try:
            if not risk_section or len(risk_section.strip()) < 100:
                raise FinancialAnalysisError("Risk factors section is too short or empty")
            
            logger.info("Starting risk factors analysis")
            
            # Truncate if too long
            if len(risk_section) > 20000:
                risk_section = risk_section[:20000] + "..."
                logger.warning("Truncated risk section to fit token limits")
            
            # Create the chain with output parser
            parser = JsonOutputParser(pydantic_object=RiskAnalysis)
            chain = self.risk_prompt | self.llm | parser
            
            # Analyze risks
            result = chain.invoke({"risk_section": risk_section})
            
            # Validate and post-process
            processed_result = self._process_risk_analysis(result)
            
            logger.info(f"Successfully analyzed {len(processed_result.get('top_risks', []))} risk factors")
            return processed_result
            
        except Exception as e:
            logger.error(f"Error analyzing risk factors: {str(e)}")
            return {"top_risks": []}
    
    def _process_financial_metrics(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate extracted financial metrics"""
        processed = {}
        
        # Ensure all required fields are present
        for field in REQUIRED_FINANCIAL_FIELDS:
            value = raw_result.get(field)
            
            if value is not None:
                try:
                    # Convert to float and handle string numbers
                    if isinstance(value, str):
                        # Remove common formatting
                        value = re.sub(r'[,$\s]', '', value)
                        value = float(value)
                    processed[field] = float(value)
                except (ValueError, TypeError):
                    processed[field] = None
            else:
                processed[field] = None
        
        # Calculate derived metrics if base metrics are available
        if processed.get('net_income_current') and processed.get('revenue_current'):
            try:
                processed['profit_margin'] = (processed['net_income_current'] / processed['revenue_current']) * 100
            except (ZeroDivisionError, TypeError):
                processed['profit_margin'] = None
        
        if processed.get('total_assets') and processed.get('total_liabilities'):
            try:
                equity = processed['total_assets'] - processed['total_liabilities']
                if equity > 0:
                    processed['debt_to_equity'] = processed['total_liabilities'] / equity
                else:
                    processed['debt_to_equity'] = None
            except (ZeroDivisionError, TypeError):
                processed['debt_to_equity'] = None
        
        return processed
    
    def _process_risk_analysis(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate risk analysis results"""
        processed = {"top_risks": []}
        
        raw_risks = raw_result.get('top_risks', [])
        
        for risk in raw_risks[:5]:  # Ensure max 5 risks
            if isinstance(risk, dict):
                processed_risk = {
                    'category': self._validate_risk_category(risk.get('category', 'Other')),
                    'description': str(risk.get('description', '')).strip()[:500],  # Limit description length
                    'severity_score': self._validate_severity_score(risk.get('severity_score', 3))
                }
                
                if processed_risk['description']:  # Only add if description exists
                    processed['top_risks'].append(processed_risk)
        
        return processed
    
    def _validate_risk_category(self, category: str) -> str:
        """Validate and normalize risk category"""
        if not category or not isinstance(category, str):
            return "Other"
        
        category = category.strip()
        
        # Check if category matches any predefined categories
        for valid_category in RISK_CATEGORIES:
            if category.lower() in valid_category.lower() or valid_category.lower() in category.lower():
                return valid_category
        
        return "Other"
    
    def _validate_severity_score(self, score: Union[int, str, float]) -> int:
        """Validate and normalize severity score"""
        try:
            score = int(float(score))
            return max(1, min(5, score))  # Clamp between 1 and 5
        except (ValueError, TypeError):
            return 3  # Default to medium severity

def extract_financial_metrics(financial_statements: str) -> Dict[str, Any]:
    """
    Main function to extract financial metrics from financial statements
    
    Args:
        financial_statements: Raw financial statements section text
        
    Returns:
        Dictionary with extracted financial metrics
    """
    analyzer = FinancialAnalyzer()
    return analyzer.extract_financial_metrics(financial_statements)

def analyze_risk_factors(risk_section: str) -> Dict[str, Any]:
    """
    Main function to analyze risk factors from Risk Factors section
    
    Args:
        risk_section: Raw risk factors section text
        
    Returns:
        Dictionary with analyzed risk factors
    """
    analyzer = FinancialAnalyzer()
    return analyzer.analyze_risk_factors(risk_section)

def analyze_financial_performance(current_metrics: Dict[str, Any], 
                                previous_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Analyze financial performance and trends
    
    Args:
        current_metrics: Current period financial metrics
        previous_metrics: Previous period metrics for comparison (optional)
        
    Returns:
        Dictionary with performance analysis
    """
    try:
        analysis = {
            'performance_indicators': {},
            'growth_rates': {},
            'financial_health': {},
            'key_insights': []
        }
        
        # Performance indicators
        revenue_current = current_metrics.get('revenue_current')
        net_income_current = current_metrics.get('net_income_current')
        profit_margin = current_metrics.get('profit_margin')
        debt_to_equity = current_metrics.get('debt_to_equity')
        
        if profit_margin is not None:
            analysis['performance_indicators']['profitability'] = (
                'Strong' if profit_margin > 15 else 
                'Moderate' if profit_margin > 5 else 'Weak'
            )
        
        if debt_to_equity is not None:
            analysis['performance_indicators']['leverage'] = (
                'Conservative' if debt_to_equity < 0.5 else
                'Moderate' if debt_to_equity < 1.0 else 'High'
            )
        
        # Growth rates (if previous period data available)
        if previous_metrics:
            revenue_previous = previous_metrics.get('revenue_previous')
            net_income_previous = previous_metrics.get('net_income_previous')
            
            if revenue_current and revenue_previous and revenue_previous > 0:
                revenue_growth = ((revenue_current - revenue_previous) / revenue_previous) * 100
                analysis['growth_rates']['revenue_growth'] = revenue_growth
                
                if revenue_growth > 10:
                    analysis['key_insights'].append("Strong revenue growth observed")
                elif revenue_growth < -5:
                    analysis['key_insights'].append("Revenue decline indicates potential challenges")
            
            if net_income_current and net_income_previous and net_income_previous > 0:
                income_growth = ((net_income_current - net_income_previous) / net_income_previous) * 100
                analysis['growth_rates']['income_growth'] = income_growth
        
        # Financial health indicators
        if revenue_current and revenue_current > 1000:  # > $1B revenue
            analysis['financial_health']['size'] = 'Large Cap'
        elif revenue_current and revenue_current > 100:  # > $100M revenue
            analysis['financial_health']['size'] = 'Mid Cap'
        else:
            analysis['financial_health']['size'] = 'Small Cap'
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing financial performance: {str(e)}")
        return {
            'performance_indicators': {},
            'growth_rates': {},
            'financial_health': {},
            'key_insights': [f"Analysis error: {str(e)}"]
        }

def validate_financial_data(financial_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the quality and completeness of extracted financial data"""
    validation_result = {
        'valid': True,
        'completeness_score': 0.0,
        'issues': [],
        'warnings': []
    }
    
    try:
        # Check completeness
        total_fields = len(REQUIRED_FINANCIAL_FIELDS)
        completed_fields = sum(1 for field in REQUIRED_FINANCIAL_FIELDS 
                             if financial_data.get(field) is not None)
        
        validation_result['completeness_score'] = (completed_fields / total_fields) * 100
        
        # Check for critical missing fields
        critical_fields = ['revenue_current', 'net_income_current']
        missing_critical = [field for field in critical_fields 
                          if financial_data.get(field) is None]
        
        if missing_critical:
            validation_result['issues'].append(f"Missing critical fields: {missing_critical}")
            validation_result['valid'] = False
        
        # Validation checks
        revenue_current = financial_data.get('revenue_current')
        net_income_current = financial_data.get('net_income_current')
        total_assets = financial_data.get('total_assets')
        
        # Sanity checks
        if revenue_current is not None and revenue_current < 0:
            validation_result['warnings'].append("Revenue is negative - please verify")
        
        if total_assets is not None and total_assets < 0:
            validation_result['issues'].append("Total assets cannot be negative")
            validation_result['valid'] = False
        
        if (revenue_current is not None and net_income_current is not None and 
            abs(net_income_current) > revenue_current * 2):
            validation_result['warnings'].append("Net income seems disproportionate to revenue")
        
        # Overall validity
        if validation_result['completeness_score'] < 40:
            validation_result['valid'] = False
            validation_result['issues'].append("Insufficient data extracted")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Error validating financial data: {str(e)}")
        return {
            'valid': False,
            'completeness_score': 0.0,
            'issues': [f"Validation error: {str(e)}"],
            'warnings': []
        } 