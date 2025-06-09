"""
Configuration constants for the LLM-Powered SEC 10-K Analyst
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv('local.env')

# SEC EDGAR API Configuration
SEC_BASE_URL = "https://data.sec.gov"
SEC_TICKERS_BASE_URL = "https://www.sec.gov"  # Company tickers endpoint uses www subdomain
SEC_ARCHIVES_BASE_URL = "https://www.sec.gov"  # Filing archives use www subdomain
SEC_SUBMISSIONS_URL = f"{SEC_BASE_URL}/submissions"
SEC_FILING_URL = f"{SEC_ARCHIVES_BASE_URL}/Archives/edgar/data"
SEC_ARCHIVES_URL = f"{SEC_ARCHIVES_BASE_URL}/Archives/edgar/data"
SEC_API_USER_AGENT = os.getenv("SEC_API_USER_AGENT", "AI-SEC-Analyst admin@example.com")
SEC_RATE_LIMIT_DELAY = 0.1  # 10 requests per second max

# Google Gemini Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_PRIMARY = "gemini-2.0-flash"
GEMINI_MODEL_FALLBACK = "gemini-1.5-flash"
GEMINI_MAX_TOKENS = 4000
GEMINI_TEMPERATURE = 0.1

# Application Configuration
APP_TITLE = "LLM-Powered SEC 10-K Analyst"
APP_PAGE_ICON = "📊"
DEFAULT_FORM_TYPE = "10-K"

# Performance Configuration
ANALYSIS_TIMEOUT = 20  # seconds
MAX_FILING_SIZE = 50 * 1024 * 1024  # 50MB

# Data Schema Constants
REQUIRED_FINANCIAL_FIELDS = [
    "revenue_current", "revenue_previous", 
    "net_income_current", "net_income_previous",
    "total_assets", "total_liabilities"
]

RISK_CATEGORIES = [
    "Market Risk", "Credit Risk", "Operational Risk", 
    "Regulatory Risk", "Technology Risk", "Other"
]

SENTIMENT_THRESHOLDS = {
    "positive": 0.1,
    "negative": -0.1
}

# HTTP Request Configuration
REQUEST_HEADERS = {
    "User-Agent": SEC_API_USER_AGENT,
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate"
}

REQUEST_TIMEOUT = 30  # seconds

def validate_config() -> Dict[str, Any]:
    """Validate required configuration parameters"""
    issues = []
    
    if not GOOGLE_API_KEY:
        issues.append("GOOGLE_API_KEY environment variable not set")
    
    if not SEC_API_USER_AGENT or SEC_API_USER_AGENT == "AI-SEC-Analyst admin@example.com":
        issues.append("SEC_API_USER_AGENT should be set to include your name and email")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues
    } 