"""
SEC EDGAR API Client for fetching 10-K filings
"""

import time
import requests
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import re

from config import (
    SEC_BASE_URL, SEC_TICKERS_BASE_URL, SEC_ARCHIVES_BASE_URL, SEC_SUBMISSIONS_URL, SEC_ARCHIVES_URL,
    REQUEST_HEADERS, REQUEST_TIMEOUT, SEC_RATE_LIMIT_DELAY,
    MAX_FILING_SIZE, DEFAULT_FORM_TYPE
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SECAPIError(Exception):
    """Custom exception for SEC API errors"""
    pass

class SECEdgarClient:
    """Client for interacting with SEC EDGAR API"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(REQUEST_HEADERS)
        self.last_request_time = 0
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting to comply with SEC requirements"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < SEC_RATE_LIMIT_DELAY:
            time.sleep(SEC_RATE_LIMIT_DELAY - time_since_last)
        
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, timeout: int = REQUEST_TIMEOUT) -> requests.Response:
        """Make HTTP request with rate limiting and error handling"""
        self._rate_limit()
        
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for URL {url}: {str(e)}")
            raise SECAPIError(f"Failed to fetch data from SEC API: {str(e)}")
    
    def ticker_to_cik(self, ticker: str) -> str:
        """Convert stock ticker to CIK (Central Index Key)"""
        ticker = ticker.upper().strip()
        
        # First try to find the ticker in the company tickers JSON
        try:
            tickers_url = f"{SEC_TICKERS_BASE_URL}/files/company_tickers.json"
            response = self._make_request(tickers_url)
            tickers_data = response.json()
            
            # Search for ticker in the data
            for entry in tickers_data.values():
                if entry.get('ticker', '').upper() == ticker:
                    cik = str(entry.get('cik_str', ''))
                    logger.info(f"Found CIK {cik} for ticker {ticker}")
                    return cik.zfill(10)  # Pad with zeros to 10 digits
            
            raise SECAPIError(f"Ticker '{ticker}' not found in SEC database")
            
        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            logger.error(f"Error converting ticker {ticker} to CIK: {str(e)}")
            raise SECAPIError(f"Failed to convert ticker '{ticker}' to CIK: {str(e)}")
    
    def get_company_submissions(self, cik: str) -> Dict[str, Any]:
        """Get company submission data from SEC API"""
        try:
            # Ensure CIK is properly formatted (10 digits with leading zeros)
            cik_padded = cik.zfill(10)
            submissions_url = f"{SEC_SUBMISSIONS_URL}/CIK{cik_padded}.json"
            
            logger.info(f"Fetching submissions for CIK {cik_padded}")
            response = self._make_request(submissions_url)
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching submissions for CIK {cik}: {str(e)}")
            raise SECAPIError(f"Failed to fetch company submissions: {str(e)}")
    
    def find_latest_filing(self, submissions_data: Dict[str, Any], 
                          form_type: str = DEFAULT_FORM_TYPE) -> Optional[Dict[str, Any]]:
        """Find the latest filing of specified form type"""
        try:
            recent_filings = submissions_data.get('filings', {}).get('recent', {})
            
            if not recent_filings:
                raise SECAPIError("No recent filings found")
            
            forms = recent_filings.get('form', [])
            filing_dates = recent_filings.get('filingDate', [])
            accession_numbers = recent_filings.get('accessionNumber', [])
            
            # Find the most recent filing of the specified type
            latest_filing = None
            latest_date = None
            
            for i, form in enumerate(forms):
                if form == form_type:
                    filing_date = filing_dates[i]
                    if latest_date is None or filing_date > latest_date:
                        latest_date = filing_date
                        latest_filing = {
                            'form': form,
                            'filingDate': filing_date,
                            'accessionNumber': accession_numbers[i]
                        }
            
            if latest_filing is None:
                raise SECAPIError(f"No {form_type} filings found")
            
            logger.info(f"Found latest {form_type} filing: {latest_filing['filingDate']}")
            return latest_filing
            
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing submissions data: {str(e)}")
            raise SECAPIError(f"Failed to parse submissions data: {str(e)}")
    
    def fetch_filing_text(self, cik: str, accession_number: str) -> str:
        """Fetch the actual filing text from SEC archives"""
        try:
            # Format accession number (remove hyphens)
            accession_clean = accession_number.replace('-', '')
            
            # Construct filing URL
            filing_url = f"{SEC_ARCHIVES_URL}/{cik}/{accession_clean}/{accession_number}.txt"
            
            logger.info(f"Fetching filing text from {filing_url}")
            response = self._make_request(filing_url)
            
            # Check file size
            content_length = len(response.content)
            if content_length > MAX_FILING_SIZE:
                raise SECAPIError(f"Filing too large: {content_length} bytes")
            
            # Return the text content
            return response.text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching filing text: {str(e)}")
            raise SECAPIError(f"Failed to fetch filing text: {str(e)}")

def fetch_company_filing(ticker: str, form_type: str = DEFAULT_FORM_TYPE) -> Dict[str, Any]:
    """
    Main function to fetch latest SEC filing for a given ticker
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        form_type: Type of SEC form to fetch (default: '10-K')
    
    Returns:
        Dictionary containing filing data and metadata
    """
    client = SECEdgarClient()
    
    try:
        # Step 1: Convert ticker to CIK
        logger.info(f"Starting filing fetch for ticker: {ticker}")
        cik = client.ticker_to_cik(ticker)
        
        # Step 2: Get company submissions
        submissions_data = client.get_company_submissions(cik)
        company_name = submissions_data.get('name', 'Unknown Company')
        
        # Step 3: Find latest filing
        latest_filing = client.find_latest_filing(submissions_data, form_type)
        
        if not latest_filing:
            raise SECAPIError(f"No {form_type} filing found for {ticker}")
        
        # Step 4: Fetch filing text
        filing_text = client.fetch_filing_text(cik, latest_filing['accessionNumber'])
        
        # Step 5: Return structured data
        result = {
            'ticker': ticker.upper(),
            'company_name': company_name,
            'cik': cik,
            'form_type': form_type,
            'filing_date': latest_filing['filingDate'],
            'accession_number': latest_filing['accessionNumber'],
            'filing_text': filing_text,
            'filing_size': len(filing_text),
            'fetch_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Successfully fetched {form_type} filing for {ticker} ({company_name})")
        return result
        
    except SECAPIError as e:
        logger.error(f"SEC API error for {ticker}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching filing for {ticker}: {str(e)}")
        raise SECAPIError(f"Unexpected error: {str(e)}")

def validate_ticker(ticker: str) -> bool:
    """Validate ticker format"""
    if not ticker or not isinstance(ticker, str):
        return False
    
    # Basic validation - alphanumeric, 1-5 characters
    ticker_clean = ticker.strip().upper()
    return bool(re.match(r'^[A-Z]{1,5}$', ticker_clean))

def get_company_info(ticker: str) -> Dict[str, Any]:
    """Get basic company information without fetching full filing"""
    if not validate_ticker(ticker):
        raise SECAPIError(f"Invalid ticker format: {ticker}")
    
    client = SECEdgarClient()
    
    try:
        cik = client.ticker_to_cik(ticker)
        submissions_data = client.get_company_submissions(cik)
        
        return {
            'ticker': ticker.upper(),
            'company_name': submissions_data.get('name', 'Unknown Company'),
            'cik': cik,
            'sic': submissions_data.get('sic', ''),
            'sic_description': submissions_data.get('sicDescription', ''),
            'business_address': submissions_data.get('addresses', {}).get('business', {}),
            'phone': submissions_data.get('phone', ''),
            'website': submissions_data.get('website', ''),
            'investor_website': submissions_data.get('investorWebsite', ''),
            'entity_type': submissions_data.get('entityType', ''),
            'fiscal_year_end': submissions_data.get('fiscalYearEnd', '')
        }
        
    except Exception as e:
        logger.error(f"Error getting company info for {ticker}: {str(e)}")
        raise SECAPIError(f"Failed to get company info: {str(e)}") 