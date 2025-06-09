"""
Enhanced Document Parser for SEC 10-K Filings
Extracts and processes key sections from SEC filings with improved accuracy
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from bs4 import BeautifulSoup
import pandas as pd
import html

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentParseError(Exception):
    """Custom exception for document parsing errors"""
    pass

class EnhancedSECDocumentParser:
    """Enhanced parser for SEC 10-K filing documents with improved section extraction"""
    
    def __init__(self):
        # Enhanced section patterns for 10-K filings with multiple variations
        self.section_patterns = {
            'business': [
                r'(?i)<(?:p|div)[^>]*>\s*item\s+1\s*[\.\-\s]*business',
                r'(?i)<(?:p|div)[^>]*>\s*part\s+i\s*[\-\s]*item\s+1\s*[\.\-\s]*business',
                r'(?i)item\s+1\s*[\.\-\s]*business(?:\s+overview)?',
                r'(?i)item\s+1[\s\-\.]*business\s+overview',
                r'(?i)business\s+overview',
                r'(?i)description\s+of\s+(?:the\s+)?business'
            ],
            'risk_factors': [
                r'(?i)<(?:p|div)[^>]*>\s*item\s+1a\s*[\.\-\s]*risk\s+factors',
                r'(?i)item\s+1a\s*[\.\-\s]*risk\s+factors',
                r'(?i)item\s+1a[\s\-\.]*risk\s+factors',
                r'(?i)risk\s+factors',
                r'(?i)factors\s+that\s+may\s+affect',
                r'(?i)principal\s+risks\s+and\s+uncertainties'
            ],
            'md_and_a': [
                r'(?i)<(?:p|div)[^>]*>\s*item\s+7\s*[\.\-\s]*management',
                r'(?i)item\s+7\s*[\.\-\s]*management[\s\']s\s+discussion\s+and\s+analysis',
                r'(?i)management[\s\']s\s+discussion\s+and\s+analysis',
                r'(?i)item\s+7[\s\-\.]*management.*discussion.*analysis',
                r'(?i)md&a',
                r'(?i)discussion\s+and\s+analysis\s+of\s+financial\s+condition'
            ],
            'financial_statements': [
                r'(?i)<(?:p|div)[^>]*>\s*item\s+8\s*[\.\-\s]*financial\s+statements',
                r'(?i)item\s+8\s*[\.\-\s]*financial\s+statements',
                r'(?i)item\s+8[\s\-\.]*financial\s+statements',
                r'(?i)consolidated\s+financial\s+statements',
                r'(?i)consolidated\s+balance\s+sheets',
                r'(?i)consolidated\s+statements\s+of\s+operations',
                r'(?i)consolidated\s+statements\s+of\s+income'
            ],
            'controls_procedures': [
                r'(?i)<(?:p|div)[^>]*>\s*item\s+9a\s*[\.\-\s]*controls\s+and\s+procedures',
                r'(?i)item\s+9a\s*[\.\-\s]*controls\s+and\s+procedures',
                r'(?i)item\s+9a[\s\-\.]*controls\s+and\s+procedures',
                r'(?i)controls\s+and\s+procedures'
            ]
        }
        
        # Financial statement table patterns
        self.financial_table_patterns = {
            'income_statement': [
                r'(?i)consolidated\s+statements?\s+of\s+operations',
                r'(?i)consolidated\s+statements?\s+of\s+income',
                r'(?i)statements?\s+of\s+operations',
                r'(?i)statements?\s+of\s+income',
                r'(?i)income\s+statements?'
            ],
            'balance_sheet': [
                r'(?i)consolidated\s+balance\s+sheets?',
                r'(?i)balance\s+sheets?',
                r'(?i)statements?\s+of\s+financial\s+position'
            ],
            'cash_flow': [
                r'(?i)consolidated\s+statements?\s+of\s+cash\s+flows?',
                r'(?i)statements?\s+of\s+cash\s+flows?',
                r'(?i)cash\s+flows?\s+statements?'
            ]
        }

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content with improved handling"""
        if not text:
            return ""
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Collapse multiple line breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Collapse horizontal whitespace
        
        # Remove special characters but keep financial notation and common punctuation
        text = re.sub(r'[^\w\s\.\,\-\$\%\(\)\[\]\/\:\;\|\+\=\<\>\n\']', ' ', text)
        
        # Final cleanup
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def extract_html_content(self, filing_text: str) -> str:
        """Extract and clean HTML content from filing with enhanced processing"""
        try:
            # Check if this is HTML content
            if '<html' not in filing_text.lower() and '<div' not in filing_text.lower():
                # If it's plain text, return as-is with basic cleaning
                return self.clean_text(filing_text)
            
            # Parse as HTML
            soup = BeautifulSoup(filing_text, 'html.parser')
            
            # Remove script, style, and other non-content elements
            for script in soup(["script", "style", "meta", "link"]):
                script.decompose()
            
            # Extract text content
            text = soup.get_text(separator=' ', strip=True)
            
            # If extraction resulted in very short content, try alternative method
            if len(text.strip()) < 10000:
                # Try extracting from specific document sections
                document_sections = soup.find_all(['div', 'section', 'table'], 
                                                class_=re.compile(r'(?i)(document|filing|content)', re.I))
                if document_sections:
                    text = ' '.join([section.get_text(separator=' ', strip=True) 
                                   for section in document_sections])
                
                # If still too short, use raw text
                if len(text.strip()) < 10000:
                    text = filing_text
            
            return self.clean_text(text)
            
        except Exception as e:
            logger.warning(f"HTML parsing failed, using raw text: {str(e)}")
            return self.clean_text(filing_text)

    def find_section_boundaries_enhanced(self, text: str, section_name: str) -> List[Tuple[int, int]]:
        """Find all possible section boundaries for a given section"""
        patterns = self.section_patterns.get(section_name, [])
        
        all_matches = []
        
        # Find all possible section starts
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
            for match in matches:
                all_matches.append(match.start())
        
        # Remove duplicates and sort
        start_positions = sorted(list(set(all_matches)))
        
        if not start_positions:
            return []
        
        # For each start position, find the best end position
        result_boundaries = []
        
        for start_pos in start_positions:
            end_pos = self._find_section_end(text, section_name, start_pos)
            
            # Validate the section content
            section_length = end_pos - start_pos
            if section_length >= 1000:  # Minimum meaningful section length
                result_boundaries.append((start_pos, end_pos))
        
        return result_boundaries

    def _find_section_end(self, text: str, section_name: str, start_pos: int) -> int:
        """Find the end position of a section with improved logic"""
        
        # Define next section patterns based on current section
        next_section_patterns = {
            'business': [
                r'(?i)item\s+1a[\s\.\-]*risk\s+factors',
                r'(?i)item\s+1b[\s\.\-]*unresolved\s+staff\s+comments',
                r'(?i)item\s+2[\s\.\-]*properties'
            ],
            'risk_factors': [
                r'(?i)item\s+1b[\s\.\-]*unresolved\s+staff\s+comments',
                r'(?i)item\s+2[\s\.\-]*properties',
                r'(?i)item\s+3[\s\.\-]*legal\s+proceedings'
            ],
            'md_and_a': [
                r'(?i)item\s+7a[\s\.\-]*quantitative\s+and\s+qualitative\s+disclosures',
                r'(?i)item\s+8[\s\.\-]*financial\s+statements',
                r'(?i)consolidated\s+financial\s+statements'
            ],
            'financial_statements': [
                r'(?i)item\s+9[\s\.\-]*changes\s+in\s+and\s+disagreements',
                r'(?i)item\s+9a[\s\.\-]*controls\s+and\s+procedures',
                r'(?i)part\s+iii'
            ],
            'controls_procedures': [
                r'(?i)item\s+9b[\s\.\-]*other\s+information',
                r'(?i)part\s+iii',
                r'(?i)item\s+10[\s\.\-]*directors'
            ]
        }
        
        # Get patterns for next sections
        end_patterns = next_section_patterns.get(section_name, [])
        
        # Add general end patterns
        end_patterns.extend([
            r'(?i)part\s+iii',
            r'(?i)part\s+iv', 
            r'(?i)signatures?',
            r'(?i)exhibit\s+index',
            r'(?i)index\s+to\s+consolidated\s+financial\s+statements'
        ])
        
        # Search for next section (skip first 2000 chars to avoid false matches)
        search_start = start_pos + 2000
        min_end_pos = len(text)
        
        for pattern in end_patterns:
            matches = list(re.finditer(pattern, text[search_start:], re.IGNORECASE))
            if matches:
                candidate_end = search_start + matches[0].start()
                min_end_pos = min(min_end_pos, candidate_end)
        
        # Ensure minimum section length
        min_section_length = 3000
        if min_end_pos - start_pos < min_section_length:
            min_end_pos = min(len(text), start_pos + min_section_length)
        
        return min_end_pos

    def extract_sections(self, filing_text: str) -> Dict[str, str]:
        """
        Extract major sections from 10-K filing with enhanced accuracy
        
        Args:
            filing_text: Raw filing text
            
        Returns:
            Dictionary with section names as keys and extracted text as values
        """
        try:
            # Clean and extract text content
            clean_text = self.extract_html_content(filing_text)
            logger.info(f"Processing document with {len(clean_text)} characters")
            
            sections = {}
            
            for section_name in self.section_patterns.keys():
                # Find all possible boundaries for this section
                boundaries = self.find_section_boundaries_enhanced(clean_text, section_name)
                
                if boundaries:
                    # Take the longest section (most likely to be complete)
                    best_boundary = max(boundaries, key=lambda x: x[1] - x[0])
                    start_pos, end_pos = best_boundary
                    
                    section_text = clean_text[start_pos:end_pos]
                    
                    # Final validation
                    if len(section_text.strip()) > 500:
                        sections[section_name] = section_text.strip()
                        logger.info(f"Extracted {section_name}: {len(section_text)} characters")
                    else:
                        logger.warning(f"Section {section_name} too short after extraction: {len(section_text)} characters")
                else:
                    logger.warning(f"Section {section_name} not found")
            
            # Validate that we found at least some key sections
            required_sections = ['business', 'risk_factors', 'md_and_a']
            found_sections = [s for s in required_sections if s in sections]
            
            if len(found_sections) < 2:
                raise DocumentParseError(f"Could not find required sections. Found: {list(sections.keys())}")
            
            return sections
            
        except Exception as e:
            logger.error(f"Error extracting sections: {str(e)}")
            raise DocumentParseError(f"Failed to extract sections: {str(e)}")

    def find_financial_tables(self, financial_section: str) -> Dict[str, str]:
        """Find and extract financial statement tables with improved detection"""
        try:
            tables = {}
            
            for table_type, patterns in self.financial_table_patterns.items():
                for pattern in patterns:
                    matches = list(re.finditer(pattern, financial_section, re.IGNORECASE))
                    
                    if matches:
                        # Take the first substantial match
                        for match in matches:
                            start_pos = match.start()
                            
                            # Find end of table (look for next table or section)
                            next_table_pos = len(financial_section)
                            
                            for other_type, other_patterns in self.financial_table_patterns.items():
                                if other_type != table_type:
                                    for other_pattern in other_patterns:
                                        other_matches = list(re.finditer(other_pattern, 
                                                           financial_section[start_pos + 500:], re.IGNORECASE))
                                        if other_matches:
                                            next_table_pos = min(next_table_pos, 
                                                                start_pos + 500 + other_matches[0].start())
                            
                            # Extract table content
                            table_content = financial_section[start_pos:next_table_pos]
                            
                            if len(table_content.strip()) > 1000:  # Substantial table content
                                tables[table_type] = table_content.strip()
                                logger.info(f"Found {table_type} table: {len(table_content)} characters")
                                break  # Found substantial table, move to next type
            
            return tables
            
        except Exception as e:
            logger.error(f"Error finding financial tables: {str(e)}")
            return {}

    def extract_financial_numbers(self, table_text: str) -> Dict[str, List[str]]:
        """Extract financial numbers from table text with enhanced patterns"""
        try:
            # Enhanced patterns for financial numbers
            number_patterns = [
                r'\$\s*[\d,]+\.?\d*',  # Dollar amounts
                r'\(\$\s*[\d,]+\.?\d*\)',  # Negative dollar amounts in parentheses
                r'[\d,]+\.?\d*\s*%',  # Percentages
                r'(?<!\d)[\d,]+\.?\d*(?=\s*(?:million|billion|thousand))',  # Numbers with units
                r'(?<!\d)[\d,]+\.?\d*(?=\s*(?:shares|per\s+share))',  # Share-related numbers
            ]
            
            extracted_numbers = {}
            
            for i, pattern in enumerate(number_patterns):
                matches = re.findall(pattern, table_text, re.IGNORECASE)
                if matches:
                    extracted_numbers[f'pattern_{i}'] = matches[:20]  # Limit to prevent overflow
            
            return extracted_numbers
            
        except Exception as e:
            logger.error(f"Error extracting financial numbers: {str(e)}")
            return {}

    def clean_financial_tables(self, section_text: str) -> Dict[str, Any]:
        """
        Clean and structure financial tables from section text
        
        Args:
            section_text: Raw financial statements section text
            
        Returns:
            Dictionary with structured financial data
        """
        try:
            # Find financial tables
            tables = self.find_financial_tables(section_text)
            
            structured_data = {
                'raw_tables': tables,
                'extracted_numbers': {},
                'table_summaries': {}
            }
            
            # Process each table
            for table_type, table_content in tables.items():
                # Extract numbers from table
                numbers = self.extract_financial_numbers(table_content)
                structured_data['extracted_numbers'][table_type] = numbers
                
                # Create table summary
                summary = {
                    'length': len(table_content),
                    'number_count': sum(len(nums) for nums in numbers.values()),
                    'preview': table_content[:500] + '...' if len(table_content) > 500 else table_content
                }
                structured_data['table_summaries'][table_type] = summary
                
                logger.info(f"Processed {table_type} table with {summary['number_count']} numbers")
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Error cleaning financial tables: {str(e)}")
            raise DocumentParseError(f"Failed to clean financial tables: {str(e)}")

# Maintain backward compatibility with existing API
SECDocumentParser = EnhancedSECDocumentParser

def extract_sections(filing_text: str) -> Dict[str, str]:
    """
    Extract major sections from 10-K filing as specified in PRD
    
    Args:
        filing_text: Raw filing text
        
    Returns:
        Dictionary with section names as keys and extracted text as values
    """
    parser = EnhancedSECDocumentParser()
    return parser.extract_sections(filing_text)

def clean_financial_tables(section_text: str) -> Dict[str, Any]:
    """
    Extract and structure financial tables from section text as specified in PRD
    
    Args:
        section_text: Raw financial statements section text
        
    Returns:
        Dictionary with structured financial data
    """
    parser = EnhancedSECDocumentParser()
    return parser.clean_financial_tables(section_text)

def parse_filing_document(filing_text: str) -> Dict[str, Any]:
    """
    Main function to parse SEC filing document with enhanced processing
    
    Args:
        filing_text: Raw filing text
        
    Returns:
        Dictionary with parsed sections and structured data
    """
    parser = EnhancedSECDocumentParser()
    
    try:
        logger.info("Starting enhanced document parsing")
        
        # Extract major sections
        sections = parser.extract_sections(filing_text)
        
        # Process financial statements if available
        financial_data = {}
        if 'financial_statements' in sections:
            financial_data = parser.clean_financial_tables(sections['financial_statements'])
        
        # Create structured result
        result = {
            'sections': sections,
            'financial_data': financial_data,
            'metadata': {
                'total_sections': len(sections),
                'total_length': len(filing_text),
                'sections_found': list(sections.keys()),
                'financial_tables_found': list(financial_data.get('raw_tables', {}).keys()),
                'section_lengths': {k: len(v) for k, v in sections.items()}
            }
        }
        
        logger.info(f"Successfully parsed document with {len(sections)} sections")
        for section_name, section_text in sections.items():
            logger.info(f"  {section_name}: {len(section_text)} characters")
        
        return result
        
    except Exception as e:
        logger.error(f"Error parsing filing document: {str(e)}")
        raise DocumentParseError(f"Failed to parse document: {str(e)}")

def validate_extracted_data(parsed_data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the quality of extracted data with enhanced checks"""
    validation_result = {
        'valid': True,
        'issues': [],
        'quality_score': 0.0
    }
    
    try:
        sections = parsed_data.get('sections', {})
        required_sections = ['business', 'risk_factors', 'md_and_a']
        
        found_required = [s for s in required_sections if s in sections]
        
        # Check section completeness
        if len(found_required) < len(required_sections):
            missing = [s for s in required_sections if s not in sections]
            validation_result['issues'].append(f"Missing required sections: {missing}")
        
        # Check section quality with higher standards
        for section_name, section_text in sections.items():
            if len(section_text) < 1000:
                validation_result['issues'].append(f"Section {section_name} is too short ({len(section_text)} chars)")
        
        # Calculate quality score
        base_score = (len(found_required) / len(required_sections)) * 100
        
        # Add bonus for good section lengths
        length_bonus = 0
        for section_name, section_text in sections.items():
            if len(section_text) > 5000:
                length_bonus += 5
        
        # Deduct points for issues
        penalty = min(len(validation_result['issues']) * 15, 60)
        validation_result['quality_score'] = min(100, max(0, base_score + length_bonus - penalty))
        
        # Set overall validity
        validation_result['valid'] = (
            len(found_required) >= 2 and 
            validation_result['quality_score'] >= 60
        )
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Error validating extracted data: {str(e)}")
        return {
            'valid': False,
            'issues': [f"Validation error: {str(e)}"],
            'quality_score': 0.0
        } 