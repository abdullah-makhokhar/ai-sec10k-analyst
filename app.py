"""
LLM-Powered SEC 10-K Financial Document Analyst
Main Streamlit Application

Provides web interface for automated analysis of SEC 10-K filings using AI/LLM technology.
"""

import streamlit as st
import traceback
import time
from datetime import datetime
from typing import Dict, Any, Optional

# Import our custom modules
from sec_edgar_client import fetch_company_filing, get_company_info, validate_ticker
from document_parser import parse_filing_document, validate_extracted_data
from financial_analyzer import extract_financial_metrics, analyze_risk_factors, validate_financial_data
from sentiment_analyzer import analyze_management_sentiment, validate_sentiment_analysis
from results_formatter import format_analysis_results, generate_executive_summary

# Import visualization libraries
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Import configuration
from config import APP_TITLE, APP_PAGE_ICON, DEFAULT_FORM_TYPE

def main():
    """Main application function"""
    
    # Page configuration
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # App header
    st.title(APP_TITLE)
    st.markdown("*Transform hours of manual SEC filing analysis into actionable insights in seconds*")
    
    # Sidebar for input and controls
    with st.sidebar:
        st.header("Analysis Configuration")
        
        # Ticker input
        ticker = st.text_input(
            "Stock Ticker Symbol",
            value="AAPL",
            help="Enter the stock ticker symbol (e.g., AAPL, MSFT, GOOGL)",
            max_chars=10
        ).upper().strip()
        
        # Form type selection
        form_type = st.selectbox(
            "SEC Form Type",
            options=["10-K", "10-Q"],
            index=0,
            help="Select the type of SEC filing to analyze"
        )
        
        # Analysis options
        st.subheader("Analysis Options")
        
        include_sentiment = st.checkbox(
            "Include Sentiment Analysis",
            value=True,
            help="Analyze management tone and sentiment from MD&A section"
        )
        
        include_risk_analysis = st.checkbox(
            "Include Risk Factor Analysis", 
            value=True,
            help="Extract and categorize top risk factors"
        )
        
        # Analysis button
        analyze_button = st.button(
            "🔍 Analyze Filing",
            type="primary",
            use_container_width=True,
            disabled=not ticker
        )
        
        # Quick company info display
        if ticker and validate_ticker(ticker):
            try:
                with st.spinner("Fetching company info..."):
                    company_info = get_company_info(ticker)
                    st.success(f"**{company_info.get('company_name', 'Unknown')}**")
                    st.caption(f"CIK: {company_info.get('cik', 'Unknown')}")
            except Exception as e:
                st.warning(f"Could not fetch company info: {str(e)}")
    
    # Main analysis workflow
    if analyze_button:
        if not ticker:
            st.error("Please enter a valid ticker symbol")
            return
        
        if not validate_ticker(ticker):
            st.error(f"Ticker '{ticker}' not found in SEC database. Please verify the symbol.")
            return
        
        # Start analysis
        run_analysis(ticker, form_type, include_sentiment, include_risk_analysis)
    
    # Show demo/instructions when no analysis is running
    else:
        show_app_instructions()

def run_analysis(ticker: str, form_type: str, include_sentiment: bool, include_risk_analysis: bool):
    """Run the complete analysis workflow"""
    
    analysis_data = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()
    
    try:
        # Step 1: Fetch SEC Filing Data
        status_text.text("📥 Fetching SEC filing data...")
        progress_bar.progress(10)
        
        filing_data = fetch_company_filing(ticker, form_type)
        analysis_data['filing_data'] = filing_data
        
        st.success(f"✅ Successfully fetched {form_type} filing for {filing_data['company_name']}")
        progress_bar.progress(25)
        
        # Step 2: Parse Document Sections
        status_text.text("📄 Parsing document sections...")
        
        parsed_data = parse_filing_document(filing_data['filing_text'])
        analysis_data['parsed_data'] = parsed_data
        
        # Validate parsing quality
        validation_result = validate_extracted_data(parsed_data)
        if not validation_result['valid']:
            st.warning(f"⚠️ Document parsing issues detected: {', '.join(validation_result['issues'])}")
        
        st.success(f"✅ Extracted {len(parsed_data['sections'])} document sections")
        progress_bar.progress(40)
        
        # Step 3: Financial Metrics Analysis
        status_text.text("💰 Analyzing financial metrics...")
        
        financial_statements = parsed_data['sections'].get('financial_statements', '')
        if financial_statements:
            financial_metrics = extract_financial_metrics(financial_statements)
            analysis_data['financial_metrics'] = financial_metrics
            
            # Validate financial data
            financial_validation = validate_financial_data(financial_metrics)
            if financial_validation['completeness_score'] < 50:
                st.warning(f"⚠️ Limited financial data extracted ({financial_validation['completeness_score']:.0f}% complete)")
            else:
                st.success(f"✅ Financial metrics extracted ({financial_validation['completeness_score']:.0f}% complete)")
        else:
            st.warning("⚠️ Financial statements section not found")
            analysis_data['financial_metrics'] = {}
        
        progress_bar.progress(60)
        
        # Step 4: Risk Factor Analysis (optional)
        if include_risk_analysis:
            status_text.text("⚠️ Analyzing risk factors...")
            
            risk_section = parsed_data['sections'].get('risk_factors', '')
            if risk_section:
                risk_analysis = analyze_risk_factors(risk_section)
                analysis_data['risk_analysis'] = risk_analysis
                
                risk_count = len(risk_analysis.get('top_risks', []))
                st.success(f"✅ Identified {risk_count} key risk factors")
            else:
                st.warning("⚠️ Risk factors section not found")
                analysis_data['risk_analysis'] = {'top_risks': []}
        else:
            analysis_data['risk_analysis'] = {'top_risks': []}
        
        progress_bar.progress(80)
        
        # Step 5: Sentiment Analysis (optional)
        if include_sentiment:
            status_text.text("😊 Analyzing management sentiment...")
            
            mda_section = parsed_data['sections'].get('md_and_a', '')
            if mda_section:
                sentiment_analysis = analyze_management_sentiment(mda_section)
                analysis_data['sentiment_analysis'] = sentiment_analysis
                
                # Validate sentiment analysis
                sentiment_validation = validate_sentiment_analysis(sentiment_analysis)
                confidence = sentiment_analysis.get('confidence_score', 0.0)
                
                if confidence > 0.7:
                    st.success(f"✅ Sentiment analysis complete (confidence: {confidence:.1%})")
                else:
                    st.warning(f"⚠️ Sentiment analysis complete with low confidence ({confidence:.1%})")
            else:
                st.warning("⚠️ MD&A section not found for sentiment analysis")
                analysis_data['sentiment_analysis'] = {}
        else:
            analysis_data['sentiment_analysis'] = {}
        
        progress_bar.progress(90)
        
        # Step 6: Format and Display Results
        status_text.text("📊 Formatting results...")
        
        formatted_results = format_analysis_results(analysis_data)
        
        progress_bar.progress(100)
        elapsed_time = time.time() - start_time
        
        status_text.text(f"✅ Analysis complete in {elapsed_time:.1f} seconds!")
        
        # Display results
        display_analysis_results(formatted_results)
        
    except Exception as e:
        # Handle specific error types as per PRD
        error_msg = str(e)
        if "SEC" in error_msg or "API" in error_msg:
            st.error("Unable to retrieve SEC filing data. Please try again later.")
        elif "Google" in error_msg or "Gemini" in error_msg or "LLM" in error_msg:
            st.error("AI analysis temporarily unavailable. Raw filing data is still accessible.")
        elif "parse" in error_msg.lower() or "format" in error_msg.lower():
            st.error("Unable to process filing format. Manual review may be required.")
        else:
            st.error(f"❌ Analysis failed: {error_msg}")
            st.error("Please try again or contact support if the issue persists.")
        
        # Show error details in expander for debugging
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
    
    finally:
        # Clean up progress indicators
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()

def display_analysis_results(results: Dict[str, Any]):
    """Display formatted analysis results in tabs"""
    
    # Create tabs for different result sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Executive Summary", 
        "💰 Financial Metrics", 
        "⚠️ Risk Analysis", 
        "😊 Sentiment Analysis", 
        "📋 Raw Data"
    ])
    
    with tab1:
        display_executive_summary(results)
    
    with tab2:
        display_financial_metrics(results)
    
    with tab3:
        display_risk_analysis(results)
    
    with tab4:
        display_sentiment_analysis(results)
    
    with tab5:
        display_raw_data(results)

def display_executive_summary(results: Dict[str, Any]):
    """Display executive summary tab"""
    
    st.header("Executive Summary")
    
    # Company header
    metadata = results.get('metadata', {})
    company_info = results.get('company_info', {}).get('basic_info', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Company", company_info.get('company_name', 'Unknown'))
        st.metric("Ticker", metadata.get('ticker', 'Unknown'))
    
    with col2:
        st.metric("Filing Date", metadata.get('filing_date', 'Unknown'))
        st.metric("Form Type", metadata.get('form_type', 'Unknown'))
    
    with col3:
        quality_score = results.get('data_quality', {}).get('overall_quality_score', 0)
        st.metric("Analysis Quality", f"{quality_score:.0f}%")
        
        recommendation = results.get('executive_summary', {}).get('recommendation_level', 'Neutral')
        st.metric("Overall Assessment", recommendation)
    
    # Executive summary text
    summary_text = results.get('executive_summary', {}).get('summary_text', '')
    if summary_text:
        st.subheader("Summary")
        st.write(summary_text)
    
    # Key highlights and concerns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Key Highlights")
        highlights = results.get('executive_summary', {}).get('key_highlights', [])
        for highlight in highlights:
            st.success(f"• {highlight}")
        
        if not highlights:
            st.info("No significant highlights identified")
    
    with col2:
        st.subheader("⚠️ Key Concerns")
        concerns = results.get('executive_summary', {}).get('key_concerns', [])
        for concern in concerns:
            st.warning(f"• {concern}")
        
        if not concerns:
            st.info("No major concerns identified")
    
    # Investment considerations
    st.subheader("💡 Investment Considerations")
    considerations = results.get('executive_summary', {}).get('investment_considerations', [])
    for consideration in considerations:
        st.info(f"• {consideration}")

def display_financial_metrics(results: Dict[str, Any]):
    """Display financial metrics tab"""
    
    st.header("Financial Metrics Analysis")
    
    financial_metrics = results.get('financial_metrics', {})
    
    # Revenue metrics
    st.subheader("📊 Revenue Performance")
    revenue_data = financial_metrics.get('revenue', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        current_revenue = revenue_data.get('current')
        if current_revenue is not None:
            st.metric("Current Revenue", f"${current_revenue:,.0f}M")
        else:
            st.metric("Current Revenue", "Not Available")
    
    with col2:
        previous_revenue = revenue_data.get('previous')
        if previous_revenue is not None:
            st.metric("Previous Revenue", f"${previous_revenue:,.0f}M")
        else:
            st.metric("Previous Revenue", "Not Available")
    
    with col3:
        growth_rate = revenue_data.get('growth_rate')
        if growth_rate is not None:
            st.metric("Revenue Growth", f"{growth_rate:+.1f}%")
        else:
            st.metric("Revenue Growth", "Not Available")
    
    # Profitability metrics
    st.subheader("💰 Profitability Analysis")
    profitability_data = financial_metrics.get('profitability', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        net_income = profitability_data.get('net_income_current')
        if net_income is not None:
            st.metric("Net Income", f"${net_income:,.0f}M")
        else:
            st.metric("Net Income", "Not Available")
    
    with col2:
        profit_margin = profitability_data.get('profit_margin')
        if profit_margin is not None:
            st.metric("Profit Margin", f"{profit_margin:.1f}%")
        else:
            st.metric("Profit Margin", "Not Available")
    
    with col3:
        assessment = profitability_data.get('profitability_assessment', 'Unknown')
        st.metric("Profitability Level", assessment)
    
    # Financial position
    st.subheader("🏦 Financial Position")
    position_data = financial_metrics.get('financial_position', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_assets = position_data.get('total_assets')
        if total_assets is not None:
            st.metric("Total Assets", f"${total_assets:,.0f}M")
        else:
            st.metric("Total Assets", "Not Available")
    
    with col2:
        total_liabilities = position_data.get('total_liabilities')
        if total_liabilities is not None:
            st.metric("Total Liabilities", f"${total_liabilities:,.0f}M")
        else:
            st.metric("Total Liabilities", "Not Available")
    
    with col3:
        leverage = position_data.get('financial_leverage', 'Unknown')
        st.metric("Leverage Level", leverage)
    
    # Create financial charts if data is available
    create_financial_charts(financial_metrics)

def create_financial_charts(financial_metrics: Dict[str, Any]):
    """Create financial performance charts"""
    
    # Revenue trend chart
    revenue_data = financial_metrics.get('revenue', {})
    current_revenue = revenue_data.get('current')
    previous_revenue = revenue_data.get('previous')
    
    if current_revenue is not None and previous_revenue is not None:
        st.subheader("📈 Revenue Trend")
        
        revenue_df = pd.DataFrame({
            'Period': ['Previous Year', 'Current Year'],
            'Revenue': [previous_revenue, current_revenue]
        })
        
        fig = px.bar(revenue_df, x='Period', y='Revenue', 
                    title='Revenue Comparison ($ Millions)')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def display_risk_analysis(results: Dict[str, Any]):
    """Display risk analysis tab"""
    
    st.header("Risk Factor Analysis")
    
    risk_analysis = results.get('risk_analysis', {})
    risk_summary = risk_analysis.get('risk_summary', {})
    top_risks = risk_analysis.get('top_risks', [])
    
    # Risk summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_risks = risk_summary.get('total_risks_identified', 0)
        st.metric("Risks Identified", total_risks)
    
    with col2:
        avg_severity = risk_summary.get('average_severity', 0)
        st.metric("Average Severity", f"{avg_severity:.1f}/5")
    
    with col3:
        overall_level = risk_summary.get('overall_risk_level', 'Unknown')
        st.metric("Overall Risk Level", overall_level)
    
    # Top risks display
    if top_risks:
        st.subheader("🎯 Top Risk Factors")
        
        for risk in top_risks:
            severity = risk.get('severity_score', 3)
            severity_level = risk.get('severity_level', 'Medium')
            category = risk.get('category', 'Unknown')
            description = risk.get('description', 'No description available')
            rank = risk.get('rank', 0)
            
            # Color-code by severity
            if severity >= 4:
                st.error(f"**#{rank} - {category}** (Severity: {severity_level})")
            elif severity >= 3:
                st.warning(f"**#{rank} - {category}** (Severity: {severity_level})")
            else:
                st.info(f"**#{rank} - {category}** (Severity: {severity_level})")
            
            st.write(description)
            st.write("---")
        
        # Risk distribution chart
        create_risk_distribution_chart(risk_analysis)
    else:
        st.info("No risk factors were identified in the analysis.")

def create_risk_distribution_chart(risk_analysis: Dict[str, Any]):
    """Create risk distribution visualization"""
    
    risk_distribution = risk_analysis.get('risk_distribution', {})
    
    if risk_distribution:
        st.subheader("📊 Risk Category Distribution")
        
        categories = list(risk_distribution.keys())
        counts = list(risk_distribution.values())
        
        fig = px.pie(values=counts, names=categories, 
                    title="Distribution of Risk Factors by Category")
        st.plotly_chart(fig, use_container_width=True)

def display_sentiment_analysis(results: Dict[str, Any]):
    """Display sentiment analysis tab"""
    
    st.header("Management Sentiment Analysis")
    
    sentiment_analysis = results.get('sentiment_analysis', {})
    overall_sentiment = sentiment_analysis.get('overall_sentiment', {})
    
    # Sentiment overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment_score = overall_sentiment.get('score', 0.0)
        st.metric("Sentiment Score", f"{sentiment_score:+.2f}")
    
    with col2:
        sentiment_category = overall_sentiment.get('category', 'neutral')
        st.metric("Sentiment Category", sentiment_category.title())
    
    with col3:
        confidence = overall_sentiment.get('confidence', 0.0)
        st.metric("Analysis Confidence", f"{confidence:.1%}")
    
    # Sentiment interpretation
    interpretation = overall_sentiment.get('interpretation', 'Unknown')
    st.subheader(f"Interpretation: {interpretation}")
    
    # Sentiment indicators
    sentiment_indicators = sentiment_analysis.get('sentiment_indicators', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("😊 Positive Signals")
        positive_signals = sentiment_indicators.get('positive_signals', [])
        for signal in positive_signals:
            st.success(f"• {signal}")
        
        if not positive_signals:
            st.info("No strong positive signals detected")
    
    with col2:
        st.subheader("😟 Negative Signals")  
        negative_signals = sentiment_indicators.get('negative_signals', [])
        for signal in negative_signals:
            st.error(f"• {signal}")
        
        if not negative_signals:
            st.info("No strong negative signals detected")
    
    # Key themes
    st.subheader("🎯 Key Themes")
    key_themes = sentiment_indicators.get('key_themes', [])
    if key_themes:
        theme_cols = st.columns(len(key_themes))
        for i, theme in enumerate(key_themes):
            with theme_cols[i]:
                st.info(theme.title())
    else:
        st.info("No key themes identified")
    
    # Management outlook
    st.subheader("🔮 Management Outlook")
    management_outlook = sentiment_analysis.get('management_outlook', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        optimism_level = management_outlook.get('optimism_level', 'Unknown')
        st.metric("Optimism Level", optimism_level)
    
    with col2:
        confidence_level = management_outlook.get('confidence_level', 'Unknown')
        st.metric("Management Confidence", confidence_level)

def display_raw_data(results: Dict[str, Any]):
    """Display raw data tab"""
    
    st.header("Raw Analysis Data")
    
    st.subheader("📋 Complete Analysis Results")
    st.json(results)
    
    # Data quality metrics
    st.subheader("📊 Data Quality Assessment")
    data_quality = results.get('data_quality', {})
    
    quality_metrics = [
        ("Parsing Quality", data_quality.get('parsing_quality', 0)),
        ("Financial Data Completeness", data_quality.get('financial_data_completeness', 0)),
        ("Risk Analysis Quality", data_quality.get('risk_analysis_quality', 0)),
        ("Sentiment Analysis Quality", data_quality.get('sentiment_analysis_quality', 0))
    ]
    
    for metric_name, score in quality_metrics:
        st.metric(metric_name, f"{score:.0f}%")
    
    # Export option
    st.subheader("💾 Export Data")
    
    if st.button("📥 Download Analysis Results (JSON)"):
        # Convert results to JSON string
        import json
        json_str = json.dumps(results, indent=2, default=str)
        
        # Create download button
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ticker = results.get('metadata', {}).get('ticker', 'unknown')
        filename = f"sec_analysis_{ticker}_{timestamp}.json"
        
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name=filename,
            mime="application/json"
        )

def show_app_instructions():
    """Show app instructions and demo"""
    
    st.header("🚀 Get Started")
    
    st.markdown("""
    **This AI-powered tool analyzes SEC 10-K filings to extract key insights in seconds:**
    
    1. **Enter a stock ticker** (e.g., AAPL, MSFT, GOOGL) in the sidebar
    2. **Choose analysis options** (sentiment analysis, risk factors)
    3. **Click "Analyze Filing"** to start the AI analysis
    4. **View results** across multiple tabs with charts and insights
    
    ### ✨ What You'll Get:
    - **Executive Summary**: Key highlights, concerns, and investment considerations
    - **Financial Metrics**: Revenue, profitability, financial position with growth analysis
    - **Risk Analysis**: Top 5 risk factors categorized and ranked by severity
    - **Sentiment Analysis**: Management tone and outlook from MD&A sections
    """)
    
    # Demo companies
    st.subheader("💡 Try These Popular Stocks:")
    
    demo_companies = [
        ("AAPL", "Apple Inc.", "Technology"),
        ("MSFT", "Microsoft Corporation", "Technology"), 
        ("GOOGL", "Alphabet Inc.", "Technology"),
        ("AMZN", "Amazon.com Inc.", "Consumer Discretionary"),
        ("TSLA", "Tesla Inc.", "Consumer Discretionary")
    ]
    
    cols = st.columns(len(demo_companies))
    for i, (ticker, name, sector) in enumerate(demo_companies):
        with cols[i]:
            st.info(f"**{ticker}**\n{name}\n*{sector}*")
    
    # Performance info
    st.subheader("⚡ Performance & Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Analysis Speed", "< 20 seconds")
        st.metric("Supported Forms", "10-K, 10-Q")
        st.metric("Coverage", "All SEC-registered companies")
    
    with col2:
        st.metric("Financial Accuracy", "> 95%")
        st.metric("Risk Factor Detection", "Top 5 risks")
        st.metric("Sentiment Analysis", "MD&A sections")

if __name__ == "__main__":
    main() 