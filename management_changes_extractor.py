import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import pandas as pd
from openai import OpenAI
import time
import textwrap

st.set_page_config(page_title="Flexible 8-K Data Extractor", layout="centered")
st.title("📄 Flexible 8-K Data Extractor")

# Inputs
ticker = st.text_input("Enter Stock Ticker (e.g., MSFT, ORCL)", "MSFT").upper()
api_key = st.text_input("Enter OpenAI API Key", type="password")

# New input for data extraction specification
st.subheader("📊 Data Extraction Configuration")
extraction_type = st.text_input(
    "What data do you want to extract from 8-K filings?", 
    "Total transaction data",
    help="Examples: 'Total transaction data', 'Management changes', 'Acquisition details', 'Financial metrics', 'Legal proceedings', etc."
)

# Additional context for better extraction
extraction_context = st.text_area(
    "Additional context or specific fields (optional)",
    placeholder="e.g., 'Include transaction amounts, dates, parties involved' or 'Focus on dollar amounts, effective dates, and counterparties'",
    help="Provide specific details about what information you want captured"
)

year_input = st.text_input("How many years back to search? (Leave blank for most recent only)", "")
quarter_input = st.text_input("OR enter specific quarter (e.g., 2Q25, Q4FY24)", "")
model_choice = st.selectbox("Select OpenAI Model", ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4"], index=0)

@st.cache_data(show_spinner=False)
def lookup_cik(ticker):
    headers = {'User-Agent': 'Financial Research contact@example.com'}
    res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
    data = res.json()
    for entry in data.values():
        if entry["ticker"].upper() == ticker:
            return str(entry["cik_str"]).zfill(10)
    return None

@st.cache_data(show_spinner=False)
def get_fiscal_year_end(cik, display_message=True):
    try:
        headers = {'User-Agent': 'Financial Research contact@example.com'}
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = requests.get(url, headers=headers)
        data = resp.json()
        
        if 'fiscalYearEnd' in data:
            fiscal_year_end = data['fiscalYearEnd']
            if len(fiscal_year_end) == 4:
                month = int(fiscal_year_end[:2])
                day = int(fiscal_year_end[2:])
                month_name = datetime(2000, month, 1).strftime('%B')
                if display_message:
                    st.success(f"✅ Fiscal year end: {month_name} {day}")
                return month, day
        
        if display_message:
            st.warning("⚠️ Using default fiscal year end: December 31")
        return 12, 31
    except Exception as e:
        if display_message:
            st.warning(f"⚠️ Error: {str(e)}. Using December 31.")
        return 12, 31

def generate_fiscal_quarters(fiscal_year_end_month):
    fiscal_year_start_month = (fiscal_year_end_month % 12) + 1
    quarters = {}
    current_month = fiscal_year_start_month
    
    for q in range(1, 5):
        start_month = current_month
        end_month = (start_month + 2) % 12
        if end_month == 0:
            end_month = 12
        quarters[q] = {'start_month': start_month, 'end_month': end_month}
        current_month = (end_month % 12) + 1
    
    return quarters

def get_fiscal_dates(quarter_num, year_num, fiscal_year_end_month):
    quarters = generate_fiscal_quarters(fiscal_year_end_month)
    
    if quarter_num < 1 or quarter_num > 4:
        st.error(f"Invalid quarter number: {quarter_num}")
        return None
        
    quarter_info = quarters[quarter_num]
    start_month = quarter_info['start_month']
    end_month = quarter_info['end_month']
    spans_calendar_years = end_month < start_month
    
    if fiscal_year_end_month == 12:
        start_calendar_year = year_num
    else:
        fiscal_year_start_month = (fiscal_year_end_month % 12) + 1
        if start_month >= fiscal_year_start_month:
            start_calendar_year = year_num - 1
        else:
            start_calendar_year = year_num
    
    end_calendar_year = start_calendar_year
    if spans_calendar_years:
        end_calendar_year = start_calendar_year + 1
    
    start_date = datetime(start_calendar_year, start_month, 1)
    
    if end_month == 2:
        if (end_calendar_year % 4 == 0 and end_calendar_year % 100 != 0) or (end_calendar_year % 400 == 0):
            end_day = 29
        else:
            end_day = 28
    elif end_month in [4, 6, 9, 11]:
        end_day = 30
    else:
        end_day = 31
    
    end_date = datetime(end_calendar_year, end_month, end_day)
    quarter_period = f"Q{quarter_num} FY{year_num}"
    
    st.write(f"Quarter {quarter_num}: {datetime(2000, start_month, 1).strftime('%B')}-{datetime(2000, end_month, 1).strftime('%B')}")
    
    return {
        'quarter_period': quarter_period,
        'start_date': start_date,
        'end_date': end_date
    }

def get_accessions(cik, years_back=None, specific_quarter=None):
    headers = {'User-Agent': 'Financial Research contact@example.com'}
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=headers)
    data = resp.json()
    filings = data["filings"]["recent"]
    accessions = []
    
    fiscal_year_end_month, fiscal_year_end_day = get_fiscal_year_end(cik, display_message=False)
    
    if years_back:
        cutoff = datetime.today() - timedelta(days=(365 * years_back))
        st.write(f"Looking for filings since {cutoff.strftime('%Y-%m-%d')}")
        
        for form, date_str, accession in zip(filings["form"], filings["filingDate"], filings["accessionNumber"]):
            if form == "8-K":
                date = datetime.strptime(date_str, "%Y-%m-%d")
                if date >= cutoff:
                    accessions.append((accession, date_str))
    
    elif specific_quarter:
        match = re.search(r'(?:Q?(\d)Q?|Q(\d))(?:\s*FY\s*|\s*)?(\d{2}|\d{4})', specific_quarter.upper())
        if match:
            quarter = match.group(1) or match.group(2)
            year = match.group(3)
            
            if len(year) == 2:
                year = '20' + year
                
            quarter_num = int(quarter)
            year_num = int(year)
            
            fiscal_info = get_fiscal_dates(quarter_num, year_num, fiscal_year_end_month)
            
            if not fiscal_info:
                return []
            
            st.write(f"Looking for {ticker} {fiscal_info['quarter_period']} filings")
            
            start_date = fiscal_info['start_date']
            end_date = fiscal_info['end_date']
            
            for form, date_str, accession in zip(filings["form"], filings["filingDate"], filings["accessionNumber"]):
                if form == "8-K":
                    date = datetime.strptime(date_str, "%Y-%m-%d")
                    if start_date <= date <= end_date:
                        accessions.append((accession, date_str))
                        st.write(f"Found filing from {date_str}")
    
    else:  # Default: most recent only
        for form, date_str, accession in zip(filings["form"], filings["filingDate"], filings["accessionNumber"]):
            if form == "8-K":
                accessions.append((accession, date_str))
                break
    
    if accessions:
        st.write(f"Found {len(accessions)} relevant 8-K filings")
    
    return accessions

def get_8k_links(cik, accessions):
    links = []
    for accession, date_str in accessions:
        base_folder = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"
        filing_url = base_folder + accession + ".txt"
        links.append((date_str, accession, filing_url))
    return links

def determine_fiscal_quarter(date_str, fiscal_year_end_month):
    date = datetime.strptime(date_str, "%Y-%m-%d")
    quarters = generate_fiscal_quarters(fiscal_year_end_month)
    
    if fiscal_year_end_month == 12:
        fiscal_year = date.year
    else:
        if date.month > fiscal_year_end_month:
            fiscal_year = date.year + 1
        else:
            fiscal_year = date.year
    
    quarter_num = None
    for q, q_info in quarters.items():
        start_month = q_info['start_month']
        end_month = q_info['end_month']
        
        if end_month < start_month:  # Quarter spans calendar years
            if date.month <= end_month or date.month >= start_month:
                quarter_num = q
                break
        else:  # Quarter within same calendar year
            if start_month <= date.month <= end_month:
                quarter_num = q
                break
    
    if quarter_num is None:
        quarter_num = 1
    
    fiscal_year_short = fiscal_year % 100
    return f"Q{quarter_num} FY{fiscal_year_short}"

def extract_relevant_sections(text, extraction_type):
    """
    Extract potentially relevant sections from 8-K text based on the extraction type.
    Uses a lightweight approach to identify relevant content for the LLM.
    """
    # Split text into sections by common 8-K item patterns
    sections = re.split(r'(?i)Item\s+\d+\.\d+', text)
    
    # Always include the full document but limit to reasonable size to avoid token limits
    if len(text) > 12000:
        # Take first part and last part to capture summary and details
        relevant_text = text[:8000] + "\n\n...[middle content truncated]...\n\n" + text[-4000:]
    else:
        relevant_text = text
    
    return relevant_text

def extract_data_with_llm(text, ticker, client, fiscal_quarter, extraction_type, extraction_context="", model="gpt-3.5-turbo"):
    """
    Generic function to extract any specified data from 8-K filings using LLM.
    """
    
    # Build the prompt dynamically based on user input
    context_instruction = f"\nAdditional focus: {extraction_context}" if extraction_context.strip() else ""
    
    prompt = f"""Extract all information related to "{extraction_type}" from this SEC 8-K filing for {ticker} (Quarter: {fiscal_quarter}).{context_instruction}

Please analyze the document thoroughly and identify any information that relates to: {extraction_type}

For each relevant item found, extract the following details:
- Category: What type of information this is
- Description: Brief description of the item/event
- Key Details: Important specifics (amounts, dates, parties, etc.)
- Effective Date: When applicable (in MM/DD/YYYY format)
- Additional Context: Any other relevant information

FORMAT: Return each item as a JSON object with these exact fields:
{{
  "category": "Type/Category of information",
  "description": "Brief description",
  "key_details": "Important specifics",
  "effective_date": "MM/DD/YYYY or Not Specified",
  "additional_context": "Other relevant details"
}}

If multiple items are found, separate each JSON object with triple hyphens (---).
If no relevant information is found, respond with "No {extraction_type.lower()} information found."

Be thorough and look for any information that could be related to "{extraction_type}", even if it's not explicitly labeled as such.

DOCUMENT TEXT:
{text}"""

    # Add retry logic for API rate limits
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            st.warning(f"⚠️ Attempt {attempt+1}/{max_retries}: {error_msg}")
            
            if "rate_limit_exceeded" in error_msg:
                st.info(f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                # For non-rate limit errors, try with a smaller text chunk
                if attempt < max_retries - 1:
                    st.info("Trying with a smaller text chunk...")
                    # Reduce text size by half for next attempt
                    truncated_text = text[:len(text)//2] + "...[truncated]"
                    
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt.replace(text, truncated_text)}],
                            temperature=0.2,
                            max_tokens=2000,
                        )
                        return response.choices[0].message.content
                    except Exception as inner_e:
                        st.warning(f"⚠️ Failed with smaller text chunk: {str(inner_e)}")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                else:
                    break
    
    st.error("Failed to extract data after multiple attempts")
    return f"No {extraction_type.lower()} information could be extracted due to API limitations. Try again later or with a different model."

def parse_extracted_data_to_df(text, fiscal_quarter, extraction_type):
    """
    Parse LLM response into a pandas DataFrame with flexible structure.
    """
    if not text or f"No {extraction_type.lower()}" in text.lower():
        return None
    
    data = []
    
    # Check if we have JSON formatted responses
    if text.strip().startswith("{"):
        # Split by the separator if multiple entries
        json_objects = text.split("---")
        
        import json
        for json_str in json_objects:
            try:
                # Clean up the JSON string
                json_str = json_str.strip()
                if not json_str:
                    continue
                    
                # Parse the JSON
                item = json.loads(json_str)
                
                # Standardize the item with our expected fields
                standardized_item = {
                    "Fiscal Quarter": fiscal_quarter,
                    "Category": item.get("category", "Not Specified"),
                    "Description": item.get("description", "Not Specified"),
                    "Key Details": item.get("key_details", "Not Specified"),
                    "Effective Date": item.get("effective_date", "Not Specified"),
                    "Additional Context": item.get("additional_context", "Not Specified")
                }
                
                data.append(standardized_item)
                
            except json.JSONDecodeError:
                # If JSON parsing fails, store as raw info
                if json_str.strip():
                    item = {
                        "Fiscal Quarter": fiscal_quarter,
                        "Category": "Raw Data",
                        "Description": json_str.strip()[:200] + "..." if len(json_str.strip()) > 200 else json_str.strip(),
                        "Key Details": "See Description",
                        "Effective Date": "Not Specified",
                        "Additional Context": json_str.strip()
                    }
                    data.append(item)
    else:
        # Handle non-JSON responses by treating as single item
        if text.strip():
            item = {
                "Fiscal Quarter": fiscal_quarter,
                "Category": extraction_type,
                "Description": text.strip()[:200] + "..." if len(text.strip()) > 200 else text.strip(),
                "Key Details": "See Additional Context",
                "Effective Date": "Not Specified", 
                "Additional Context": text.strip()
            }
            data.append(item)
    
    # Create DataFrame
    if data:
        df = pd.DataFrame(data)
        
        # Ensure all required columns exist
        required_columns = [
            "Fiscal Quarter", "Category", "Description", "Key Details", 
            "Effective Date", "Additional Context"
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = "Not Specified"
                
        # Reorder columns
        df = df[required_columns]
        
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        return df
    else:
        return None

if st.button("🔍 Extract Data from 8-K Filings"):
    if not api_key:
        st.error("Please enter your OpenAI API key.")
    elif not extraction_type.strip():
        st.error("Please specify what data you want to extract.")
    else:
        cik = lookup_cik(ticker)
        if not cik:
            st.error("CIK not found for ticker.")
        else:
            client = OpenAI(api_key=api_key)
            
            # Get fiscal year end to determine quarters
            fiscal_year_end_month, _ = get_fiscal_year_end(cik, display_message=True)
            
            # Handle different filtering options
            if quarter_input.strip():
                accessions = get_accessions(cik, specific_quarter=quarter_input.strip())
                if not accessions:
                    st.warning(f"No 8-K filings found for {quarter_input}.")
            elif year_input.strip():
                try:
                    years_back = int(year_input.strip())
                    accessions = get_accessions(cik, years_back=years_back)
                except:
                    st.error("Invalid year input. Must be a number.")
                    accessions = []
            else:
                accessions = get_accessions(cik)

            links = get_8k_links(cik, accessions)
            results = []

            # Create progress bar
            progress_bar = st.progress(0)
            
            st.info(f"🔍 Looking for: **{extraction_type}**")
            if extraction_context:
                st.info(f"📋 Additional context: {extraction_context}")
            
            with st.spinner("Processing 8-K filings..."):
                # Process each filing
                for i, (date_str, acc, url) in enumerate(links):
                    # Update progress
                    progress_bar.progress((i / len(links)))
                    
                    # Create expander for this filing
                    with st.expander(f"📄 8-K Filing from {date_str}", expanded=False):
                        try:
                            # Determine the fiscal quarter for this filing
                            fiscal_quarter = determine_fiscal_quarter(date_str, fiscal_year_end_month)
                            st.write(f"📅 Filing is in: **{fiscal_quarter}**")
                            
                            # Get the text content of the filing
                            response = requests.get(url, headers={"User-Agent": "Financial Research contact@example.com"})
                            text = response.text
                            
                            # Extract relevant sections
                            relevant_text = extract_relevant_sections(text, extraction_type)
                            
                            # Extract data using LLM
                            with st.spinner(f"Extracting {extraction_type} using AI..."):
                                extracted_data = extract_data_with_llm(
                                    relevant_text, 
                                    ticker, 
                                    client, 
                                    fiscal_quarter,
                                    extraction_type,
                                    extraction_context,
                                    model=model_choice
                                )
                            
                            if extracted_data and f"No {extraction_type.lower()}" not in extracted_data.lower():
                                # Parse to DataFrame
                                df = parse_extracted_data_to_df(extracted_data, fiscal_quarter, extraction_type)
                                
                                if df is not None:
                                    # Add metadata
                                    df["Filing Date"] = date_str
                                    df["8K Link"] = url
                                    results.append(df)
                                    st.success(f"✅ {extraction_type} data extracted from this 8-K")
                                    
                                    # Show preview of extracted data
                                    st.dataframe(df.drop(columns=["Filing Date", "8K Link"]), use_container_width=True)
                                else:
                                    st.warning("⚠️ No structured data found")
                            else:
                                st.info(f"ℹ️ No {extraction_type.lower()} information found in this filing")
                                
                        except Exception as e:
                            st.error(f"❌ Could not process: {url}. Error: {str(e)}")
                
                # Set progress to 100% when done
                progress_bar.progress(1.0)

            if results:
                # Combine all results
                combined = pd.concat(results, ignore_index=True)
                
                # Display summary
                st.subheader(f"📊 Extracted {extraction_type} Data")
                st.write(f"Found **{len(combined)}** relevant items across **{len(results)}** filings")
                
                # Display the combined table
                st.dataframe(combined, use_container_width=True)
                
                # Download button
                import io
                csv_buffer = io.StringIO()
                combined.to_csv(csv_buffer, index=False)
                st.download_button(
                    "📥 Download CSV", 
                    data=csv_buffer.getvalue(), 
                    file_name=f"{ticker}_{extraction_type.replace(' ', '_').lower()}_data.csv", 
                    mime="text/csv"
                )
                
                # Show summary statistics
                with st.expander("📈 Summary Statistics", expanded=False):
                    st.write(f"**Total Items Found:** {len(combined)}")
                    st.write(f"**Filings Processed:** {len(links)}")
                    st.write(f"**Filings with Data:** {len(results)}")
                    
                    if "Category" in combined.columns:
                        category_counts = combined["Category"].value_counts()
                        st.write("**Data by Category:**")
                        st.dataframe(category_counts)
                        
            else:
                st.warning(f"No {extraction_type.lower()} data extracted from any of the filings.")
