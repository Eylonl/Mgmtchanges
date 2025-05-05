import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import pandas as pd
from openai import OpenAI
import time
import textwrap

st.set_page_config(page_title="SEC 8-K Management Changes Extractor", layout="centered")
st.title("📄 SEC 8-K Management Changes Extractor")

# Inputs
ticker = st.text_input("Enter Stock Ticker (e.g., MSFT, ORCL)", "MSFT").upper()
api_key = st.text_input("Enter OpenAI API Key", type="password")
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

def find_management_change_paragraphs(text):
    # First, try to extract Item 5.02 section which is specifically for management changes
    item_502_pattern = re.compile(r'(?i)Item\s+5\.02.+?(?=Item\s+\d\.\d{2}|$)', re.DOTALL)
    match = item_502_pattern.search(text)
    
    if match:
        section_text = match.group(0)
        # Limit to a reasonable length (8000 chars) to avoid token limits
        if len(section_text) > 8000:
            section_text = section_text[:8000] + "...[truncated for length]"
        return section_text, True
    
    # If Item 5.02 section not found, use keyword search as fallback
    management_change_patterns = [
        r'(?i)appoint(?:ed|ment|ing|s)?',
        r'(?i)nam(?:ed|ing|es) .* (?:as|to)',
        r'(?i)(?:new|interim|acting) (?:CEO|CFO|COO|CTO|CMO|president|chief|officer|director)',
        r'(?i)(?:CEO|CFO|COO|CTO|CMO|president|chief|officer|director) transition',
        r'(?i)(?:management|leadership|executive) (?:change|transition|appointment)',
        r'(?i)(?:resignation|resigned|retiring|retire|departure) of (?:CEO|CFO|COO|CTO|CMO|president|chief|officer|director)',
        r'(?i)promot(?:ed|ion|ing)',
        r'(?i)succeed(?:ed|ing|s)? (?:as|to)',
        r'(?i)(?:change|new) (?:of|in) leadership',
        r'(?i)(?:CEO|CFO|COO|CTO|CMO|president|chief|officer|director) (?:search|succession)',
        r'(?i)step(?:ping|ped) down',
        r'(?i)(?:join|joined) (?:the|as)',
        r'(?i)(?:board|executive|management) (?:appointment|change)',
        r'(?i)(?:hire|hired|hiring) (?:as|to)',
        r'(?i)(?:elect|elected|electing) (?:as|to)'
    ]
    
    paragraphs = re.split(r'\n\s*\n|\.\s+(?=[A-Z])', text)
    management_change_paragraphs = []
    
    for para in paragraphs:
        if any(re.search(pattern, para) for pattern in management_change_patterns):
            management_change_paragraphs.append(para)
    
    found_paragraphs = len(management_change_paragraphs) > 0
    
    formatted_paragraphs = "\n\n".join(management_change_paragraphs)
    
    # Limit to a reasonable length (8000 chars) to avoid token limits
    if len(formatted_paragraphs) > 8000:
        formatted_paragraphs = formatted_paragraphs[:8000] + "...[truncated for length]"
    
    if management_change_paragraphs:
        formatted_paragraphs = (
            f"DOCUMENT TYPE: SEC 8-K Filing for {ticker}\n\n"
            f"POTENTIAL MANAGEMENT CHANGE INFORMATION (extracted from full document):\n\n{formatted_paragraphs}\n\n"
            "Note: These are selected paragraphs that may contain information about executive management transitions and changes."
        )
    
    return formatted_paragraphs, found_paragraphs

def extract_management_changes(text, ticker, client, fiscal_quarter, model="gpt-3.5-turbo"):
    # Create a shorter prompt to reduce token count
    prompt = f"""Extract ALL management change information in this 8-K filing for {ticker} (Quarter: {fiscal_quarter}).
Focus only on executive management transitions, new appointments, resignations, retirements, and board changes.

For each change, identify:
- Type (Appointment, Resignation, Retirement, Promotion, Other)
- Name of Executive
- New Role (if applicable)
- Previous Role (if applicable) 
- Effective Date (in MM/DD/YYYY format when available)
- Key Details (background, reason for change)

FORMAT: Return each management change as a JSON object with the fields above. Use this exact format:
{{
  "type": "Appointment/Resignation/etc",
  "name": "Executive Name",
  "new_role": "New Position",
  "previous_role": "Previous Position",
  "effective_date": "MM/DD/YYYY",
  "details": "Brief background/reason"
}}

If multiple changes, separate each JSON object with triple hyphens (---).
If no management changes are found, respond with "No management changes found."

TEXT:
{text}"""

    # Add retry logic for API rate limits
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500,  # Limit response size
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
                            temperature=0.3,
                            max_tokens=1500,
                        )
                        return response.choices[0].message.content
                    except Exception as inner_e:
                        st.warning(f"⚠️ Failed with smaller text chunk: {str(inner_e)}")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                else:
                    break
    
    st.error("Failed to extract management changes after multiple attempts")
    return "No management changes could be extracted due to API limitations. Try again later or with a different model."

def parse_management_changes_to_df(text, fiscal_quarter):
    if not text or "No management changes" in text:
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
                
                # Standardize keys
                standardized_item = {
                    "Fiscal Quarter": fiscal_quarter,
                    "Type": item.get("type", "Not Specified"),
                    "Name of Executive": item.get("name", "Not Specified"),
                    "New Role": item.get("new_role", "Not Specified"),
                    "Previous Role": item.get("previous_role", "Not Specified"),
                    "Effective Date": item.get("effective_date", "Not Specified"),
                    "Other Key Details": item.get("details", "Not Specified")
                }
                
                data.append(standardized_item)
            except json.JSONDecodeError:
                # If JSON parsing fails, fall back to regex parsing for this entry
                item = {"Fiscal Quarter": fiscal_quarter, "Raw Info": json_str}
                
                # Try to identify type
                if re.search(r'(?i)appoint|promot|nam[ed]|hir[ed]|join', json_str):
                    item["Type"] = "Appointment"
                elif re.search(r'(?i)resign|step down|depart', json_str):
                    item["Type"] = "Resignation"
                elif re.search(r'(?i)retir', json_str):
                    item["Type"] = "Retirement"
                else:
                    item["Type"] = "Other Change"
                    
                # Try to extract name
                name_match = re.search(r'(?:[A-Z][a-z]+ ){1,3}[A-Z][a-z]+', json_str)
                if name_match:
                    item["Name of Executive"] = name_match.group(0)
                    
                if len(item) > 2:  # More than just Fiscal Quarter and Raw Info
                    data.append(item)
    else:
        # Fall back to bullet point parsing for backward compatibility
        bullet_points = re.split(r'\n\s*-\s*|\n\s*•\s*', text)
        bullet_points = [bp.strip() for bp in bullet_points if bp.strip()]
        
        patterns = {
            "Type": r'(?i)Type:?\s*([A-Za-z\s]+)',
            "Name of Executive": r'(?i)Name:?\s*([A-Za-z\s\.]+)',
            "New Role": r'(?i)New Role:?\s*([^,\n]+)',
            "Previous Role": r'(?i)Previous Role:?\s*([^,\n]+)',
            "Effective Date": r'(?i)(?:Effective|Date):?\s*([^,\n]+)',
            "Other Key Details": r'(?i)(?:Other Key Details|Details|Key Details):?\s*(.+)'
        }
        
        for bullet in bullet_points:
            item = {"Fiscal Quarter": fiscal_quarter}
            
            # Check for structured format first
            for key, pattern in patterns.items():
                match = re.search(pattern, bullet)
                if match:
                    item[key] = match.group(1).strip()
            
            # If no structured data was found, try to extract as best as possible
            if len(item) <= 1:  # Only has Fiscal Quarter
                item["Raw Info"] = bullet
                
                # Try to identify type
                if re.search(r'(?i)appoint|promot|nam[ed]|hir[ed]|join', bullet):
                    item["Type"] = "Appointment"
                elif re.search(r'(?i)resign|step down|depart', bullet):
                    item["Type"] = "Resignation"
                elif re.search(r'(?i)retir', bullet):
                    item["Type"] = "Retirement"
                else:
                    item["Type"] = "Other Change"
                    
                # Try to extract name
                name_match = re.search(r'(?:[A-Z][a-z]+ ){1,3}[A-Z][a-z]+', bullet)
                if name_match:
                    item["Name of Executive"] = name_match.group(0)
            
            # Only add if we have at least some useful information
            if len(item) > 2:  # More than just Fiscal Quarter and Raw Info
                data.append(item)
    
    # Create DataFrame
    if data:
        df = pd.DataFrame(data)
        
        # Ensure required columns exist
        required_columns = [
            "Type", "Name of Executive", "New Role", "Previous Role", 
            "Effective Date", "Other Key Details", "Fiscal Quarter"
        ]
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = "Not Specified"
                
        # Reorder columns
        df = df[["Fiscal Quarter"] + [col for col in required_columns if col != "Fiscal Quarter"]]
        
        # Remove duplicate rows based on all columns except Filing Date and 8K Link
        # since those are added later
        df = df.drop_duplicates()
        
        return df
    else:
        return None

def install_required_packages():
    """Make sure openpyxl is installed for Excel support"""
    try:
        import importlib.util
        if importlib.util.find_spec("openpyxl") is None:
            st.warning("⚠️ Required package 'openpyxl' is not installed")
            st.info("To enable Excel export, please install openpyxl using: pip install openpyxl")
            return False
        else:
            return True
    except Exception as e:
        st.error(f"❌ Error checking for dependencies: {str(e)}")
        st.info("If you're seeing Excel-related errors, please ensure the openpyxl package is installed")
        return False

if st.button("🔍 Extract Management Changes"):
    # Make sure dependencies are installed
    install_required_packages()
    
    if not api_key:
        st.error("Please enter your OpenAI API key.")
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
                            
                            # Find paragraphs containing management change patterns
                            management_paragraphs, found_management = find_management_change_paragraphs(text)
                            
                            if found_management:
                                st.success("✅ Found potential management change information")
                                
                                # Extract management changes from the highlighted text
                                with st.spinner("Extracting management changes using AI..."):
                                    management_changes = extract_management_changes(
                                        management_paragraphs, 
                                        ticker, 
                                        client, 
                                        fiscal_quarter,
                                        model=model_choice
                                    )
                                
                                if management_changes and "No management changes" not in management_changes:
                                    # Format the display for JSON output
                                    if management_changes.strip().startswith("{"):
                                        import json
                                        json_objects = management_changes.split("---")
                                        
                                        for idx, json_str in enumerate(json_objects):
                                            try:
                                                json_str = json_str.strip()
                                                if not json_str:
                                                    continue
                                                    
                                                change_data = json.loads(json_str)
                                                
                                                # Display in a clean card format
                                                st.markdown(f"""
                                                <div style="border:1px solid #d0d7de; border-radius:6px; padding:16px; margin-bottom:16px;">
                                                    <h4 style="margin-top:0; color:#0969da;">
                                                        {change_data.get('type', 'Management Change')} - {change_data.get('name', 'Executive')}
                                                    </h4>
                                                    <table style="width:100%;">
                                                        <tr>
                                                            <td style="width:25%; color:#57606a;"><strong>New Role:</strong></td>
                                                            <td>{change_data.get('new_role', 'Not Specified')}</td>
                                                        </tr>
                                                        <tr>
                                                            <td style="width:25%; color:#57606a;"><strong>Previous Role:</strong></td>
                                                            <td>{change_data.get('previous_role', 'Not Specified')}</td>
                                                        </tr>
                                                        <tr>
                                                            <td style="width:25%; color:#57606a;"><strong>Effective Date:</strong></td>
                                                            <td>{change_data.get('effective_date', 'Not Specified')}</td>
                                                        </tr>
                                                        <tr>
                                                            <td style="width:25%; color:#57606a;"><strong>Details:</strong></td>
                                                            <td>{change_data.get('details', 'Not Specified')}</td>
                                                        </tr>
                                                    </table>
                                                </div>
                                                """, unsafe_allow_html=True)
                                            except json.JSONDecodeError:
                                                st.markdown(f"**Raw Change Data {idx+1}:**\n{json_str}")
                                    else:
                                        # Fall back to displaying as markdown for non-JSON format
                                        st.markdown("### Extracted Management Changes")
                                        st.markdown(management_changes)
                                    
                                    # Parse to DataFrame
                                    df = parse_management_changes_to_df(management_changes, fiscal_quarter)
                                    
                                    if df is not None:
                                        # Add metadata
                                        df["Filing Date"] = date_str
                                        df["8K Link"] = url
                                        results.append(df)
                                        st.success("✅ Management changes extracted from this 8-K")
                                    else:
                                        st.warning("⚠️ No structured management change data found")
                                else:
                                    st.info("ℹ️ No management changes found in this filing")
                            else:
                                st.info("ℹ️ No management change information detected in this filing")
                                
                        except Exception as e:
                            st.error(f"❌ Could not process: {url}. Error: {str(e)}")
                
                # Set progress to 100% when done
                progress_bar.progress(1.0)

            if results:
                # Combine all results
                combined = pd.concat(results, ignore_index=True)
                
                # Show summary statistics
                st.subheader("📊 Management Changes Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Changes", len(combined))
                with col2:
                    # Count by type
                    type_counts = combined["Type"].value_counts()
                    most_common_type = type_counts.index[0] if not type_counts.empty else "None"
                    st.metric("Most Common Type", f"{most_common_type} ({type_counts.iloc[0]})" if not type_counts.empty else "None")
                with col3:
                    # Count by quarter
                    quarter_counts = combined["Fiscal Quarter"].value_counts()
                    most_active_quarter = quarter_counts.index[0] if not quarter_counts.empty else "None"
                    st.metric("Most Active Quarter", f"{most_active_quarter} ({quarter_counts.iloc[0]})" if not quarter_counts.empty else "None")
                
                # Preview the table with filters
                st.subheader("🔍 Management Changes Table")
                
                # Add filtering options
                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    filter_type = st.multiselect(
                        "Filter by Type",
                        options=["All"] + sorted(combined["Type"].unique().tolist()),
                        default=["All"]
                    )
                with filter_col2:
                    filter_quarter = st.multiselect(
                        "Filter by Quarter",
                        options=["All"] + sorted(combined["Fiscal Quarter"].unique().tolist()),
                        default=["All"]
                    )
                
                # Apply filters
                filtered_df = combined.copy()
                if filter_type and "All" not in filter_type:
                    filtered_df = filtered_df[filtered_df["Type"].isin(filter_type)]
                if filter_quarter and "All" not in filter_quarter:
                    filtered_df = filtered_df[filtered_df["Fiscal Quarter"].isin(filter_quarter)]
                
                # Display the filtered table
                st.dataframe(filtered_df, use_container_width=True)
                
                # Add download buttons
                st.subheader("📥 Download Results")
                
                download_col1, download_col2 = st.columns(2)
                
                with download_col1:
                    # CSV export for all data
                    import io
                    csv_buffer = io.StringIO()
                    combined.to_csv(csv_buffer, index=False)
                    st.download_button(
                        "📥 Download All Data (CSV)", 
                        data=csv_buffer.getvalue(), 
                        file_name=f"{ticker}_management_changes.csv", 
                        mime="text/csv"
                    )
                
                with download_col2:
                    # Download filtered data
                    if not filtered_df.equals(combined):
                        csv_buffer_filtered = io.StringIO()
                        filtered_df.to_csv(csv_buffer_filtered, index=False)
                        st.download_button(
                            "📥 Download Filtered Data (CSV)", 
                            data=csv_buffer_filtered.getvalue(), 
                            file_name=f"{ticker}_filtered_management_changes.csv", 
                            mime="text/csv"
                        )
            else:
                st.warning("No management change data extracted from any of the filings.")
