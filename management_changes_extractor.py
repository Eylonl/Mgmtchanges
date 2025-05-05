import streamlit as st
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import re
import pandas as pd
from openai import OpenAI
import os

st.set_page_config(page_title="SEC 8-K Management Changes Extractor", layout="centered")
st.title("📄 SEC 8-K Management Changes Extractor")

# Inputs
ticker = st.text_input("Enter Stock Ticker (e.g., MSFT, ORCL)", "MSFT").upper()
api_key = st.text_input("Enter OpenAI API Key", type="password")

# Both filter options displayed at the same time
year_input = st.text_input("How many years back to search for 8-K filings? (Leave blank for most recent only)", "")
quarter_input = st.text_input("OR enter specific quarter (e.g., 2Q25, Q4FY24)", "")


@st.cache_data(show_spinner=False)
def lookup_cik(ticker):
    headers = {'User-Agent': 'Your Name Contact@domain.com'}
    res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=headers)
    data = res.json()
    for entry in data.values():
        if entry["ticker"].upper() == ticker:
            return str(entry["cik_str"]).zfill(10)


def get_fiscal_year_end(ticker, cik):
    """
    Get the fiscal year end month for a company from SEC data.
    Returns the month (1-12) and day.
    """
    try:
        headers = {'User-Agent': 'Your Name Contact@domain.com'}
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        resp = requests.get(url, headers=headers)
        data = resp.json()
        
        # Extract fiscal year end info - format is typically "MMDD" 
        if 'fiscalYearEnd' in data:
            fiscal_year_end = data['fiscalYearEnd']
            if len(fiscal_year_end) == 4:  # MMDD format
                month = int(fiscal_year_end[:2])
                day = int(fiscal_year_end[2:])
                
                month_name = datetime(2000, month, 1).strftime('%B')
                st.success(f"✅ Retrieved fiscal year end for {ticker}: {month_name} {day}")
                
                return month, day
        
        # If not found, default to December 31 (calendar year)
        st.warning(f"⚠️ Could not determine fiscal year end for {ticker} from SEC data. Using December 31 (calendar year).")
        return 12, 31
        
    except Exception as e:
        st.warning(f"⚠️ Error retrieving fiscal year end: {str(e)}. Using December 31 (calendar year).")
        return 12, 31


def generate_fiscal_quarters(fiscal_year_end_month):
    """
    Dynamically generate fiscal quarters based on the fiscal year end month.
    """
    # Calculate the first month of the fiscal year (month after fiscal year end)
    fiscal_year_start_month = (fiscal_year_end_month % 12) + 1
    
    # Generate all four quarters
    quarters = {}
    current_month = fiscal_year_start_month
    
    for q in range(1, 5):
        start_month = current_month
        
        # Each quarter is 3 months
        end_month = (start_month + 2) % 12
        if end_month == 0:  # Handle December (month 0 becomes month 12)
            end_month = 12
            
        quarters[q] = {'start_month': start_month, 'end_month': end_month}
        
        # Move to next quarter's start month
        current_month = (end_month % 12) + 1
    
    return quarters


def get_fiscal_dates(ticker, quarter_num, year_num, fiscal_year_end_month, fiscal_year_end_day):
    """
    Calculate the appropriate date range for a fiscal quarter
    based on the fiscal year end month.
    """
    # Generate quarters dynamically based on fiscal year end
    quarters = generate_fiscal_quarters(fiscal_year_end_month)
    
    # Get the specified quarter
    if quarter_num < 1 or quarter_num > 4:
        st.error(f"Invalid quarter number: {quarter_num}. Must be 1-4.")
        return None
        
    quarter_info = quarters[quarter_num]
    start_month = quarter_info['start_month']
    end_month = quarter_info['end_month']
    
    # Determine if the quarter spans calendar years
    spans_calendar_years = end_month < start_month
    
    # Determine the calendar year for each quarter
    if fiscal_year_end_month == 12:
        # Simple case: Calendar year matches fiscal year
        start_calendar_year = year_num
    else:
        # For non-calendar fiscal years, determine which calendar year the quarter falls in
        fiscal_year_start_month = (fiscal_year_end_month % 12) + 1
        
        if start_month >= fiscal_year_start_month:
            # This quarter starts in the previous calendar year
            # Example: For fiscal year ending in June (FY2024 = Jul 2023-Jun 2024)
            # Q1 (Jul-Sep) and Q2 (Oct-Dec) start in calendar year 2023
            start_calendar_year = year_num - 1
        else:
            # This quarter starts in the current calendar year
            # Example: For fiscal year ending in June (FY2024 = Jul 2023-Jun 2024)
            # Q3 (Jan-Mar) and Q4 (Apr-Jun) start in calendar year 2024
            start_calendar_year = year_num
    
    # For quarters that span calendar years, the end date is in the next calendar year
    end_calendar_year = start_calendar_year
    if spans_calendar_years:
        end_calendar_year = start_calendar_year + 1
    
    # Create actual date objects
    start_date = datetime(start_calendar_year, start_month, 1)
    
    # Calculate end date (last day of the end month)
    if end_month == 2:
        # Handle February and leap years
        if (end_calendar_year % 4 == 0 and end_calendar_year % 100 != 0) or (end_calendar_year % 400 == 0):
            end_day = 29  # Leap year
        else:
            end_day = 28
    elif end_month in [4, 6, 9, 11]:
        end_day = 30
    else:
        end_day = 31
    
    end_date = datetime(end_calendar_year, end_month, end_day)
    
    # Calculate expected filing dates window (entire quarter)
    filing_start = start_date
    filing_end = end_date
    
    # Output info about the dates
    quarter_period = f"Q{quarter_num} FY{year_num}"
    period_description = f"{start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}"
    filing_window = f"{filing_start.strftime('%B %d, %Y')} to {filing_end.strftime('%B %d, %Y')}"
    
    # Display fiscal quarter information
    st.write(f"Fiscal year ends in {datetime(2000, fiscal_year_end_month, 1).strftime('%B')} {fiscal_year_end_day}")
    st.write(f"Quarter {quarter_num} spans: {datetime(2000, start_month, 1).strftime('%B')}-{datetime(2000, end_month, 1).strftime('%B')}")
    
    # Show all quarters
    st.write("All quarters for this fiscal pattern:")
    for q, q_info in quarters.items():
        st.write(f"Q{q}: {datetime(2000, q_info['start_month'], 1).strftime('%B')}-{datetime(2000, q_info['end_month'], 1).strftime('%B')}")
    
    return {
        'quarter_period': quarter_period,
        'start_date': start_date,
        'end_date': end_date,
        'filing_start': filing_start,
        'filing_end': filing_end,
        'period_description': period_description,
        'filing_window': filing_window
    }


def get_accessions(cik, ticker, years_back=None, specific_quarter=None):
    """General function for finding filings"""
    headers = {'User-Agent': 'Your Name Contact@domain.com'}
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=headers)
    data = resp.json()
    filings = data["filings"]["recent"]
    accessions = []
    
    # Auto-detect fiscal year end from SEC data
    fiscal_year_end_month, fiscal_year_end_day = get_fiscal_year_end(ticker, cik)
    
    if years_back:
        # Look for filings from X years back
        cutoff = datetime.today() - timedelta(days=(365 * years_back))
        
        st.write(f"Looking for filings from the past {years_back} years (from {cutoff.strftime('%Y-%m-%d')} to present)")
        
        for form, date_str, accession in zip(filings["form"], filings["filingDate"], filings["accessionNumber"]):
            if form == "8-K":
                date = datetime.strptime(date_str, "%Y-%m-%d")
                if date >= cutoff:
                    accessions.append((accession, date_str))
    
    elif specific_quarter:
        # Parse quarter and year from input - handle various formats
        # Examples: 2Q25, Q4FY24, Q3 2024, Q1 FY 2025, etc.
        match = re.search(r'(?:Q?(\d)Q?|Q(\d))(?:\s*FY\s*|\s*)?(\d{2}|\d{4})', specific_quarter.upper())
        if match:
            quarter = match.group(1) or match.group(2)
            year = match.group(3)
            
            # Convert 2-digit year to 4-digit year
            if len(year) == 2:
                year = '20' + year
                
            quarter_num = int(quarter)
            year_num = int(year)
            
            # Get fiscal dates based on fiscal year end month
            fiscal_info = get_fiscal_dates(ticker, quarter_num, year_num, fiscal_year_end_month, fiscal_year_end_day)
            
            if not fiscal_info:
                return []
            
            # Display fiscal quarter information
            st.write(f"Looking for {ticker} {fiscal_info['quarter_period']} filings")
            st.write(f"Fiscal quarter period: {fiscal_info['period_description']}")
            st.write(f"Looking for filings between: {fiscal_info['filing_window']}")
            
            # We want to find filings within the quarter
            start_date = fiscal_info['start_date']
            end_date = fiscal_info['end_date']
            
            # Find filings in this date range
            for form, date_str, accession in zip(filings["form"], filings["filingDate"], filings["accessionNumber"]):
                if form == "8-K":
                    date = datetime.strptime(date_str, "%Y-%m-%d")
                    if start_date <= date <= end_date:
                        accessions.append((accession, date_str))
                        st.write(f"Found filing from {date_str}: {accession}")
    
    else:  # Default: most recent only
        for form, date_str, accession in zip(filings["form"], filings["filingDate"], filings["accessionNumber"]):
            if form == "8-K":
                accessions.append((accession, date_str))
                break
    
    # Show debug info about the selected accessions
    if accessions:
        st.write(f"Found {len(accessions)} relevant 8-K filings")
    else:
        # Show all available dates for reference
        available_dates = []
        for form, date_str in zip(filings["form"], filings["filingDate"]):
            if form == "8-K":
                available_dates.append(date_str)
        
        if available_dates:
            available_dates.sort(reverse=True)  # Show most recent first
            st.write("All available 8-K filing dates:")
            for date in available_dates[:15]:  # Show only the first 15 to avoid cluttering
                st.write(f"- {date}")
            if len(available_dates) > 15:
                st.write(f"... and {len(available_dates) - 15} more")
    
    return accessions


def get_8k_links(cik, accessions):
    """Get links to the 8-K filings (not just the exhibits)"""
    links = []
    headers = {'User-Agent': 'Your Name Contact@domain.com'}
    for accession, date_str in accessions:
        base_folder = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/"
        filing_url = base_folder + accession + ".txt"
        links.append((date_str, accession, filing_url))
    return links


def determine_fiscal_quarter(date_str, fiscal_year_end_month):
    """
    Determine the fiscal quarter for a given date based on fiscal year end.
    Returns a string like "Q1 FY24" based on the date and fiscal year end month.
    """
    date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Generate fiscal quarters
    quarters = generate_fiscal_quarters(fiscal_year_end_month)
    
    # Determine fiscal year
    if fiscal_year_end_month == 12:
        # Calendar year end (Dec 31) - fiscal year matches calendar year
        fiscal_year = date.year
    else:
        # Non-calendar year end - fiscal year may differ from calendar year
        # If date is after fiscal year end, then fiscal year is calendar year + 1
        if date.month > fiscal_year_end_month:
            fiscal_year = date.year + 1
        else:
            fiscal_year = date.year
    
    # Determine which quarter this date falls into
    quarter_num = None
    for q, q_info in quarters.items():
        start_month = q_info['start_month']
        end_month = q_info['end_month']
        
        # Handle quarters that span calendar years
        if end_month < start_month:  # Quarter spans calendar years
            # Date is in the end portion (Jan-end_month of following year)
            if date.month <= end_month:
                if date.month > 0:  # Ensure date is not in the start portion
                    quarter_num = q
                    break
            # Date is in the start portion (start_month-Dec of current year)
            elif date.month >= start_month:
                quarter_num = q
                break
        else:  # Quarter within same calendar year
            if start_month <= date.month <= end_month:
                quarter_num = q
                break
    
    # If we couldn't determine the quarter, default to 1
    if quarter_num is None:
        quarter_num = 1
    
    # Format fiscal year as 2-digit year
    fiscal_year_short = fiscal_year % 100
    
    # Return formatted quarter string
    return f"Q{quarter_num} FY{fiscal_year_short}"


def find_management_change_paragraphs(text):
    """
    Extract paragraphs from text that are likely to contain management change information.
    Returns both the filtered paragraphs and a boolean indicating if any were found.
    """
    # Define patterns to identify management change sections
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
        r'(?i)(?:elect|elected|electing) (?:as|to)',
        r'(?i)Item 5.02'  # SEC 8-K item number for management changes
    ]
    
    # Split text into paragraphs
    paragraphs = re.split(r'\n\s*\n|\.\s+(?=[A-Z])', text)
    
    # Find paragraphs matching management change patterns
    management_change_paragraphs = []
    
    for para in paragraphs:
        if any(re.search(pattern, para) for pattern in management_change_patterns):
            management_change_paragraphs.append(para)
    
    # Check if we found any management change paragraphs
    found_paragraphs = len(management_change_paragraphs) > 0
    
    # If no management change paragraphs found, get a small sample around item 5.02
    if not found_paragraphs:
        # Look specifically for Item 5.02 section
        item_502_pattern = re.compile(r'(?i)Item\s+5\.02.+?(?=Item\s+\d\.\d{2}|$)', re.DOTALL)
        match = item_502_pattern.search(text)
        if match:
            section_text = match.group(0)
            # Split this section into paragraphs
            item_502_paragraphs = re.split(r'\n\s*\n|\.\s+(?=[A-Z])', section_text)
            management_change_paragraphs.extend(item_502_paragraphs)
            found_paragraphs = True
    
    # Combine paragraphs and add a note about the original document
    formatted_paragraphs = "\n\n".join(management_change_paragraphs)
    
    # Add metadata about the document to help GPT understand the context
    if management_change_paragraphs:
        formatted_paragraphs = (
            f"DOCUMENT TYPE: SEC 8-K Filing for {ticker}\n\n"
            f"POTENTIAL MANAGEMENT CHANGE INFORMATION (extracted from full document):\n\n{formatted_paragraphs}\n\n"
            "Note: These are selected paragraphs that may contain information about executive management transitions and changes."
        )
    
    return formatted_paragraphs, found_paragraphs


def extract_management_changes(text, ticker, client, fiscal_quarter):
    """
    Extract management changes from SEC filings.
    """
    prompt = f"""You are a financial analyst assistant. Extract ALL management change information in this 8-K filing for {ticker}. Focus on executive management transitions, new appointments, resignations, retirements, and board changes.

For the quarter: {fiscal_quarter}

Return the information as bullet points with these categories:
- Type (Appointment, Resignation, Retirement, Promotion, Other Change)
- Name of Executive
- New Role (if applicable)
- Previous Role (if applicable)
- Effective Date
- Other Key Details (e.g., reason for change, background of new appointee)

IMPORTANT INSTRUCTIONS:
- List each management change on a separate bullet point
- Include ALL management changes mentioned in the document
- Focus on C-suite executives, Presidents, Board Members, and other high-level executives
- Include both incoming and outgoing executives
- If exact dates are given, include them in MM/DD/YYYY format
- If only a month and year is mentioned, format as MM/YYYY
- If no date is specified, note as "Effective Immediately" or "Not Specified"
- For "Other Key Details", include notable background information, succession details, or reasons for the change
- If certain information is not available, use "N/A" or "Not Specified"

Respond in bullet point format with clear headings for each type of information.\n\n{text}"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning(f"⚠️ Error extracting management changes: {str(e)}")
        return None


def parse_management_changes_to_df(text, fiscal_quarter):
    """
    Parse the bullet point management changes into a DataFrame.
    """
    if not text or "No management changes found" in text:
        return None
    
    data = []
    
    # Split into bullet points
    bullet_points = re.split(r'\n\s*-\s*|\n\s*•\s*', text)
    bullet_points = [bp.strip() for bp in bullet_points if bp.strip()]
    
    current_item = {}
    
    for bullet in bullet_points:
        # Check if this is the start of a new item
        if bullet.startswith("Type:") or "Type:" in bullet[:20]:
            # Save previous item if it exists
            if current_item and "Type" in current_item:
                current_item["Fiscal Quarter"] = fiscal_quarter
                data.append(current_item)
                current_item = {}
            
            # Parse the bullet point into fields
            fields = re.split(r'\n', bullet)
            for field in fields:
                if ":" in field:
                    key, value = field.split(":", 1)
                    current_item[key.strip()] = value.strip()
    
    # Add the last item
    if current_item and "Type" in current_item:
        current_item["Fiscal Quarter"] = fiscal_quarter
        data.append(current_item)
    
    # If no structured data was found but we have bullet points, try a simpler parsing
    if not data and bullet_points:
        for bullet in bullet_points:
            item = {
                "Raw Info": bullet,
                "Fiscal Quarter": fiscal_quarter
            }
            
            # Try to extract some structure
            if re.search(r'appoint|promot|nam[ed]|hir[ed]|join', bullet, re.I):
                item["Type"] = "Appointment"
            elif re.search(r'resign|step down|depart', bullet, re.I):
                item["Type"] = "Resignation"
            elif re.search(r'retir', bullet, re.I):
                item["Type"] = "Retirement"
            else:
                item["Type"] = "Other Change"
                
            # Try to extract name
            name_match = re.search(r'(?:[A-Z][a-z]+ ){1,3}[A-Z][a-z]+', bullet)
            if name_match:
                item["Name of Executive"] = name_match.group(0)
                
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
        
        return df
    else:
        return None


if st.button("🔍 Extract Management Changes"):
    if not api_key:
        st.error("Please enter your OpenAI API key.")
    else:
        cik = lookup_cik(ticker)
        if not cik:
            st.error("CIK not found for ticker.")
        else:
            client = OpenAI(api_key=api_key)
            
            # Get fiscal year end to determine quarters
            fiscal_year_end_month, fiscal_year_end_day = get_fiscal_year_end(ticker, cik)
            
            # Handle different filtering options
            if quarter_input.strip():
                # Quarter input takes precedence
                accessions = get_accessions(cik, ticker, specific_quarter=quarter_input.strip())
                if not accessions:
                    st.warning(f"No 8-K filings found for {quarter_input}. Please check the format (e.g., 2Q25, Q4FY24).")
            elif year_input.strip():
                try:
                    years_back = int(year_input.strip())
                    accessions = get_accessions(cik, ticker, years_back=years_back)
                except:
                    st.error("Invalid year input. Must be a number.")
                    accessions = []
            else:
                # Default to most recent if neither input is provided
                accessions = get_accessions(cik, ticker)

            links = get_8k_links(cik, accessions)
            results = []

            for date_str, acc, url in links:
                st.write(f"📄 Processing 8-K from {date_str}: {url}")
                try:
                    # Determine the fiscal quarter for this filing
                    fiscal_quarter = determine_fiscal_quarter(date_str, fiscal_year_end_month)
                    st.write(f"Filing is in: {fiscal_quarter}")
                    
                    # Get the text content of the filing
                    response = requests.get(url, headers={"User-Agent": "MyCompanyName Data Research Contact@mycompany.com"})
                    text = response.text
                    
                    # Find paragraphs containing management change patterns
                    management_paragraphs, found_management = find_management_change_paragraphs(text)
                    
                    # Check if we found any management change paragraphs
                    if found_management:
                        st.success(f"✅ Found potential management change information.")
                        
                        # Extract management changes from the highlighted text
                        management_changes = extract_management_changes(management_paragraphs, ticker, client, fiscal_quarter)
                        
                        if management_changes:
                            # Display the extracted information
                            st.markdown("### Extracted Management Changes")
                            st.markdown(management_changes)
                            
                            # Parse to DataFrame
                            df = parse_management_changes_to_df(management_changes, fiscal_quarter)
                            
                            if df is not None:
                                # Add metadata
                                df["Filing Date"] = date_str
                                df["8K Link"] = url
                                results.append(df)
                                st.success("✅ Management changes extracted from this 8-K.")
                            else:
                                st.warning(f"⚠️ No structured management change data found in {url}")
                        else:
                            st.warning(f"⚠️ No management changes found in {url}")
                    else:
                        st.info(f"ℹ️ No management change information detected in this filing.")
                        
                except Exception as e:
                    st.warning(f"Could not process: {url}. Error: {str(e)}")

            if results:
                combined = pd.concat(results, ignore_index=True)
                
                # Preview the table
                st.subheader("🔍 Preview of Extracted Management Changes")
                st.dataframe(combined, use_container_width=True)
                
                # Add download button
                import io
                excel_buffer = io.BytesIO()
                combined.to_excel(excel_buffer, index=False)
                st.download_button("📥 Download Excel", data=excel_buffer.getvalue(), file_name=f"{ticker}_management_changes.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.warning("No management change data extracted from any of the filings.")
