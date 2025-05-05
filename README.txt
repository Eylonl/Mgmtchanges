SEC 8-K Management Changes Extractor
A Streamlit application that extracts information about executive management changes and transitions from SEC 8-K filings.

Overview
This tool allows users to search for and analyze executive management changes (appointments, resignations, retirements, etc.) from companies' SEC 8-K filings. The application uses natural language processing to identify and extract relevant information about management transitions, delivering structured output for easy analysis.

Features
Search for 8-K filings by company ticker
Filter by specific fiscal quarter or number of years back
Automatically detect and extract paragraphs related to management changes
Parse management transitions into structured data with categories:
Type of change (Appointment, Resignation, Retirement, etc.)
Name of executive
New role
Previous role
Effective date
Other key details
Export results to Excel
Fiscal quarter labeling for each change
Installation
Clone this repository:
git clone https://github.com/yourusername/sec-management-changes.git
cd sec-management-changes
Install the required dependencies:
pip install -r requirements.txt
Run the Streamlit app:
streamlit run streamlit_app.py
Usage
Enter a stock ticker (e.g., MSFT, ORCL)
Enter your OpenAI API key (required for extracting information using AI)
Choose filtering options:
Enter a number of years back to search for filings
OR specify a particular fiscal quarter (e.g., 2Q25, Q4FY24)
Click "Extract Management Changes"
Review the extracted management changes
Download the results as an Excel file
How It Works
The application retrieves SEC 8-K filings for the specified company
It identifies sections of the document likely to contain management change information using pattern matching
These specific paragraphs are sent to OpenAI's language model (reducing API costs)
The AI extracts structured information about management changes
Results are parsed into a clean, tabular format with fiscal quarter labeling
Users can review and download the structured data
Requirements
Python 3.8+
Streamlit
Requests
BeautifulSoup4
Pandas
OpenAI API key
Python-dateutil
Notes
SEC API usage is subject to SEC fair access policies
Use a proper User-Agent when making requests to SEC.gov
The application processes 8-K filings, which may contain other information beyond management changes
The AI extraction process may occasionally miss or misinterpret information
License
[Your license information here]

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

