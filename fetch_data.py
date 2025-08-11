import os
import requests
import time
from bs4 import BeautifulSoup

# --- Configuration ---
# List of company tickers to fetch filings for
TARGET_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "JNJ", "WMT", 
    "PG", "UNH", "HD", "V", "MA", "BAC", "XOM", "CVX", "KO", "PEP", "MCD", 
    "DIS", "NFLX", "ADBE", "CRM", "ORCL", "CSCO", "INTC", "PFE", "MRK"
] 
# Types of filings to download
FILING_TYPES = [
    "S-1", "S-4", "10-K", "10-Q", "8-K", "3", "4", "5", "13D", 
    "DEF 14A", "144", "11-K", "13F", "1-F", "20-F"
]
# Number of recent filings to get for each type
NUM_FILINGS_TO_GET = 5
# Directory to save the filings
DATA_DIR = "data"
# SEC EDGAR base URLs
SEC_CIK_LOOKUP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{}.json"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

# --- Helper Functions ---

def get_company_ciks(target_tickers):
    """
    Downloads the CIK lookup file from the SEC and maps tickers to CIKs.
    """
    headers = {"User-Agent": "MALAVATH HANMANTH NAYAK hanmanthnayak.95@gmail.com"}
    response = requests.get(SEC_CIK_LOOKUP_URL, headers=headers)
    response.raise_for_status()
    all_companies = response.json()
    
    cik_map = {}
    for company in all_companies.values():
        if company['ticker'] in target_tickers:
            cik_map[company['ticker']] = str(company['cik_str']).zfill(10)
    return cik_map

def download_and_clean_filing(url, ticker, form_type, accession_no):
    """
    Downloads a single filing, extracts the text content, and saves it
    into a nested directory structure.
    """
    headers = {"User-Agent": "MALAVATH HANMANTH NAYAK hanmanthnayak.95@gmail.com"}
    try:
        # Create the specific directory for the filing
        filing_dir = os.path.join(DATA_DIR, ticker, form_type.replace(" ", "_"))
        os.makedirs(filing_dir, exist_ok=True)
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        filing_text = soup.get_text()
        
        filename = f"{filing_dir}/{accession_no}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(filing_text)
        print(f"Successfully downloaded: {filename}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
    except Exception as e:
        print(f"Error processing {url}: {e}")

# --- Main Execution ---

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("Fetching CIKs for target companies...")
    cik_map = get_company_ciks(TARGET_TICKERS)
    
    if not cik_map:
        print("Could not find CIKs for the target tickers. Exiting.")
        exit()
        
    print(f"Found CIKs: {cik_map}")

    for ticker, cik in cik_map.items():
        print(f"\n--- Processing {ticker} (CIK: {cik}) ---")
        
        headers = {"User-Agent": "MALAVATH HANMANTH NAYAK hanmanthnayak.95@gmail.com"}
        submissions_url = SEC_SUBMISSIONS_URL.format(cik)
        
        try:
            response = requests.get(submissions_url, headers=headers)
            response.raise_for_status()
            submissions_data = response.json()
            
            for filing_type in FILING_TYPES:
                print(f"  Fetching recent {filing_type} filings...")
                
                recent_filings = submissions_data['filings']['recent']
                filings_found = 0
                
                for i in range(len(recent_filings['accessionNumber'])):
                    if filings_found >= NUM_FILINGS_TO_GET:
                        break
                        
                    if recent_filings['form'][i] == filing_type:
                        accession_no_raw = recent_filings['accessionNumber'][i]
                        accession_no_clean = accession_no_raw.replace('-', '')
                        primary_document = recent_filings['primaryDocument'][i]
                        
                        filing_url = f"{SEC_ARCHIVES_URL}/{cik}/{accession_no_clean}/{primary_document}"
                        
                        print(f"    Found filing: {primary_document}. Downloading...")
                        download_and_clean_filing(filing_url, ticker, filing_type, accession_no_raw)
                        filings_found += 1
                        
                        # Respect SEC's rate limits (10 requests per second)
                        time.sleep(0.1) 
                
                if filings_found == 0:
                    print(f"  No recent {filing_type} filings found for {ticker}.")

        except requests.exceptions.RequestException as e:
            print(f"  Could not fetch submissions for {ticker}: {e}")
        except Exception as e:
            print(f"  An error occurred while processing {ticker}: {e}")
            
    print("\nData fetching complete.")

