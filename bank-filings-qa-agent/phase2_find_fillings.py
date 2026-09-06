"""
Phase 2, Step 1: Find a bank's recent 10-K filings on SEC EDGAR.
 
Concept: every company has a CIK (Central Index Key) - a unique ID
number the SEC uses instead of company names. We look up the CIK,
then ask EDGAR "show me this company's filing history," then filter
that list down to just 10-K filings.
 
SEC EDGAR requires every request to include a User-Agent header
identifying who's making the request (name + email). This is not
optional - requests without it get blocked.
"""

import requests
import json 

# SEC requires a real identifying User-Agent - replace with your info

HEADERS = {"User-Agent": "Palash palashaggarwal@gmail.com"}

# Example CIKs for a few major banks
# https://www.sec.gov/cgi-bin/browse-edgar - search company name,

BANK_CIKS = {
    "JPM": "0000019617",   # JPMorgan Chase
    "BAC": "0000070858",   # Bank of America
    "WFC": "0000072971",   # Wells Fargo
}

def get_filling_history(cik: str):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()

def get_recent_10k_filings(cik: str, limit: int =1):
    """
    Filter a company's filing history to just 10-Ks.
    """
    data = get_filling_history(cik)
    recent = data["filings"]["recent"]
    
    # The API returns parallel lists (all same length, same order) -
    # recent["form"][i] corresponds to recent["accessionNumber"][i],
    # recent["filingDate"][i], etc. for the SAME filing at index i.
    
    filings = []
    for i, form_type in enumerate(recent["form"]):
        if form_type == "10-K":
            filings.append({
                "accessionNumber": recent["accessionNumber"][i],
                "filingDate": recent["filingDate"][i],
                "reportDate": recent["reportDate"][i],
                "acceptanceDateTime": recent["acceptanceDateTime"][i],
                "act": recent["act"][i],
                "fileNumber": recent["fileNumber"][i],
                "filmNumber": recent["filmNumber"][i],
                "size": recent["size"][i],
                "primaryDocument": recent["primaryDocument"][i],
            })
        
        if len(filings) >= limit:
            break
    
    return filings 

if __name__ == "__main__":
    for ticker, cik in BANK_CIKS.items():
        print(f"\nFetching 10-K filings for {ticker} (CIK{cik})...")
        filings = get_recent_10k_filings(cik, limit=1)
        for f in filings:
            print(json.dumps(f, indent=2))