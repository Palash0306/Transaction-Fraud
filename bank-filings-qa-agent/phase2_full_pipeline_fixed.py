"""
Phase 2 (fixed): uses the full submission .txt file (proven to work),
and correctly unwraps the list returned by get_recent_10k_filings
instead of treating it as a single dict.
"""

import os
import json
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Palash Aggarwal palashaggarwal@gmail.com"
}

BANK_CIKS = {
    "JPM": "0000019617",
    "BAC": "0000070858",
    "WFC": "0000072971",
}


# ---------------------------------------------------------------
# Step 1: Find the real filing
# ---------------------------------------------------------------
def get_filing_history(cik: str):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()


def get_recent_10k_filings(cik: str, limit: int = 1):
    """
    Returns a LIST of dicts describing the most recent 10-K filing(s).
    Note the plural name and plural return type - this is what was
    tripping up ingest_bank() before: it expected a single dict but
    this correctly returns a list (even when limit=1, it's a list
    containing one dict).
    """
    data = get_filing_history(cik)
    recent = data["filings"]["recent"]

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


# ---------------------------------------------------------------
# Step 2: Download the full submission .txt file (proven to work)
# ---------------------------------------------------------------
def build_filing_txt_url(cik: str, accession_number: str, primary_document: str) -> str:
    """
    The "complete submission text file" lives at a predictable URL:
    .../data/<cik-no-padding>/<accession-no-dashes>.txt
    This is the same pattern that already worked for you manually.
    """
    cik_no_padding = str(int(cik))
    accession_no_dashes = accession_number.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_no_padding}/{accession_no_dashes}/{primary_document}"
    )


def fetch_and_clean_text(url: str) -> str:
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    return cleaned


# ---------------------------------------------------------------
# Step 3: Chunk the real text
# ---------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------
# Step 4: Save the real chunks
# ---------------------------------------------------------------
def save_chunks(chunks: list, ticker: str, filing_date: str, output_dir: str = "data/chunks"):
    os.makedirs(output_dir, exist_ok=True)
    for i, chunk in enumerate(chunks):
        record = {
            "ticker": ticker,
            "filing_date": filing_date,
            "chunk_index": i,
            "text": chunk,
        }
        filename = f"{ticker}_{filing_date}_chunk{i}.json"
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump(record, f, indent=2)
    print(f"Saved {len(chunks)} REAL chunks for {ticker} to {output_dir}/")


# ---------------------------------------------------------------
# Run the full pipeline, one bank at a time
# ---------------------------------------------------------------
def ingest_bank(ticker: str, cik: str):
    print(f"\n=== Ingesting {ticker} ===")

    filings = get_recent_10k_filings(cik, limit=1)

    # get_recent_10k_filings returns a LIST - even with limit=1,
    # it's a list containing one dict, not the dict itself.
    # This is the exact unwrap step that was missing before.
    if not filings:
        print(f"No 10-K found for {ticker}, skipping.")
        return

    filing = filings[0]   # unwrap: grab the single filing dict out of the list

    print(f"Found filing: accession {filing['accessionNumber']}, "
          f"filed {filing['filingDate']}")

    accession_number = filing["accessionNumber"]
    primary_document = filing["primaryDocument"]
    url = build_filing_txt_url(cik, accession_number, primary_document)
    print(f"Downloading: {url}")

    text = fetch_and_clean_text(url)
    print(f"Extracted {len(text)} characters of REAL filing text.")

    chunks = chunk_text(text, chunk_size=500, overlap=50)
    print(f"Created {len(chunks)} chunks.")

    save_chunks(chunks, ticker=ticker, filing_date=filing["filingDate"])


if __name__ == "__main__":
    for ticker, cik in BANK_CIKS.items():
        ingest_bank(ticker, cik)