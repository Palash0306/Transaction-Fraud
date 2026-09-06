"""
Debug script: fetch ONE real filing and inspect the raw extracted
text directly, before any chunking happens. This tells us whether
the problem is in fetching/parsing, or somewhere later.
"""

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Palash Aggarwal palash@example.com"
}


def get_filing_history(cik: str):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()


def get_recent_10k_filings(cik: str, limit: int = 1):
    data = get_filing_history(cik)
    recent = data["filings"]["recent"]
    filings = []
    for i, form_type in enumerate(recent["form"]):
        if form_type == "10-K":
            filings.append({
                "accessionNumber": recent["accessionNumber"][i],
                "filingDate": recent["filingDate"][i],
                "primaryDocument": recent["primaryDocument"][i],
            })
        if len(filings) >= limit:
            break
    return filings


def build_filing_url(cik: str, accession_number: str, primary_document: str) -> str:
    cik_no_padding = str(int(cik))
    accession_no_dashes = accession_number.replace("-", "")
    return (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_no_padding}/{accession_no_dashes}/{primary_document}"
    )


if __name__ == "__main__":
    cik = "0000019617"  # JPM

    filings = get_recent_10k_filings(cik, limit=1)
    filing = filings[0]
    print("Filing metadata found:")
    print(filing)

    url = build_filing_url(cik, filing["accessionNumber"], filing["primaryDocument"])
    print("\nFetching URL:", url)

    response = requests.get(url, headers=HEADERS)
    print("HTTP status code:", response.status_code)
    print("Response length (raw bytes):", len(response.content))

    # Save the RAW html so you can open it in a browser and eyeball it directly
    with open("debug_raw.html", "wb") as f:
        f.write(response.content)
    print("\nSaved raw HTML to debug_raw.html - open this in a browser to check "
          "whether it's really the 10-K or a redirect/error page.")

    soup = BeautifulSoup(response.content, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)

    print("\nTotal cleaned text length (characters):", len(cleaned))
    print("\n--- First 1000 characters ---")
    print(cleaned[:1000])
    print("\n--- Characters 5000 to 6000 (further into the doc) ---")
    print(cleaned[5000:6000])

    with open("debug_cleaned.txt", "w", encoding="utf-8") as f:
        f.write(cleaned)
    print("\nSaved full cleaned text to debug_cleaned.txt - open this and "
          "search (Cmd+F) for a term like 'loan loss' or 'net income' to "
          "confirm real financial content is actually in there.")