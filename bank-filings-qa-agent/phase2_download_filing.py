"""
Phase 2, Step 2: Download a filing and extract clean text.
 
Concept: A 10-K filing is just a big HTML webpage - full of tags
like <div>, <table>, <p> that are meant for browsers to render,
not for humans (or search systems) to read directly. We use a
library called BeautifulSoup to strip all that markup away and
keep just the visible text, similar to what you'd see if you
selected "Reader Mode" in a browser.
"""

import requests 
from bs4 import BeautifulSoup 

HEADERS = {"User-Agent": "Palash palashaggarwal@gmail.com"}

def build_filling_url(clk:str, accession_number:str, primary_document:str) -> str:
    """
    SEC filing URLs follow a predictable pattern once you have the
    three pieces of info: CIK, accession number (unique ID for this
    specific filing), and the primary document filename.
 
    Accession numbers come formatted with dashes (e.g. 0000019617-24-000123)
    but the URL path needs them WITHOUT dashes - a common gotcha.
    """
    
    cik_no_padding = str(int(cik))
    accession_no_dashes = accession_number.replace("-", "")
    return (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{cik_no_padding}/{accession_no_dashes}/{primary_document}"
        )

def fetch_and_clean_text(url:str) -> str:
    """
    Download the HTML page, then extract just the readable text - 
    dropping scripts, styles, and all the tag clutter. 
    """
    
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.content, "html.parser")
    
     # Remove elements that contain no useful text content
    for tag in soup(["script", "style"]):
        tag.decompose()
 
    # get_text() pulls out just the visible text, "\n" joins pieces
    # that were in separate tags so we don't get one giant run-on line
    text = soup.get_text(separator="\n")
 
    # Collapse multiple blank lines down to one, for readability
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
 
    return cleaned

if __name__ == "__main__":
    
    cik = "0000019617"
    accession_number = "0000019617-24-000123"
    primary_document = "0000019617-24-000123.txt"
    url = build_filling_url(cik, accession_number, primary_document)
    print(f"Downloading filing from: {url}")
    text = fetch_and_clean_text(url)
    print(f"\nFirst 500 characters of cleaned text:")
    print(text[:500])
    print(f"\n...truncated for brevity. Full text saved to 'filing.txt'.")
    with open("filing.txt", "w", encoding="utf-8") as f:
        f.write(text)
    