import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote # Added quote for better URL handling
from mistralai import Mistral
from dotenv import load_dotenv
import re

# --- Configuration ---
# !!! IMPORTANT: Set your Mistral API Key as an environment variable MISTRAL_API_KEY !!!
# You can get an API key from https://console.mistral.ai/
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_OCR_MODEL = "mistral-ocr-latest" # Check Mistral documentation for the latest model

# !!! IMPORTANT: Replace with the list of URLs you want to scrape !!!
URLS_TO_SCRAPE = [
    "https://e-biblioteka.mk/%d0%b1%d0%b8%d0%b1%d0%bb%d0%b8%d0%be%d1%82%d0%b5%d0%ba%d0%b0/",
    # "https://another-example.org/research-papers",
    # Add more URLs here. For testing, you might use a page you know has direct PDF links.
    # e.g., an arXiv abstract page might link to its PDF.
]

def find_pdf_links_on_page(page_url):
    """
    Fetches a webpage and extracts all links that lead to PDF files.
    Uses multiple strategies:
    1. Direct .pdf links
    2. Links that redirect to PDFs (checks content-type)
    3. Common academic site patterns (arXiv, etc.)
    4. Links with PDF-related keywords
    """
    pdf_links = set()
    print(f"Attempting to fetch: {page_url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(page_url, headers=headers, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Strategy 1: Direct PDF links
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if href.lower().endswith('.pdf'):
                absolute_pdf_url = urljoin(page_url, quote(href, safe='/:?=&%'))
                pdf_links.add(absolute_pdf_url)

        # Strategy 2: Check for arXiv specific patterns
        if 'arxiv.org' in page_url:
            # Convert arXiv abstract URLs to PDF URLs
            arxiv_match = re.search(r'arxiv\.org/abs/([0-9]+\.[0-9]+)', page_url)
            if arxiv_match:
                arxiv_id = arxiv_match.group(1)
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                pdf_links.add(pdf_url)
                print(f"  ArXiv pattern detected, added: {pdf_url}")

        # Strategy 3: Look for links that might lead to PDFs (broader search)
        potential_pdf_links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            link_text = a_tag.get_text().lower()
            
            # Check for PDF-related keywords in link text or href
            pdf_indicators = ['pdf', 'download', 'paper', 'document', 'full text', 'view pdf', 'get pdf']
            if any(indicator in link_text for indicator in pdf_indicators) or any(indicator in href.lower() for indicator in pdf_indicators):
                absolute_url = urljoin(page_url, href)
                potential_pdf_links.append(absolute_url)
        
        # Strategy 4: Test potential links by checking their content-type
        print(f"  Found {len(potential_pdf_links)} potential PDF links to verify...")
        for potential_url in potential_pdf_links[:10]:  # Limit to avoid too many requests
            if is_pdf_url(potential_url):
                pdf_links.add(potential_url)
                print(f"  Verified PDF: {potential_url}")
        
        # Strategy 5: Look for common academic site patterns
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            absolute_url = urljoin(page_url, href)
            
            # Common patterns for academic sites
            patterns = [
                r'\.pdf$',  # Direct PDF
                r'/pdf/',   # PDF in path
                r'download.*pdf',  # Download PDF
                r'viewPDF',  # View PDF
                r'getPDF',   # Get PDF
                r'fulltext.*pdf',  # Fulltext PDF
            ]
            
            for pattern in patterns:
                if re.search(pattern, href, re.IGNORECASE):
                    pdf_links.add(absolute_url)
                    break

        if not pdf_links:
            print(f"No PDF links found on {page_url}")
        else:
            print(f"Found {len(pdf_links)} unique PDF link(s) on {page_url}:")
            for link in pdf_links:
                print(f"  - {link}")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error fetching {page_url}: {http_err} - Status: {http_err.response.status_code}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error fetching {page_url}: {conn_err}")
    except requests.exceptions.Timeout:
        print(f"Timeout error fetching {page_url}")
    except requests.exceptions.RequestException as req_err:
        print(f"Error fetching {page_url}: {req_err}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {page_url}: {e}")
    
    return list(pdf_links)

def is_pdf_url(url):
    """
    Check if a URL points to a PDF by making a HEAD request and checking content-type.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
        content_type = response.headers.get('content-type', '').lower()
        
        # Check if content-type indicates PDF
        if 'application/pdf' in content_type:
            return True
        
        # Some servers don't set proper content-type, check URL patterns as fallback
        if url.lower().endswith('.pdf'):
            return True
            
        # Check for PDF in content-disposition header
        content_disposition = response.headers.get('content-disposition', '').lower()
        if 'pdf' in content_disposition:
            return True
            
    except Exception as e:
        print(f"    Could not verify {url}: {e}")
        # If we can't verify, but URL looks like a PDF, include it anyway
        if url.lower().endswith('.pdf'):
            return True
    
    return False

def ocr_pdf_with_mistral(pdf_url, client):
    """
    Sends a PDF URL to Mistral's OCR API and returns the extracted markdown content.
    """
    print(f"\n📄 Processing PDF with Mistral OCR: {pdf_url}")
    try:
        ocr_response = client.ocr.process(
            model=MISTRAL_OCR_MODEL,
            document={
                "type": "document_url",
                "document_url": pdf_url
            }
            # Optional: include_image_base64=True if you want image data
        )

        full_markdown = []
        if hasattr(ocr_response, 'pages') and ocr_response.pages:
            print(f"✅ Successfully received OCR response with {len(ocr_response.pages)} page(s) for {pdf_url}.")
            for i, page in enumerate(ocr_response.pages):
                page_content = f"--- Page {i+1} of {len(ocr_response.pages)} (from {pdf_url}) ---\n"
                if hasattr(page, 'markdown') and page.markdown:
                    page_content += page.markdown
                    # print(page.markdown) # Uncomment to print live page content
                else:
                    page_content += "(No markdown content for this page)"
                full_markdown.append(page_content)
            return "\n\n".join(full_markdown)
        else:
            print(f"⚠️ Mistral OCR did not return any pages or expected content for {pdf_url}.")
            print(f"   Raw Mistral OCR response object: {ocr_response}") # Helps in debugging
            return None
    except Exception as e:
        print(f"❌ An unexpected error occurred during Mistral OCR for {pdf_url}: {e}")
    return None

def main():
    print("🚀 Starting PDF scraping and OCR process...")

    if not MISTRAL_API_KEY:
        print("🛑 Error: MISTRAL_API_KEY environment variable not set.")
        print("   Please set your Mistral API key and try again.")
        print("   You can obtain an API key from https://console.mistral.ai/")
        return

    if not URLS_TO_SCRAPE or all("example.com" in url for url in URLS_TO_SCRAPE):
        print("⚠️ Warning: The `URLS_TO_SCRAPE` list is empty or contains only example URLs.")
        print("   Please update it with the actual URLs you want to process.")
        print("   Script will exit now.")
        return

    try:
        client = Mistral(api_key=MISTRAL_API_KEY)
        print("🤖 Mistral client initialized successfully.")
    except Exception as e:
        print(f"🔥 Failed to initialize Mistral client: {e}")
        return

    all_ocr_results = {} # {pdf_url: ocr_markdown_content}

    for web_page_url in URLS_TO_SCRAPE:
        print(f"\n🔎 Scraping page for PDF links: {web_page_url}")
        pdf_links_on_page = find_pdf_links_on_page(web_page_url)
        
        if pdf_links_on_page:
            print(f"Found {len(pdf_links_on_page)} PDF(s) on {web_page_url}. Starting OCR...")
            for pdf_url in pdf_links_on_page:
                ocr_markdown = ocr_pdf_with_mistral(pdf_url, client)
                if ocr_markdown:
                    all_ocr_results[pdf_url] = ocr_markdown
                    print(f"👍 OCR successful for: {pdf_url}")
                else:
                    print(f"👎 OCR failed or produced no content for: {pdf_url}")
                # Consider adding a small delay here if you encounter rate limiting issues,
                # though the Mistral client library might handle some retry logic.
                # import time
                # time.sleep(1) # e.g., 1-second delay
        else:
            print(f"No PDF links found on {web_page_url} to process.")

    print("\n\n🏁 --- OCR Processing Complete --- 🏁")
    if all_ocr_results:
        print(f"Successfully processed and retrieved OCR content for {len(all_ocr_results)} PDF(s).")
        
        # Example: Save each OCR result to a Markdown file
        output_dir = "ocr_outputs"
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n💾 Saving OCR results to '{output_dir}/' directory...")

        for pdf_url, markdown_content in all_ocr_results.items():
            try:
                # Create a somewhat safe filename from the URL
                file_name_base = "".join(c if c.isalnum() else "_" for c in pdf_url.split('/')[-1])
                if not file_name_base.endswith(".pdf"): # Should always be true based on earlier logic
                    file_name_base += "_pdf"
                file_name = os.path.join(output_dir, file_name_base.replace('.pdf', '.md'))
                
                with open(file_name, 'w', encoding='utf-8') as f:
                    f.write(f"# OCR Result for: {pdf_url}\n\n")
                    f.write(markdown_content)
                print(f"   Saved: {file_name}")
            except Exception as e:
                print(f"   Error saving file for {pdf_url}: {e}")
    else:
        print("No PDFs were successfully processed by OCR, or no PDF links were found.")

if __name__ == "__main__":
    # Example: Add a test URL if URLS_TO_SCRAPE is empty for quick testing.
    # Make sure this URL actually has public PDF links.
    # This is just for a placeholder if the user doesn't modify the list.
    if not URLS_TO_SCRAPE or all("example.com" in url for url in URLS_TO_SCRAPE):
         print("\n🔔 Reminder: `URLS_TO_SCRAPE` is empty or has placeholder URLs.")
         print("   Please edit the script to add the URLs you want to scrape.")
         # You could add a default test URL here if you have one, e.g.:
         # URLS_TO_SCRAPE.append("https://arxiv.org/abs/2310.06825") # Mistral-7B paper, has PDF link

    main()