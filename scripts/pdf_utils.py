import sys
import os
import base64
from datetime import datetime
from urllib.parse import urlparse
from mistralai import Mistral
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, quote

load_dotenv()

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

def save_images_from_page(page, page_index, output_dir):
    """
    Extract and save images from a page to the specified directory.
    Returns a mapping of image IDs to saved file paths.
    """
    image_paths = {}
    
    if hasattr(page, 'images') and page.images:
        print(f"  Found {len(page.images)} image(s) on page {page_index + 1}")
        
        for img_idx, image in enumerate(page.images):
            if hasattr(image, 'image_base64') and image.image_base64:
                # Create filename using image ID if available, otherwise use index
                if hasattr(image, 'id') and image.id:
                    filename = image.id
                else:
                    filename = f"{img_idx}.jpeg"
                
                filepath = os.path.join(output_dir, filename)
                
                try:
                    # Decode base64 and save image
                    image_data = base64.b64decode(image.image_base64)
                    with open(filepath, 'wb') as f:
                        f.write(image_data)
                    
                    # Store mapping of image ID to relative filename for markdown
                    image_id = image.id if hasattr(image, 'id') and image.id else f"img_{img_idx}"
                    image_paths[image_id] = filename
                    
                    print(f"    Saved image: {filename}")
                    
                    # Print image annotation if available
                    if hasattr(image, 'image_annotation') and image.image_annotation:
                        print(f"    Annotation: {image.image_annotation}")
                        
                except Exception as e:
                    print(f"    Error saving image {filename}: {e}")
    
    return image_paths

def create_pdf_output_dir(pdf_url, run_name, pdf_index):
    """
    Create a unique output directory for a specific PDF within the run directory.
    Returns the directory path.
    """
    # Clean the PDF URL to create a valid directory name
    parsed_url = urlparse(pdf_url)
    pdf_filename = os.path.basename(parsed_url.path)
    if pdf_filename.endswith('.pdf'):
        pdf_filename = pdf_filename[:-4]  # Remove .pdf extension
    
    # Clean filename for directory use
    cleaned_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in pdf_filename)
    if not cleaned_name:
        cleaned_name = f"pdf_{pdf_index + 1}"
    
    # Create directory path
    run_dir = run_name
    pdf_dir = os.path.join(run_dir, f"{cleaned_name}_{pdf_index + 1}")
    
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
    
    return pdf_dir

def ocr_pdf_with_mistral_custom(pdf_url, output_dir, client):
    """
    Modified version of OCR function that saves to a specified directory.
    Based on mistral_ocr from ocr_pdf.py but with custom output directory.
    """
    print(f"\n📄 Processing PDF with Mistral OCR: {pdf_url}")
    try:
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": pdf_url
            },
            include_image_base64=True
        )

        full_markdown = []
        all_image_paths = {}
        
        if hasattr(ocr_response, 'pages') and ocr_response.pages:
            print(f"✅ Successfully received OCR response with {len(ocr_response.pages)} page(s).")
            
            for i, page in enumerate(ocr_response.pages):
                # Extract and save images from this page
                page_image_paths = save_images_from_page(page, i, output_dir)
                all_image_paths.update(page_image_paths)
                
                if hasattr(page, 'markdown') and page.markdown:
                    full_markdown.append(page.markdown)
            
            # Combine all markdown content
            complete_markdown = "\n\n".join(full_markdown)
            
            # Save markdown to file
            markdown_path = os.path.join(output_dir, "extracted_text.md")
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(f"# OCR Result from: {pdf_url}\n\n")
                f.write(complete_markdown)
            
            print(f"✅ Processed {len(all_image_paths)} total image(s) across all pages.")
            print(f"✅ Results saved to: {output_dir}")
            
            return complete_markdown
        else:
            print("⚠️ No pages found in OCR response")
            return ""
            
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return ""