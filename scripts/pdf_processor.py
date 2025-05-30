#!/usr/bin/env python3
"""
PDF Processor Script

This script takes a run name and URL as command line arguments,
finds all PDFs on the given webpage, and processes each one with 
Mistral OCR, saving results in an organized directory structure.

Usage:
    python pdf_processor.py <run_name> <url>

Example:
    python pdf_processor.py "research_batch_1" "https://arxiv.org/abs/2310.06825"

Directory structure:
    <run_name>/
    ├── pdf_1_subdirectory/
    │   ├── extracted_text.md
    │   └── [images...]
    ├── pdf_2_subdirectory/
    │   ├── extracted_text.md
    │   └── [images...]
    └── ...
"""

import sys
import os
from mistralai import Mistral
from dotenv import load_dotenv

# Import functions from the merged ocr_utils module
from ocr_utils import find_pdf_links_on_page, ocr_pdf_with_mistral, create_pdf_output_dir

load_dotenv()

def main():
    if len(sys.argv) != 3:
        print("Usage: python pdf_processor.py <run_name> <url>")
        print("\nExample:")
        print('  python pdf_processor.py "research_batch_1" "https://arxiv.org/abs/2310.06825"')
        sys.exit(1)
    
    run_name = sys.argv[1]
    url = sys.argv[2]
    
    print(f"🚀 Starting PDF processing for run: {run_name}")
    print(f"🔍 Target URL: {url}")
    
    # Check for Mistral API key
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("🛑 Error: MISTRAL_API_KEY environment variable not set.")
        print("   Please set your Mistral API key and try again.")
        print("   You can obtain an API key from https://console.mistral.ai/")
        sys.exit(1)
    
    # Initialize Mistral client
    try:
        client = Mistral(api_key=api_key)
        print("🤖 Mistral client initialized successfully.")
    except Exception as e:
        print(f"🔥 Failed to initialize Mistral client: {e}")
        sys.exit(1)
    
    # Create run directory
    if not os.path.exists(run_name):
        os.makedirs(run_name)
        print(f"📁 Created run directory: {run_name}")
    
    # Find PDF links on the page
    print(f"\n🔎 Searching for PDF links on: {url}")
    pdf_links = find_pdf_links_on_page(url)
    
    if not pdf_links:
        print(f"❌ No PDF links found on {url}")
        sys.exit(1)
    
    print(f"✅ Found {len(pdf_links)} PDF(s) to process:")
    for i, pdf_url in enumerate(pdf_links):
        print(f"  {i + 1}. {pdf_url}")
    
    # Process each PDF
    successful_ocr = 0
    for i, pdf_url in enumerate(pdf_links):
        print(f"\n🔄 Processing PDF {i + 1}/{len(pdf_links)}: {pdf_url}")
        
        # Create output directory for this PDF
        pdf_output_dir = create_pdf_output_dir(pdf_url, run_name, i)
        print(f"📁 Created PDF directory: {pdf_output_dir}")
        
        # Run OCR on the PDF
        markdown_content = ocr_pdf_with_mistral(pdf_url, pdf_output_dir, client)
        
        if markdown_content:
            successful_ocr += 1
            print(f"👍 Successfully processed PDF {i + 1}")
        else:
            print(f"👎 Failed to process PDF {i + 1}")
    
    # Summary
    print(f"\n🏁 Processing complete!")
    print(f"📊 Successfully processed: {successful_ocr}/{len(pdf_links)} PDFs")
    print(f"📁 Results saved in: {run_name}/")
    
    if successful_ocr > 0:
        print(f"\n💡 You can find the extracted content in individual subdirectories:")
        print(f"   - Each PDF has its own subdirectory in {run_name}/")
        print(f"   - Look for 'extracted_text.md' files for the OCR content")
        print(f"   - Images (if any) are saved alongside the markdown files")

if __name__ == "__main__":
    main() 