import os
import base64
import re
from datetime import datetime
from urllib.parse import urlparse
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)


def create_unique_output_dir(pdf_url):
    """
    Create a unique output directory for the PDF processing.
    Returns the directory path.
    """
    # Clean the URL to create a valid directory name
    parsed_url = urlparse(pdf_url)
    cleaned_url = parsed_url.path.replace('/', '_').replace('.', '_')
    if cleaned_url.startswith('_'):
        cleaned_url = cleaned_url[1:]
    
    # Add timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{cleaned_url}_{timestamp}"
    
    output_dir = os.path.join("ocr_outputs", dir_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return output_dir


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


def mistral_ocr(pdf_url):
    """
    Process a PDF using Mistral OCR and save the results to a unique directory.
    Returns a tuple of (markdown_content, output_directory_path).
    """
    # Create unique output directory
    output_dir = create_unique_output_dir(pdf_url)
    
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
                    # print(page.markdown) # Uncomment to print live page content
            
            # Combine all markdown content
            complete_markdown = "\n\n".join(full_markdown)
            
            # Save markdown to file
            markdown_path = os.path.join(output_dir, "extracted_text.md")
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(f"# OCR Result from: {pdf_url}\n\n")
                f.write(complete_markdown)
            
            print(f"✅ Processed {len(all_image_paths)} total image(s) across all pages.")
            print(f"✅ Results saved to: {output_dir}")
            
            return complete_markdown, output_dir
        else:
            print("⚠️ No pages found in OCR response")
            return "", output_dir
            
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return "", output_dir


def process_pdf_to_markdown(pdf_url):
    """
    Convenience function to process a PDF and return just the markdown content.
    """
    markdown_content, _ = mistral_ocr(pdf_url)
    return markdown_content

if __name__ == "__main__":
    mistral_ocr('https://arxiv.org/pdf/2503.09573')