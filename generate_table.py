#!/usr/bin/env python3
import json

def generate_markdown_table(json_file):
    """Generate markdown table from summary stats JSON"""
    
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Start building the markdown table
    markdown = "| Link | Number of Tokens | Number of Characters | Number of Images |\n"
    markdown += "|------|------------------|---------------------|------------------|\n"
    
    # Add each entry as a row
    for link, stats in data.items():
        tokens = stats['number_of_tokens']
        images = stats['total_images']
        characters = stats['number_of_characters']
        markdown += f"| {link} | {tokens:,} | {characters:,} | {images} |\n"
    
    return markdown

if __name__ == "__main__":
    # Generate the table
    table = generate_markdown_table("univerzitetski_uchebnici/summary_stats.json")
    
    # Print to console
    print(table)
    
    # Also save to file
    with open("summary_table.md", "w") as f:
        f.write(table)
    
    print("\nTable saved to summary_table.md") 