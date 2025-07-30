"""
This script is used to benchmark the MMMU dataset with the OpenRouter API.
It will save the results to a JSON file and calculate accuracy by comparing
model predictions with ground truth answers.

Usage:
python benchmark_mmmu.py --models google/gemma-3-4b-it qwen/qwen-2.5-vl-7b-instruct mistralai/pixtral-12b google/gemma-3-12b-it google/gemma-3-27b-it meta-llama/llama-3.2-11b-vision-instruct --output overnight_run_results.json

"""


import os
import json
import time
from pathlib import Path
from scripts.openrouter import answer_question
import re
from tqdm import tqdm
import argparse

def parse_options_string(options_str):
    """Parse the options string and format it nicely"""
    try:
        # Remove brackets and split by quotes
        options_str = options_str.strip("[]'\"")
        options = [opt.strip("'\"") for opt in options_str.split("', '")]
        
        # Format as A, B, C, D
        formatted_options = ""
        for i, opt in enumerate(options):
            letter = chr(ord('A') + i)
            formatted_options += f"{letter}. {opt}\n"
        
        return formatted_options.strip()
    except:
        return options_str

def extract_answer_letter(response_text):
    """Extract the answer letter (A, B, C, D) from the response"""
    if isinstance(response_text, dict):
        if 'choices' in response_text and len(response_text['choices']) > 0:
            response_text = response_text['choices'][0]['message']['content']
        else:
            return None
    
    # First, look for answers formatted between ### markers (e.g., ### A ###)
    match = re.search(r'###\s*([A-E])\s*###', response_text.upper())
    if match:
        return match.group(1)
    
    # Fallback: Look for patterns like "A", "B", "C", "D" in the response
    match = re.search(r'\b([A-D])\b', response_text.upper())
    if match:
        return match.group(1)
    
    return None

def run_benchmark(mmmu_path, model, all_results, output_file, max_questions_per_subject=None, subjects_filter=None):
    """Run the MMMU benchmark for a specific model"""
    
    # Initialize results for this model if not already present
    if model not in all_results:
        all_results[model] = {
            'model': model,
            'total_questions': 0,
            'correct_answers': 0,
            'failed_requests': 0,
            'subject_results': {},
            'detailed_results': []
        }
    
    results = all_results[model]
    
    # Get all subject folders
    subject_folders = [d for d in os.listdir(mmmu_path) 
                      if os.path.isdir(os.path.join(mmmu_path, d)) and d.startswith('mmmu_')]
    
    if subjects_filter:
        subject_folders = [s for s in subject_folders if s in subjects_filter]
    
    print(f"Found {len(subject_folders)} subjects to evaluate")
    
    for subject_folder in tqdm(subject_folders, desc="Processing subjects"):
        subject_path = os.path.join(mmmu_path, subject_folder)
        
        # Find the JSONL file
        jsonl_files = [f for f in os.listdir(subject_path) if f.endswith('.jsonl')]
        if not jsonl_files:
            print(f"No JSONL file found in {subject_folder}")
            continue
            
        jsonl_file = jsonl_files[0]
        jsonl_path = os.path.join(subject_path, jsonl_file)
        
        subject_results = {
            'total': 0,
            'correct': 0,
            'failed': 0
        }
        
        print(f"\nProcessing {subject_folder}...")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            questions_processed = 0
            
            for line_num, line in enumerate(f):
                if max_questions_per_subject and questions_processed >= max_questions_per_subject:
                    break
                    
                try:
                    data = json.loads(line)
                    
                    # Get the Macedonian question and options
                    question = data.get('translated_question_mk', data.get('question', ''))
                    options_raw = data.get('translated_options_mk', data.get('options', ''))
                    
                    # Format the options
                    options = parse_options_string(options_raw)
                    
                    # Get the image path
                    image_1 = data.get('image_1')
                    if not image_1:
                        continue  # Skip questions without images
                    
                    # Construct full image path
                    image_path = os.path.join(subject_path, 'mmmu_media', os.path.basename(image_1))
                    
                    if not os.path.exists(image_path):
                        print(f"Image not found: {image_path}")
                        continue
                    
                    print(f"  Question {questions_processed + 1}: {data.get('id', 'unknown')}")
                    
                    # Call the OpenRouter API
                    try:
                        response = answer_question(question, options, image_path, model)
                        print(f"Response: {response}")
                        predicted_answer = extract_answer_letter(response)
                        
                        # Get the ground truth answer
                        ground_truth = data.get('answer', '').strip().upper()
                        
                        if predicted_answer:
                            subject_results['total'] += 1
                            results['total_questions'] += 1
                            
                            # Check if the predicted answer is correct
                            is_correct = predicted_answer == ground_truth
                            if is_correct:
                                subject_results['correct'] += 1
                                results['correct_answers'] += 1
                            
                            # Store detailed result
                            result_detail = {
                                'subject': subject_folder,
                                'question_id': data.get('id', f"{subject_folder}_{line_num}"),
                                'question': question,
                                'options': options,
                                'predicted_answer': predicted_answer,
                                'ground_truth_answer': ground_truth,
                                'is_correct': is_correct,
                                'response': response,
                                'image_path': image_path
                            }
                            results['detailed_results'].append(result_detail)
                            
                            status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
                            print(f"    Predicted: {predicted_answer}, Ground Truth: {ground_truth} - {status}")
                        else:
                            subject_results['failed'] += 1
                            results['failed_requests'] += 1
                            print(f"    Failed to extract answer from response")
                            
                    except Exception as e:
                        subject_results['failed'] += 1
                        results['failed_requests'] += 1
                        print(f"    API call failed: {str(e)}")
                    
                    questions_processed += 1
                    
                    # Save results continuously after each question
                    save_results_continuously(all_results, output_file)
                    
                    # Add a small delay to avoid rate limiting
                    time.sleep(1)
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON in {subject_folder}, line {line_num}: {e}")
                    continue
        
        results['subject_results'][subject_folder] = subject_results
        subject_accuracy = (subject_results['correct'] / subject_results['total'] * 100) if subject_results['total'] > 0 else 0
        print(f"  {subject_folder}: {subject_results['total']} successful, {subject_results['correct']} correct ({subject_accuracy:.1f}%), {subject_results['failed']} failed")
        
        # Save results after completing each subject
        save_results_continuously(all_results, output_file)

def save_results(results, output_file):
    """Save benchmark results to a file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def load_existing_results(output_file):
    """Load existing results from file if it exists"""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}

def save_results_continuously(all_results, output_file):
    """Save results continuously after each question"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Failed to save results continuously: {e}")

def print_summary(results):
    """Print a summary of the benchmark results"""
    print("\n" + "="*50)
    print(f"BENCHMARK SUMMARY - {results['model']}")
    print("="*50)
    print(f"Total questions processed: {results['total_questions']}")
    print(f"Correct answers: {results['correct_answers']}")
    print(f"Failed requests: {results['failed_requests']}")
    
    # Calculate accuracy
    if results['total_questions'] > 0:
        accuracy = (results['correct_answers'] / results['total_questions']) * 100
        print(f"Accuracy: {accuracy:.1f}% ({results['correct_answers']}/{results['total_questions']})")
    else:
        print("Accuracy: 0% (no successful questions)")
    
    # Calculate success rate (API calls that returned valid responses)
    total_attempted = results['total_questions'] + results['failed_requests']
    if total_attempted > 0:
        success_rate = (results['total_questions'] / total_attempted) * 100
        print(f"API Success rate: {success_rate:.1f}% ({results['total_questions']}/{total_attempted})")
    else:
        print("API Success rate: 0% (no attempts)")
    
    print("\nPer-subject results:")
    for subject, subject_results in results['subject_results'].items():
        total = subject_results['total'] + subject_results['failed']
        api_success_rate = (subject_results['total'] / total * 100) if total > 0 else 0
        
        # Calculate subject accuracy
        if subject_results['total'] > 0:
            subject_accuracy = (subject_results['correct'] / subject_results['total']) * 100
            print(f"  {subject}: {subject_results['correct']}/{subject_results['total']} correct ({subject_accuracy:.1f}% accuracy), {subject_results['failed']} failed ({api_success_rate:.1f}% API success)")
        else:
            print(f"  {subject}: 0/0 correct (0% accuracy), {subject_results['failed']} failed ({api_success_rate:.1f}% API success)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MMMU benchmark with OpenRouter')
    parser.add_argument('--mmmu_path', default='/root/synglot/evaluation', help='Path to MMMU dataset')
    parser.add_argument('--max_per_subject', type=int, default=None, help='Maximum questions per subject (None for all questions, use a number for testing)')
    parser.add_argument('--subjects', nargs='+', help='Specific subjects to test (e.g., mmmu_Art_Theory)')
    parser.add_argument('--models', nargs='+', default=['google/gemma-3-4b-it'], help='Models to benchmark (e.g., google/gemma-3-4b-it anthropic/claude-3-haiku)')
    parser.add_argument('--output', default='benchmark_results.json', help='Output file for results')
    
    args = parser.parse_args()
    
    print("Starting MMMU benchmark...")
    print(f"Dataset path: {args.mmmu_path}")
    print(f"Max questions per subject: {args.max_per_subject}")
    print(f"Subjects filter: {args.subjects}")
    print(f"Models to test: {args.models}")
    print(f"Output file: {args.output}")
    
    # Load existing results if the file exists
    all_results = load_existing_results(args.output)
    if all_results:
        print(f"Resuming from existing results file: {args.output}")
        print(f"Found existing results for models: {list(all_results.keys())}")
        for existing_model in all_results.keys():
            if existing_model in args.models:
                existing_q = all_results[existing_model]['total_questions']
                existing_c = all_results[existing_model]['correct_answers']
                existing_f = all_results[existing_model]['failed_requests']
                accuracy = (existing_c / existing_q * 100) if existing_q > 0 else 0
                print(f"  {existing_model}: {existing_q} successful, {existing_c} correct ({accuracy:.1f}%), {existing_f} failed questions already processed")
    else:
        print(f"Starting fresh benchmark (no existing results file)")
    
    for model in args.models:
        print(f"\n{'='*60}")
        print(f"Testing model: {model}")
        print(f"{'='*60}")
        
        run_benchmark(
            mmmu_path=args.mmmu_path,
            model=model,
            all_results=all_results,
            output_file=args.output,
            max_questions_per_subject=args.max_per_subject,
            subjects_filter=args.subjects
        )
        
        print_summary(all_results[model])
    
    # Final save of all results (redundant but safe)
    save_results(all_results, args.output)
    print(f"\nFinal results saved to: {args.output}")
    
    # Print comparative summary
    if len(args.models) > 1:
        print(f"\n{'='*60}")
        print("COMPARATIVE SUMMARY")
        print(f"{'='*60}")
        for model in args.models:
            if model in all_results:
                results = all_results[model]
                total_attempted = results['total_questions'] + results['failed_requests']
                api_success_rate = (results['total_questions'] / total_attempted * 100) if total_attempted > 0 else 0
                accuracy = (results['correct_answers'] / results['total_questions'] * 100) if results['total_questions'] > 0 else 0
                print(f"{model}:")
                print(f"  API Success: {results['total_questions']}/{total_attempted} ({api_success_rate:.1f}%)")
                print(f"  Accuracy: {results['correct_answers']}/{results['total_questions']} ({accuracy:.1f}%)")
            else:
                print(f"{model}: No results (not processed)") 