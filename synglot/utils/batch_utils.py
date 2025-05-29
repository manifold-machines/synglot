import json
import logging
from typing import Dict, Any, Union, Optional


def retrieve_batch(client, batch_job_or_result, save_results=True, batch_type="generation"):
    """
    Retrieve batch output content when the batch job is done.
    Handles both simple batch jobs and dataset batch jobs for both translation and generation.
    
    Args:
        client: OpenAI client instance
        batch_job_or_result: Batch job object or result dict from translate_batch/translate_dataset
        save_results (bool): Whether to save results to file automatically (for dataset batches)
        batch_type (str): Type of batch operation - "generation" or "translation"
        
    Returns:
        File content, translations, or processing summary depending on input type and batch_type
    """
    if not client:
        raise RuntimeError("OpenAI client not initialized properly.")
    
    try:
        # Handle both simple batch jobs and dataset batch results
        if hasattr(batch_job_or_result, 'id'):
            # Simple batch job object
            batch_id = batch_job_or_result.id
            output_path = None
            is_dataset_batch = False
        elif isinstance(batch_job_or_result, dict) and 'batch_job' in batch_job_or_result:
            # Dataset batch result
            batch_id = batch_job_or_result['batch_job'].id
            output_path = batch_job_or_result.get('output_path')
            is_dataset_batch = True
        else:
            raise ValueError("Invalid batch job or result object")
        
        # Check batch status
        current_batch = client.batches.retrieve(batch_id)
        batch_status = current_batch.status
        
        print(f"Batch ID: {batch_id}")
        print(f"Status: {batch_status}")
        
        if batch_status == "completed":
            # Get output file
            output_file_id = current_batch.output_file_id
            file_content = client.files.content(output_file_id)
            
            if not is_dataset_batch:
                # Simple batch - return file content
                return file_content
            
            # Dataset batch - process and organize results based on batch type
            if batch_type == "translation":
                return _process_translation_batch_results(
                    file_content, batch_job_or_result, output_path, save_results
                )
            elif batch_type == "generation":
                return _process_generation_batch_results(
                    file_content, batch_job_or_result, output_path, save_results
                )
            else:
                raise ValueError(f"Unsupported batch_type: {batch_type}")
                
        elif batch_status == "failed":
            raise RuntimeError(f"Batch job {batch_id} failed")
        elif batch_status == "cancelled":
            raise RuntimeError(f"Batch job {batch_id} was cancelled")
        else:
            # Batch is still in progress (validating, in_progress, finalizing)
            if is_dataset_batch:
                print(f"Batch not completed; current status is {batch_status}.")
                return {
                    "batch_id": batch_id,
                    "status": batch_status,
                    "message": f"Batch is still processing. Current status: {batch_status}"
                }
            else:
                logging.info(f"Batch not completed; current status is {batch_status}.")
                return None
                
    except Exception as e:
        if "openai" in str(type(e).__module__.lower()):
            raise RuntimeError(f"OpenAI API error during batch retrieval: {e}")
        else:
            raise RuntimeError(f"An unexpected error occurred during batch retrieval: {e}")


def _process_translation_batch_results(file_content, batch_job_or_result, output_path, save_results):
    """Process translation batch results."""
    print("Batch completed! Processing translation results...")
    
    # Parse results
    results = []
    for line in file_content.text.strip().split('\n'):
        if line.strip():
            results.append(json.loads(line))
    
    print(f"Retrieved {len(results)} translation results")
    
    # Organize results by request metadata
    translations_by_request = {}
    success_count = 0
    error_count = 0
    
    for result in results:
        custom_id = result.get("custom_id", "")
        # Parse custom_id: "req-{request_id}-{column}-{sample_index}"
        parts = custom_id.split("-")
        if len(parts) >= 4:
            request_id = int(parts[1])
            column = parts[2]
            sample_index = int(parts[3])
            
            if result.get("response") and result["response"].get("body"):
                # Success
                response_body = result["response"]["body"]
                if response_body.get("choices") and len(response_body["choices"]) > 0:
                    translated_text = response_body["choices"][0]["message"]["content"]
                    translations_by_request[custom_id] = {
                        "translation": translated_text,
                        "column": column,
                        "sample_index": sample_index,
                        "request_id": request_id,
                        "status": "success"
                    }
                    success_count += 1
                else:
                    error_count += 1
                    translations_by_request[custom_id] = {
                        "translation": None,
                        "column": column,
                        "sample_index": sample_index,
                        "request_id": request_id,
                        "status": "error",
                        "error": "No translation in response"
                    }
            else:
                # Error
                error_count += 1
                error_msg = result.get("error", {}).get("message", "Unknown error")
                translations_by_request[custom_id] = {
                    "translation": None,
                    "column": column,
                    "sample_index": sample_index,
                    "request_id": request_id,
                    "status": "error",
                    "error": error_msg
                }
    
    if save_results and output_path:
        print(f"Saving results to: {output_path}")
        
        # Create a mapping file for easier post-processing
        translation_map_path = output_path.replace('.jsonl', '_translation_map.json')
        with open(translation_map_path, 'w', encoding='utf-8') as f:
            json.dump(translations_by_request, f, ensure_ascii=False, indent=2)
        
        # Also create a simplified results file
        with open(output_path, 'w', encoding='utf-8') as f:
            for request_id, translation_data in translations_by_request.items():
                f.write(json.dumps(translation_data, ensure_ascii=False) + '\n')
        
        print(f"Translation map saved to: {translation_map_path}")
        print(f"Translation results saved to: {output_path}")
    
    summary = {
        "batch_id": batch_job_or_result['batch_job'].id,
        "status": "completed",
        "total_requests": len(results),
        "successful_translations": success_count,
        "errors": error_count,
        "success_rate": success_count / len(results) if results else 0,
        "translations": translations_by_request,
        "output_path": output_path,
        "columns_translated": batch_job_or_result.get("columns_translated", []),
        "source_language": batch_job_or_result.get("source_language"),
        "target_language": batch_job_or_result.get("target_language")
    }
    
    print(f"\nBatch translation retrieval complete!")
    print(f"Total requests: {len(results)}")
    print(f"Successful translations: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    
    return summary


def _process_generation_batch_results(file_content, batch_job_or_result, output_path, save_results):
    """Process generation batch results."""
    print("Batch completed! Processing generation results...")
    
    # Parse results
    results = []
    for line in file_content.text.strip().split('\n'):
        if line.strip():
            results.append(json.loads(line))
    
    print(f"Retrieved {len(results)} generation results")
    
    # Organize results by request ID
    generations_by_request = {}
    success_count = 0
    error_count = 0
    
    for result in results:
        custom_id = result.get("custom_id", "")
        
        if result.get("response") and result["response"].get("body"):
            # Success
            response_body = result["response"]["body"]
            if response_body.get("choices") and len(response_body["choices"]) > 0:
                generated_text = response_body["choices"][0]["message"]["content"]
                generations_by_request[custom_id] = {
                    "generated_text": generated_text,
                    "status": "success"
                }
                success_count += 1
            else:
                error_count += 1
                generations_by_request[custom_id] = {
                    "generated_text": None,
                    "status": "error",
                    "error": "No generation in response"
                }
        else:
            # Error
            error_count += 1
            error_msg = result.get("error", {}).get("message", "Unknown error")
            generations_by_request[custom_id] = {
                "generated_text": None,
                "status": "error",
                "error": error_msg
            }
    
    if save_results and output_path:
        print(f"Saving results to: {output_path}")
        
        # Create results file
        with open(output_path, 'w', encoding='utf-8') as f:
            for request_id, generation_data in generations_by_request.items():
                f.write(json.dumps(generation_data, ensure_ascii=False) + '\n')
        
        print(f"Generation results saved to: {output_path}")
    
    summary = {
        "batch_id": batch_job_or_result.get('batch_job', {}).get('id', 'unknown'),
        "status": "completed",
        "total_requests": len(results),
        "successful_generations": success_count,
        "errors": error_count,
        "success_rate": success_count / len(results) if results else 0,
        "generations": generations_by_request,
        "output_path": output_path
    }
    
    print(f"\nBatch generation retrieval complete!")
    print(f"Total requests: {len(results)}")
    print(f"Successful generations: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    
    return summary 