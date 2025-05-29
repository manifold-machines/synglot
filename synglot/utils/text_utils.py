import os
import re
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import hashlib


def load_material_files(material_paths: Union[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Load and parse material files (.txt, .md).
    
    Args:
        material_paths: Path(s) to .txt or .md files containing source material
        
    Returns:
        List of dictionaries containing file content and metadata
    """
    if isinstance(material_paths, str):
        material_paths = [material_paths]
    
    loaded_materials = []
    
    for path_str in material_paths:
        path = Path(path_str)
        
        if not path.exists():
            raise FileNotFoundError(f"Material file not found: {path}")
        
        if not path.suffix.lower() in ['.txt', '.md']:
            raise ValueError(f"Unsupported file format: {path.suffix}. Only .txt and .md files are supported.")
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            material = {
                "file_path": str(path),
                "file_name": path.name,
                "file_extension": path.suffix,
                "content": content,
                "content_length": len(content),
                "word_count": len(content.split()),
                "line_count": content.count('\n') + 1
            }
            
            loaded_materials.append(material)
            
        except Exception as e:
            raise RuntimeError(f"Error reading file {path}: {e}")
    
    return loaded_materials


def chunk_text(text: str, chunk_size: int, overlap: int = 0, preserve_words: bool = True) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to split into chunks
        chunk_size: Size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        preserve_words: Whether to preserve word boundaries
        
    Returns:
        List of dictionaries containing chunk data and metadata
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Extract chunk
        chunk_text = text[start:end]
        
        # Preserve word boundaries if requested and not at the end of text
        if preserve_words and end < len(text):
            # Find the last space within the chunk
            last_space = chunk_text.rfind(' ')
            if last_space > 0:
                # Adjust end to the last complete word
                end = start + last_space
                chunk_text = text[start:end]
        
        chunk = {
            "chunk_id": chunk_id,
            "text": chunk_text,
            "start_pos": start,
            "end_pos": end,
            "length": len(chunk_text),
            "word_count": len(chunk_text.split()),
            "overlap_start": start > 0,
            "overlap_end": end < len(text) and overlap > 0
        }
        
        chunks.append(chunk)
        
        # Calculate next start position with overlap
        next_start = end - overlap
        if next_start <= start:  # Prevent infinite loop
            next_start = start + 1
        
        start = next_start
        chunk_id += 1
        
        # Break if we've reached the end
        if end >= len(text):
            break
    
    return chunks


def filter_generated_content(generated_texts: List[str], 
                           min_length: int = 10,
                           max_length: Optional[int] = None,
                           min_quality_score: float = 0.5,
                           remove_duplicates: bool = True,
                           similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:
    """
    Filter generated content based on quality metrics.
    
    Args:
        generated_texts: List of generated text samples
        min_length: Minimum length in characters
        max_length: Maximum length in characters (None for no limit)
        min_quality_score: Minimum quality score (0.0 to 1.0)
        remove_duplicates: Whether to remove duplicate or near-duplicate texts
        similarity_threshold: Threshold for considering texts as duplicates (0.0 to 1.0)
        
    Returns:
        List of dictionaries containing filtered texts and their quality metrics
    """
    filtered_results = []
    seen_hashes = set()
    
    for i, text in enumerate(generated_texts):
        if not isinstance(text, str):
            continue
        
        text = text.strip()
        if not text:
            continue
        
        # Length filtering
        if len(text) < min_length:
            continue
        
        if max_length and len(text) > max_length:
            continue
        
        # Calculate quality metrics
        quality_metrics = _calculate_quality_metrics(text)
        
        # Quality filtering
        if quality_metrics["overall_score"] < min_quality_score:
            continue
        
        # Duplicate filtering
        if remove_duplicates:
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            
            if text_hash in seen_hashes:
                continue
            
            # Check for near-duplicates using simple similarity
            is_duplicate = False
            for existing_result in filtered_results:
                similarity = _calculate_text_similarity(text, existing_result["text"])
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    break
            
            if is_duplicate:
                continue
            
            seen_hashes.add(text_hash)
        
        result = {
            "text": text,
            "original_index": i,
            "length": len(text),
            "word_count": len(text.split()),
            "quality_metrics": quality_metrics
        }
        
        filtered_results.append(result)
    
    # Sort by quality score (descending)
    filtered_results.sort(key=lambda x: x["quality_metrics"]["overall_score"], reverse=True)
    
    return filtered_results


def _calculate_quality_metrics(text: str) -> Dict[str, float]:
    """
    Calculate quality metrics for a text sample.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary containing various quality metrics
    """
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Basic metrics
    word_count = len(words)
    sentence_count = len(sentences)
    char_count = len(text)
    
    # Diversity metrics
    unique_words = len(set(word.lower() for word in words))
    word_diversity = unique_words / word_count if word_count > 0 else 0
    
    # Structure metrics
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    
    # Readability proxy (simple heuristic)
    readability_score = min(1.0, (avg_sentence_length / 20.0) * (avg_word_length / 6.0))
    
    # Coherence proxy (ratio of complete sentences)
    complete_sentences = sum(1 for s in sentences if len(s.split()) >= 3)
    coherence_score = complete_sentences / sentence_count if sentence_count > 0 else 0
    
    # Overall quality score (weighted combination)
    overall_score = (
        word_diversity * 0.3 +
        min(1.0, readability_score) * 0.3 +
        coherence_score * 0.4
    )
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "word_diversity": word_diversity,
        "avg_sentence_length": avg_sentence_length,
        "avg_word_length": avg_word_length,
        "readability_score": readability_score,
        "coherence_score": coherence_score,
        "overall_score": overall_score
    }


def _calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using simple character-based comparison.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Normalize texts
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()
    
    if text1 == text2:
        return 1.0
    
    # Simple character-based similarity using set intersection
    chars1 = set(text1)
    chars2 = set(text2)
    
    if not chars1 and not chars2:
        return 1.0
    
    if not chars1 or not chars2:
        return 0.0
    
    intersection = chars1.intersection(chars2)
    union = chars1.union(chars2)
    
    return len(intersection) / len(union)


def extract_topics_from_text(text: str, max_topics: int = 10) -> List[str]:
    """
    Extract potential topics from text content.
    
    Args:
        text: Text to extract topics from
        max_topics: Maximum number of topics to return
        
    Returns:
        List of extracted topic strings
    """
    # Simple topic extraction using keyword frequency
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter out common stop words
    stop_words = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her', 'was', 'one',
        'our', 'had', 'have', 'has', 'what', 'were', 'said', 'each', 'which', 'she', 'how',
        'will', 'may', 'who', 'oil', 'its', 'now', 'find', 'way', 'use', 'his', 'they',
        'this', 'that', 'with', 'from', 'would', 'been', 'more', 'when', 'where', 'why'
    }
    
    # Count word frequencies
    word_freq = {}
    for word in words:
        if word not in stop_words and len(word) > 3:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top topics
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    topics = [word for word, freq in sorted_words[:max_topics] if freq > 1]
    
    return topics 