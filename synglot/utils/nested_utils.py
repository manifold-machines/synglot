"""
Utilities for handling nested field operations in datasets.
"""

from typing import List, Dict, Any, Optional, Union


def find_nested_text_fields(sample_data: List[Dict], nested_field_separator: str = ".", 
                           prefix: str = "", max_depth: int = 3) -> List[str]:
    """Recursively find nested text fields in the dataset."""
    if max_depth <= 0:
        return []
    
    text_fields = []
    
    for item in sample_data[:3]:  # Sample first 3 items
        if isinstance(item, dict):
            for key, value in item.items():
                field_path = f"{prefix}{key}" if prefix else key
                
                if isinstance(value, str) and len(value.strip()) > 0:
                    if any(c.isalpha() for c in value):
                        if field_path not in text_fields:
                            text_fields.append(field_path)
                elif isinstance(value, list) and value:
                    # Handle list of dicts (like QA pairs)
                    if isinstance(value[0], dict):
                        nested_fields = find_nested_text_fields(
                            value, nested_field_separator, f"{field_path}{nested_field_separator}", max_depth - 1
                        )
                        text_fields.extend(nested_fields)
                elif isinstance(value, dict):
                    # Handle nested dict
                    nested_fields = find_nested_text_fields(
                        [value], nested_field_separator, f"{field_path}{nested_field_separator}", max_depth - 1
                    )
                    text_fields.extend(nested_fields)
    
    return list(set(text_fields))  # Remove duplicates


def get_nested_value(item: Dict, field_path: str, nested_field_separator: str = ".") -> Any:
    """Get value from nested field path like 'qa.question'."""
    if nested_field_separator not in field_path:
        return item.get(field_path) if isinstance(item, dict) else None
    
    parts = field_path.split(nested_field_separator)
    current = item
    
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            # Handle list of items (like QA pairs)
            results = []
            for list_item in current:
                if isinstance(list_item, dict) and part in list_item:
                    results.append(list_item[part])
            return results if results else None
        else:
            return None
    
    return current


def set_nested_value(item: Dict, field_path: str, value: Any, nested_field_separator: str = ".") -> None:
    """Set value in nested field path like 'qa.question'."""
    if nested_field_separator not in field_path:
        if isinstance(item, dict):
            item[field_path] = value
        return
    
    parts = field_path.split(nested_field_separator)
    current = item
    
    # Navigate to the parent of the target field
    for part in parts[:-1]:
        if isinstance(current, dict):
            if part not in current:
                current[part] = {}
            current = current[part]
        elif isinstance(current, list):
            # This is more complex for lists - we'd need index information
            # For now, skip this case
            return
    
    # Set the final value
    final_key = parts[-1]
    if isinstance(current, dict):
        current[final_key] = value


def extract_texts_from_field(sample: Dict, field_path: str, nested_field_separator: str = ".") -> List[Dict[str, Any]]:
    """Extract all translatable texts from a field, handling nested structures."""
    texts = []
    
    if nested_field_separator in field_path:
        # Handle nested field like "qa.question"
        parts = field_path.split(nested_field_separator)
        current = sample
        
        # Navigate to the target field
        for i, part in enumerate(parts[:-1]):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return texts  # Path doesn't exist
        
        final_key = parts[-1]
        
        if isinstance(current, list):
            # Handle list of items (like QA pairs)
            for idx, item in enumerate(current):
                if isinstance(item, dict) and final_key in item:
                    text = item[final_key]
                    if isinstance(text, str) and text.strip():
                        texts.append({
                            'text': text,
                            'path': field_path,
                            'index': idx
                        })
        elif isinstance(current, dict) and final_key in current:
            text = current[final_key]
            if isinstance(text, str) and text.strip():
                texts.append({
                    'text': text,
                    'path': field_path
                })
    else:
        # Simple field
        if isinstance(sample, dict) and field_path in sample:
            text = sample[field_path]
            if isinstance(text, str) and text.strip():
                texts.append({
                    'text': text,
                    'path': field_path
                })
    
    return texts


def set_translated_nested_value(record: Dict, field_path: str, value: Any, 
                               nested_field_separator: str = ".", list_index: Optional[int] = None) -> None:
    """Set translated value in nested structure."""
    parts = field_path.split(nested_field_separator)
    current = record
    
    # Navigate/create the nested structure
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        elif not isinstance(current[part], dict):
            # If it exists but isn't a dict, we need to handle list case
            if isinstance(current[part], list):
                # For list structures, we might need special handling
                pass
            else:
                current[part] = {}
        current = current[part]
    
    final_key = parts[-1]
    
    # Handle list index if provided
    if list_index is not None:
        if final_key not in current:
            current[final_key] = []
        
        # Ensure list is long enough
        while len(current[final_key]) <= list_index:
            current[final_key].append(None)
        
        current[final_key][list_index] = value
    else:
        current[final_key] = value


def is_column_translatable(sample_data: List[Dict], column: str, nested_field_separator: str = ".") -> bool:
    """Check if a column contains translatable text."""
    for item in sample_data:
        value = get_nested_value(item, column, nested_field_separator)
        if isinstance(value, str) and len(value.strip()) > 0:
            if any(c.isalpha() for c in value):
                return True
    return False 