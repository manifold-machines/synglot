from .nested_utils import (
    find_nested_text_fields,
    is_column_translatable
)
import os


def save_media_file(media_data, media_output_dir, sample_index, media_field_name="image"):
    """Save media file (image, etc.) and return relative path."""
    if media_data is None:
        return None
    
    try:
        # Determine file extension and format
        if hasattr(media_data, 'save'):  # PIL Image
            extension = "jpg"
            format_type = "JPEG"
        else:
            # Could add more media type detection here
            return None
        
        filename = f"{media_field_name}_{sample_index:06d}.{extension}"
        full_path = os.path.join(media_output_dir, filename)
        relative_path = f"media/{filename}"
        
        # Save the file
        if hasattr(media_data, 'save'):
            media_data.save(full_path, format_type, quality=95)
        
        return relative_path
        
    except Exception as e:
        print(f"Warning: Failed to save media file for sample {sample_index}: {e}")
        return None


def auto_detect_translatable_columns(dataset, streaming_mode, nested_field_separator):
    """Auto-detect translatable columns in the dataset."""
    if streaming_mode:
        # For streaming datasets, sample first few items
        sample_data = []
        dataset_iter = iter(dataset)
        for _ in range(min(10, 100)):  # Sample up to 10 items
            try:
                sample_data.append(next(dataset_iter))
            except StopIteration:
                break
    else:
        if hasattr(dataset, 'columns') and dataset.columns:
            all_columns = list(dataset.columns)
            sample_size = min(10, len(dataset))
            sample_data = list(dataset)[:sample_size]
        else:
            sample_data = list(dataset)[:10]
            all_columns = None
    
    if not sample_data:
        raise ValueError("No samples found in dataset for auto-detection")
    
    columns_to_translate = []
    
    if all_columns:
        # Standard column-based detection
        for column in all_columns:
            is_translatable = is_column_translatable(sample_data, column, nested_field_separator)
            if is_translatable:
                columns_to_translate.append(column)
    else:
        # Nested structure detection
        nested_columns = find_nested_text_fields(sample_data, nested_field_separator)
        columns_to_translate.extend(nested_columns)
    
    print(f"No columns specified, auto-detecting translatable columns: {columns_to_translate}")
    if not columns_to_translate:
        raise ValueError("No translatable text columns found in dataset")
    
    return columns_to_translate


def column_exists(dataset, column, nested_field_separator):
    """Check if a column exists in the dataset, supporting nested fields."""
    if nested_field_separator in column:
        return True  # for nested fields, we can't easily check without sampling; check is during processing instead
    return hasattr(dataset, 'columns') and column in dataset.columns 