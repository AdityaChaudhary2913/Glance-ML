"""
Shared Utilities Module

Common utilities, logging, and configuration used by both indexer and retriever.
"""

from .utils import (
    load_fashionpedia_data,
    get_image_annotations,
    extract_dominant_colors,
    build_grounded_string,
    save_json,
    load_json,
    ensure_dir,
    closest_color_name
)

from .logger import (
    indexer_logger,
    caption_logger,
    retriever_logger,
    evaluator_logger,
    utils_logger
)

__all__ = [
    'load_fashionpedia_data',
    'get_image_annotations',
    'extract_dominant_colors',
    'build_grounded_string',
    'save_json',
    'load_json',
    'ensure_dir',
    'closest_color_name',
    'indexer_logger',
    'caption_logger',
    'retriever_logger',
    'evaluator_logger',
    'utils_logger'
]
