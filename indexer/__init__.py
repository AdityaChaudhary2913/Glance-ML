"""
Indexer Module: Multi-Stream Fashion Image Vectorization

This module processes raw fashion images into a searchable triple-stream
vector database using Fashionpedia annotations, BLIP-2 captions, and CLIP encodings.
"""

from .indexer import MultiStreamIndexer
from .caption_generator import VibeCaptionGenerator

__all__ = ['MultiStreamIndexer', 'VibeCaptionGenerator']
