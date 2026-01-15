"""
Logging configuration for Triple-Stream Fashion Search Engine
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str,
    log_file: str = None,
    level: int = logging.INFO,
    console_output: bool = True
):
    """
    Setup logger with console and optional file output
    
    Args:
        name: Logger name (typically module name)
        log_file: Optional log file name (will be created in logs/ directory)
        level: Logging level (default: INFO)
        console_output: Whether to output to console (default: True)
    
    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_path = log_dir / log_file
        file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str, log_file: str = None):
    """
    Get or create a logger instance
    
    Args:
        name: Logger name
        log_file: Optional log file name
    
    Returns:
        Logger instance
    """
    return setup_logger(name, log_file)


# Create default loggers for each module
caption_logger = setup_logger('caption_generator', 'caption_generator.log')
indexer_logger = setup_logger('indexer', 'indexer.log')
retriever_logger = setup_logger('retriever', 'retriever.log')
evaluator_logger = setup_logger('evaluator', 'evaluator.log')
utils_logger = setup_logger('utils', 'utils.log')


# Create a main logger for general purpose
main_logger = setup_logger('main', 'main.log')
