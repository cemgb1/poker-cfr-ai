#!/usr/bin/env python3
"""
Logging Configuration for Natural CFR Training

Provides centralized logging setup for the natural CFR training system.
Logs to both timestamped files in logs/ directory and to stdout.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(name="natural_cfr", log_level=logging.INFO):
    """
    Set up logging to both file and console with proper formatting.
    
    Args:
        name: Logger name (used for log filename)
        log_level: Logging level (default: INFO)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"{name}_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler - detailed logging
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler - cleaner format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Ensure immediate flushing
    for handler in logger.handlers:
        if hasattr(handler, 'flush'):
            handler.flush()
    
    logger.info(f"Logging initialized - file: {log_filename}")
    
    return logger


def get_logger(name="natural_cfr"):
    """
    Get an existing logger or create a new one if it doesn't exist.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Logger doesn't exist yet, create it
        return setup_logging(name)
    return logger


def log_exception(logger, message="Exception occurred"):
    """
    Log an exception with full stack trace.
    
    Args:
        logger: Logger instance
        message: Custom message to include with the exception
    """
    logger.exception(f"{message}")


def flush_logs(logger):
    """
    Force flush all log handlers.
    
    Args:
        logger: Logger instance
    """
    for handler in logger.handlers:
        if hasattr(handler, 'flush'):
            handler.flush()