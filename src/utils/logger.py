#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logging utility for the RAG system.
Provides consistent logging throughout the application.
"""

import os
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "rag_system",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True
) -> logging.Logger:
    """
    Setup and configure a logger.
    
    Args:
        name: Name of the logger.
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Path to the log file. If None, a default path is used.
        console_output: Whether to output logs to console.
        file_output: Whether to output logs to a file.
        
    Returns:
        Configured logger object.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # Define log format
    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        console_handler.setLevel(level)
        # Set UTF-8 encoding for console handler
        if hasattr(console_handler.stream, 'reconfigure'):
            try:
                console_handler.stream.reconfigure(encoding='utf-8')
            except Exception:
                pass  # Ignore encoding errors
        logger.addHandler(console_handler)
    
    # Add file handler if requested
    if file_output:
        if log_file is None:
            # Create default log file in logs directory at project root
            try:
                # Determine project root
                current_dir = Path(__file__).resolve().parent
                project_root = current_dir.parent.parent
                
                # Create logs directory if it doesn't exist
                logs_dir = project_root / "logs"
                os.makedirs(logs_dir, exist_ok=True)
                
                # Create log file with date-time stamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = logs_dir / f"{name}_{timestamp}.log"
            except Exception as e:
                print(f"Error creating default log file: {e}")
                # Fallback to current directory
                log_file = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        try:
            # Ensure the log directory exists
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            
            # Add file handler
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(log_format)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Error setting up file handler: {e}")
            # Add a warning to the logger if available
            if console_output and logger.handlers:
                logger.warning(f"Could not setup file logging: {e}")
    
    return logger


def get_logger(name: str = "rag_system") -> logging.Logger:
    """
    Get an existing logger or create a new one.
    
    Args:
        name: Name of the logger.
        
    Returns:
        Logger object.
    """
    logger = logging.getLogger(name)
    
    # If logger has no handlers, set it up
    if not logger.handlers:
        logger = setup_logger(name)
    
    return logger


if __name__ == "__main__":
    # Test logging
    logger = setup_logger(level=logging.DEBUG)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    
    # Get the same logger
    same_logger = get_logger()
    same_logger.info("Using same logger")
    
    # Get a different logger
    other_logger = get_logger("other_module")
    other_logger.info("Using other logger")
