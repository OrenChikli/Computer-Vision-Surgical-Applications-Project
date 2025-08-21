"""
Centralized logging utility for the project.
Provides consistent logging setup across all modules.
"""

import logging
import sys
from typing import Dict, Any


def setup_logger(name: str, config: Dict[str, Any] = None) -> logging.Logger:
    """
    Set up a logger with configuration from config file.
    
    Args:
        name: Logger name (usually __name__)
        config: Configuration dictionary with logging settings
        
    Returns:
        Configured logger instance
    """
    if config is None:
        config = {}
    
    logging_config = config.get('logging', {})
    level = getattr(logging, logging_config.get('level', 'INFO'))
    format_str = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Configure root logger if not already configured
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    # Return named logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    return logger