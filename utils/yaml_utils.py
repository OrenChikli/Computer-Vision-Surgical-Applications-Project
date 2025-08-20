"""
Centralized YAML utilities for the project.
Provides consistent YAML loading and saving across all modules.
"""

import yaml
from pathlib import Path
from typing import Any, Dict


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    Load YAML file safely.
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        Dictionary containing YAML data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file is invalid YAML
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: Dict[str, Any], file_path: str, sort_keys: bool = False) -> None:
    """
    Save data to YAML file.
    
    Args:
        data: Dictionary to save
        file_path: Path to save YAML file
        sort_keys: Whether to sort keys alphabetically
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=sort_keys, indent=2)


# Alias for backward compatibility
load_config = load_yaml