"""Configuration utilities."""

# Simple YAML config loading
import yaml

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

__all__ = ['load_config']
