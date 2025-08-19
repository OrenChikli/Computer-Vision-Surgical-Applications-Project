import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Handles loading and validation of YAML configuration files."""
    
    def __init__(self):
        self.config_dir = Path(__file__).parent
        self.default_config_path = self.config_dir / "default_config.yaml"
    
    def load_config(self, config_path: Optional[str] = None, overrides: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file with optional overrides.
        
        Args:
            config_path: Path to config YAML file. If None, uses default.
            overrides: Dictionary of override values
            
        Returns:
            Merged configuration dictionary
        """
        # Load default config
        default_config = self._load_yaml(self.default_config_path)
        
        # Load user config if provided
        if config_path and os.path.exists(config_path):
            user_config = self._load_yaml(config_path)
            config = self._deep_merge(default_config, user_config)
        else:
            config = default_config.copy()
        
        # Apply overrides
        if overrides:
            config = self._deep_merge(config, overrides)
        
        # Validate and resolve paths
        config = self._resolve_paths(config)
        self._validate_config(config)
        
        return config
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {path}: {e}")
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _resolve_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve relative paths and environment variables."""
        path_keys = [
            'tools_path', 'annotations_path', 'camera_params', 
            'output_dir', 'hdri_path'
        ]
        
        for key in path_keys:
            if key in config and config[key]:
                # Expand environment variables
                path = os.path.expandvars(config[key])
                # Convert to absolute path
                config[key] = os.path.abspath(path)
        
        return config
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration values."""
        required_paths = ['tools_path', 'annotations_path', 'camera_params']
        
        # Check required paths exist
        for path_key in required_paths:
            if path_key not in config:
                raise ValueError(f"Missing required config key: {path_key}")
            
            path = config[path_key]
            if not os.path.exists(path):
                raise ValueError(f"Path does not exist: {path_key} = {path}")
        
        # Validate numeric ranges
        if config.get('num_images', 0) <= 0:
            raise ValueError("num_images must be positive")
        
        if config.get('poses_per_workspace', 0) <= 0:
            raise ValueError("poses_per_workspace must be positive")
        
        if config.get('workspace_size', 0) <= 0:
            raise ValueError("workspace_size must be positive")
        
        # Validate probability values
        for prob_key in ['motion_blur_prob', 'occlusion_prob']:
            if prob_key in config:
                prob = config[prob_key]
                if not 0 <= prob <= 1:
                    raise ValueError(f"{prob_key} must be between 0 and 1")
        
        print("✅ Configuration validation passed")


def load_config(config_path: Optional[str] = None, **overrides) -> Dict[str, Any]:
    """Convenience function to load configuration."""
    loader = ConfigLoader()
    return loader.load_config(config_path, overrides)
