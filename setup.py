#!/usr/bin/env python3
"""
Setup script to verify installation and data paths.
Run this script after setting up your environment to check everything is working.
"""

import os
import sys
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major != 3 or version.minor < 7:
        print("❌ Python 3.7+ required. Current version:", sys.version)
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'yaml', 'numpy', 'cv2', 'tqdm'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    # Special check for BlenderProc
    try:
        import blenderproc as bproc
        print("✅ BlenderProc is installed")
    except ImportError:
        missing_packages.append('blenderproc')
        print("❌ BlenderProc is missing")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_config_file():
    """Check if config file exists and is valid."""
    config_file = Path("config.yaml")
    if not config_file.exists():
        print("❌ config.yaml not found")
        print("Copy and modify config/default_config.yaml to config.yaml")
        return False
    
    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ config.yaml is valid YAML")
        return config
    except Exception as e:
        print(f"❌ config.yaml is invalid: {e}")
        return False

def check_data_paths(config):
    """Check if all required data paths exist."""
    required_paths = {
        'tools_path': 'Surgical tool 3D models directory',
        'annotations_path': 'Keypoint annotations directory', 
        'camera_params': 'Camera parameters file'
    }
    
    all_paths_ok = True
    for path_key, description in required_paths.items():
        if path_key not in config:
            print(f"❌ {path_key} not specified in config")
            all_paths_ok = False
            continue
            
        path = Path(config[path_key])
        if not path.exists():
            print(f"❌ {description} not found: {path}")
            all_paths_ok = False
        else:
            print(f"✅ {description}: {path}")
    
    return all_paths_ok

def check_tool_data(config):
    """Check if tool data is properly structured."""
    if 'tools_path' not in config:
        return False
        
    tools_path = Path(config['tools_path'])
    if not tools_path.exists():
        return False
    
    # Check for .obj files
    obj_files = list(tools_path.glob("**/*.obj"))
    if not obj_files:
        print("❌ No .obj files found in tools directory")
        return False
    
    print(f"✅ Found {len(obj_files)} .obj files")
    
    # Check annotations
    if 'annotations_path' not in config:
        return False
        
    annotations_path = Path(config['annotations_path'])
    if not annotations_path.exists():
        return False
        
    annotation_files = list(annotations_path.glob("*_keypoints.json"))
    print(f"✅ Found {len(annotation_files)} annotation files")
    
    # Check camera params
    if 'camera_params' not in config:
        return False
        
    camera_file = Path(config['camera_params'])
    try:
        with open(camera_file, 'r') as f:
            camera_data = json.load(f)
        
        required_camera_params = ['fx', 'fy', 'cx', 'cy', 'width', 'height']
        missing_params = [p for p in required_camera_params if p not in camera_data]
        
        if missing_params:
            print(f"❌ Camera params missing: {missing_params}")
            return False
        
        print("✅ Camera parameters are valid")
        
    except Exception as e:
        print(f"❌ Invalid camera parameters: {e}")
        return False
    
    return True

def check_output_directory(config):
    """Check if output directory can be created."""
    if 'output_dir' not in config:
        print("❌ output_dir not specified in config")
        return False
    
    output_dir = Path(config['output_dir'])
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Output directory: {output_dir}")
        return True
    except Exception as e:
        print(f"❌ Cannot create output directory {output_dir}: {e}")
        return False

def test_blenderproc():
    """Test basic BlenderProc functionality."""
    try:
        import blenderproc as bproc
        print("✅ BlenderProc import successful")
        
        # Test basic initialization (this might take a moment)
        print("Testing BlenderProc initialization...")
        bproc.init()
        print("✅ BlenderProc initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ BlenderProc test failed: {e}")
        return False

def main():
    """Run all setup checks."""
    print("🔧 Setting up Surgical Instrument Pose Estimation Project")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Configuration File", check_config_file),
    ]
    
    config = None
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n🔍 Checking {check_name}...")
        result = check_func()
        if check_name == "Configuration File":
            config = result
        if not result:
            all_passed = False
            break
    
    if config and all_passed:
        print(f"\n🔍 Checking Data Paths...")
        if not check_data_paths(config):
            all_passed = False
        
        if all_passed:
            print(f"\n🔍 Checking Tool Data...")
            if not check_tool_data(config):
                all_passed = False
            
            if all_passed:
                print(f"\n🔍 Checking Output Directory...")
                if not check_output_directory(config):
                    all_passed = False
    
    # Optional BlenderProc test (can be slow)
    if all_passed:
        print(f"\n🔍 Testing BlenderProc (this may take a moment)...")
        test_blenderproc()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 Setup complete! You're ready to generate synthetic data.")
        print("\nTo get started, run:")
        print("python synthetic_data_generator.py --config config.yaml")
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
