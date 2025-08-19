#!/usr/bin/env python3
"""
Test script to verify synthetic data generation is working correctly.
Generates a small test dataset and validates the output.
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_test_config(temp_dir, data_paths):
    """Create a minimal test configuration."""
    config_content = f"""
# Test configuration
tools_path: "{data_paths['tools_path']}"
annotations_path: "{data_paths['annotations_path']}"
camera_params: "{data_paths['camera_params']}"
output_dir: "{temp_dir}/test_output"
hdri_path: ""

# Minimal test dataset
num_images: 3
poses_per_workspace: 2
workspace_size: 2.0

# Fast rendering
render_width: 320
render_height: 240
render_samples: 25

# No effects for speed
motion_blur_prob: 0.0
occlusion_prob: 0.0
visualize_keypoints: true
debug: false
seed: 42
"""
    
    config_path = Path(temp_dir) / "test_config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    return config_path

def check_data_availability():
    """Check if required data is available by looking for config.yaml."""
    config_file = project_root / "config.yaml"
    if not config_file.exists():
        print("❌ config.yaml not found. Please set up your configuration first.")
        print("Run: python setup.py")
        return None
    
    try:
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract data paths
        data_paths = {
            'tools_path': config.get('tools_path'),
            'annotations_path': config.get('annotations_path'),
            'camera_params': config.get('camera_params')
        }
        
        # Validate paths exist
        for key, path in data_paths.items():
            if not path or not Path(path).exists():
                print(f"❌ {key} not found: {path}")
                return None
        
        print("✅ Data paths validated")
        return data_paths
        
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None

def run_generation_test(config_path):
    """Run the synthetic data generator with test configuration."""
    try:
        # Import and run the main generation function
        from synthetic_data_generator import main as generate_main
        
        # Temporarily modify sys.argv to pass the test config
        original_argv = sys.argv.copy()
        sys.argv = ['synthetic_data_generator.py', '--config', str(config_path)]
        
        print("🚀 Running synthetic data generation test...")
        result = generate_main()
        
        # Restore original argv
        sys.argv = original_argv
        
        return result == 0
        
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_test_output(output_dir):
    """Validate the generated test dataset."""
    output_path = Path(output_dir)
    
    # Check basic file structure
    coco_file = output_path / "coco_annotations.json"
    if not coco_file.exists():
        print("❌ COCO annotations file not created")
        return False
    
    # Check for generated images
    image_files = list(output_path.glob("*.jpg")) + list(output_path.glob("*.png"))
    if not image_files:
        print("❌ No image files generated")
        return False
    
    print(f"✅ Generated {len(image_files)} images")
    
    # Validate COCO file content
    try:
        import json
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)
        
        required_keys = ['images', 'annotations', 'categories']
        missing_keys = [key for key in required_keys if key not in coco_data]
        
        if missing_keys:
            print(f"❌ COCO file missing keys: {missing_keys}")
            return False
        
        print(f"✅ COCO file structure valid")
        print(f"  Images: {len(coco_data['images'])}")
        print(f"  Annotations: {len(coco_data['annotations'])}")
        print(f"  Categories: {len(coco_data['categories'])}")
        
        # Check for keypoint annotations
        keypoint_annotations = [ann for ann in coco_data['annotations'] if 'keypoints' in ann]
        if keypoint_annotations:
            print(f"✅ Found {len(keypoint_annotations)} keypoint annotations")
        else:
            print("⚠️  No keypoint annotations found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating COCO file: {e}")
        return False

def main():
    """Run the complete test suite."""
    print("🧪 Testing Synthetic Data Generation")
    print("=" * 50)
    
    # Check if data is available
    data_paths = check_data_availability()
    if not data_paths:
        return 1
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"📁 Using temporary directory: {temp_dir}")
        
        # Create test configuration
        config_path = create_test_config(temp_dir, data_paths)
        print(f"⚙️  Created test configuration: {config_path}")
        
        # Run generation test
        if not run_generation_test(config_path):
            print("❌ Generation test failed")
            return 1
        
        # Validate output
        output_dir = Path(temp_dir) / "test_output"
        if not validate_test_output(output_dir):
            print("❌ Output validation failed")
            return 1
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed!")
        print("✅ Synthetic data generation is working correctly")
        
        # Optional: Copy test results to project directory
        test_output_dir = project_root / "test_results"
        if test_output_dir.exists():
            shutil.rmtree(test_output_dir)
        shutil.copytree(output_dir, test_output_dir)
        print(f"📋 Test results saved to: {test_output_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main())
