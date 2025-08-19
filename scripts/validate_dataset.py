#!/usr/bin/env python3
"""
Validate generated synthetic dataset.
Checks dataset integrity, statistics, and generates summary reports.
"""

import json
import cv2
import numpy as np
import argparse
from pathlib import Path
from collections import Counter
import seaborn as sns

def load_coco_data(coco_file):
    """Load COCO annotations file."""
    try:
        with open(coco_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading COCO file: {e}")
        return None

def validate_coco_structure(coco_data):
    """Validate COCO file structure."""
    required_keys = ['images', 'annotations', 'categories']
    missing_keys = [key for key in required_keys if key not in coco_data]
    
    if missing_keys:
        print(f"❌ Missing required keys: {missing_keys}")
        return False
    
    print(f"✅ COCO structure is valid")
    return True

def analyze_dataset_statistics(coco_data, output_dir):
    """Analyze and print dataset statistics."""
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']
    
    print(f"\n📊 Dataset Statistics")
    print(f"=" * 40)
    print(f"Total images: {len(images)}")
    print(f"Total annotations: {len(annotations)}")
    print(f"Total categories: {len(categories)}")
    
    # Category analysis
    category_names = {cat['id']: cat['name'] for cat in categories}
    category_counts = Counter(ann['category_id'] for ann in annotations)
    
    print(f"\n🏷️  Category Distribution:")
    for cat_id, count in category_counts.items():
        name = category_names.get(cat_id, f"Unknown_{cat_id}")
        print(f"  {name}: {count} annotations")
    
    # Keypoint analysis
    keypoint_annotations = [ann for ann in annotations if 'keypoints' in ann]
    total_keypoints = sum(ann.get('num_keypoints', 0) for ann in keypoint_annotations)
    
    print(f"\n🎯 Keypoint Statistics:")
    print(f"  Annotations with keypoints: {len(keypoint_annotations)}")
    print(f"  Total visible keypoints: {total_keypoints}")
    if keypoint_annotations:
        avg_keypoints = total_keypoints / len(keypoint_annotations)
        print(f"  Average keypoints per annotation: {avg_keypoints:.2f}")
    
    # Image size analysis
    if images:
        widths = [img['width'] for img in images]
        heights = [img['height'] for img in images]
        
        print(f"\n🖼️  Image Properties:")
        print(f"  Resolution: {widths[0]}x{heights[0]}")
        if len(set(widths)) > 1 or len(set(heights)) > 1:
            print(f"  ⚠️  Mixed resolutions detected")
    
    return {
        'total_images': len(images),
        'total_annotations': len(annotations),
        'total_categories': len(categories),
        'keypoint_annotations': len(keypoint_annotations),
        'total_keypoints': total_keypoints,
        'category_distribution': dict(category_counts)
    }

def check_file_integrity(dataset_dir, coco_data):
    """Check if all referenced image files exist."""
    dataset_path = Path(dataset_dir)
    missing_files = []
    
    for image_info in coco_data['images']:
        image_path = dataset_path / image_info['file_name']
        if not image_path.exists():
            missing_files.append(image_info['file_name'])
    
    if missing_files:
        print(f"❌ Missing {len(missing_files)} image files")
        for file in missing_files[:5]:  # Show first 5
            print(f"  Missing: {file}")
        if len(missing_files) > 5:
            print(f"  ... and {len(missing_files) - 5} more")
        return False
    else:
        print(f"✅ All {len(coco_data['images'])} image files exist")
        return True

def validate_keypoint_annotations(coco_data):
    """Validate keypoint annotation format and consistency."""
    categories = {cat['id']: cat for cat in coco_data['categories']}
    issues = []
    
    for ann in coco_data['annotations']:
        if 'keypoints' not in ann:
            continue
            
        keypoints = ann['keypoints']
        category = categories.get(ann['category_id'])
        
        if not category:
            issues.append(f"Annotation {ann['id']}: Unknown category {ann['category_id']}")
            continue
        
        expected_keypoints = len(category.get('keypoints', []))
        actual_keypoint_triplets = len(keypoints) // 3
        
        if expected_keypoints != actual_keypoint_triplets:
            issues.append(f"Annotation {ann['id']}: Expected {expected_keypoints} keypoints, got {actual_keypoint_triplets}")
        
        # Check keypoint format (x, y, visibility triplets)
        if len(keypoints) % 3 != 0:
            issues.append(f"Annotation {ann['id']}: Keypoints not in triplets (x,y,v)")
        
        # Check visibility values
        for i in range(2, len(keypoints), 3):
            visibility = keypoints[i]
            if visibility not in [0, 1, 2]:
                issues.append(f"Annotation {ann['id']}: Invalid visibility value {visibility}")
    
    if issues:
        print(f"❌ Found {len(issues)} keypoint validation issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  {issue}")
        if len(issues) > 10:
            print(f"  ... and {len(issues) - 10} more")
        return False
    else:
        print(f"✅ All keypoint annotations are valid")
        return True

def generate_visualizations(coco_data, output_dir):
    """Generate visualization plots for dataset analysis."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("⚠️  Matplotlib/Seaborn not available, skipping visualizations")
        return
    
    viz_dir = Path(output_dir) / "validation_plots"
    viz_dir.mkdir(exist_ok=True)
    
    # Category distribution
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    category_counts = Counter(ann['category_id'] for ann in coco_data['annotations'])
    
    if category_counts:
        plt.figure(figsize=(10, 6))
        names = [categories[cat_id] for cat_id in category_counts.keys()]
        counts = list(category_counts.values())
        
        plt.bar(names, counts)
        plt.title('Category Distribution')
        plt.xlabel('Tool Type')
        plt.ylabel('Number of Annotations')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(viz_dir / "category_distribution.png", dpi=150)
        plt.close()
    
    # Keypoints per annotation
    keypoint_counts = [ann.get('num_keypoints', 0) for ann in coco_data['annotations'] if 'keypoints' in ann]
    
    if keypoint_counts:
        plt.figure(figsize=(8, 6))
        plt.hist(keypoint_counts, bins=range(max(keypoint_counts) + 2), alpha=0.7)
        plt.title('Distribution of Visible Keypoints per Annotation')
        plt.xlabel('Number of Visible Keypoints')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(viz_dir / "keypoints_distribution.png", dpi=150)
        plt.close()
    
    print(f"✅ Visualizations saved to {viz_dir}")

def sample_images_check(dataset_dir, coco_data, num_samples=5):
    """Check a sample of images for basic quality."""
    dataset_path = Path(dataset_dir)
    images = coco_data['images']
    
    if len(images) < num_samples:
        num_samples = len(images)
    
    sample_indices = np.random.choice(len(images), num_samples, replace=False)
    
    print(f"\n🔍 Checking {num_samples} sample images:")
    
    for idx in sample_indices:
        image_info = images[idx]
        image_path = dataset_path / image_info['file_name']
        
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"  ❌ {image_info['file_name']}: Cannot load image")
                continue
            
            h, w, c = img.shape
            expected_w, expected_h = image_info['width'], image_info['height']
            
            if w != expected_w or h != expected_h:
                print(f"  ❌ {image_info['file_name']}: Size mismatch. Expected {expected_w}x{expected_h}, got {w}x{h}")
            else:
                print(f"  ✅ {image_info['file_name']}: {w}x{h}, {c} channels")
                
        except Exception as e:
            print(f"  ❌ {image_info['file_name']}: Error - {e}")

def main():
    parser = argparse.ArgumentParser(description="Validate synthetic surgical instrument dataset")
    parser.add_argument('dataset_dir', help='Path to dataset directory')
    parser.add_argument('--coco-file', default='coco_annotations.json', help='COCO annotations filename')
    parser.add_argument('--visualizations', action='store_true', help='Generate visualization plots')
    parser.add_argument('--samples', type=int, default=5, help='Number of sample images to check')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_dir)
    coco_file = dataset_path / args.coco_file
    
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_path}")
        return 1
    
    if not coco_file.exists():
        print(f"❌ COCO file not found: {coco_file}")
        return 1
    
    print(f"🔍 Validating dataset: {dataset_path}")
    print(f"📄 COCO file: {coco_file}")
    print("=" * 60)
    
    # Load COCO data
    coco_data = load_coco_data(coco_file)
    if not coco_data:
        return 1
    
    # Run validation checks
    checks_passed = 0
    total_checks = 5
    
    if validate_coco_structure(coco_data):
        checks_passed += 1
    
    if check_file_integrity(dataset_path, coco_data):
        checks_passed += 1
    
    if validate_keypoint_annotations(coco_data):
        checks_passed += 1
    
    # Generate statistics
    stats = analyze_dataset_statistics(coco_data, dataset_path)
    checks_passed += 1
    
    # Sample image check
    sample_images_check(dataset_path, coco_data, args.samples)
    checks_passed += 1
    
    # Generate visualizations if requested
    if args.visualizations:
        generate_visualizations(coco_data, dataset_path)
    
    # Final summary
    print(f"\n" + "=" * 60)
    print(f"🏁 Validation Complete")
    print(f"Passed {checks_passed}/{total_checks} validation checks")
    
    if checks_passed == total_checks:
        print("✅ Dataset is valid and ready for use!")
        return 0
    else:
        print("❌ Dataset has issues that should be addressed.")
        return 1

if __name__ == "__main__":
    exit(main())
