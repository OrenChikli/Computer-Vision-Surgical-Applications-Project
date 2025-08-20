#!/usr/bin/env python3
"""
Convert COCO format dataset to YOLO format for ultralytics YOLO training.

This script converts COCO annotations (including keypoints) to YOLO format:
- Creates YOLO directory structure
- Converts bounding boxes to normalized format
- Handles keypoint annotations with flip_idx for augmentation
- Creates dataset.yaml configuration file
- Reads flip_pairs directly from COCO categories (self-contained)
"""

import json
import argparse
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple
from yaml_utils import save_yaml


class COCOToYOLOConverter:
    """Convert COCO format dataset to YOLO format with flip_idx support."""

    def __init__(self, dataset_path: str):
        # Statistics tracking
        self.stats = {
            'total_annotations': 0,
            'annotations_with_keypoints': 0,
            'images_processed': 0,
            'labels_created': 0
        }
        """
        Initialize converter.

        Args:
            dataset_path: Path to the dataset directory containing COCO JSON and images
        """
        self.dataset_path = Path(dataset_path)
        self.output_dir = self.dataset_path / "yolo_format"
        self.output_dir.mkdir(parents=True, exist_ok=True)



        # Load COCO data
        self.coco_data = self._load_coco_json()

        # Create output directories
        self.setup_directories()

        # Category and image mappings
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        self.images = {img['id']: img for img in self.coco_data['images']}

        # Generate unified keypoint schema and flip_idx from COCO data only
        self.all_keypoint_names = self._generate_unified_keypoints()
        self.flip_idx = self._generate_flip_idx()

    def _load_coco_json(self) -> Dict:
        """Automatically find and load COCO JSON file from dataset directory."""
        print(f"🔍 Searching for COCO JSON file in: {self.dataset_path}")

        # Find JSON files
        json_files = list(self.dataset_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.dataset_path}")

        # Load and validate first JSON file
        json_file = json_files[0]
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Validate COCO format
        required_keys = {'images', 'annotations', 'categories'}
        if not required_keys.issubset(data.keys()):
            raise ValueError(f"Invalid COCO format in {json_file.name}. Missing keys: {required_keys - data.keys()}")

        print(f"✅ Found COCO format file: {json_file.name}")
        return data


    def _generate_unified_keypoints(self) -> List[str]:
        """Generate unified keypoint schema across all categories."""
        print("🛠️ Creating unified skeleton for all tool types...")

        # Collect unique keypoints using set comprehension
        all_keypoints = {
            kp_name for cat in self.coco_data['categories']
            for kp_name in cat.get('keypoints', [])
        }

        keypoint_list = sorted(all_keypoints)  # Sort for consistency
        print(f"  - Found {len(keypoint_list)} unique keypoints across all categories")
        return keypoint_list

    def _generate_flip_idx(self) -> List[int]:
        """Generate flip_idx array for keypoint augmentation from COCO categories."""
        print("🔄 Generating flip_idx for keypoint augmentation...")

        num_keypoints = len(self.all_keypoint_names)
        flip_idx = list(range(num_keypoints))  # Default: each keypoint maps to itself
        kpt_name_to_idx = {name: i for i, name in enumerate(self.all_keypoint_names)}

        # Read flip_pairs from COCO categories (primary method)
        all_flip_pairs_indices = []
        coco_pairs_found = 0

        for category in self.coco_data['categories']:
            category_keypoints = category.get('keypoints', [])
            flip_pairs = category.get('flip_pairs', [])

            if not flip_pairs or not category_keypoints:
                continue

            # Create mapping from category keypoint indices to unified indices
            category_to_unified = {}
            for cat_idx, kpt_name in enumerate(category_keypoints):
                if kpt_name in kpt_name_to_idx:
                    category_to_unified[cat_idx + 1] = kpt_name_to_idx[kpt_name]  # COCO uses 1-based indices

            # Convert category flip_pairs to unified indices
            for pair in flip_pairs:
                if len(pair) == 2:
                    cat_idx1, cat_idx2 = pair
                    if cat_idx1 in category_to_unified and cat_idx2 in category_to_unified:
                        unified_idx1 = category_to_unified[cat_idx1]
                        unified_idx2 = category_to_unified[cat_idx2]
                        all_flip_pairs_indices.append([unified_idx1, unified_idx2])
                        coco_pairs_found += 1

        # Apply flip mappings from COCO categories
        if coco_pairs_found > 0:
            print(f"  - Reading flip_pairs from COCO categories")
            for idx1, idx2 in all_flip_pairs_indices:
                flip_idx[idx1] = idx2
                flip_idx[idx2] = idx1
                kpt1_name = self.all_keypoint_names[idx1]
                kpt2_name = self.all_keypoint_names[idx2]
                print(f"  - Mapped '{kpt1_name}' (idx {idx1}) ↔ '{kpt2_name}' (idx {idx2})")
        else:
            print("  - No flip_pairs found in COCO categories, using default flip_idx")

        print(f"  - Applied {coco_pairs_found} flip pairs")
        print(f"  - Generated flip_idx: {flip_idx}")
        return flip_idx

    def setup_directories(self):
        """Create YOLO dataset directory structure."""
        dirs = [
            self.output_dir / "images" / "train",
            self.output_dir / "images" / "val",
            self.output_dir / "images" / "test",
            self.output_dir / "labels" / "train",
            self.output_dir / "labels" / "val",
            self.output_dir / "labels" / "test"
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        print(f"✅ Created YOLO directory structure in: {self.output_dir}")

    def convert_bbox_to_yolo(self, bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """Convert COCO bbox to YOLO format (normalized center coordinates)."""
        x, y, w, h = bbox
        return [
            (x + w / 2) / img_width,  # x_center
            (y + h / 2) / img_height,  # y_center
            w / img_width,  # width
            h / img_height  # height
        ]

    def convert_keypoints_to_yolo(self, keypoints: List[float], img_width: int, img_height: int,
                                  category_id: int) -> Tuple[List[float], int]:
        """Convert COCO keypoints to YOLO format with unified schema."""
        category = self.categories[category_id]
        original_keypoints = category.get('keypoints', [])

        # Create mapping from original to unified indices
        kpt_mapping = {
            i: self.all_keypoint_names.index(kp_name)
            for i, kp_name in enumerate(original_keypoints)
            if kp_name in self.all_keypoint_names
        }

        # Initialize unified keypoint array
        num_unified = len(self.all_keypoint_names)
        yolo_keypoints = [0.0] * (num_unified * 3)

        # Map keypoints to unified schema
        visible_count = 0
        for i in range(0, len(keypoints), 3):
            if i // 3 in kpt_mapping:
                x, y, v = keypoints[i:i + 3]
                unified_idx = kpt_mapping[i // 3]

                # Store normalized coordinates
                yolo_keypoints[unified_idx * 3:unified_idx * 3 + 3] = [
                    x / img_width, y / img_height, v
                ]

                if v > 0:
                    visible_count += 1

        return yolo_keypoints, visible_count

    def process_annotations(self, split: str, split_ratio: float, test_ratio: float):
        """Process annotations for a specific split."""
        print(f"📝 Processing {split} split...")

        # Split images into train/val/test
        all_images = list(self.images.values())
        random.shuffle(all_images)

        total_images = len(all_images)
        train_idx = int(total_images * split_ratio)
        val_idx = int(total_images * (1.0 - test_ratio))  # Fixed val calculation

        if split == "train":
            split_images = all_images[:train_idx]
        elif split == "val":
            split_images = all_images[train_idx:val_idx]
        else:  # test
            split_images = all_images[val_idx:]

        processed_count = 0
        for img_info in split_images:
            if self._process_single_image(img_info, split):
                processed_count += 1

        print(f"✅ Processed {processed_count} images for {split} split")

    def _process_single_image(self, img_info: Dict, split: str) -> bool:
        """Process a single image and its annotations."""
        img_filename = img_info['file_name']

        # Handle relative paths in COCO JSON (e.g., "images/000528.jpg")
        # Extract just the filename without the directory part
        img_basename = Path(img_filename).name

        # Source path: dataset_path + relative path from COCO JSON
        src_img_path = self.dataset_path / img_filename

        # Destination path: output_dir/images/split/filename (without directory part)
        dst_img_path = self.output_dir / "images" / split / img_basename

        # Copy image
        if not src_img_path.exists():
            print(f"⚠️  Warning: Image not found: {src_img_path}")
            return False

        shutil.copy2(src_img_path, dst_img_path)

        # Process annotations
        img_annotations = [
            ann for ann in self.coco_data['annotations']
            if ann['image_id'] == img_info['id']
        ]

        # Create label file - use basename for consistency
        label_filename = img_basename.replace(Path(img_basename).suffix, '.txt')
        label_path = self.output_dir / "labels" / split / label_filename

        # Filter out empty labels and write valid ones
        valid_labels = []
        for ann in img_annotations:
            label_line = self._create_yolo_label(ann, img_info)
            if label_line:
                valid_labels.append(label_line)

        # Only write label file if there are valid annotations
        if valid_labels:
            with open(label_path, 'w') as f:
                f.write('\n'.join(valid_labels))
            self.stats['labels_created'] += 1
        else:
            # If no valid annotations, create empty label file
            with open(label_path, 'w') as f:
                pass  # Create empty file

        self.stats['images_processed'] += 1
        return True

    def _create_yolo_label(self, ann: Dict, img_info: Dict) -> str:
        """Create a single YOLO label line."""
        if 'bbox' not in ann:
            return ""

        # Track total annotations
        self.stats['total_annotations'] += 1

        # Class ID (0-based for YOLO)
        class_id = ann['category_id'] - 1

        # Convert bbox
        bbox_yolo = self.convert_bbox_to_yolo(
            ann['bbox'], img_info['width'], img_info['height']
        )

        # Start with bbox line
        label_parts = [str(class_id)] + [f"{x:.6f}" for x in bbox_yolo]

        # Convert keypoints if available
        if 'keypoints' in ann and ann['keypoints']:
            keypoints_yolo, visible_count = self.convert_keypoints_to_yolo(
                ann['keypoints'], img_info['width'], img_info['height'], ann['category_id']
            )
        else:
            # Create zero-filled keypoints if none exist
            num_keypoints = len(self.all_keypoint_names)
            keypoints_yolo = [0.0] * (num_keypoints * 3)
            visible_count = 0

        # Only include annotation if there are visible keypoints
        if visible_count > 0:
            # Track annotations with keypoints
            self.stats['annotations_with_keypoints'] += 1

            # Include keypoints
            label_parts.extend(f"{x:.6f}" for x in keypoints_yolo)
            return " ".join(label_parts)

        return ""  # Return empty string if no visible keypoints

    def create_dataset_yaml(self):
        """Create dataset.yaml configuration file."""
        yaml_content = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.categories),
            'names': [cat['name'] for cat in self.categories.values()],
            'kpt_shape': [len(self.all_keypoint_names), 3],
            'flip_idx': self.flip_idx
        }

        yaml_path = self.output_dir / "dataset.yaml"
        save_yaml(yaml_content, yaml_path, sort_keys=False)

        print(f"📄 Created dataset.yaml: {yaml_path}")
        print(f"  - Keypoints: {len(self.all_keypoint_names)}")
        print(f"  - Flip indices: {self.flip_idx}")

    def convert(self, split_ratio: float = 0.8, test_ratio: float = 0.1):
        """Convert complete dataset."""
        print("🚀 Starting COCO to YOLO conversion...")
        print(f"Dataset path: {self.dataset_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Train/Val split ratio: {split_ratio}")
        print(f"Test ratio: {test_ratio}")
        print(f"Categories: {[cat['name'] for cat in self.categories.values()]}")
        print(f"Unified keypoints: {self.all_keypoint_names}")
        print("-" * 50)

        # Process splits
        self.process_annotations("train", split_ratio, test_ratio)
        self.process_annotations("val", split_ratio, test_ratio)
        self.process_annotations("test", split_ratio, test_ratio)

        # Create configuration
        self.create_dataset_yaml()
        self.print_summary()

        print("\n🎉 Conversion completed successfully!")
        print(f"YOLO dataset saved to: {self.output_dir}")
        print(f"Use dataset.yaml for training with ultralytics YOLO")
        print(f"Generated flip_idx: {self.flip_idx}")

    def print_summary(self):
        """Print conversion summary."""

        def count_files(directory: Path, extensions: List[str]) -> int:
            return sum(len(list(directory.glob(f"*{ext}"))) for ext in extensions)

        train_dir = self.output_dir / "images" / "train"
        val_dir = self.output_dir / "images" / "val"
        test_dir = self.output_dir / "images" / "test"
        train_labels_dir = self.output_dir / "labels" / "train"
        val_labels_dir = self.output_dir / "labels" / "val"
        test_labels_dir = self.output_dir / "labels" / "test"

        train_images = count_files(train_dir, ['.jpg', '.png', '.jpeg'])
        val_images = count_files(val_dir, ['.jpg', '.png', '.jpeg'])
        test_images = count_files(test_dir, ['.jpg', '.png', '.jpeg'])
        train_labels = count_files(train_labels_dir, ['.txt'])
        val_labels = count_files(val_labels_dir, ['.txt'])
        test_labels = count_files(test_labels_dir, ['.txt'])

        print("\n" + "=" * 50)
        print("CONVERSION SUMMARY")
        print("=" * 50)
        print(f"Total categories: {len(self.categories)}")
        print(f"Categories: {[cat['name'] for cat in self.categories.values()]}")
        print(f"Unified keypoints: {len(self.all_keypoint_names)}")
        print(f"Keypoint names: {self.all_keypoint_names}")
        print(f"Flip indices: {self.flip_idx}")
        print(f"Training images: {train_images}")
        print(f"Training labels: {train_labels}")
        print(f"Validation images: {val_images}")
        print(f"Validation labels: {val_labels}")
        print(f"Test images: {test_images}")
        print(f"Test labels: {test_labels}")
        print(f"Total images: {train_images + val_images + test_images}")
        print("=" * 50)

        # Print conversion statistics
        print(f"\n📊 Conversion Statistics:")
        print(f"   Total annotations processed: {self.stats['total_annotations']}")
        print(f"   Annotations with keypoints: {self.stats['annotations_with_keypoints']}")
        print(f"   Images processed: {self.stats['images_processed']}")
        print(f"   Label files created: {self.stats['labels_created']}")
        print(f"   Unified keypoint schema: {len(self.all_keypoint_names)} keypoints")
        print(f"   Categories: {[cat['name'] for cat in self.categories.values()]}")
        print(f"   Keypoint names: {self.all_keypoint_names}")

        # Validate generated annotations
        print("\n🔍 Validating generated annotations...")
        self._validate_generated_annotations()

    def _validate_generated_annotations(self):
        """Validate that generated YOLO annotations have consistent keypoint structure."""
        import glob

        label_files = list(self.output_dir.glob("labels/**/*.txt"))

        if not label_files:
            print("   No label files found to validate!")
            return

        # Check first 10 files
        sample_files = label_files[:10]
        expected_kpt_values = len(self.all_keypoint_names) * 3  # x,y,visibility per keypoint

        issues = []
        for file_path in sample_files:
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    if line.strip():
                        parts = line.strip().split()

                        if len(parts) < 5:  # class + bbox minimum
                            continue

                        keypoint_values = parts[5:]  # skip class and bbox
                        actual_kpt_values = len(keypoint_values)

                        if actual_kpt_values != expected_kpt_values:
                            issues.append(f"{file_path.name}:{line_num} - "
                                          f"Expected {expected_kpt_values} keypoint values, got {actual_kpt_values}")

                        # Check if divisible by 3
                        if actual_kpt_values % 3 != 0:
                            issues.append(f"{file_path.name}:{line_num} - "
                                          f"Keypoint values ({actual_kpt_values}) not divisible by 3")

            except Exception as e:
                issues.append(f"{file_path.name} - Read error: {e}")

        if issues:
            print(f"   ⚠️  Found {len(issues)} validation issues:")
            for issue in issues[:5]:  # Show first 5
                print(f"      {issue}")
            if len(issues) > 5:
                print(f"      ... and {len(issues) - 5} more issues")
        else:
            print(
                f"   ✅ All sampled annotations have consistent structure ({expected_kpt_values} values per annotation)")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Convert COCO dataset to YOLO format")
    parser.add_argument("dataset_path", help="Path to the dataset directory containing COCO JSON and images")
    parser.add_argument("--split-ratio", type=float, default=0.8,
                        help="Ratio of training data (default: 0.8)")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Ratio of test data (default: 0.1)")

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.dataset_path).exists():
        print(f"❌ Error: Dataset directory not found: {args.dataset_path}")
        return 1

    if not Path(args.dataset_path).is_dir():
        print(f"❌ Error: Dataset path must be a directory: {args.dataset_path}")
        return 1


    if not 0 < args.split_ratio < 1:
        print(f"❌ Error: Split ratio must be between 0 and 1, got: {args.split_ratio}")
        return 1

    if not 0 < args.test_ratio < 1:
        print(f"❌ Error: Test ratio must be between 0 and 1, got: {args.test_ratio}")
        return 1

    # Ensure train + test + val <= 1.0
    if args.split_ratio + args.test_ratio > 1.0:
        print(f"❌ Error: Train ratio ({args.split_ratio}) + test ratio ({args.test_ratio}) cannot exceed 1.0")
        return 1

    try:
        converter = COCOToYOLOConverter(args.dataset_path)
        converter.convert(args.split_ratio, args.test_ratio)
        return 0
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
