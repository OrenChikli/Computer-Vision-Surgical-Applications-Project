#!/usr/bin/env python3
"""
Train Ultralytics YOLO model directly on COCO format dataset.

This script sets up the configuration and training commands for training
a YOLO model on your synthetic COCO dataset without conversion.
"""

import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, List


def create_training_config(coco_json_path: str, output_dir: str, model_size: str = "n") -> str:
    """
    Create a training configuration file for Ultralytics YOLO.

    Args:
        coco_json_path: Path to the COCO annotations JSON file
        output_dir: Directory to save training outputs
        model_size: YOLO model size (n, s, m, l, x)

    Returns:
        Path to the created config file
    """

    # Load COCO data to get category information
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Extract categories
    categories = coco_data['categories']
    category_names = [cat['name'] for cat in categories]

    # Check if keypoints exist
    has_keypoints = any('keypoints' in cat for cat in categories)

    # Create training configuration
    config = {
        'task': 'pose' if has_keypoints else 'detect',
        'mode': 'train',
        'model': f'yolov8{model_size}.pt',
        'data': coco_json_path,
        'epochs': 100,
        'patience': 50,
        'batch': 16,
        'imgsz': 640,
        'save': True,
        'cache': False,
        'device': 'auto',
        'workers': 8,
        'project': output_dir,
        'name': 'yolo_training',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0 if has_keypoints else None,
        'kobj': 2.0 if has_keypoints else None,
        'label_smoothing': 0.0,
        'nbs': 64,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True,
        'save_period': -1,
        'local_rank': -1,
        'entity': None,
        'upload_dataset': False,
        'bbox_interval': -1,
        'artifact_alias': 'latest'
    }

    # Remove None values
    config = {k: v for k, v in config.items() if v is not None}

    # Create config file
    config_path = Path(output_dir) / 'training_config.yaml'
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    return str(config_path)


def create_dataset_splits(coco_json_path: str, output_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.2,
                          test_ratio: float = 0.1) -> tuple:
    """
    Create train/val/test splits from COCO dataset.

    Args:
        coco_json_path: Path to the COCO annotations JSON file
        output_dir: Directory to save the split files
        train_ratio: Training data ratio (default: 0.7)
        val_ratio: Validation data ratio (default: 0.2)
        test_ratio: Test data ratio (default: 0.1)

    Returns:
        Tuple of (train_path, val_path, test_path)
    """

    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    # Load COCO data
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Get all images
    images = coco_data['images']
    annotations = coco_data['annotations']

    # Shuffle images for random split
    import random
    random.shuffle(images)

    # Calculate split indices
    total_images = len(images)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    # Split images
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    print(f"📊 Dataset split:")
    print(f"   Train: {len(train_images)} images ({len(train_images) / total_images * 100:.1f}%)")
    print(f"   Val: {len(val_images)} images ({len(val_images) / total_images * 100:.1f}%)")
    print(f"   Test: {len(test_images)} images ({len(test_images) / total_images * 100:.1f}%)")

    # Create split files
    def create_split_file(split_images, split_name):
        # Get image IDs for this split
        split_image_ids = {img['id'] for img in split_images}

        # Filter annotations for this split
        split_annotations = [ann for ann in annotations if ann['image_id'] in split_image_ids]

        # Create split data
        split_data = {
            'images': split_images,
            'annotations': split_annotations,
            'categories': coco_data['categories']
        }

        # Save split file
        split_path = Path(output_dir) / f'{split_name}.json'
        with open(split_path, 'w') as f:
            json.dump(split_data, f, indent=2)

        return str(split_path)

    # Create all splits
    train_path = create_split_file(train_images, 'train')
    val_path = create_split_file(val_images, 'val')
    test_path = create_split_file(test_images, 'test')

    return train_path, val_path, test_path


def generate_flip_idx(categories: List[Dict], skeleton_config_path: str = None) -> tuple:
    """
    Generate proper flip_idx for keypoint augmentation using skeleton configuration.

    Args:
        categories: List of COCO categories
        skeleton_config_path: Path to tool_skeletons.json file

    Returns:
        Tuple of (keypoint_names, flip_idx, has_keypoints)
    """

    # Check if keypoints exist
    has_keypoints = any('keypoints' in cat for cat in categories)
    if not has_keypoints:
        return [], [], False

    # Find the category with the most keypoints as reference
    max_keypoints = 0
    reference_category = None

    for cat in categories:
        if 'keypoints' in cat and len(cat['keypoints']) > max_keypoints:
            max_keypoints = len(cat['keypoints'])
            reference_category = cat

    if not reference_category:
        return [], [], False

    keypoint_names = reference_category['keypoints']
    num_keypoints = len(keypoint_names)

    # Initialize flip_idx - each keypoint maps to itself by default
    flip_idx = list(range(num_keypoints))

    # Load skeleton configuration if provided
    if skeleton_config_path and Path(skeleton_config_path).exists():
        try:
            with open(skeleton_config_path, 'r') as f:
                skeleton_config = json.load(f)

            print(f"📋 Loaded skeleton configuration for {len(skeleton_config)} tool types")

            # Create name to index mapping
            kpt_name_to_idx = {name: i for i, name in enumerate(keypoint_names)}

            # Collect all flip pairs from all tool types
            all_flip_pairs = []
            for tool_type, config in skeleton_config.items():
                flip_pairs = config.get('flip_pairs', [])
                all_flip_pairs.extend(flip_pairs)

            # Apply flip mappings
            pairs_found = 0
            for pair in all_flip_pairs:
                if len(pair) == 2:
                    kpt1_name, kpt2_name = pair

                    if kpt1_name in kpt_name_to_idx and kpt2_name in kpt_name_to_idx:
                        idx1 = kpt_name_to_idx[kpt1_name]
                        idx2 = kpt_name_to_idx[kpt2_name]

                        # Create mutual mapping
                        flip_idx[idx1] = idx2
                        flip_idx[idx2] = idx1
                        pairs_found += 1

                        print(f"  - Mapped '{kpt1_name}' (idx {idx1}) ↔ '{kpt2_name}' (idx {idx2})")

            print(f"  - Found {pairs_found} flip pairs")

        except Exception as e:
            print(f"⚠️  Warning: Could not load skeleton config: {e}")
            print(f"  - Using default flip_idx (no augmentation)")

    else:
        print(f"📋 No skeleton config provided, using default flip_idx (no augmentation)")

    print(f"  - Generated flip_idx: {flip_idx}")
    return keypoint_names, flip_idx, True


def create_dataset_yaml(coco_json_path: str, output_dir: str, train_path: str, val_path: str, test_path: str,
                        skeleton_config_path: str = None) -> str:
    """
    Create a dataset.yaml file for Ultralytics YOLO training with splits.

    Args:
        coco_json_path: Path to the original COCO annotations JSON file
        output_dir: Directory to save the dataset config
        train_path: Path to training split JSON
        val_path: Path to validation split JSON
        test_path: Path to test split JSON
        skeleton_config_path: Path to tool_skeletons.json file

    Returns:
        Path to the created dataset.yaml file
    """

    # Load COCO data to get category information
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Extract categories
    categories = coco_data['categories']
    category_names = [cat['name'] for cat in categories]

    # Generate keypoint information with proper flip_idx
    keypoint_names, flip_idx, has_keypoints = generate_flip_idx(categories, skeleton_config_path)

    # Create dataset configuration with splits
    dataset_config = {
        'path': str(Path(coco_json_path).parent.absolute()),
        'train': str(Path(train_path).name),  # Training split JSON
        'val': str(Path(val_path).name),  # Validation split JSON
        'test': str(Path(test_path).name),  # Test split JSON
        'nc': len(categories),
        'names': category_names
    }

    # Add keypoint information if available
    if has_keypoints:
        keypoint_info = {
            'kpt_shape': [len(keypoint_names), 3],  # [num_keypoints, 3] for x, y, visibility
            'flip_idx': flip_idx
        }
        dataset_config.update(keypoint_info)

    # Create dataset.yaml file
    dataset_yaml_path = Path(output_dir) / 'dataset.yaml'
    dataset_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    with open(dataset_yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, indent=2)

    return str(dataset_yaml_path)


def print_training_commands(coco_json_path: str, config_path: str, dataset_yaml_path: str, model_size: str = "n"):
    """Print the training commands for the user."""

    print("\n" + "=" * 60)
    print("🚀 YOLO TRAINING SETUP COMPLETE!")
    print("=" * 60)

    print(f"\n📁 Files created:")
    print(f"   Training config: {config_path}")
    print(f"   Dataset config: {dataset_yaml_path}")

    print(f"\n🎯 Training commands:")
    print(f"\n# Option 1: Train with custom config file (recommended)")
    print(f"yolo train cfg={config_path}")

    print(f"\n# Option 2: Train with dataset.yaml (uses splits)")
    print(f"yolo train model=yolov8{model_size}.pt data={dataset_yaml_path} epochs=100 imgsz=640")

    print(f"\n# Option 3: Train with original COCO file (no splits)")
    print(f"yolo train model=yolov8{model_size}.pt data={coco_json_path} epochs=100")

    print(f"\n# Option 4: Resume training from checkpoint")
    print(f"yolo train resume runs/train/yolo_training/weights/last.pt")

    print(f"\n📊 Monitor training:")
    print(f"   # View training progress")
    print(f"   tensorboard --logdir runs/train")

    print(f"\n🔍 Validate model:")
    print(f"   yolo val model=runs/train/yolo_training/weights/best.pt data={coco_json_path}")

    print(f"\n📸 Predict on new images:")
    print(f"   yolo predict model=runs/train/yolo_training/weights/best.pt source=path/to/images")

    print(f"\n💡 Tips:")
    print(f"   - Start with yolov8n.pt for faster training")
    print(f"   - Use yolov8s.pt or yolov8m.pt for better accuracy")
    print(f"   - Adjust batch size based on your GPU memory")
    print(f"   - Monitor training with tensorboard")
    print(f"   - Use early stopping with patience=50")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Setup Ultralytics YOLO training on COCO dataset")
    parser.add_argument("coco_json", help="Path to COCO annotations JSON file")
    parser.add_argument("output_dir", help="Output directory for training configs")
    parser.add_argument("--model-size", choices=['n', 's', 'm', 'l', 'x'], default='n',
                        help="YOLO model size (default: n)")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Training data ratio (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation data ratio (default: 0.2)")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Test data ratio (default: 0.1)")
    parser.add_argument("--skeleton-json",
                        help="Path to tool_skeletons.json file for flip_idx generation (auto-detected if in same directory as COCO JSON)")

    args = parser.parse_args()

    # Validate inputs
    if not Path(args.coco_json).exists():
        print(f"❌ Error: COCO JSON file not found: {args.coco_json}")
        return 1

    if not Path(args.coco_json).suffix == '.json':
        print(f"❌ Error: File must be a JSON file: {args.coco_json}")
        return 1

    # Auto-detect skeleton configuration if not provided
    if not args.skeleton_json:
        coco_dir = Path(args.coco_json).parent
        auto_skeleton_path = coco_dir / "tool_skeletons.json"
        if auto_skeleton_path.exists():
            args.skeleton_json = str(auto_skeleton_path)
            print(f"🔍 Auto-detected skeleton config: {auto_skeleton_path}")
        else:
            print(f"💡 No skeleton config found. For proper augmentation, provide --skeleton-json")

    # Validate split ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        print(f"❌ Error: Split ratios must sum to 1.0, got {total_ratio}")
        return 1

    try:
        # Create dataset splits
        print("🔀 Creating dataset splits...")
        train_path, val_path, test_path = create_dataset_splits(
            args.coco_json, args.output_dir,
            args.train_ratio, args.val_ratio, args.test_ratio
        )
        print(f"✅ Dataset splits created:")
        print(f"   Train: {Path(train_path).name}")
        print(f"   Val: {Path(val_path).name}")
        print(f"   Test: {Path(test_path).name}")

        # Create training configuration
        print("🔧 Creating training configuration...")
        config_path = create_training_config(args.coco_json, args.output_dir, args.model_size)
        print(f"✅ Training config created: {config_path}")

        # Create dataset configuration with splits
        print("📋 Creating dataset configuration...")
        dataset_yaml_path = create_dataset_yaml(args.coco_json, args.output_dir, train_path, val_path, test_path,
                                                args.skeleton_json)
        print(f"✅ Dataset config created: {dataset_yaml_path}")

        # Print training commands
        print_training_commands(args.coco_json, config_path, dataset_yaml_path, args.model_size)

        return 0

    except Exception as e:
        print(f"❌ Error during setup: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
