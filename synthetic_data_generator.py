import blenderproc as bproc

import argparse
import os
import sys
import time
import random
import numpy as np
from pathlib import Path
import json
import logging
from typing import Optional

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from utils.yaml_utils import load_yaml
from utils.logger_utils import setup_logger
from utils.tool_manager import ToolManager
from utils.camera_utils import setup_camera, sample_camera_pose
from utils.lighting_utils import setup_lighting
from utils.visualization import visualize_keypoints_on_images, get_hdri_files
from utils.coco_utils import (
    save_keypoint_annotations,
    update_coco_categories
)
from utils.statistics_tracker import StatisticsTracker

# Initialize logger
logger = logging.getLogger(__name__)


def set_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try:
        bpy.context.scene.frame_set(seed)
    except:
        pass

    logger.info(f"All seeds set to: {seed}")


def generate_workspace_images(config: dict, tool_manager: ToolManager, hdri_files: list,
                              output_dir: Path, frame_idx: int, workspace_idx: int,
                              stats_tracker: Optional[StatisticsTracker] = None) -> int:
    """Generate images for a single workspace setup."""
    logger.info(f"Generating workspace setup {workspace_idx}")

    # Clear previous camera poses
    bproc.utility.reset_keyframes()

    # Create occlusion blobs
    tool_manager.create_occlusion_blobs()

    # Setup lighting
    setup_lighting(config)

    # Change background
    random_hdr = None
    if hdri_files:
        random_hdr = random.choice(hdri_files)
        bproc.world.set_world_background_hdr_img(random_hdr)
        logger.info(f"Using HDRI: {os.path.basename(random_hdr)}")

    # Collect workspace statistics
    workspace_stats = {}
    if stats_tracker:
        workspace_stats = stats_tracker.collect_workspace_statistics(
            tool_manager, workspace_idx, 
            os.path.basename(random_hdr) if random_hdr else None
        )
        # Update workspace counter in summary stats
        stats_tracker.summary_stats['workspace_setups'] = workspace_idx

    # Motion blur setup
    use_motion_blur = random.random() < config['motion_blur_prob']

    if use_motion_blur:
        logger.info(f"Using motion blur for workspace {workspace_idx}")
        blur_length = random.uniform(*config['motion_blur_length_range'])
        bproc.renderer.enable_motion_blur(motion_blur_length=blur_length)
        motion_offset = np.random.uniform(
            config['motion_offset_range']['min'],
            config['motion_offset_range']['max']
        )
    else:
        import bpy
        bpy.context.scene.render.use_motion_blur = False

    # Generate camera poses
    start_frame = frame_idx
    poses_generated = 0

    for pose_idx in range(config['poses_per_workspace']):
        if not use_motion_blur or pose_idx == 0:
            cam2world_matrix = sample_camera_pose(config)
        else:
            # Apply motion blur offset
            current_location = np.array(cam2world_matrix[:3, 3])
            new_location = current_location + motion_offset * (pose_idx / config['poses_per_workspace'])
            cam2world_matrix = bproc.math.build_transformation_mat(new_location, cam2world_matrix[:3, :3])

        bproc.camera.add_camera_pose(cam2world_matrix)
        poses_generated += 1

    blur_status = "with motion blur" if use_motion_blur else "without motion blur"
    logger.info(f"Added {poses_generated} camera poses {blur_status}")

    # Render
    data = bproc.renderer.render()

    # Write COCO annotations
    bproc.writer.write_coco_annotations(
        output_dir,
        instance_segmaps=data["instance_segmaps"],
        instance_attribute_maps=data["instance_attribute_maps"],
        color_file_format="JPEG",
        colors=data["colors"],
        append_to_existing_output=True
    )

    # Add keypoints
    save_keypoint_annotations(data, output_dir, start_frame, tool_manager)

    # Collect per-frame statistics
    if stats_tracker:
        for i in range(poses_generated):
            stats_tracker.collect_frame_statistics(
                start_frame + i, workspace_stats, data, tool_manager
            )

    return poses_generated


def main():
    """Main function to generate synthetic dataset."""


    # Simple argument parser for config file path
    parser = argparse.ArgumentParser(description="Generate synthetic surgical instrument dataset")
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_yaml(args.config)
        # Setup logging with config
        setup_logger(__name__, config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    # Set seeds for reproducibility
    set_seeds(config['seed'])

    # Setup output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Output directory: {output_dir}")

    # Initialize BlenderProc
    bproc.init()

    # Setup camera
    if not setup_camera(config['camera_params']):
        logger.error("Failed to setup camera")
        return 1

    # Initialize tool manager
    tool_manager = ToolManager(config)
    if not tool_manager.load_all_data():
        logger.error("Failed to load tool data")
        return 1

    # Setup surgical workspace
    tool_manager.setup_surgical_workspace()
    logger.info(f"Setting up surgical workspace with size {config['workspace_size']}m...")

    # Get HDRI files
    hdri_files = get_hdri_files(config.get('hdri_path', ''))
    if hdri_files:
        logger.info(f"Found {len(hdri_files)} HDRI files")

    # Initialize statistics tracker
    stats_enabled = config.get('enable_statistics', True)  # Default to enabled
    stats_tracker = StatisticsTracker(config, output_dir, enabled=stats_enabled)
    if stats_enabled:
        logger.info("Statistics tracking enabled")

    # Setup renderer
    bproc.renderer.set_max_amount_of_samples(config['render_samples'])
    bproc.renderer.set_output_format(enable_transparency=False)
    bproc.renderer.enable_segmentation_output(map_by=["instance", "class", "name"])

    # Override render resolution if specified
    if 'render_width' in config and 'render_height' in config:
        import bpy
        bpy.context.scene.render.resolution_x = config['render_width']
        bpy.context.scene.render.resolution_y = config['render_height']
        logger.info(f"Set render resolution to {config['render_width']}x{config['render_height']}")

    # Generate images
    frame_idx = 0
    workspace_idx = 0
    start_time = time.time()

    logger.info(f"Generating {config['num_images']} images...")

    while frame_idx < config['num_images']:
        workspace_idx += 1

        poses_generated = generate_workspace_images(
            config, tool_manager, hdri_files, output_dir, frame_idx, workspace_idx,
            stats_tracker
        )

        frame_idx += poses_generated

        # Progress update
        progress = (frame_idx / config['num_images']) * 100
        elapsed = time.time() - start_time
        if frame_idx > 0:
            eta = (elapsed / frame_idx) * (config['num_images'] - frame_idx)
            logger.info(f"Progress: {progress:.1f}% ({frame_idx}/{config['num_images']}), ETA: {eta / 60:.1f} minutes")

    # Update COCO categories
    update_coco_categories(output_dir, tool_manager)

    # Save skeleton configuration for training setup
    logger.info("Saving skeleton configuration...")
    skeleton_output_path = output_dir / "tool_skeletons.json"
    if tool_manager.skeletons:
        # Convert skeletons back to the format expected by training scripts
        training_skeletons = {}
        for tool_type, skeleton_data in tool_manager.skeletons.items():
            if "keypoints" in skeleton_data:
                training_skeletons[tool_type] = {
                    "keypoints": skeleton_data["keypoints"],
                    "skeleton": skeleton_data.get("skeleton", []),
                    "flip_pairs": skeleton_data.get("flip_pairs", [])
                }

        with open(skeleton_output_path, 'w') as f:
            json.dump(training_skeletons, f, indent=2)
        logger.info(f"Skeleton configuration saved to: {skeleton_output_path}")
        logger.info(f"   - {len(training_skeletons)} tool types with keypoints")
    else:
        logger.warning("No skeleton data available to save")
        
    # Save YOLO dataset configuration
    logger.info("Generating YOLO dataset configuration...")
    try:
        yolo_config = tool_manager.generate_yolo_dataset_config(str(output_dir.absolute()))
        yolo_config_path = output_dir / "dataset.yaml"
        
        from utils.yaml_utils import save_yaml
        save_yaml(yolo_config, yolo_config_path, sort_keys=False)
            
        logger.info(f"YOLO dataset configuration saved to: {yolo_config_path}")
        logger.info(f"   - {yolo_config['nc']} tool categories")
        logger.info(f"   - {yolo_config['kpt_shape'][0]} total keypoints")
        logger.info(f"   - flip_idx: {yolo_config['flip_idx']}")
        logger.info("Use this config for YOLO training with proper flip augmentations!")
        
    except Exception as e:
        logger.warning(f"Error generating YOLO dataset config: {e}")

    # Generate keypoint visualizations if requested
    if config.get('visualize_keypoints', False):
        logger.info("Generating keypoint visualizations...")
        coco_file = output_dir / "coco_annotations.json"
        visualize_keypoints_on_images(output_dir, coco_file, config)

    # Final summary
    coco_file = output_dir / "coco_annotations.json"
    if coco_file.exists():
        with open(coco_file, 'r') as f:
            final_coco = json.load(f)

        keypoint_count = sum(1 for ann in final_coco["annotations"] if "keypoints" in ann)
        total_time = time.time() - start_time

        logger.info("Dataset generation complete!")
        logger.info(f"Generated: {len(final_coco['images'])} images")
        logger.info(f"Total annotations: {len(final_coco['annotations'])}")
        logger.info(f"Annotations with keypoints: {keypoint_count}")
        logger.info(f"Categories: {[cat['name'] for cat in final_coco['categories']]}")
        logger.info(f"Workspace size: {config['workspace_size']}m")
        logger.info(f"Motion blur probability: {config['motion_blur_prob']}")
        logger.info(f"Total time: {total_time / 60:.1f} minutes")
        logger.info(f"Dataset saved to: {coco_file}")

        # Save statistics
        if stats_tracker:
            detailed_saved, summary_saved = stats_tracker.save_statistics()
            if detailed_saved:
                logger.info(f"Detailed statistics: {output_dir / 'dataset_stats.json'}")
            if summary_saved:
                logger.info(f"Summary statistics: {output_dir / 'dataset_summary.txt'}")

        # Check if skeleton config was saved
        skeleton_file = output_dir / "tool_skeletons.json"
        if skeleton_file.exists():
            logger.info(f"Skeleton config saved to: {skeleton_file}")
            logger.info(f"Use this for training: --skeleton-json {skeleton_file}")

        logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
