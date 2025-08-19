import json
import bpy
from pathlib import Path
from typing import Dict, List


def merge_keypoints_into_coco(output_dir: Path, frame_annotations: List[Dict], curr_frame: int):
    """Merge keypoint annotations into the existing COCO file."""
    coco_file = output_dir / "coco_annotations.json"

    if not coco_file.exists():
        return

    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    image_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == curr_frame]

    # Create a more robust matching system
    for coco_ann in image_annotations:
        # Try to find matching keypoint annotation by both category and position/instance
        best_match = None
        for kp_ann in frame_annotations:
            if (coco_ann["category_id"] == kp_ann["category_id"] and
                    "keypoints" not in coco_ann):
                # If we have instance info, use it for better matching
                if "instance_id" in coco_ann and "instance_id" in kp_ann:
                    if coco_ann["instance_id"] == kp_ann["instance_id"]:
                        best_match = kp_ann
                        break
                else:
                    # Fallback: use the first unmatched annotation of the same category
                    if kp_ann not in [a.get("_matched_keypoints") for a in image_annotations]:
                        best_match = kp_ann
                        break

        if best_match:
            coco_ann["keypoints"] = best_match["keypoints"]
            coco_ann["num_keypoints"] = best_match["num_keypoints"]
            coco_ann["_matched_keypoints"] = best_match  # Mark as matched

    with open(coco_file, 'w') as f:
        json.dump(coco_data, f, indent=4)


def save_keypoint_annotations(data: Dict, output_dir: Path,
                              start_frame: int, tool_manager) -> None:
    """Save keypoint annotations for rendered frames."""
    if not tool_manager.keypoint_extractor:
        return

    for i in range(len(data["colors"])):
        bpy.context.scene.frame_set(i)
        bpy.context.view_layer.update()

        frame_annotations = tool_manager.keypoint_extractor.extract_frame_keypoints()

        if frame_annotations:
            merge_keypoints_into_coco(output_dir, frame_annotations, start_frame + i)


def update_coco_categories(output_dir: Path, tool_manager) -> None:
    """Update COCO file with proper categories."""
    coco_file = output_dir / "coco_annotations.json"

    if not coco_file.exists():
        return

    try:
        with open(coco_file, 'r') as f:
            coco_data = json.load(f)

        coco_data["categories"] = tool_manager.get_coco_categories()

        with open(coco_file, 'w') as f:
            json.dump(coco_data, f, indent=2)

        print("Updated COCO categories with keypoint and skeleton information")

    except Exception as e:
        print(f"Error updating COCO categories: {e}")
