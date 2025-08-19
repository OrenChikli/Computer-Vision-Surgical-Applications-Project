import cv2
import json
import numpy as np
import os
import glob
from pathlib import Path
from typing import List


def get_keypoint_colors(num_keypoints):
    """Generate distinct colors for each keypoint."""
    colors = []
    for i in range(num_keypoints):
        # Generate distinct colors using HSV color space
        # OpenCV uses H: 0-179, S: 0-255, V: 0-255
        hue = int((i * 180 / num_keypoints) % 180)  # Distribute hues evenly
        saturation = 255
        value = 255

        # Convert HSV to BGR for OpenCV
        hsv = np.uint8([[[hue, saturation, value]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, bgr)))
    return colors


def visualize_keypoints_on_images(output_dir: Path, coco_file: Path, config: dict) -> None:
    """
    Visualize keypoints on rendered images and save to visualization folder.
    Correctly handles multiple tools per image.
    """
    max_keypoints = 10  # Adjust based on your maximum number of keypoints
    keypoint_colors = get_keypoint_colors(max_keypoints)
    viz_dir = output_dir / "keypoint_visualizations"
    viz_dir.mkdir(exist_ok=True)

    # Load COCO annotations
    if not coco_file.exists():
        print("No COCO annotations file found for visualization")
        return

    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    # Group annotations by image_id for efficient processing
    annotations_by_image = {}
    for ann in coco_data.get("annotations", []):
        if "keypoints" not in ann:
            continue
        image_id = ann["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # Get category info
    categories = {cat["id"]: cat for cat in coco_data.get("categories", [])}
    skeleton_color = (255, 255, 255)  # White color for all skeleton lines
    print(f"Visualizing keypoints for {len(coco_data.get('images', []))} images...")

    # Process each image
    for image_info in coco_data.get("images", []):
        image_id = image_info["id"]
        if image_id not in annotations_by_image:
            continue

        # Load the image once
        image_path = output_dir / image_info["file_name"]
        if not image_path.exists():
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            continue

        # Draw all annotations for this image
        for ann in annotations_by_image[image_id]:
            category_id = ann["category_id"]
            category = categories.get(category_id, {})
            keypoints = ann["keypoints"]

            visible_keypoints = {}
            for i in range(0, len(keypoints), 3):
                x, y, visibility = keypoints[i], keypoints[i + 1], keypoints[i + 2]
                if visibility == config.get('keypoint_visible_value', 2):
                    # Assign a unique color per keypoint index
                    keypoint_index = i // 3
                    point_color = keypoint_colors[keypoint_index % len(keypoint_colors)]

                    # Draw keypoint circle with a black outline for better visibility
                    cv2.circle(img, (int(x), int(y)), 4, (0, 0, 0), -1)  # Black outline
                    cv2.circle(img, (int(x), int(y)), 3, point_color, -1)  # Colored point

                    # Store visible keypoints for drawing the skeleton
                    kp_idx_1_based = keypoint_index + 1
                    visible_keypoints[kp_idx_1_based] = (int(x), int(y))

            # Draw skeleton connections for the current tool
            skeleton = category.get("skeleton", [])
            for connection in skeleton:
                if len(connection) == 2:
                    p1_idx, p2_idx = connection
                    if p1_idx in visible_keypoints and p2_idx in visible_keypoints:
                        pt1 = visible_keypoints[p1_idx]
                        pt2 = visible_keypoints[p2_idx]
                        cv2.line(img, pt1, pt2, skeleton_color, 1)

        # Save the final image with all visualizations
        viz_filename = f"viz_{image_path.name}"
        viz_path = viz_dir / viz_filename
        cv2.imwrite(str(viz_path), img)

    print(f"Keypoint visualizations saved to: {viz_dir}")


def get_hdri_files(hdri_path: str) -> List[str]:
    """Get all HDR files from directory with better error handling."""
    if not hdri_path or not os.path.exists(hdri_path):
        return []

    try:
        # Try direct pattern first
        hdr_files = glob.glob(os.path.join(hdri_path, "*.hdr"))
        hdr_files.extend(glob.glob(os.path.join(hdri_path, "*.exr")))  # Support EXR files too

        # Try nested directories
        if not hdr_files:
            hdr_files = glob.glob(os.path.join(hdri_path, "*", "*.hdr"))
            hdr_files.extend(glob.glob(os.path.join(hdri_path, "*", "*.exr")))

        return sorted(hdr_files)
    except Exception as e:
        print(f"Warning: Error scanning HDRI directory: {e}")
        return []
