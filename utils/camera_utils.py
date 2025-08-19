import blenderproc as bproc

import json
import random
import numpy as np


def setup_camera(camera_params_file: str) -> bool:
    """Setup camera intrinsics from JSON file."""
    try:
        with open(camera_params_file, 'r') as f:
            camera_params = json.load(f)

        required_params = ["fx", "fy", "cx", "cy", "width", "height"]
        if not all(param in camera_params for param in required_params):
            print(f"Error: Camera params missing required fields: {required_params}")
            return False

        K = np.array([
            [camera_params["fx"], 0, camera_params["cx"]],
            [0, camera_params["fy"], camera_params["cy"]],
            [0, 0, 1]
        ])
        bproc.camera.set_intrinsics_from_K_matrix(K, camera_params["width"], camera_params["height"])
        return True
    except Exception as e:
        print(f"Error setting camera parameters: {e}")
        return False


def sample_camera_pose(config: dict) -> np.ndarray:
    """Sample camera pose with improved validation and variety."""
    max_attempts = config['camera_pose_max_attempts']
    workspace_center = [0, 0, 0]  # Could be configurable

    for attempt in range(max_attempts):
        # Camera position
        height = random.uniform(*config['camera_height_range'])
        offset_x = random.uniform(*config['camera_offset_range'])
        offset_y = random.uniform(*config['camera_offset_range'])

        camera_location = np.array([
            workspace_center[0] + offset_x,
            workspace_center[1] + offset_y,
            workspace_center[2] + height
        ])

        # Look at point within workspace
        look_at_x = random.uniform(-config['workspace_size'] / 2, config['workspace_size'] / 2)
        look_at_y = random.uniform(-config['workspace_size'] / 2, config['workspace_size'] / 2)
        look_at_point = np.array([
            workspace_center[0] + look_at_x,
            workspace_center[1] + look_at_y,
            workspace_center[2]
        ])

        # Calculate direction and validate angle
        direction = look_at_point - camera_location
        direction_norm = np.linalg.norm(direction)

        if direction_norm < config['camera_min_distance_threshold']:  # Too close
            continue

        direction = direction / direction_norm

        # Check if angle is reasonable (not too extreme)
        angle_from_vertical = np.arccos(-direction[2])  # Angle from straight down
        max_angle_radians = np.radians(config['camera_max_angle_from_vertical'])
        if angle_from_vertical > max_angle_radians:
            continue

        # Create rotation matrix
        rotation_variation = random.uniform(*config['camera_rotation_range'])
        rotation_matrix = bproc.camera.rotation_from_forward_vec(direction, inplane_rot=rotation_variation)

        return bproc.math.build_transformation_mat(camera_location, rotation_matrix)

    # Fallback: simple overhead shot
    camera_location = np.array([
        workspace_center[0],
        workspace_center[1],
        workspace_center[2] + config['camera_fallback_height']
    ])
    look_at_point = np.array(workspace_center)
    direction = (look_at_point - camera_location) / np.linalg.norm(look_at_point - camera_location)
    rotation_matrix = bproc.camera.rotation_from_forward_vec(direction)

    return bproc.math.build_transformation_mat(camera_location, rotation_matrix)
