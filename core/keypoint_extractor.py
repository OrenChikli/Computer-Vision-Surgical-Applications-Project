import bpy
import bpy_extras
import numpy as np
from mathutils import Vector
from typing import Dict, List, Optional, Tuple


class KeypointExtractor:
    """Dedicated class for keypoint extraction logic."""

    def __init__(self, tool_manager, config: Dict):
        self.tool_manager = tool_manager
        self.config = config

    def world_to_camera_2d(self, world_point: np.ndarray) -> Tuple[float, float, bool]:
        """Convert 3D world point to 2D camera coordinates."""
        try:
            scene = bpy.context.scene
            camera = scene.camera
            point2d_normalized = bpy_extras.object_utils.world_to_camera_view(
                scene, camera, Vector(world_point[:3])
            )

            width, height = scene.render.resolution_x, scene.render.resolution_y
            x_pixel = point2d_normalized[0] * width
            y_pixel = (1 - point2d_normalized[1]) * height

            # Check if point is visible (within image bounds and in front of camera)
            is_visible = (point2d_normalized[2] > 0 and
                          0 <= x_pixel < width and
                          0 <= y_pixel < height)

            return float(x_pixel), float(y_pixel), is_visible

        except Exception as e:
            print(f"Warning: Error in coordinate conversion: {e}")
            return 0.0, 0.0, False

    def extract_tool_keypoints(self, obj) -> Optional[Dict]:
        """Extract keypoints for a single tool."""
        tool_name = obj.get_cp("tool_name")
        tool_type = self.tool_manager.tool_types.get(tool_name)

        if tool_name not in self.tool_manager.annotations or not tool_type:
            return None

        annotation_data = self.tool_manager.annotations[tool_name]
        keypoints_3d = annotation_data['keypoints']

        # Get standard keypoints and mapping
        standard_keypoints = self.tool_manager.get_standard_keypoints_for_type(tool_type)
        keypoint_mapping = self.tool_manager.map_tool_keypoints_to_standard(tool_name, tool_type)

        # Transform keypoints to world coordinates
        local_to_world = np.array(obj.get_local2world_mat())

        keypoints_2d_flat = []
        visible_count = 0

        for std_kp_name in standard_keypoints:
            tool_kp_name = keypoint_mapping.get(std_kp_name)

            if tool_kp_name and tool_kp_name in keypoints_3d:
                kp_coords = keypoints_3d[tool_kp_name]
                local_point = np.array([kp_coords['x'], kp_coords['y'], kp_coords['z'], 1.0])
                world_point = local_to_world @ local_point

                x_pixel, y_pixel, is_visible = self.world_to_camera_2d(world_point)

                if is_visible:
                    keypoints_2d_flat.extend([x_pixel, y_pixel, self.config['keypoint_visible_value']])
                    visible_count += 1
                else:
                    keypoints_2d_flat.extend([0, 0, self.config['keypoint_not_visible_value']])
            else:
                keypoints_2d_flat.extend([0, 0, self.config['keypoint_not_available_value']])

        if visible_count > 0:
            return {
                "category_id": self.tool_manager.category_mapping[tool_name],
                "keypoints": keypoints_2d_flat,
                "num_keypoints": visible_count,
            }

        return None

    def extract_frame_keypoints(self) -> List[Dict]:
        """Extract keypoints for all visible tools in a frame."""
        annotations = []

        for obj in self.tool_manager.loaded_tools.values():
            keypoint_data = self.extract_tool_keypoints(obj)
            if keypoint_data:
                annotations.append(keypoint_data)

        return annotations
