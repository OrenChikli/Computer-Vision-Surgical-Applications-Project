import blenderproc as bproc
import json
import os
import glob
import random
import numpy as np
from typing import Dict, List

from .keypoint_extractor import KeypointExtractor
from utils.material_utils import setup_tool_material
from utils.workspace_utils import generate_random_positions


class ToolManager:
    """Manages surgical tool loading, annotation, and workspace setup."""

    def __init__(self, config: Dict):
        self.config = config
        self.tools_path = config['tools_path']
        self.annotations_path = config['annotations_path']
        self.skeleton_path = os.path.join(self.annotations_path, "tool_skeletons.json")

        # Data storage
        self.loaded_tools = {}
        self.annotations = {}
        self.tool_types = {}
        self.category_mapping = {}
        self.type_to_category_id = {}
        self.skeletons = {}

        # Initialize keypoint extractor
        self.keypoint_extractor = None

    def load_all_data(self) -> bool:
        """Load all required data with comprehensive error checking."""
        try:
            if not self._load_skeleton_data():
                print("Warning: No skeleton data loaded")

            if not self._load_annotations():
                print("Error: No annotations loaded")
                return False

            if not self._load_tools():
                print("Error: No tools loaded")
                return False

            self.keypoint_extractor = KeypointExtractor(self, self.config)
            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def _load_skeleton_data(self) -> bool:
        """Load skeleton data from JSON file and precompute skeleton connections."""
        if not os.path.exists(self.skeleton_path):
            return False

        try:
            with open(self.skeleton_path, 'r') as f:
                raw_skeletons = json.load(f)

            # Precompute skeleton connections with 1-based indices
            self.skeletons = {}
            for tool_type, skeleton_data in raw_skeletons.items():
                if "keypoints" not in skeleton_data:
                    self.skeletons[tool_type] = skeleton_data
                    continue

                keypoint_names = skeleton_data["keypoints"]
                keypoint_to_index = {name: i + 1 for i, name in enumerate(keypoint_names)}

                # Precompute skeleton connections with indices
                skeleton_connections = []
                if "skeleton" in skeleton_data:
                    for connection in skeleton_data["skeleton"]:
                        if len(connection) == 2:
                            p1, p2 = connection
                            if p1 in keypoint_to_index and p2 in keypoint_to_index:
                                skeleton_connections.append([keypoint_to_index[p1], keypoint_to_index[p2]])

                # Precompute flip_pairs with indices  
                flip_pairs_indices = []
                if "flip_pairs" in skeleton_data:
                    for pair in skeleton_data["flip_pairs"]:
                        if len(pair) == 2:
                            p1, p2 = pair
                            if p1 in keypoint_to_index and p2 in keypoint_to_index:
                                flip_pairs_indices.append([keypoint_to_index[p1], keypoint_to_index[p2]])

                # Store both original keypoints and precomputed skeleton
                self.skeletons[tool_type] = {
                    "keypoints": keypoint_names,
                    "skeleton": skeleton_data.get("skeleton", []),  # Original skeleton (names)
                    "skeleton_indices": skeleton_connections,  # Precomputed skeleton (indices)
                    "flip_pairs": skeleton_data.get("flip_pairs", []),  # Original flip pairs (names)
                    "flip_pairs_indices": flip_pairs_indices  # Precomputed flip pairs (indices)
                }

            print(f"Loaded skeleton data for {len(self.skeletons)} tool types")
            return True
        except Exception as e:
            print(f"Error loading skeleton data: {e}")
            return False

    def _load_annotations(self) -> bool:
        """Load manual keypoint annotations."""
        annotation_files = glob.glob(os.path.join(self.annotations_path, "*_keypoints.json"))

        if not annotation_files:
            print("No annotation files found")
            return False

        loaded_count = 0
        for annotation_file in annotation_files:
            try:
                with open(annotation_file, 'r') as f:
                    data = json.load(f)

                if 'tool_name' in data and 'tool_type' in data:
                    tool_name = data['tool_name']
                    self.annotations[tool_name] = data
                    self.tool_types[tool_name] = data['tool_type']
                    loaded_count += 1
                else:
                    print(f"Warning: Invalid annotation format in {annotation_file}")

            except Exception as e:
                print(f"Error loading {annotation_file}: {e}")

        print(f"Loaded {loaded_count} annotations")
        return loaded_count > 0

    def _load_tools(self) -> bool:
        """Load all tools with annotations into the scene."""
        loaded_count = 0

        self._create_category_mapping()
        for tool_name, annotation_data in self.annotations.items():
            obj_file = annotation_data.get('obj_file')

            if not obj_file:
                print(f"Warning: No obj_file specified for {tool_name}")
                continue

            if not os.path.exists(obj_file):
                print(f"Warning: .obj file not found for {tool_name}: {obj_file}")
                continue

            try:
                objs = bproc.loader.load_obj(obj_file)
                if not objs:
                    print(f"Warning: No objects loaded from {obj_file}")
                    continue

                obj = objs[0]
                obj.set_name(tool_name)
                category_id = self.category_mapping[tool_name]
                obj.set_cp("category_id", category_id)
                obj.set_cp("tool_type", annotation_data['tool_type'])
                obj.set_cp("tool_name", tool_name)

                self.loaded_tools[tool_name] = obj
                loaded_count += 1

            except Exception as e:
                print(f"Error loading {obj_file}: {e}")

        print(f"Loaded {loaded_count} tools into scene")
        return loaded_count > 0

    def _create_category_mapping(self) -> None:
        """Create category mapping based on tool types."""
        unique_tool_types = set(self.tool_types.values())

        self.type_to_category_id = {
            tool_type: i + 1
            for i, tool_type in enumerate(sorted(unique_tool_types))
        }

        self.category_mapping = {
            tool_name: self.type_to_category_id[tool_type]
            for tool_name, tool_type in self.tool_types.items()
        }

        print(f"Created {len(self.type_to_category_id)} tool type categories")

    def get_standard_keypoints_for_type(self, tool_type: str) -> List[str]:
        """Get standardized keypoint names for a tool type."""
        skeleton_data = self.skeletons.get(tool_type.lower(), {})
        return skeleton_data.get("keypoints", [])

    def map_tool_keypoints_to_standard(self, tool_name: str, tool_type: str) -> Dict[str, str]:
        """Map tool-specific keypoint names to standard keypoint names."""
        standard_keypoints = self.get_standard_keypoints_for_type(tool_type)
        tool_annotation = self.annotations.get(tool_name, {})
        tool_keypoints = tool_annotation.get('keypoint_names', [])

        # Simple exact name matching for now
        mapping = {std_kp: std_kp for std_kp in standard_keypoints if std_kp in tool_keypoints}
        return mapping

    def get_coco_categories(self) -> List[Dict]:
        """Generate COCO categories with keypoint and skeleton information."""
        categories = []

        for tool_type, category_id in sorted(self.type_to_category_id.items(), key=lambda x: x[1]):
            keypoint_names = self.get_standard_keypoints_for_type(tool_type)

            if not keypoint_names:
                continue

            # Use precomputed skeleton connections and flip pairs
            skeleton_data = self.skeletons.get(tool_type.lower(), {})
            skeleton_connections = skeleton_data.get("skeleton_indices", [])
            flip_pairs_indices = skeleton_data.get("flip_pairs_indices", [])

            category = {
                "id": category_id,
                "name": tool_type,
                "supercategory": "surgical_instrument",
                "keypoints": keypoint_names,
                "skeleton": skeleton_connections
            }
            
            # Add flip_pairs indices if available (COCO format uses indices, not names)
            if flip_pairs_indices:
                category["flip_pairs"] = flip_pairs_indices
                
            categories.append(category)

        return categories

    def generate_yolo_dataset_config(self, dataset_path: str) -> Dict:
        """Generate YOLO dataset configuration with proper flip_idx for all tools."""
        # Count total keypoints across all tool types to create unified schema
        unified_keypoints = []
        unified_flip_pairs = []
        tool_keypoint_ranges = {}
        
        current_kpt_idx = 0
        
        # Process each tool type to build unified keypoint schema
        for tool_type in sorted(self.type_to_category_id.keys()):
            skeleton_data = self.skeletons.get(tool_type.lower(), {})
            keypoints = skeleton_data.get("keypoints", [])
            flip_pairs = skeleton_data.get("flip_pairs", [])
            
            if not keypoints:
                continue
                
            # Track the range of keypoint indices for this tool type    
            start_idx = current_kpt_idx
            tool_keypoint_ranges[tool_type] = {
                'start': start_idx,
                'keypoints': keypoints,
                'original_flip_pairs': flip_pairs
            }
            
            # Add keypoints to unified schema
            unified_keypoints.extend(keypoints)
            
            # Convert flip_pairs to global indices
            keypoint_to_global_idx = {kp: start_idx + i for i, kp in enumerate(keypoints)}
            
            for pair in flip_pairs:
                if len(pair) == 2 and pair[0] in keypoint_to_global_idx and pair[1] in keypoint_to_global_idx:
                    global_pair = [keypoint_to_global_idx[pair[0]], keypoint_to_global_idx[pair[1]]]
                    unified_flip_pairs.append(global_pair)
            
            current_kpt_idx += len(keypoints)
        
        # Generate flip_idx array (maps keypoint index to its flipped version)
        flip_idx = list(range(len(unified_keypoints)))  # Default: no flipping
        
        for pair in unified_flip_pairs:
            if len(pair) == 2:
                idx1, idx2 = pair
                if 0 <= idx1 < len(flip_idx) and 0 <= idx2 < len(flip_idx):
                    flip_idx[idx1] = idx2
                    flip_idx[idx2] = idx1
        
        # Create YOLO dataset configuration
        config = {
            'path': dataset_path,
            'train': 'images/train',
            'val': 'images/val', 
            'test': 'images/test',
            'nc': len(self.type_to_category_id),
            'names': {cat_id: tool_type for tool_type, cat_id in self.type_to_category_id.items()},
            'kpt_shape': [len(unified_keypoints), 3],  # [num_keypoints, 3] (x,y,visibility)
            'flip_idx': flip_idx
        }
        
        return config

    def setup_surgical_workspace(self):
        """Setup all tools in a surgical workspace with improved positioning algorithm."""
        if not len(self.loaded_tools):
            return

        print(f"Setting up surgical workspace with {len(self.loaded_tools)} tools")

        # Generate positions with collision avoidance
        positions = generate_random_positions(
            len(self.loaded_tools),
            self.config['workspace_size'],
            self.config['min_tool_distance'],
            self.config['max_placement_attempts']
        )

        workspace_center = [0, 0, 0]  # Could be configurable

        for i, (tool_name, obj) in enumerate(self.loaded_tools.items()):
            # Set position
            pos = positions[i] + np.array(workspace_center)
            obj.set_location(pos.tolist())

            # Set varied rotations
            rotation_x = random.uniform(*self.config['tool_rotation_range'])
            rotation_y = random.uniform(*self.config['tool_rotation_range'])
            rotation_z = random.uniform(0, 2 * np.pi)
            obj.set_rotation_euler([rotation_x, rotation_y, rotation_z])

            # Set scale
            scale_factor = random.uniform(*self.config['tool_scale_range'])
            obj.set_scale([scale_factor] * 3)

            # Setup materials
            setup_tool_material(obj, self.config)

            print(f"  {tool_name}: pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
                  f"rot=({rotation_x:.2f}, {rotation_y:.2f}, {rotation_z:.2f}), scale={scale_factor:.2f}")

    def create_occlusion_blobs(self):
        """Create random occlusion blobs at specified keypoints."""
        if not hasattr(self, 'occlusion_objects'):
            self.occlusion_objects = []

        # Clear existing occlusion objects
        for obj in self.occlusion_objects:
            obj.delete()
        self.occlusion_objects = []

        for tool_name, tool_obj in self.loaded_tools.items():
            tool_type = self.tool_types.get(tool_name)
            if not tool_type or random.random() > self.config['occlusion_prob']:
                continue

            # Get keypoints for this tool type
            occlusion_keypoints = self.config['occlusion_keypoints'].get(tool_type.lower(), [])
            if not occlusion_keypoints:
                continue

            annotation_data = self.annotations.get(tool_name)
            if not annotation_data:
                continue

            keypoints_3d = annotation_data['keypoints']
            local_to_world = np.array(tool_obj.get_local2world_mat())

            # Randomly select keypoints to occlude
            num_blobs = random.randint(*self.config['occlusion_blob_count_range'])
            selected_keypoints = random.sample(
                occlusion_keypoints,
                min(num_blobs, len(occlusion_keypoints))
            )

            for kp_name in selected_keypoints:
                if kp_name not in keypoints_3d:
                    continue

                # Get keypoint world position
                kp_coords = keypoints_3d[kp_name]
                local_point = np.array([kp_coords['x'], kp_coords['y'], kp_coords['z'], 1.0])
                world_point = local_to_world @ local_point

                # Create blob
                blob_radius = random.uniform(*self.config['occlusion_blob_size_range'])
                blob = self._create_blob(world_point[:3], blob_radius)
                self.occlusion_objects.append(blob)

                print(f"  Created occlusion blob at {kp_name} for {tool_name}")

    def _create_blob(self, position, radius):
        """Create a single occlusion blob (sphere) simulating medical gloves."""
        # Create sphere using BlenderProc
        sphere_obj = bproc.object.create_primitive('SPHERE', radius=radius, location=position)

        # Add some randomness to position only
        offset = np.random.uniform(-radius * 0.3, radius * 0.3, 3)
        sphere_obj.set_location(position + offset)

        # Create medical glove material
        try:
            material = bproc.material.create(f"MedicalGlove_{sphere_obj.get_name()}")
            glove_color = random.choice(self.config['occlusion_glove_colors']) + [1.0]

            material.set_principled_shader_value("Base Color", glove_color)
            material.set_principled_shader_value("Roughness", random.uniform(*self.config['glove_roughness_range']))
            material.set_principled_shader_value("Metallic", 0.0)
            material.set_principled_shader_value("IOR", 1.4)

            sphere_obj.add_material(material)

        except Exception as e:
            print(f"Warning: Failed to setup material for medical glove: {e}")

        return sphere_obj
