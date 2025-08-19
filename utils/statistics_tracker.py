"""
Statistics tracking module for synthetic dataset generation.
Collects comprehensive statistics about generated images and dataset properties.
"""

import json
import time
import numpy as np
import bpy
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


class StatisticsTracker:
    """Tracks comprehensive statistics during synthetic data generation."""
    
    def __init__(self, config: Dict, output_dir: Path, enabled: bool = True):
        """
        Initialize statistics tracker.
        
        Args:
            config: Configuration dictionary
            output_dir: Output directory for saving statistics
            enabled: Whether statistics collection is enabled
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        
        # Statistics storage
        self.per_image_stats = []
        self.generation_start_time = time.time()
        
        # Initialize summary counters
        self.summary_stats = {
            'total_images': 0,
            'total_objects': 0,
            'total_tool_instances': 0,  # Total instances across all images
            'tool_type_counts': {},
            'tool_type_instances': {},  # Count instances per tool type
            'obj_file_usage': {},
            'occlusion_stats': {'fully_visible': 0, 'partially_occluded': 0},
            'scale_factors': [],
            'lighting_settings': [],
            'background_images': [],
            'camera_distances': [],
            'object_sizes': [],
            'tool_sizes_by_type': {},  # Track sizes per tool type for averaging
            'workspace_setups': 0
        }
        
        print(f"Statistics tracking {'enabled' if enabled else 'disabled'}")
    
    def collect_workspace_statistics(self, tool_manager, workspace_idx: int, 
                                   hdri_file: Optional[str] = None) -> Dict:
        """
        Collect statistics about the current workspace setup.
        
        Args:
            tool_manager: ToolManager instance with loaded tools
            workspace_idx: Current workspace index
            hdri_file: HDRI file being used for lighting
            
        Returns:
            Dictionary containing workspace statistics
        """
        if not self.enabled:
            return {}
        
        workspace_stats = {
            'workspace_idx': workspace_idx,
            'num_objects': len(tool_manager.loaded_tools),
            'tool_types_present': [],
            'obj_files_used': [],
            'object_properties': [],
            'lighting_setup': {},
            'background_image': hdri_file,
            'occlusion_objects_count': 0
        }
        
        # Collect tool information
        for tool_name, tool_obj in tool_manager.loaded_tools.items():
            tool_type = tool_manager.tool_types.get(tool_name, 'unknown')
            workspace_stats['tool_types_present'].append(tool_type)
            workspace_stats['obj_files_used'].append(tool_name)
            
            # Get object properties
            location = tool_obj.get_location()
            rotation = tool_obj.get_rotation_euler()
            scale = tool_obj.get_scale()
            
            obj_props = {
                'name': tool_name,
                'type': tool_type,
                'location': location,
                'rotation': rotation,
                'scale': scale[0],  # Assuming uniform scaling
                'scale_factor': scale[0]
            }
            workspace_stats['object_properties'].append(obj_props)
        
        # Collect lighting information
        workspace_stats['lighting_setup'] = self._get_lighting_stats()
        
        # Count occlusion objects if they exist
        if hasattr(tool_manager, 'occlusion_objects'):
            workspace_stats['occlusion_objects_count'] = len(tool_manager.occlusion_objects)
        
        return workspace_stats
    
    def collect_frame_statistics(self, frame_idx: int, workspace_stats: Dict,
                               data: Dict, tool_manager) -> None:
        """
        Collect statistics for a single rendered frame by analyzing what's actually visible.
        
        Args:
            frame_idx: Frame index  
            workspace_stats: Statistics from workspace setup
            data: Rendered data from BlenderProc
            tool_manager: ToolManager instance
        """
        if not self.enabled:
            return
        
        # Get camera information
        camera = bpy.context.scene.camera
        camera_location = camera.location
        camera_rotation = camera.rotation_euler
        
        # Analyze what's actually visible in this frame
        visible_objects = self._analyze_visible_objects(data, frame_idx, workspace_stats, tool_manager)
        
        frame_stats = {
            'frame_idx': frame_idx,
            'workspace_idx': workspace_stats.get('workspace_idx', 0),
            'timestamp': time.time(),
            'num_objects': len(visible_objects),  # Only count visible objects
            'tool_types_present': [obj['type'] for obj in visible_objects],  # Only visible types
            'obj_files_used': [obj['name'] for obj in visible_objects],  # Only visible objects
            'background_image': workspace_stats.get('background_image'),
            'lighting_setup': workspace_stats.get('lighting_setup', {}),
            'camera_info': {
                'location': list(camera_location),
                'rotation': list(camera_rotation),
                'distances_to_objects': [obj['distance_from_camera'] for obj in visible_objects]
            },
            'object_details': visible_objects,
            'occlusion_info': {
                'occlusion_objects_count': workspace_stats.get('occlusion_objects_count', 0),
                'objects_with_occlusion': [
                    {'name': obj['name'], 'occlusion_percentage': obj['occlusion_percentage']} 
                    for obj in visible_objects if obj['occlusion_percentage'] > 0
                ]
            }
        }
        
        self.per_image_stats.append(frame_stats)
        self._update_summary_stats(frame_stats)
    
    def _get_lighting_stats(self) -> Dict:
        """Extract lighting statistics from current scene."""
        lighting_stats = {
            'light_objects': [],
            'world_background': None
        }
        
        # Get light objects
        for obj in bpy.data.objects:
            if obj.type == 'LIGHT':
                lighting_stats['light_objects'].append({
                    'name': obj.name,
                    'type': obj.data.type,
                    'energy': obj.data.energy,
                    'location': list(obj.location)
                })
        
        # Get world background
        if bpy.context.scene.world and bpy.context.scene.world.node_tree:
            world_nodes = bpy.context.scene.world.node_tree.nodes
            for node in world_nodes:
                if node.type == 'TEX_ENVIRONMENT' and node.image:
                    lighting_stats['world_background'] = node.image.name
                    break
        
        return lighting_stats
    
    def _analyze_visible_objects(self, data: Dict, frame_idx: int, 
                               workspace_stats: Dict, tool_manager) -> List[Dict]:
        """
        Analyze rendered data to determine which objects are actually visible in the frame.
        
        Args:
            data: Rendered data from BlenderProc
            frame_idx: Frame index (relative to current batch)  
            workspace_stats: Workspace statistics containing all placed objects
            tool_manager: ToolManager instance
            
        Returns:
            List of visible object statistics
        """
        visible_objects = []
        
        try:
            batch_frame_idx = frame_idx % len(data.get("colors", []))
            
            if (batch_frame_idx >= len(data.get("instance_segmaps", [])) or
                batch_frame_idx >= len(data.get("instance_attribute_maps", []))):
                return visible_objects
                
            instance_map = data["instance_segmaps"][batch_frame_idx]
            instance_attrs = data["instance_attribute_maps"][batch_frame_idx]
            
            if instance_map is None or instance_attrs is None:
                return visible_objects
            
            # Get unique visible instance IDs (excluding background = 0)
            unique_instances = np.unique(instance_map)
            visible_instance_ids = [int(inst_id) for inst_id in unique_instances if inst_id > 0]
            
            # Get camera position for distance calculations
            camera = bpy.context.scene.camera
            camera_pos = np.array(camera.location)
            
            # Map visible instances to tool objects
            workspace_objects = workspace_stats.get('object_properties', [])
            
            # For each visible instance, try to match it with workspace objects
            for inst_id in visible_instance_ids:
                # Get pixels for this instance
                mask = instance_map == inst_id
                if not np.any(mask):
                    continue
                
                # Calculate bounding box
                y_coords, x_coords = np.where(mask)
                bbox = [
                    int(np.min(x_coords)),  # x_min
                    int(np.min(y_coords)),  # y_min  
                    int(np.max(x_coords) - np.min(x_coords)),  # width
                    int(np.max(y_coords) - np.min(y_coords))   # height
                ]
                bbox_area = bbox[2] * bbox[3]
                
                # Try to match instance with workspace object
                # Since we can't easily map BlenderProc instance IDs to object names,
                # we'll use a simplified approach based on the instance attribute data
                matched_object = self._match_instance_to_object(
                    inst_id, instance_attrs, workspace_objects, tool_manager
                )
                
                if matched_object:
                    obj_pos = np.array(matched_object['location'])
                    distance = np.linalg.norm(camera_pos - obj_pos)
                    
                    # Calculate occlusion (simplified - based on bbox size vs expected size)
                    expected_area = 10000  # Rough estimate, could be improved
                    occlusion_pct = max(0, min(50, (expected_area - bbox_area) / expected_area * 100))
                    
                    visible_obj = {
                        'name': matched_object['name'],
                        'type': matched_object['type'], 
                        'location': matched_object['location'],
                        'rotation': matched_object['rotation'],
                        'scale_factor': matched_object['scale_factor'],
                        'distance_from_camera': distance,
                        'bbox_pixels': bbox,
                        'bbox_area': bbox_area,
                        'visibility': 'visible',
                        'occlusion_percentage': occlusion_pct,
                        'pixel_count': int(np.sum(mask))
                    }
                    visible_objects.append(visible_obj)
            
        except Exception as e:
            print(f"Warning: Could not analyze visible objects for frame {frame_idx}: {e}")
            # Fallback: return empty list so we get accurate "0 objects visible" stats
            
        return visible_objects
    
    def _match_instance_to_object(self, instance_id: int, instance_attrs: np.ndarray,
                                 workspace_objects: List[Dict], tool_manager) -> Optional[Dict]:
        """
        Match a visible instance ID to a workspace object.
        This is a simplified matching - in practice you might need more sophisticated logic.
        """
        try:
            # For now, we'll match instances to objects sequentially
            # This is not perfect but gives us reasonable results
            if instance_id <= len(workspace_objects):
                return workspace_objects[instance_id - 1]  # -1 because instance IDs start at 1
        except:
            pass
        return None

    
    def _update_summary_stats(self, frame_stats: Dict) -> None:
        """Update running summary statistics with new frame data."""
        self.summary_stats['total_images'] += 1
        self.summary_stats['total_objects'] += frame_stats['num_objects']
        self.summary_stats['total_tool_instances'] += frame_stats['num_objects']
        
        # Update tool type counts - count unique tool types per image (not instances)
        unique_tool_types = set(frame_stats['tool_types_present'])
        for tool_type in unique_tool_types:
            self.summary_stats['tool_type_counts'][tool_type] = (
                self.summary_stats['tool_type_counts'].get(tool_type, 0) + 1
            )
        
        # Update tool type instances - count all instances per tool type
        for tool_type in frame_stats['tool_types_present']:
            self.summary_stats['tool_type_instances'][tool_type] = (
                self.summary_stats['tool_type_instances'].get(tool_type, 0) + 1
            )
        
        # Update obj file usage - count unique obj files per image (not instances)
        unique_obj_files = set(frame_stats['obj_files_used'])
        for obj_file in unique_obj_files:
            self.summary_stats['obj_file_usage'][obj_file] = (
                self.summary_stats['obj_file_usage'].get(obj_file, 0) + 1
            )
        
        # Update scale factors and tool sizes by type
        for obj_detail in frame_stats['object_details']:
            self.summary_stats['scale_factors'].append(obj_detail['scale_factor'])
            
            # Track sizes by tool type
            tool_type = obj_detail['type']
            if obj_detail['bbox_area'] > 0:
                if tool_type not in self.summary_stats['tool_sizes_by_type']:
                    self.summary_stats['tool_sizes_by_type'][tool_type] = []
                self.summary_stats['tool_sizes_by_type'][tool_type].append(obj_detail['bbox_area'])
        
        # Update camera distances
        if frame_stats['camera_info']['distances_to_objects']:
            self.summary_stats['camera_distances'].extend(
                frame_stats['camera_info']['distances_to_objects']
            )
        
        # Update object sizes
        for obj_detail in frame_stats['object_details']:
            if obj_detail['bbox_area'] > 0:
                self.summary_stats['object_sizes'].append(obj_detail['bbox_area'])
        
        # Update occlusion stats
        occluded_objects = len(frame_stats['occlusion_info']['objects_with_occlusion'])
        fully_visible = frame_stats['num_objects'] - occluded_objects
        self.summary_stats['occlusion_stats']['fully_visible'] += fully_visible
        self.summary_stats['occlusion_stats']['partially_occluded'] += occluded_objects
        
        # Track backgrounds
        if frame_stats['background_image'] and frame_stats['background_image'] not in self.summary_stats['background_images']:
            self.summary_stats['background_images'].append(frame_stats['background_image'])
    
    def save_statistics(self) -> Tuple[bool, bool]:
        """
        Save collected statistics to files.
        
        Returns:
            Tuple of (detailed_saved, summary_saved) booleans
        """
        if not self.enabled:
            print("Statistics tracking disabled - no files saved")
            return False, False
        
        detailed_saved = self._save_detailed_statistics()
        summary_saved = self._save_summary_statistics()
        
        return detailed_saved, summary_saved
    
    def _save_detailed_statistics(self) -> bool:
        """Save detailed per-image statistics to JSON file."""
        try:
            output_file = self.output_dir / "dataset_stats.json"
            
            detailed_stats = {
                'generation_info': {
                    'start_time': self.generation_start_time,
                    'end_time': time.time(),
                    'duration_seconds': time.time() - self.generation_start_time,
                    'config_used': self.config,
                    'total_images_generated': len(self.per_image_stats)
                },
                'per_image_statistics': self.per_image_stats
            }
            
            with open(output_file, 'w') as f:
                json.dump(detailed_stats, f, indent=2, default=str)
            
            print(f"✅ Detailed statistics saved to: {output_file}")
            return True
        
        except Exception as e:
            print(f"❌ Failed to save detailed statistics: {e}")
            return False
    
    def _save_summary_statistics(self) -> bool:
        """Save human-readable summary statistics to text file."""
        try:
            output_file = self.output_dir / "dataset_summary.txt"
            
            # Calculate derived statistics
            total_time = time.time() - self.generation_start_time
            avg_objects_per_image = (self.summary_stats['total_objects'] / 
                                   max(self.summary_stats['total_images'], 1))
            
            # Calculate scale factor statistics
            scale_stats = {}
            if self.summary_stats['scale_factors']:
                scale_stats = {
                    'min': min(self.summary_stats['scale_factors']),
                    'max': max(self.summary_stats['scale_factors']),
                    'mean': np.mean(self.summary_stats['scale_factors']),
                    'std': np.std(self.summary_stats['scale_factors'])
                }
            
            # Calculate distance statistics  
            distance_stats = {}
            if self.summary_stats['camera_distances']:
                distance_stats = {
                    'min': min(self.summary_stats['camera_distances']),
                    'max': max(self.summary_stats['camera_distances']),
                    'mean': np.mean(self.summary_stats['camera_distances']),
                    'std': np.std(self.summary_stats['camera_distances'])
                }
            
            # Calculate object size statistics
            size_stats = {}
            if self.summary_stats['object_sizes']:
                size_stats = {
                    'min': min(self.summary_stats['object_sizes']),
                    'max': max(self.summary_stats['object_sizes']),
                    'mean': np.mean(self.summary_stats['object_sizes']),
                    'std': np.std(self.summary_stats['object_sizes'])
                }
            
            # Create summary report
            summary_text = f"""# Synthetic Dataset Generation Summary

## Dataset Overview
- Total Images Generated: {self.summary_stats['total_images']}
- Total Object Instances Visible: {self.summary_stats['total_objects']}
- Total Tool Instances: {self.summary_stats['total_tool_instances']}
- Average Objects per Image: {avg_objects_per_image:.2f}
- Generation Time: {total_time/60:.1f} minutes
- Images per Minute: {self.summary_stats['total_images']/(total_time/60):.1f}

## Tool Type Distribution
(Number of images where each tool type appears)
"""
            
            # Add tool type distribution
            for tool_type, count in sorted(self.summary_stats['tool_type_counts'].items()):
                percentage = (count / self.summary_stats['total_images']) * 100
                instances = self.summary_stats['tool_type_instances'].get(tool_type, 0)
                summary_text += f"- {tool_type}: {count} images ({percentage:.1f}%) - {instances} total instances\n"
            
            summary_text += f"""
## Object File Usage
(Top 10 most frequently visible object files)
"""
            # Add most used obj files
            sorted_obj_files = sorted(self.summary_stats['obj_file_usage'].items(), 
                                    key=lambda x: x[1], reverse=True)[:10]
            for obj_file, count in sorted_obj_files:
                percentage = (count / self.summary_stats['total_images']) * 100
                summary_text += f"- {obj_file}: {count} images ({percentage:.1f}%)\n"
            
            # Add scale factor statistics
            if scale_stats:
                summary_text += f"""
## Scale Factor Statistics
- Range: {scale_stats['min']:.3f} to {scale_stats['max']:.3f}
- Mean: {scale_stats['mean']:.3f} ± {scale_stats['std']:.3f}
"""
            
            # Add distance statistics
            if distance_stats:
                summary_text += f"""
## Camera Distance Statistics (Blender units)
- Range: {distance_stats['min']:.2f} to {distance_stats['max']:.2f}
- Mean: {distance_stats['mean']:.2f} ± {distance_stats['std']:.2f}
"""
            
            # Add object size statistics
            if size_stats:
                summary_text += f"""
## Object Size Statistics (pixels²)
- Range: {size_stats['min']:.0f} to {size_stats['max']:.0f}
- Mean: {size_stats['mean']:.0f} ± {size_stats['std']:.0f}
"""

            # Add average tool sizes by type
            if self.summary_stats['tool_sizes_by_type']:
                summary_text += f"""
## Average Tool Sizes by Type (pixels²)
"""
                for tool_type, sizes in sorted(self.summary_stats['tool_sizes_by_type'].items()):
                    if sizes:
                        avg_size = np.mean(sizes)
                        std_size = np.std(sizes)
                        count = len(sizes)
                        summary_text += f"- {tool_type}: {avg_size:.0f} ± {std_size:.0f} pixels² ({count} instances)\n"
            
            # Add occlusion statistics
            total_object_instances = (self.summary_stats['occlusion_stats']['fully_visible'] + 
                                    self.summary_stats['occlusion_stats']['partially_occluded'])
            if total_object_instances > 0:
                occlusion_rate = (self.summary_stats['occlusion_stats']['partially_occluded'] / 
                                total_object_instances) * 100
                summary_text += f"""
## Occlusion Statistics
- Fully Visible Objects: {self.summary_stats['occlusion_stats']['fully_visible']}
- Partially Occluded Objects: {self.summary_stats['occlusion_stats']['partially_occluded']}
- Occlusion Rate: {occlusion_rate:.1f}%
"""
            
            # Add background variety
            summary_text += f"""
## Scene Variety
- Unique Background Images: {len(self.summary_stats['background_images'])}
- Workspace Configurations: {self.summary_stats.get('workspace_setups', 'N/A')}
"""
            
            # Add configuration used
            summary_text += f"""
## Generation Configuration
- Workspace Size: {self.config.get('workspace_size', 'N/A')}m
- Motion Blur Probability: {self.config.get('motion_blur_prob', 'N/A')}
- Occlusion Probability: {self.config.get('occlusion_prob', 'N/A')}
- Render Samples: {self.config.get('render_samples', 'N/A')}
- Resolution: {self.config.get('render_width', 'N/A')}x{self.config.get('render_height', 'N/A')}
- Seed: {self.config.get('seed', 'N/A')}

Generated on {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}
"""
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            
            print(f"✅ Summary statistics saved to: {output_file}")
            return True
        
        except Exception as e:
            print(f"❌ Failed to save summary statistics: {e}")
            return False