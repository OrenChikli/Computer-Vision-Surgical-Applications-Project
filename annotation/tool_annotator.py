import blenderproc as bproc
import bpy
import json
import os
import sys
import time
from pathlib import Path


class BatchAnnotator:
    """
    BlenderProc batch annotation system designed for debug mode
    """

    def __init__(self, tools_root_dir, output_dir):
        self.tools_root_dir = tools_root_dir
        self.output_dir = output_dir
        self.current_tool_index = 0
        self.tools_queue = []
        self.progress_file = os.path.join(output_dir, "annotation_progress.json")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Scan for tools
        self.scan_tools_directory()

        # Load previous progress if exists
        self.load_progress()

    def scan_tools_directory(self):
        """
        Scan the tools directory structure and build queue
        """
        print(f"🔍 Scanning tools directory: {self.tools_root_dir}")

        tool_dirs = {}

        # Use the same logic as your synthetic generation code
        for item in os.listdir(self.tools_root_dir):
            tool_path = os.path.join(self.tools_root_dir, item)
            if os.path.isdir(tool_path):
                import glob
                obj_files = glob.glob(os.path.join(tool_path, "*.obj"))
                if obj_files:
                    tool_dirs[item] = sorted(obj_files)

        # Build queue from discovered tools
        self.tools_queue = []
        for tool_type, obj_files in tool_dirs.items():
            for obj_file in obj_files:
                tool_info = {
                    'tool_type': tool_type,
                    'obj_file': obj_file,
                    'tool_name': Path(obj_file).stem,
                    'completed': False
                }
                self.tools_queue.append(tool_info)

        print(f"📋 Found {len(self.tools_queue)} tools across {len(tool_dirs)} tool types:")
        for tool_type, obj_files in tool_dirs.items():
            print(f"  • {tool_type}: {len(obj_files)} variations")

        return tool_dirs

    def load_progress(self):
        """
        Load previous annotation progress
        """
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)

                self.current_tool_index = progress_data.get('current_tool_index', 0)
                completed_tools = set(progress_data.get('completed_tools', []))

                # Mark completed tools
                for tool in self.tools_queue:
                    if tool['tool_name'] in completed_tools:
                        tool['completed'] = True

                # Find next uncompleted tool
                for i, tool in enumerate(self.tools_queue):
                    if not tool['completed']:
                        self.current_tool_index = i
                        break

                print(f"📁 Loaded progress: {len(completed_tools)} tools already completed")

            except Exception as e:
                print(f"⚠️ Could not load progress: {e}")
                self.current_tool_index = 0

    def save_progress(self):
        """
        Save current annotation progress
        """
        completed_tools = [tool['tool_name'] for tool in self.tools_queue if tool['completed']]

        progress_data = {
            'current_tool_index': self.current_tool_index,
            'completed_tools': completed_tools,
            'total_tools': len(self.tools_queue),
            'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)

    def load_current_tool(self):
        """
        Load the current tool for annotation
        """
        if self.current_tool_index >= len(self.tools_queue):
            print("🎉 ALL TOOLS COMPLETED! 🎉")
            return False

        current_tool = self.tools_queue[self.current_tool_index]

        # Skip if already completed
        if current_tool['completed']:
            self.current_tool_index += 1
            return self.load_current_tool()

        status = self.get_current_status()

        print(f"\n{'=' * 60}")
        print(f"📋 Tool {status['completed'] + 1}/{status['total']} ({status['progress_percent']:.1f}%)")
        print(f"🔧 Tool Type: {current_tool['tool_type']}")
        print(f"📄 Tool Name: {current_tool['tool_name']}")
        print(f"📁 File: {os.path.basename(current_tool['obj_file'])}")
        print(f"{'=' * 60}")

        # Load the tool using BlenderProc method
        if not self.load_tool_blenderproc(current_tool):
            print(f"❌ Failed to load tool, skipping...")
            self.current_tool_index += 1
            return self.load_current_tool()

        # Try to auto-create keypoints from skeleton
        skeleton_created = self.auto_create_keypoints_from_skeleton(current_tool['tool_type'])

        if skeleton_created:
            print(f"\n✅ AUTO-CREATED KEYPOINTS from skeleton:")
            print(f"   • Keypoints created at origin (0,0,0)")
            print(f"   • Position them at correct anatomical locations")
            print(f"   • Use finish_tool() when done")
        else:
            print(f"\n📌 MANUAL ANNOTATION REQUIRED:")
            print(f"1. Add keypoints: Shift+A > Empty > Plain Axes")
            print(f"2. Position each empty at important landmarks")
            print(f"3. Rename each empty (F2) with descriptive names")

        print(f"\n💡 Suggested keypoints for {current_tool['tool_type']}:")
        self.suggest_keypoints(current_tool['tool_type'])
        print(f"\n⌨️  CONTROLS (use in Python console):")
        print(f"   • finish_tool() - Save current tool and move to next")
        print(f"   • skip_tool() - Skip current tool")
        print(f"   • next_tool() / previous_tool() - Navigate without saving")
        print(f"   • show_status() - Show progress")
        print(f"   • rescale_tool(2.0) - Make tool bigger if too small")
        print(f"   • list_references() - Show completed tools of same type")
        print(f"   • transfer_keypoints('reference_name') - Copy from reference")
        print(f"   • recreate_keypoints() - Recreate keypoints from skeleton")

        return True

    def load_tool_blenderproc(self, tool_info):
        """
        Load a tool using BlenderProc - same method as your synthetic generation code
        """
        try:
            # Clear scene first
            bproc.utility.reset_keyframes()

            # Clear all mesh objects
            for obj in bproc.object.get_all_mesh_objects():
                obj.delete()

            # Clear all lights
            for light_obj in bpy.context.scene.objects:
                if light_obj.type == 'LIGHT':
                    bpy.data.objects.remove(light_obj, do_unlink=True)

            # Load the instrument using BlenderProc - SAME AS YOUR WORKING CODE
            print(f"  Loading {tool_info['tool_type']}: {os.path.basename(tool_info['obj_file'])}")
            objs = bproc.loader.load_obj(tool_info['obj_file'])

            if not objs:
                print(f"❌ No objects loaded from {tool_info['obj_file']}")
                return False

            # Take the first object
            obj = objs[0]

            # Set name for the object
            obj.set_name(tool_info['tool_name'])

            # Set some properties like in your synthetic generation
            obj.set_cp("tool_type", tool_info['tool_type'])
            obj.set_cp("obj_file", os.path.basename(tool_info['obj_file']))

            # Scale the object to reasonable size for annotation - LARGER for better visibility
            scale_factor = 1.0  # Much larger than synthetic generation (0.2) for easier annotation
            obj.set_scale([scale_factor, scale_factor, scale_factor])

            # Set up basic lighting - simple version
            light = bproc.types.Light()
            light.set_type("SUN")
            light.set_location([2, 2, 5])
            light.set_energy(3)

            # Position object at origin for easy annotation
            obj.set_location([0, 0, 0])
            obj.set_rotation_euler([0, 0, 0])

            # Focus viewport on the object
            self.focus_on_object(obj)

            print(f"✅ Loaded: {tool_info['tool_name']}")
            return True

        except Exception as e:
            print(f"❌ Failed to load {tool_info['obj_file']}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def focus_on_object(self, obj):
        """
        Focus the 3D viewport on the loaded object and ensure good visibility
        """
        try:
            # Get the Blender object
            blender_obj = obj.blender_obj

            # Select the object
            bpy.context.view_layer.objects.active = blender_obj
            blender_obj.select_set(True)

            # Frame the object in all 3D viewports and ensure good view
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            # Set solid shading for better visibility
                            space.shading.type = 'SOLID'
                            space.shading.show_xray = False

                            # Frame the selected object
                            override = {'area': area, 'region': space.region_3d}
                            try:
                                with bpy.context.temp_override(**override):
                                    bpy.ops.view3d.view_selected()
                                    # Zoom out a bit for working space
                                    bpy.ops.view3d.zoom(mx=0, my=0, delta=-2, use_cursor_init=False)
                            except:
                                pass  # Ignore viewport errors

        except Exception as e:
            print(f"⚠️ Could not focus on object: {e}")

    def finish_current_tool(self):
        """
        Save annotations for current tool and move to next
        """
        if self.current_tool_index >= len(self.tools_queue):
            print("🎉 ALL TOOLS COMPLETED! 🎉")
            return False

        current_tool = self.tools_queue[self.current_tool_index]

        # Get all Empty objects (keypoints)
        empty_objects = [obj for obj in bpy.data.objects if obj.type == 'EMPTY']

        if not empty_objects:
            print("⚠️ No keypoints found! Add Empty objects before saving.")
            return False

        # Extract keypoint data
        keypoints = {}
        for empty in empty_objects:
            keypoints[empty.name] = {
                'x': float(empty.location.x),
                'y': float(empty.location.y),
                'z': float(empty.location.z)
            }

        # Create annotation data
        annotation_data = {
            'tool_name': current_tool['tool_name'],
            'tool_type': current_tool['tool_type'],
            'obj_file': current_tool['obj_file'],
            'keypoints': keypoints,
            'keypoint_names': list(keypoints.keys()),
            'num_keypoints': len(keypoints),
            'annotation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        }

        # Save JSON annotation
        json_output = os.path.join(self.output_dir, f"{current_tool['tool_name']}_keypoints.json")
        with open(json_output, 'w') as f:
            json.dump(annotation_data, f, indent=2)

        # Save Blender file
        blend_output = os.path.join(self.output_dir, f"{current_tool['tool_name']}_annotated.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend_output)

        # Mark as completed
        current_tool['completed'] = True

        print(f"✅ Saved annotations for {current_tool['tool_name']}")
        print(f"   • {len(keypoints)} keypoints: {list(keypoints.keys())}")
        print(f"   • JSON: {json_output}")
        print(f"   • Blend: {blend_output}")

        # Move to next tool
        self.current_tool_index += 1
        self.save_progress()

        # Load next tool
        return self.load_current_tool()

    def skip_current_tool(self):
        """
        Skip current tool and move to next
        """
        if self.current_tool_index >= len(self.tools_queue):
            print("🎉 ALL TOOLS COMPLETED! 🎉")
            return False

        current_tool = self.tools_queue[self.current_tool_index]
        print(f"⏭️ Skipping {current_tool['tool_name']}")

        # Move to next tool
        self.current_tool_index += 1
        self.save_progress()

        # Load next tool
        return self.load_current_tool()

    def get_current_status(self):
        """
        Get current annotation status
        """
        completed = sum(1 for tool in self.tools_queue if tool['completed'])
        total = len(self.tools_queue)

        if self.current_tool_index < total:
            current_tool = self.tools_queue[self.current_tool_index]
            return {
                'completed': completed,
                'total': total,
                'current_tool': current_tool,
                'progress_percent': (completed / total) * 100
            }
        else:
            return {
                'completed': completed,
                'total': total,
                'current_tool': None,
                'progress_percent': 100.0
            }

    def transfer_keypoints_from_reference(self, reference_tool_name, target_tool):
        """
        Transfer keypoints from a reference tool to the current target tool
        """
        try:
            # Load reference annotations
            reference_file = os.path.join(self.output_dir, f"{reference_tool_name}_keypoints.json")
            if not os.path.exists(reference_file):
                print(f"❌ No reference annotations found for {reference_tool_name}")
                return False

            with open(reference_file, 'r') as f:
                reference_data = json.load(f)

            reference_keypoints = reference_data['keypoints']

            print(f"📋 Transferring {len(reference_keypoints)} keypoints from {reference_tool_name}")

            # Clear existing empties
            empties_to_remove = [obj for obj in bpy.data.objects if obj.type == 'EMPTY']
            for empty in empties_to_remove:
                bpy.data.objects.remove(empty, do_unlink=True)

            # Create new empties at transferred positions
            created_keypoints = []
            for kp_name, kp_coords in reference_keypoints.items():
                # Create empty
                bpy.ops.object.empty_add(type='PLAIN_AXES', location=(kp_coords['x'], kp_coords['y'], kp_coords['z']))
                empty = bpy.context.active_object
                empty.name = kp_name
                empty.show_name = True  # Show keypoint name in viewport

                # Scale empty for visibility
                empty.empty_display_size = 0.1

                created_keypoints.append(kp_name)
                print(f"  ✅ Created keypoint: {kp_name}")

            print(f"🎯 Transferred keypoints: {', '.join(created_keypoints)}")
            print(f"💡 Review and adjust positions as needed, then use finish_tool()")

            return True

        except Exception as e:
            print(f"❌ Error transferring keypoints: {e}")
            import traceback
            traceback.print_exc()
            return False

    def show_status(self):
        """
        Show current annotation status
        """
        status = self.get_current_status()
        print(f"📊 Progress: {status['completed']}/{status['total']} ({status['progress_percent']:.1f}%)")
        if status['current_tool']:
            print(f"🔧 Current: {status['current_tool']['tool_name']} ({status['current_tool']['tool_type']})")
        else:
            print("🎉 All tools completed!")

    def list_reference_tools(self):
        """
        List available reference tools for the current tool type
        """
        if self.current_tool_index >= len(self.tools_queue):
            print("❌ No current tool")
            return []

        current_tool = self.tools_queue[self.current_tool_index]
        current_tool_type = current_tool['tool_type']

        # Find completed tools of the same type
        reference_tools = []
        for tool in self.tools_queue:
            if (tool['completed'] and
                    tool['tool_type'] == current_tool_type and
                    tool['tool_name'] != current_tool['tool_name']):
                reference_tools.append(tool['tool_name'])

        if reference_tools:
            print(f"📋 Available reference tools for {current_tool_type}:")
            for i, ref_tool in enumerate(reference_tools):
                print(f"  {i}: {ref_tool}")
        else:
            print(f"❌ No completed reference tools found for {current_tool_type}")

        return reference_tools

    def load_tool_for_editing(self, tool_name):
        """
        Load a specific tool for editing (completed or not)
        """
        try:
            # Find the tool in the queue
            target_tool = None
            target_index = None
            for i, tool in enumerate(self.tools_queue):
                if tool['tool_name'] == tool_name:
                    target_tool = tool
                    target_index = i
                    break

            if target_tool is None:
                print(f"❌ Tool '{tool_name}' not found")
                return False

            # Load the tool
            if not self.load_tool_blenderproc(target_tool):
                print(f"❌ Failed to load tool {tool_name}")
                return False

            # If tool has existing annotations, load them
            annotation_file = os.path.join(self.output_dir, f"{tool_name}_keypoints.json")
            if os.path.exists(annotation_file):
                try:
                    with open(annotation_file, 'r') as f:
                        annotation_data = json.load(f)

                    # Clear existing empties
                    empties_to_remove = [obj for obj in bpy.data.objects if obj.type == 'EMPTY']
                    for empty in empties_to_remove:
                        bpy.data.objects.remove(empty, do_unlink=True)

                    # Recreate keypoints from saved annotations
                    keypoints = annotation_data['keypoints']
                    for kp_name, kp_coords in keypoints.items():
                        bpy.ops.object.empty_add(type='PLAIN_AXES',
                                                 location=(kp_coords['x'], kp_coords['y'], kp_coords['z']))
                        empty = bpy.context.active_object
                        empty.name = kp_name
                        empty.show_name = True
                        empty.empty_display_size = 0.1

                    print(f"✅ Loaded existing annotations: {list(keypoints.keys())}")

                except Exception as e:
                    print(f"⚠️ Could not load existing annotations: {e}")
            else:
                print(f"💡 No existing annotations found for {tool_name}")

            # Set as current tool for editing
            self.current_tool_index = target_index

            print(f"🔧 Loaded {tool_name} for editing")
            print(f"📋 Tool Type: {target_tool['tool_type']}")
            print(f"📁 File: {os.path.basename(target_tool['obj_file'])}")
            print(f"\n💡 Edit keypoints, then use finish_tool() to save changes")

            return True

        except Exception as e:
            print(f"❌ Error loading tool for editing: {e}")
            import traceback
            traceback.print_exc()
            return False

    def list_completed_tools(self):
        """
        List all completed tools
        """
        completed_tools = [tool for tool in self.tools_queue if tool['completed']]

        if not completed_tools:
            print("❌ No completed tools found")
            return []

        print(f"📋 Completed tools ({len(completed_tools)}):")
        for tool in completed_tools:
            print(f"  • {tool['tool_name']} ({tool['tool_type']})")

        return [tool['tool_name'] for tool in completed_tools]

    def list_all_tools(self):
        """
        List all tools with their status
        """
        print(f"📋 All tools ({len(self.tools_queue)}):")

        by_type = {}
        for tool in self.tools_queue:
            tool_type = tool['tool_type']
            if tool_type not in by_type:
                by_type[tool_type] = []
            by_type[tool_type].append(tool)

        for tool_type, tools in by_type.items():
            print(f"\n  {tool_type}:")
            for tool in tools:
                status = "✅ Completed" if tool['completed'] else "⏳ Pending"
                print(f"    • {tool['tool_name']} - {status}")

        return [tool['tool_name'] for tool in self.tools_queue]

    def unmark_completed(self, tool_name):
        """
        Unmark a tool as completed so it appears in the normal workflow again
        """
        for tool in self.tools_queue:
            if tool['tool_name'] == tool_name:
                if tool['completed']:
                    tool['completed'] = False
                    self.save_progress()
                    print(f"✅ Unmarked {tool_name} as completed - will appear in normal workflow")
                    return True
                else:
                    print(f"💡 {tool_name} was not marked as completed")
                    return True

        print(f"❌ Tool '{tool_name}' not found")
        return False

    def next_tool_without_saving(self):
        """
        Go to next tool without saving current one
        """
        if self.current_tool_index >= len(self.tools_queue) - 1:
            print("❌ Already at the last tool")
            return False

        self.current_tool_index += 1
        return self.load_current_tool_for_browsing()

    def previous_tool_without_saving(self):
        """
        Go to previous tool without saving current one
        """
        if self.current_tool_index <= 0:
            print("❌ Already at the first tool")
            return False

        self.current_tool_index -= 1
        return self.load_current_tool_for_browsing()

    def goto_tool_by_index(self, index):
        """
        Jump to specific tool by index (0-based)
        """
        if index < 0 or index >= len(self.tools_queue):
            print(f"❌ Invalid index. Valid range: 0-{len(self.tools_queue) - 1}")
            return False

        self.current_tool_index = index
        return self.load_current_tool_for_browsing()

    def goto_tool_by_name(self, tool_name):
        """
        Jump to specific tool by name
        """
        for i, tool in enumerate(self.tools_queue):
            if tool['tool_name'] == tool_name:
                self.current_tool_index = i
                return self.load_current_tool_for_browsing()

        print(f"❌ Tool '{tool_name}' not found")
        return False

    def load_current_tool_for_browsing(self):
        """
        Load current tool for browsing (doesn't change completion status)
        """
        if self.current_tool_index >= len(self.tools_queue):
            print("❌ No more tools")
            return False

        current_tool = self.tools_queue[self.current_tool_index]

        # Show info
        completed = sum(1 for tool in self.tools_queue if tool['completed'])
        total = len(self.tools_queue)

        print(f"\n{'=' * 60}")
        print(f"📋 Tool {self.current_tool_index + 1}/{total} | Completed: {completed}")
        print(f"🔧 Tool Type: {current_tool['tool_type']}")
        print(f"📄 Tool Name: {current_tool['tool_name']}")
        print(f"📁 File: {os.path.basename(current_tool['obj_file'])}")
        print(f"📊 Status: {'✅ Completed' if current_tool['completed'] else '⏳ Pending'}")
        print(f"{'=' * 60}")

        # Load the tool
        if not self.load_tool_blenderproc(current_tool):
            print(f"❌ Failed to load tool")
            return False

        # If tool has existing annotations, load them
        annotation_file = os.path.join(self.output_dir, f"{current_tool['tool_name']}_keypoints.json")
        if os.path.exists(annotation_file):
            try:
                with open(annotation_file, 'r') as f:
                    annotation_data = json.load(f)

                # Clear existing empties
                empties_to_remove = [obj for obj in bpy.data.objects if obj.type == 'EMPTY']
                for empty in empties_to_remove:
                    bpy.data.objects.remove(empty, do_unlink=True)

                # Recreate keypoints from saved annotations
                keypoints = annotation_data['keypoints']
                for kp_name, kp_coords in keypoints.items():
                    bpy.ops.object.empty_add(type='PLAIN_AXES',
                                             location=(kp_coords['x'], kp_coords['y'], kp_coords['z']))
                    empty = bpy.context.active_object
                    empty.name = kp_name
                    empty.show_name = True
                    empty.empty_display_size = 0.1

                print(f"💡 Loaded existing keypoints: {list(keypoints.keys())}")

            except Exception as e:
                print(f"⚠️ Could not load existing annotations: {e}")
        else:
            # Clear any existing empties for clean view
            empties_to_remove = [obj for obj in bpy.data.objects if obj.type == 'EMPTY']
            for empty in empties_to_remove:
                bpy.data.objects.remove(empty, do_unlink=True)
            print(f"💡 No existing annotations for this tool")

        print(f"\n🎮 Navigation:")
        print(f"   • next_tool() / previous_tool() - Navigate without saving")
        print(f"   • finish_tool() - Save and go to next in workflow")
        print(f"   • goto_tool(5) - Jump to tool by index")
        print(f"   • goto_tool_by_name('NH1') - Jump to tool by name")

        return True

    def suggest_keypoints(self, tool_type):
        """
        Suggest standard keypoints for different tool types
        """
        suggestions = {
            'needle_holder': ['jaw_left', 'jaw_right', 'joint', 'handle_end', 'ratchet'],
            'tweezers': ['tip_left', 'tip_right', 'joint', 'handle_end'],
            'forceps': ['tip_left', 'tip_right', 'joint', 'handle_end'],
            'grasper': ['tip_left', 'tip_right', 'joint', 'handle_end'],
            'scissors': ['blade_tip_left', 'blade_tip_right', 'pivot', 'handle_left_end', 'handle_right_end'],
            'scalpel': ['blade_tip', 'blade_base', 'handle_end'],
            'probe': ['tip', 'middle', 'handle_end']
        }

        tool_type_lower = tool_type.lower()
        for key, keypoints in suggestions.items():
            if key in tool_type_lower:
                print(f"   • {', '.join(keypoints)}")
                return

        print(f"   • tip, joint, handle_end (generic)")

    def auto_create_keypoints_from_skeleton(self, tool_type):
        """
        Automatically create keypoints based on the skeleton file
        """
        try:
            # Load skeleton file
            skeleton_file = os.path.join(self.output_dir, "tool_skeletons.json")
            if not os.path.exists(skeleton_file):
                print(f"⚠️  No skeleton file found at {skeleton_file}")
                return False

            with open(skeleton_file, 'r') as f:
                skeletons = json.load(f)

            # Check if tool type exists in skeleton
            if tool_type not in skeletons:
                print(f"⚠️  No skeleton found for tool type: {tool_type}")
                return False

            skeleton = skeletons[tool_type]
            keypoints = skeleton.get('keypoints', [])

            if not keypoints:
                print(f"⚠️  No keypoints defined for {tool_type}")
                return False

            # Clear existing keypoints
            empties_to_remove = [obj for obj in bpy.data.objects if obj.type == 'EMPTY']
            for empty in empties_to_remove:
                bpy.data.objects.remove(empty, do_unlink=True)

            # Create new keypoints at origin
            created_keypoints = []
            for kp_name in keypoints:
                # Create empty at origin
                bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
                empty = bpy.context.active_object
                empty.name = kp_name
                empty.show_name = True  # Show keypoint name in viewport
                empty.empty_display_size = 0.1  # Make visible

                created_keypoints.append(kp_name)

            print(f"   • Created {len(created_keypoints)} keypoints: {', '.join(created_keypoints)}")
            
            # If skeleton has connections, create visualization lines (optional)
            if 'skeleton' in skeleton:
                print(f"   • Skeleton connections: {len(skeleton['skeleton'])} lines defined")
                
            return True

        except Exception as e:
            print(f"❌ Error creating keypoints from skeleton: {e}")
            import traceback
            traceback.print_exc()
            return False

    def recreate_keypoints(self):
        """
        Recreate keypoints from skeleton for current tool
        """
        if self.current_tool_index >= len(self.tools_queue):
            print("❌ No current tool")
            return False

        current_tool = self.tools_queue[self.current_tool_index]
        return self.auto_create_keypoints_from_skeleton(current_tool['tool_type'])


# Global instance for easy access
annotator = None


def start_annotation():
    """
    Start the annotation process
    """
    global annotator

    if len(sys.argv) < 3:
        print("Usage: blenderproc debug debug_batch_annotation.py <tools_dir> <output_dir>")
        return

    tools_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Validate input directory
    if not os.path.exists(tools_dir):
        print(f"❌ Tools directory not found: {tools_dir}")
        return

    print("🚀 BlenderProc Debug Batch Annotation System")
    print("=" * 50)

    # Create batch annotator
    annotator = BatchAnnotator(tools_dir, output_dir)

    # Load first tool
    if annotator.load_current_tool():
        print("\n🎯 Ready for annotation!")
    else:
        print("❌ No tools to annotate")


def finish_tool():
    """
    Finish current tool and move to next
    """
    global annotator
    if annotator:
        return annotator.finish_current_tool()
    else:
        print("❌ No annotation session active. Run start_annotation() first.")
        return False


def skip_tool():
    """
    Skip current tool and move to next
    """
    global annotator
    if annotator:
        return annotator.skip_current_tool()
    else:
        print("❌ No annotation session active. Run start_annotation() first.")
        return False


def rescale_tool(scale_factor):
    """
    Rescale the current tool for better visibility
    """
    # Find the tool object (mesh objects that aren't empties)
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']

    if mesh_objects:
        for obj in mesh_objects:
            obj.scale = [scale_factor, scale_factor, scale_factor]
        print(f"🔍 Rescaled tool(s) to {scale_factor}")

        # Reframe the view
        bpy.ops.view3d.view_selected()
    else:
        print("❌ No tool objects found to rescale")


def transfer_keypoints(reference_tool_name):
    """
    Transfer keypoints from a reference tool to current tool
    """
    global annotator
    if annotator:
        return annotator.transfer_keypoints_from_reference(reference_tool_name, None)
    else:
        print("❌ No annotation session active. Run start_annotation() first.")
        return False


def list_references():
    """
    List available reference tools for current tool type
    """
    global annotator
    if annotator:
        return annotator.list_reference_tools()
    else:
        print("❌ No annotation session active. Run start_annotation() first.")
        return []


def load_tool_for_editing(tool_name):
    """
    Load a specific tool for editing
    """
    global annotator
    if annotator:
        return annotator.load_tool_for_editing(tool_name)
    else:
        print("❌ No annotation session active. Run start_annotation() first.")
        return False


def list_completed():
    """
    List all completed tools
    """
    global annotator
    if annotator:
        return annotator.list_completed_tools()
    else:
        print("❌ No annotation session active. Run start_annotation() first.")
        return []


def list_all_tools():
    """
    List all tools with their status
    """
    global annotator
    if annotator:
        return annotator.list_all_tools()
    else:
        print("❌ No annotation session active. Run start_annotation() first.")
        return []


def unmark_completed(tool_name):
    """
    Unmark a tool as completed
    """
    global annotator
    if annotator:
        return annotator.unmark_completed(tool_name)
    else:
        print("❌ No annotation session active. Run start_annotation() first.")
        return False


def next_tool():
    """
    Go to next tool without saving current
    """
    global annotator
    if annotator:
        return annotator.next_tool_without_saving()
    else:
        print("❌ No annotation session active. Run start_annotation() first.")
        return False


def previous_tool():
    """
    Go to previous tool without saving current
    """
    global annotator
    if annotator:
        return annotator.previous_tool_without_saving()
    else:
        print("❌ No annotation session active. Run start_annotation() first.")
        return False


def goto_tool(index):
    """
    Jump to specific tool by index (0-based)
    """
    global annotator
    if annotator:
        return annotator.goto_tool_by_index(index)
    else:
        print("❌ No annotation session active. Run start_annotation() first.")
        return False


def goto_tool_by_name(tool_name):
    """
    Jump to specific tool by name
    """
    global annotator
    if annotator:
        return annotator.goto_tool_by_name(tool_name)
    else:
        print("❌ No annotation session active. Run start_annotation() first.")
        return False


def show_status():
    """
    Show current annotation status
    """
    global annotator
    if annotator:
        annotator.show_status()
    else:
        print("❌ No annotation session active. Run start_annotation() first.")


def recreate_keypoints():
    """
    Recreate keypoints from skeleton for current tool
    """
    global annotator
    if annotator:
        return annotator.recreate_keypoints()
    else:
        print("❌ No annotation session active. Run start_annotation() first.")
        return False


def main():
    """
    Main function - automatically starts annotation
    """
    # Initialize BlenderProc
    bproc.init()

    # Start annotation session
    start_annotation()

    # Make functions available globally in Blender
    import builtins
    builtins.finish_tool = finish_tool
    builtins.skip_tool = skip_tool
    builtins.show_status = show_status
    builtins.rescale_tool = rescale_tool
    builtins.start_annotation = start_annotation
    builtins.transfer_keypoints = transfer_keypoints
    builtins.list_references = list_references
    builtins.load_tool_for_editing = load_tool_for_editing
    builtins.list_completed = list_completed
    builtins.list_all_tools = list_all_tools
    builtins.unmark_completed = unmark_completed
    builtins.next_tool = next_tool
    builtins.previous_tool = previous_tool
    builtins.goto_tool = goto_tool
    builtins.goto_tool_by_name = goto_tool_by_name
    builtins.recreate_keypoints = recreate_keypoints

    # Also register in Blender's driver namespace
    bpy.app.driver_namespace['finish_tool'] = finish_tool
    bpy.app.driver_namespace['skip_tool'] = skip_tool
    bpy.app.driver_namespace['show_status'] = show_status
    bpy.app.driver_namespace['rescale_tool'] = rescale_tool
    bpy.app.driver_namespace['start_annotation'] = start_annotation
    bpy.app.driver_namespace['transfer_keypoints'] = transfer_keypoints
    bpy.app.driver_namespace['list_references'] = list_references
    bpy.app.driver_namespace['load_tool_for_editing'] = load_tool_for_editing
    bpy.app.driver_namespace['list_completed'] = list_completed
    bpy.app.driver_namespace['list_all_tools'] = list_all_tools
    bpy.app.driver_namespace['unmark_completed'] = unmark_completed
    bpy.app.driver_namespace['next_tool'] = next_tool
    bpy.app.driver_namespace['previous_tool'] = previous_tool
    bpy.app.driver_namespace['goto_tool'] = goto_tool
    bpy.app.driver_namespace['goto_tool_by_name'] = goto_tool_by_name
    bpy.app.driver_namespace['recreate_keypoints'] = recreate_keypoints

    print("\n🎮 Available commands:")
    print("  • finish_tool() - Save current tool and load next")
    print("  • skip_tool() - Skip current tool")
    print("  • show_status() - Show progress")
    print("  • rescale_tool(2.0) - Make tool bigger (try 2.0, 3.0, etc.)")
    print("  • rescale_tool(0.5) - Make tool smaller")
    print("  • list_references() - Show reference tools for current type")
    print("  • transfer_keypoints('tool_name') - Copy keypoints from reference")
    print("  • recreate_keypoints() - Recreate keypoints from skeleton")
    print("\n🧭 Navigation (without saving):")
    print("  • next_tool() - Go to next tool")
    print("  • previous_tool() - Go to previous tool")
    print("  • goto_tool(5) - Jump to tool by index (0-based)")
    print("  • goto_tool_by_name('NH1') - Jump to tool by name")
    print("\n📝 Editing completed tools:")
    print("  • list_completed() - Show all completed tools")
    print("  • list_all_tools() - Show all tools with status")
    print("  • load_tool_for_editing('tool_name') - Load specific tool for editing")
    print("  • unmark_completed('tool_name') - Remove completed status")
    print("\n💡 If functions not found, try:")
    print("  • annotator.finish_current_tool() - Alternative finish")
    print("  • annotator.skip_current_tool() - Alternative skip")
    print("  • annotator.show_status() - Alternative status")


if __name__ == "__main__":
    main()