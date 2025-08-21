import blenderproc as bproc
import bpy
import logging
import random

logger = logging.getLogger(__name__)

def setup_lighting(config: dict):
    """Setup realistic surgical lighting with improved parameters."""
    workspace_center = [0, 0, 0]  # Could be configurable
    
    # Clear existing lights
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)

    # Main surgical light (overhead area light)
    main_light = bproc.types.Light()
    main_light.set_type("AREA")
    main_light.set_location([
        workspace_center[0] + random.uniform(*config['main_light_offset_range']),
        workspace_center[1] + random.uniform(*config['main_light_offset_range']),
        workspace_center[2] + random.uniform(*config['main_light_height_range'])
    ])
    main_light.set_energy(random.uniform(*config['main_light_energy_range']))
    main_light.blender_obj.data.size = random.uniform(*config['main_light_size_range'])

    # Set random color for main light
    main_color = [
        random.uniform(*config['main_light_color']['r_range']),
        random.uniform(*config['main_light_color']['g_range']),
        random.uniform(*config['main_light_color']['b_range'])
    ]
    main_light.set_color(main_color)

    # Fill light
    fill_light = bproc.types.Light()
    fill_light.set_type("POINT")
    fill_light.set_location([
        workspace_center[0] + random.uniform(*config['fill_light_offset_range']),
        workspace_center[1] + random.uniform(*config['fill_light_offset_range']),
        workspace_center[2] + random.uniform(*config['fill_light_height_range'])
    ])
    fill_light.set_energy(random.uniform(*config['fill_light_energy_range']))

    # Set random color for fill light
    fill_color = [
        random.uniform(*config['fill_light_color']['r_range']),
        random.uniform(*config['fill_light_color']['g_range']),
        random.uniform(*config['fill_light_color']['b_range'])
    ]
    fill_light.set_color(fill_color)

    # Side light for better coverage
    side_light = bproc.types.Light()
    side_light.set_type("POINT")
    side_light.set_location([
        workspace_center[0] + random.uniform(*config['side_light_offset_range']),
        workspace_center[1] + random.uniform(*config['side_light_offset_range']),
        workspace_center[2] + random.uniform(*config['side_light_height_range'])
    ])
    side_light.set_energy(random.uniform(*config['side_light_energy_range']))

    # Set random color for side light
    side_color = [
        random.uniform(*config['side_light_color']['r_range']),
        random.uniform(*config['side_light_color']['g_range']),
        random.uniform(*config['side_light_color']['b_range'])
    ]
    side_light.set_color(side_color)

    # Ambient lighting with color variation
    try:
        world_node_tree = bpy.data.worlds["World"].node_tree
        if "Background" in world_node_tree.nodes:
            bg_node = world_node_tree.nodes["Background"]
            # Set ambient intensity
            bg_node.inputs[1].default_value = random.uniform(*config['ambient_light_range'])
            # Set ambient color
            ambient_color = [
                random.uniform(*config['ambient_light_color']['r_range']),
                random.uniform(*config['ambient_light_color']['g_range']),
                random.uniform(*config['ambient_light_color']['b_range']),
                1.0  # Alpha channel
            ]
            bg_node.inputs[0].default_value = ambient_color
    except Exception as e:
        logger.warning(f"Could not set ambient lighting: {e}")
