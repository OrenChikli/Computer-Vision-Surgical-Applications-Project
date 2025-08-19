import random
import numpy as np
from colorsys import hsv_to_rgb
import bpy


def setup_tool_material(obj, config: dict):
    """Setup realistic surgical instrument material with better error handling."""
    try:
        materials = obj.get_materials()
        if not materials:
            mat = bpy.data.materials.new(name=f"SurgicalSteel_{obj.get_name()}")
            obj.add_material(mat)
            materials = obj.get_materials()

        # Primary material (steel)
        mat = materials[0]
        mat.set_principled_shader_value("IOR", random.uniform(*config['material_ior_range']))
        mat.set_principled_shader_value("Roughness", random.uniform(*config['material_roughness_range']))
        mat.set_principled_shader_value("Metallic", config['material_metallic_value'])

        # Secondary material (gold accents) if available
        if len(materials) > 1:
            mat = materials[1]
            gold_hsv = np.random.uniform(config['gold_hsv_min'], config['gold_hsv_max'])
            gold_color = list(hsv_to_rgb(*gold_hsv)) + [1.0]
            mat.set_principled_shader_value("Base Color", gold_color)
            mat.set_principled_shader_value("IOR", random.uniform(*config['material_ior_range']))
            mat.set_principled_shader_value("Roughness", random.uniform(*config['material_roughness_range']))
            mat.set_principled_shader_value("Metallic", config['material_metallic_value'])

    except Exception as e:
        print(f"Warning: Failed to setup material for {obj.get_name()}: {e}")
