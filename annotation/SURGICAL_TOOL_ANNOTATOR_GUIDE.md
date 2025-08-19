# Surgical Tool Keypoint Annotator

## Overview

The **Surgical Tool Keypoint Annotator** is a BlenderProc-based interactive system for manually annotating 3D surgical instrument models with keypoint landmarks. This tool enables researchers and developers to create high-quality training datasets for surgical instrument pose estimation models.

## Features

- 🎯 **Interactive 3D Annotation**: Place keypoints directly in Blender's 3D viewport
- 📋 **Batch Processing**: Annotate multiple tools sequentially with progress tracking
- 💾 **Progress Persistence**: Save and resume annotation sessions
- 🔄 **Keypoint Transfer**: Copy annotations between similar tools
- 📊 **Real-time Status**: Monitor progress and completion status
- 🎮 **Navigation Controls**: Browse tools without saving, jump to specific tools
- 📁 **Multiple Formats**: Save as JSON annotations and Blender files

## Prerequisites

- **Blender 3.0+** with BlenderProc addon installed
- **Python 3.7-3.11** (BlenderProc compatibility)
- **3D Models**: Surgical instrument .obj files organized by tool type
- **Output Directory**: Write permissions for saving annotations
- **Tool Skeleton File**: `tool_skeletons.json` defining keypoint structures for each tool type

## Directory Structure

Your 3D models should be organized as follows:

```
surgical_tools_models/
├── needle_holder/
│   ├── NH1.obj
│   ├── NH2.obj
│   └── ...
├── tweezers/
│   ├── TW1.obj
│   └── ...
├── forceps/
│   ├── FP1.obj
│   └── ...
└── ...

annotations/
├── tool_skeletons.json          # Tool skeleton definitions
├── NH1_keypoints.json          # Individual tool annotations
├── NH2_keypoints.json
└── ...
```

### Tool Skeleton File

The `tool_skeletons.json` file defines the standard keypoint structure for each tool type. This file should be created manually before starting annotation and will be used to automatically create keypoints for new tools.

**Example skeleton structure:**
```json
{
  "needle_holder": {
    "keypoints": ["base_left_forcep", "base_right_forcep", "joint", "tip_left_forcep", "tip_right_forcep"],
    "skeleton": [
      ["base_left_forcep", "joint"],
      ["base_right_forcep", "joint"], 
      ["joint", "tip_left_forcep"],
      ["joint", "tip_right_forcep"]
    ],
    "flip_pairs": [
      ["base_left_forcep", "base_right_forcep"],
      ["tip_left_forcep", "tip_right_forcep"]
    ]
  }
}
```

**Skeleton Components:**
- **keypoints**: Array of keypoint names for the tool type
- **skeleton**: Connections between keypoints (for visualization)
- **flip_pairs**: Symmetric keypoints that can be flipped during annotation

### Creating Your Skeleton File

Before starting annotation, create a `tool_skeletons.json` file in your annotations directory:

1. **Identify Tool Types**: List all surgical instrument types you'll be annotating
2. **Define Keypoints**: For each tool type, list the anatomical landmarks
3. **Plan Connections**: Define which keypoints should be connected (for visualization)
4. **Mark Symmetries**: Identify pairs of keypoints that are mirror images

**Example for a new tool type:**
```json
{
  "scissors": {
    "keypoints": ["blade_tip_left", "blade_tip_right", "pivot", "handle_left_end", "handle_right_end"],
    "skeleton": [
      ["pivot", "blade_tip_left"],
      ["pivot", "blade_tip_right"],
      ["pivot", "handle_left_end"],
      ["pivot", "handle_right_end"]
    ],
    "flip_pairs": [
      ["blade_tip_left", "blade_tip_right"],
      ["handle_left_end", "handle_right_end"]
    ]
  }
}
```

## Quick Start

### 1. Launch Blender with BlenderProc

```bash
blenderproc debug tool_annotator.py <tools_dir> <output_dir>
```

**Example:**
```bash
blenderproc debug tool_annotator.py /path/to/surgical_tools_models /path/to/annotations
```

### 2. Start Annotation Session

The system will automatically:
- Scan your tools directory
- Load the first unannotated tool
- Display annotation instructions
- Show progress information

### 3. Annotate Keypoints

**Automatic Keypoint Creation (Recommended):**
The system automatically creates keypoints based on your `tool_skeletons.json` file. You only need to position them correctly.

**Manual Keypoint Creation (Alternative):**
1. **Add Keypoints**: `Shift + A` → `Empty` → `Plain Axes`
2. **Position Keypoints**: Move each empty to important landmarks
3. **Rename Keypoints**: `F2` to rename with descriptive names

**Final Step:**
4. **Position Keypoints**: Move automatically created keypoints to correct anatomical locations
5. **Save Annotations**: Use `finish_tool()` in Python console

## Annotation Workflow

### Basic Workflow

**Skeleton-Based Workflow (Recommended):**
```
Load Tool → Auto-Create Keypoints → Position Keypoints → Save → Next Tool
```

**Manual Workflow (Alternative):**
```
Load Tool → Place Keypoints → Rename Keypoints → Save → Next Tool
```

### Skeleton-Based Annotation

When using the skeleton file, the system will:

1. **Detect Tool Type**: Automatically identify the tool type from the directory structure
2. **Load Skeleton**: Read the corresponding keypoint definitions from `tool_skeletons.json`
3. **Create Keypoints**: Automatically generate Empty objects with the correct names
4. **Position Keypoints**: Place keypoints at the origin (0,0,0) for easy positioning
5. **User Positioning**: You only need to move keypoints to their correct anatomical locations

### Keypoint Placement Guidelines

#### Needle Holders
- `jaw_left`, `jaw_right` - Left and right jaw tips
- `joint` - Pivot point between jaws and handle
- `handle_end` - End of the handle
- `ratchet` - Ratchet mechanism location

#### Tweezers/Forceps
- `tip_left`, `tip_right` - Left and right tip points
- `joint` - Pivot point
- `handle_end` - End of the handle

#### Scissors
- `blade_tip_left`, `blade_tip_right` - Blade tip points
- `pivot` - Scissor pivot point
- `handle_left_end`, `handle_right_end` - Handle ends

#### Generic Tools
- `tip` - Working end of the tool
- `joint` - Any pivot or bend point
- `handle_end` - End of the handle

## Console Commands

### Core Functions

| Command | Description |
|---------|-------------|
| `finish_tool()` | Save current tool and move to next |
| `skip_tool()` | Skip current tool without saving |
| `show_status()` | Display progress information |

### Navigation (Without Saving)

| Command | Description |
|---------|-------------|
| `next_tool()` | Go to next tool |
| `previous_tool()` | Go to previous tool |
| `goto_tool(5)` | Jump to tool by index (0-based) |
| `goto_tool_by_name('NH1')` | Jump to tool by name |

### Utility Functions

| Command | Description |
|---------|-------------|
| `rescale_tool(2.0)` | Make tool bigger for better visibility |
| `rescale_tool(0.5)` | Make tool smaller |
| `list_references()` | Show available reference tools |
| `transfer_keypoints('NH1')` | Copy keypoints from reference tool |

### Tool Management

| Command | Description |
|---------|-------------|
| `list_completed()` | Show all completed tools |
| `list_all_tools()` | Show all tools with status |
| `load_tool_for_editing('NH1')` | Load specific tool for editing |
| `unmark_completed('NH1')` | Remove completed status |

## Advanced Features

### Keypoint Transfer

Copy annotations from previously annotated tools:

```python
# List available references for current tool type
list_references()

# Transfer keypoints from a reference tool
transfer_keypoints('NH1')
```

### Tool Editing

Edit previously completed annotations:

```python
# Load a completed tool for editing
load_tool_for_editing('NH1')

# Make changes, then save
finish_tool()

# Or unmark as completed to re-enter workflow
unmark_completed('NH1')
```

### Progress Management

The system automatically saves progress to `annotation_progress.json`:

```json
{
  "current_tool_index": 3,
  "completed_tools": ["NH1", "NH2", "TW1"],
  "total_tools": 8,
  "last_updated": "2024-01-15 14:30:25"
}
```

## Output Files

### JSON Annotations

Each tool generates a `{tool_name}_keypoints.json` file:

```json
{
  "tool_name": "NH1",
  "tool_type": "needle_holder",
  "obj_file": "/path/to/NH1.obj",
  "keypoints": {
    "jaw_left": {"x": 0.1, "y": 0.2, "z": 0.0},
    "jaw_right": {"x": -0.1, "y": 0.2, "z": 0.0},
    "joint": {"x": 0.0, "y": 0.0, "z": 0.0},
    "handle_end": {"x": 0.0, "y": -0.3, "z": 0.0}
  },
  "keypoint_names": ["jaw_left", "jaw_right", "joint", "handle_end"],
  "num_keypoints": 4,
  "annotation_date": "2024-01-15 14:30:25"
}
```

### Blender Files

Each tool saves a `{tool_name}_annotated.blend` file containing:
- 3D model with proper positioning
- Keypoint empties with names
- Scene setup and lighting

## Troubleshooting

### Common Issues

#### 1. Functions Not Found
```python
# If functions aren't available, try:
annotator.finish_current_tool()
annotator.skip_current_tool()
annotator.show_status()
```

#### 2. Tool Too Small/Large
```python
# Rescale for better visibility
rescale_tool(3.0)  # Make bigger
rescale_tool(0.5)  # Make smaller
```

#### 3. Lost Progress
- Check `annotation_progress.json` in output directory
- Verify file permissions
- Restart annotation session

#### 4. BlenderProc Errors
- Ensure BlenderProc is properly installed
- Check Python version compatibility (3.7-3.11)
- Verify .obj files are valid 3D models

### Performance Tips

- **Use Solid Shading**: Better visibility than wireframe
- **Rescale Tools**: Larger tools are easier to annotate
- **Batch Similar Tools**: Use keypoint transfer for efficiency
- **Save Frequently**: Use `finish_tool()` after each tool

## Best Practices

### Annotation Quality

1. **Be Consistent**: Use same keypoint names across similar tools
2. **Precise Placement**: Place keypoints at exact anatomical landmarks
3. **Document Decisions**: Keep notes on keypoint placement criteria
4. **Review Regularly**: Check annotations for accuracy

### Workflow Efficiency

1. **Start with Reference Tools**: Annotate one tool of each type thoroughly
2. **Use Keypoint Transfer**: Copy annotations between similar tools
3. **Batch Process**: Annotate all tools of one type before moving to next
4. **Regular Saves**: Save progress frequently to avoid data loss

### Data Organization

1. **Consistent Naming**: Use descriptive, consistent keypoint names
2. **Tool Type Grouping**: Organize tools by surgical function
3. **Version Control**: Track changes to annotation schemes
4. **Backup Annotations**: Keep copies of annotation files

## Integration with Synthetic Data Generation

The generated annotations are compatible with the synthetic data generation pipeline:

1. **Load Annotations**: The pipeline reads `*_keypoints.json` files
2. **Generate Keypoints**: Creates synthetic images with keypoint annotations
3. **COCO Format**: Exports datasets in standard COCO format
4. **Training Ready**: Annotations are ready for model training

## Example Workflow

### Complete Annotation Session

```bash
# 1. Start annotation session
blenderproc debug surgical_tool_annotator.py /data/tools /data/annotations

# 2. Annotate first needle holder (NH1)
# - Place keypoints: jaw_left, jaw_right, joint, handle_end
# - Save: finish_tool()

# 3. Annotate second needle holder (NH2)
# - Use keypoint transfer: transfer_keypoints('NH1')
# - Adjust positions as needed
# - Save: finish_tool()

# 4. Continue with other tool types
# - Use similar workflow for tweezers, forceps, etc.
# - Leverage keypoint transfer for efficiency
```

## Support and Development

### Getting Help

1. **Check Console Output**: Look for error messages and status information
2. **Verify File Paths**: Ensure all directories and files exist
3. **Test with Simple Models**: Start with basic .obj files
4. **Check BlenderProc Installation**: Verify addon is working

### Contributing

To improve the annotation system:

1. **Report Issues**: Document problems with specific steps
2. **Suggest Features**: Propose new functionality
3. **Share Workflows**: Contribute efficient annotation strategies
4. **Test Compatibility**: Verify with different tool types

## Version History

- **v1.0**: Initial release with basic annotation functionality
- **v1.1**: Added keypoint transfer and progress persistence
- **v1.2**: Enhanced navigation and editing capabilities
- **v1.3**: Improved error handling and user feedback

---

**Note**: This annotation system is designed specifically for surgical instrument pose estimation research. For other applications, modifications may be required.
