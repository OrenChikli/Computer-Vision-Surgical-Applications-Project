# Directory Structure Setup

Create the following directory structure in your project:

```
your-project/
├── config/
│   ├── __init__.py
│   ├── config_loader.py
│   └── default_config.yaml
├── core/
│   ├── __init__.py
│   ├── tool_manager.py
│   └── keypoint_extractor.py
├── utils/
│   ├── __init__.py
│   ├── camera_utils.py
│   ├── lighting_utils.py
│   ├── material_utils.py
│   ├── workspace_utils.py
│   ├── visualization.py
│   └── coco_utils.py
├── synthetic_data_generator.py
├── predict.py
├── video.py
├── config.yaml
├── requirements.txt
└── README.md
```

## Files to Copy

Copy each of the provided files to their corresponding locations in this structure.

## Configuration

1. Update `config.yaml` with your actual data paths
2. Ensure your data follows this structure:

```
data/
├── surgical_tools_models/
│   ├── needle_holder/
│   │   ├── NH1.obj
│   │   └── NH2.obj
│   └── tweezers/
│       ├── TW1.obj
│       └── TW2.obj
├── surgical_tools_annotations/
│   ├── tool_skeletons.json
│   ├── NH1_keypoints.json
│   ├── NH2_keypoints.json
│   ├── TW1_keypoints.json
│   └── TW2_keypoints.json
└── camera.json
```

## Quick Start

1. Create the directory structure
2. Copy all provided files
3. Install dependencies: `pip install -r requirements.txt`
4. Update paths in `config.yaml`
5. Run: `python synthetic_data_generator.py --config config.yaml`
