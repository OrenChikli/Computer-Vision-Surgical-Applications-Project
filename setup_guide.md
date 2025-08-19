# Complete Setup Guide

This guide will walk you through setting up the surgical instrument pose estimation project from scratch.

## Prerequisites

- Python 3.7-3.11
- At least 8GB RAM (16GB recommended)
- NVIDIA GPU (optional but recommended for faster rendering)
- 10GB+ free disk space

## Step 1: Repository Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd surgical-instrument-pose-estimation

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Verify Installation

```bash
python setup.py
```

This will check:
- ✅ Python version compatibility
- ✅ All required packages installed
- ✅ BlenderProc functionality

## Step 3: Prepare Your Data

### Option A: Use Provided Data Structure

If you already have your data organized, ensure it follows this structure:

```
data/
├── surgical_tools_models/
│   ├── needle_holder/
│   │   ├── NH1.obj
│   │   ├── NH2.obj
│   │   └── ...
│   ├── tweezers/
│   │   ├── TW1.obj
│   │   └── ...
│   └── ...
├── surgical_tools_annotations/
│   ├── tool_skeletons.json
│   ├── NH1_keypoints.json
│   ├── NH2_keypoints.json
│   ├── TW1_keypoints.json
│   └── ...
└── camera.json
```

### Option B: Auto-Generate Data Structure

```bash
python scripts/prepare_data.py \
    --tools-dir "/path/to/your/3d/models" \
    --annotations-dir "/path/to/your/annotations" \
    --camera-file "/path/to/camera.json" \
    --create-templates
```

This will:
- 🔍 Scan for 3D models (.obj files)
- 📋 Create skeleton templates for detected tool types
- 📝 Generate annotation templates for each tool
- 📷 Create camera parameter template

## Step 4: Manual Annotation (if needed)

If you created templates in Step 3, you need to annotate your tools:

### Using the Manual Annotation Tool

```bash
# Run the manual annotation tool
blenderproc debug manual_keypoint_annotation.py \
    "/path/to/your/surgical_tools_models" \
    "/path/to/your/surgical_tools_annotations"
```

The annotation tool will:
- Load each tool sequentially
- Allow you to place keypoints interactively
- Save annotations in the correct format

### Annotation Guidelines

1. **Be Consistent**: Use the same keypoint names across similar tools
2. **Follow Skeleton**: Place keypoints according to your skeleton definition
3. **Precision Matters**: Accurate keypoints improve model performance
4. **Document Decisions**: Keep notes on keypoint placement criteria

## Step 5: Configure the Generator

### Create Your Configuration

```bash
# Copy example configuration
cp config_example.yaml config.yaml

# Edit with your paths
nano config.yaml  # or use your preferred editor
```

### Essential Configuration Settings

```yaml
# REQUIRED: Update these paths
tools_path: "/absolute/path/to/surgical_tools_models"
annotations_path: "/absolute/path/to/surgical_tools_annotations"
camera_params: "/absolute/path/to/camera.json"
output_dir: "/absolute/path/to/output"

# OPTIONAL: HDRI for realistic lighting
hdri_path: "/path/to/hdri/files"

# Start with small numbers for testing
num_images: 10
poses_per_workspace: 2
```

### Configuration Templates

Use pre-made configurations for common scenarios:

```bash
# Quick test (5 images, fast rendering)
cp config/examples/quick_test.yaml my_config.yaml

# Development work (20 images, balanced quality)
cp config/examples/development.yaml my_config.yaml

# High quality dataset (5000 images)
cp config/examples/high_quality.yaml my_config.yaml
```

## Step 6: Test Your Setup

```bash
# Run a comprehensive test
python scripts/test_generation.py
```

This will:
- ✅ Validate your configuration
- 🚀 Generate 3 test images
- 📊 Validate the output format
- 💾 Save results to `test_results/`

## Step 7: Generate Your Dataset

### Start Small

```bash
# Generate a small test dataset first
python synthetic_data_generator.py --config config.yaml
```

### Monitor Progress

The generator will show:
- 📊 Progress percentage
- ⏱️ Estimated time remaining
- 🖼️ Images generated
- 💾 Output location

### For Large Datasets

```bash
# Use batch generation for multiple configurations
python scripts/batch_generate.py --create-example
# Edit batch_config_example.yaml
python scripts/batch_generate.py --batch-config batch_config_example.yaml
```

## Step 8: Validate Your Dataset

```bash
# Comprehensive dataset validation
python scripts/validate_dataset.py /path/to/your/dataset --visualizations
```

This will:
- ✅ Check file integrity
- 📊 Generate statistics
- 🎯 Validate keypoint annotations
- 📈 Create visualization plots

## Troubleshooting

### Common Issues

#### 1. BlenderProc Installation Issues
```bash
# Ensure Python version is 3.7-3.11
python --version

# Try installing BlenderProc manually
pip install blenderproc==2.6.0

# On some systems, you may need:
pip install --no-cache-dir blenderproc
```

#### 2. Memory Issues
```bash
# Reduce rendering quality in config.yaml:
render_samples: 50        # Instead of 100
render_width: 640         # Instead of 1920
render_height: 480        # Instead of 1080
```

#### 3. Path Issues
```bash
# Always use absolute paths in config.yaml
# Verify paths exist:
ls "/path/to/your/surgical_tools_models"
ls "/path/to/your/surgical_tools_annotations"
```

#### 4. GPU Issues
```bash
# BlenderProc should auto-detect GPU
# If having issues, try CPU-only mode by reducing:
render_samples: 25
```

### Getting Help

1. **Check Logs**: Look at console output for specific error messages
2. **Validate Data**: Run `python setup.py` to check your setup
3. **Test Individual Components**: Use test scripts to isolate issues
4. **Check Examples**: Compare your config with working examples

## Performance Optimization

### For Faster Generation

```yaml
# Reduce quality settings
render_samples: 50
render_width: 640
render_height: 480
poses_per_workspace: 2

# Disable expensive effects
motion_blur_prob: 0.0
occlusion_prob: 0.0
visualize_keypoints: false
```

### For Higher Quality

```yaml
# Increase quality settings
render_samples: 200
render_width: 1920
render_height: 1080
poses_per_workspace: 10

# Add realistic effects
motion_blur_prob: 0.4
occlusion_prob: 0.5
hdri_path: "/path/to/hdri/files"
```

## What's Next?

After successful dataset generation:

1. **Phase 2**: Implement model training
2. **Phase 3**: Add domain adaptation
3. **Validation**: Test on real surgical videos
4. **Publication**: Document results and methods

## File Organization Tips

Keep your project organized:

```
project/
├── config.yaml              # Your main configuration
├── data/                    # All your data files
├── datasets/               # Generated datasets
│   ├── experiment_1/
│   ├── experiment_2/
│   └── ...
├── models/                 # Trained models (Phase 2)
├── results/               # Evaluation results
└── documentation/         # Your notes and reports
```

This structure will help you manage multiple experiments and maintain reproducibility.
