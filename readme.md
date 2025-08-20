# Surgical Instrument 2D Pose Estimation

This project implements a synthetic data generation pipeline for 2D pose estimation of surgical instruments, for the Technion course Computer vision, Surgical applications (SM-00970222).

## Project Structure

```
├── annotation/
│   ├── SURGICAL_TOOL_ANNOTATOR_GUIDE.md  # Tool annotation guide
│   ├── tool_annotator.py                  # Manual annotation tool
│   └── tool_skeletons.json               # Tool keypoint definitions
├── config/
│   ├── __init__.py
│   ├── config.yaml                       # Example configuration
│   ├── config_examples.yaml              # Additional config examples
│   ├── config_loader.py                  # YAML configuration loading
│   └── default_config.yaml               # Default configuration values
├── domain_adaptation/
│   ├── __init__.py
│   ├── README.md                         # Domain adaptation guide
│   ├── config.yaml                       # Domain adaptation config
│   ├── domain_adaptation.py              # Core adaptation logic
│   ├── evaluate_refinement.py            # Evaluation tools
│   └── run_domain_adaptation.py          # Main adaptation script
├── examples/
│   ├── 000000.jpg - 000009.jpg           # 10 example synthetic images
│   ├── results_refined.mp4               # Refined model results
│   └── results_synthetic_only.mp4        # Synthetic-only model results
├── utils/
│   ├── __init__.py
│   ├── camera_utils.py                   # Camera positioning utilities
│   ├── coco_utils.py                     # COCO format handling
│   ├── coco_to_yolo.py                   # COCO to YOLO converter
│   ├── lighting_utils.py                 # Lighting setup
│   ├── material_utils.py                 # Material properties
│   ├── statistics_tracker.py             # Dataset statistics
│   ├── visualization.py                  # Keypoint visualization
│   ├── keypoint_extractor.py             # Keypoint extraction logic
│   ├──tool_manager.py                   # Tool loading and management
│   └── workspace_utils.py                # Workspace generation
│
├── directory_structure.md                # Project structure details
├── predict.py                            # Single image prediction (Phase 2)
├── requirements.txt                      # Python dependencies
├── setup_pip.py                         # Automated environment setup
├── setup_guide.md                       # Setup instructions
├── synthetic_data_generator.py          # Main entry point (Phase 1)
└── video.py                             # Video processing (Phase 2)
```

## Quick Start

### 1. Environment Setup

**⚠️ Important**: This project requires **Python 3.10** for compatibility with BlenderProc and other dependencies.

#### Option A: Automated Setup (Recommended)

Use the automated setup script that creates a Python 3.10 virtual environment and installs all dependencies:

```bash
# Run the automated setup (requires Python 3.10 to be installed on your system)
python setup_pip.py
```

#### Option B: Manual Setup

If you prefer manual setup:

```bash
# Create a virtual environment with Python 3.10 (recommended)
py -3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

#### Activation

After setup, activate the environment:

```bash
# On Linux/Mac:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 2. Data Preparation

Prepare your surgical instrument data structure:

```
data/
├── surgical_tools_models/        # 3D models (.obj files)
│   ├── needle_holder/
│   │   ├── NH1.obj
│   │   └── NH2.obj
│   └── tweezers/
│       └── TW1.obj
├── annotations/                   # Manual keypoint annotations  
│   ├── tool_skeletons.json
│   ├── NH1_keypoints.json
│   ├── NH2_keypoints.json
│   └── TW1_keypoints.json
└── camera.json                   # Camera intrinsic parameters
```

**📚 For detailed annotation instructions, see: [annotation/SURGICAL_TOOL_ANNOTATOR_GUIDE.md](annotation/TOOL_ANNOTATOR_GUIDE.md)**

**💡 Note**: The `tool_skeletons.json` file should be created manually before starting annotation. See the annotation guide for details.

### 3. Configuration

Copy and modify the configuration file:

```bash
# Option 1: Use the provided example configuration
cp config/config.yaml my_config.yaml

# Option 2: Start from the default configuration  
cp config/default_config.yaml my_config.yaml
```

Edit `my_config.yaml` to set your specific paths:

```yaml
# Update these paths to match your data location
tools_path: "/path/to/your/surgical_tools_models"
annotations_path: "/path/to/your/annotations"
camera_params: "/path/to/your/camera.json"
output_dir: "/path/to/output/dataset"
hdri_path: "/path/to/hdri/files"  # Optional
```



## Usage

### Phase 1: Synthetic Data Generation

Generate synthetic surgical instrument images with keypoint annotations:

```bash
# Use blenderproc to run the generator (required for BlenderProc)
blenderproc run synthetic_data_generator.py --config my_config.yaml
```

**⚠️ Important**: You must use `blenderproc run` instead of `python` for the synthetic data generator since it uses BlenderProc.

#### Configuration Options

Key parameters you can adjust in `config.yaml`:

- `num_images`: Number of images to generate (default: 1000)
- `poses_per_workspace`: Camera poses per workspace setup (default: 5)
- `workspace_size`: Size of the surgical workspace in meters (default: 3.5)
- `render_width`/`render_height`: Output image resolution
- `motion_blur_prob`: Probability of applying motion blur (default: 0.3)
- `occlusion_prob`: Probability of adding occlusion (default: 0.3)
- `visualize_keypoints`: Generate visualization images (default: false)

#### Output

The generator creates:
- Synthetic images in JPEG format
- COCO-format annotations with keypoints
- Segmentation masks
- Optional keypoint visualizations

### Phase 2: Model Training and Prediction

**Phase 2 Implementation**: This phase uses ultralytics YOLO for training. You have two options for training:

## ** YOLO Format Training **

#### Step 1: Convert COCO to YOLO Format

Convert your synthetic dataset to YOLO format:

```bash
# Basic conversion (flip_pairs automatically read from COCO categories)
python utils/coco_to_yolo.py path/to/your/dataset path/to/yolo_output

# Custom train/val/test split ratios
python utils/coco_to_yolo.py path/to/your/dataset path/to/yolo_output --split-ratio 0.8 --test-ratio 0.1
```

#### Step 2: Train YOLO Model

Train the pose estimation model using YOLO CLI:

```bash
# Train YOLO pose model with your converted dataset
yolo pose train \
  data="path/to/yolo_output/dataset.yaml" \
  model=yolo11n-pose.pt \
  epochs=100 \
  imgsz=1280 \
  batch=16 \
  project="path/to/training_results" \
  name="surgical_pose_v1" \
  save_period=5 \
  visualize=True \
  flipud=0.5 \
  fliplr=0.5
```


#### **Step 3: Run Predictions**

###### Single Image Prediction

```bash
python predict.py  --model path/to/best.pt --image_path path/to/image.jpg 
```

###### Video Processing

```bash
python video.py path/to/video.mp4 --model path/to/best.pt --output predicted_video.mp4
```

### Phase 3: Domain Adaptation

**Phase 3 Implementation**: Uses iterative refinement and pseudo-labeling to adapt your synthetic-trained model to real surgical data through unsupervised domain adaptation.

#### What Domain Adaptation Does

- **Extracts pseudo-labels** from real surgical video using your Phase 2 model
- **Combines synthetic + pseudo-labeled real data** for improved training
- **Iteratively refines** the model through multiple adaptation cycles
- **Automatically creates annotated videos** to visualize improvements
- **Provides comprehensive evaluation** comparing original vs refined models

#### Step 1: Configure Domain Adaptation

```bash
# Copy example configuration and customize for your setup
cp domain_adaptation/config.yaml domain_adaptation/my_config.yaml

# Edit the configuration file with your specific paths
nano domain_adaptation/config.yaml
```

**Key Configuration Settings:**
```yaml
paths:
  model_path: "path/to/your/phase2_model/weights/best.pt"        # Phase 2 trained model
  synthetic_data_path: "path/to/your/synthetic/yolo_dataset"     # Phase 1 dataset (YOLO format)
  real_video_path: "path/to/your/real_surgical_video.mp4"       # Real surgical video
  output_dir: "output/domain_adaptation_results"                # Results directory

refinement:
  iterations: 3                    # Number of refinement iterations
  accumulate_data: true           # Accumulate data across iterations
  
pseudo_labeling:
  confidence_threshold: 0.8       # Minimum confidence for pseudo-labels
  max_pseudo_labels: 500         # Maximum pseudo-labels per iteration
```

#### Step 2: Run Domain Adaptation

```bash
# Basic domain adaptation with default settings
python domain_adaptation/run_domain_adaptation.py --config domain_adaptation/config.yaml

# Skip model retraining (only extract pseudo-labels for analysis)
python domain_adaptation/run_domain_adaptation.py --config domain_adaptation/config.yaml --no-retrain
```
#### Step 3: Results and Evaluation

**Automatic Evaluation (Recommended):**
Domain adaptation automatically runs evaluation when `evaluation.run_evaluation: true` in config.yaml. This provides immediate feedback on model improvements without needing separate commands.

**Manual Evaluation (Optional):**
```bash
# Run detailed evaluation separately (creates additional videos)
python domain_adaptation/evaluate_refinement.py domain_adaptation/config.yaml
```

**Note**: Automatic evaluation focuses on metrics only to avoid video duplication, while manual evaluation creates additional annotated videos for detailed analysis.

**Evaluation Output:**
- `results_synthetic_only.mp4` - Original model predictions
- `results_refined.mp4` - Refined model predictions  
- `comparison_report.json` - Detailed performance metrics
- `detailed_frame_analysis.json` - Frame-by-frame statistics

#### Domain Adaptation Results Structure

```
output/domain_adaptation_results/
├── baseline_video_annotated.mp4          # Original model performance
├── iteration_1/
│   ├── images/                           # Combined training images
│   ├── labels/                           # Combined training labels
│   ├── pseudo_labeled/                   # Pseudo-labeled real data
│   ├── refined_model/weights/best.pt     # Iteration 1 refined model
│   ├── video_annotated.mp4               # Iteration 1 performance
│   └── iteration_summary.json            # Iteration 1 statistics
├── iteration_2/
│   └── ...                               # Same structure for iteration 2
├── iteration_3/
│   └── ...                               # Same structure for iteration 3
├── overall_refinement_summary.json       # Summary across all iterations
├── automatic_evaluation_results/         # Automatic evaluation metrics (no videos)
│   ├── comparison_report.json            # Performance comparison metrics
│   └── detailed_frame_analysis.json     # Frame-by-frame statistics
└── evaluation_results/                   # Manual evaluation outputs (created only if run separately)
    ├── results_synthetic_only.mp4        # Created only by manual evaluation
    ├── results_refined.mp4               # Created only by manual evaluation
    └── comparison_report.json             # Detailed comparison with videos
```

#### Advanced Usage

**Custom Iteration Parameters:**
```yaml
refinement:
  iterations: 5                    # More iterations for better adaptation
  save_intermediate_models: true   # Save model after each iteration
  accumulate_data: false          # Don't accumulate (use only current iteration data)

training:
  epochs: 15                      # More training epochs per iteration
  batch_size: 8                   # Adjust for GPU memory
  device: "cuda:0"                # Specific GPU selection
```

**Memory Optimization:**
```yaml
memory_management:
  chunk_size: 25                  # Process fewer frames at once
  force_cpu_inference: true       # Use CPU if GPU memory limited
  max_frames_in_memory: 100       # Reduce memory usage

video_annotation:
  enabled: false                  # Disable video creation to save time
```

**Evaluation Options:**
```bash
# Evaluate all iterations to see progression
python domain_adaptation/evaluate_refinement.py config.yaml --evaluate-all-iterations

# Process every 10th frame for faster evaluation
python domain_adaptation/evaluate_refinement.py config.yaml --sample-rate 10
```

#### Expected Results

**Successful Domain Adaptation Shows:**
-  **Increased confidence scores** on real surgical data
-  **More consistent detections** across video frames  
-  **Better keypoint localization** on real instruments
-  **Reduced domain gap** between synthetic and real data

**Monitoring Progress:**
```bash
# Watch training logs
tail -f output/domain_adaptation_results/iteration_*/refined_model/train/weights/last.pt

# Check iteration summaries
cat output/domain_adaptation_results/iteration_*/iteration_summary.json

# View overall results
cat output/domain_adaptation_results/overall_refinement_summary.json
```

#### Troubleshooting

**Common Issues:**

1. **Import Errors**: Run from project root directory
2. **Path Issues**: Use absolute paths or check config.yaml paths
3. **GPU Memory**: Reduce batch_size or enable CPU inference
4. **No Pseudo-Labels**: Lower confidence_threshold or check video quality
5. **Training Failures**: Verify dataset format and paths

**Performance Tips:**
- Start with 1-2 iterations to test the pipeline
- Use confidence_threshold 0.7-0.9 for good pseudo-labels
- Enable video annotation to visualize improvements
- Monitor GPU usage during training iterations

## Dataset Output Format

The generated dataset follows COCO format with keypoint annotations:

```json
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "keypoints": [x1, y1, v1, x2, y2, v2, ...],
      "num_keypoints": 5,
      "area": 1234,
      "bbox": [x, y, width, height]
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "needle_holder",
      "keypoints": ["tip", "joint", "handle", ...],
      "skeleton": [[1, 2], [2, 3], ...]
    }
  ]
}
```

Where keypoint visibility values are:
- `0`: Not available
- `1`: Not visible (occluded)
- `2`: Visible

## Converting to YOLO Format

After generating your COCO dataset, you can convert it to YOLO format for training with ultralytics YOLO:

### Installation

The YOLO conversion dependencies are included in the main requirements.txt file. No additional installation needed.

### Conversion

Convert your COCO dataset to YOLO format:

```bash
# Basic conversion (automatically finds COCO JSON and images)
python scripts/coco_to_yolo.py path/to/dataset path/to/yolo_output

# Custom train/validation split ratio (default: 0.8)
python scripts/coco_to_yolo.py path/to/dataset path/to/yolo_output --split-ratio 0.7

# Custom test split ratio (default: 0.1)
python scripts/coco_to_yolo.py path/to/dataset path/to/yolo_output --test-ratio 0.15

# Custom split ratios (train: 70%, test: 15%, val: 15%)
python scripts/coco_to_yolo.py path/to/dataset path/to/yolo_output --split-ratio 0.7 --test-ratio 0.15

# With skeleton configuration for flip_idx augmentation
python scripts/coco_to_yolo.py path/to/dataset path/to/yolo_output --skeleton-json path/to/tool_skeletons.json
```

### What Gets Created

The converter creates a complete YOLO dataset structure:

```
yolo_output/
├── images/
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   └── test/           # Test images
├── labels/
│   ├── train/          # Training labels (.txt files)
│   ├── val/            # Validation labels (.txt files)
│   └── test/           # Test labels (.txt files)
└── dataset.yaml         # YOLO configuration file
```

### Training with YOLO

After conversion, train your model using the YOLO CLI (recommended for pose estimation):

```bash
# Train pose estimation model
yolo pose train \
  data="path/to/yolo_output/dataset.yaml" \
  model=yolo11n-pose.pt \
  epochs=100 \
  imgsz=1280 \
  batch=16 \
  project="path/to/results" \
  save_period=5 \
  visualize=True \
  flipud=0.5 \
  fliplr=0.5

# For smaller models, use yolo11n-pose.pt
# For better accuracy, try yolo11s-pose.pt or yolo11m-pose.pt
```

**Alternative Python API:**

```python
from ultralytics import YOLO

# Load a pre-trained pose model
model = YOLO('yolo11n-pose.pt')

# Train on your converted dataset
model.train(
    data='path/to/yolo_output/dataset.yaml',
    epochs=100,
    imgsz=1280,
    batch=16,
    flipud=0.5,
    fliplr=0.5
)
```

### Key Features

- ✅ **Bounding Box Conversion**: COCO [x, y, w, h] → YOLO [x_center, y_center, w, h] (normalized)
- ✅ **Keypoint Support**: Preserves surgical tool keypoint annotations with unified schema
- ✅ **Flip Index Generation**: Creates flip_idx for proper keypoint augmentation during training
- ✅ **Skeleton Configuration**: Supports tool_skeletons.json for advanced keypoint mapping
- ✅ **Automatic Splitting**: Creates train/validation/test splits
- ✅ **YOLO Compatibility**: Generates dataset.yaml ready for training
- ✅ **Flexible Paths**: Handles images in different directories

## Model Weights

*Model weights will be available after Phase 2 completion.*

Download links will be provided here for:
- **Synthetic-only model**: Trained only on synthetic data
- **Refined model**: Fine-tuned with domain adaptation

## Advanced Usage

### Dataset Validation

Validate your generated datasets:

```bash
# Validate dataset integrity and generate statistics
python scripts/validate_dataset.py /path/to/your/dataset --visualizations

# This will check:
# - File integrity
# - COCO format compliance  
# - Keypoint annotation validity
# - Generate summary statistics and plots
```

### COCO to YOLO Conversion

Convert your COCO dataset to YOLO format for training:

```bash
# Convert to YOLO format
python scripts/coco_to_yolo.py path/to/coco_annotations.json path/to/yolo_output
```

### Setup Verification

```bash
# Verify your setup and check dependencies
python setup.py
```

### Available Configurations

The project includes several configuration files:

- `config/default_config.yaml`: Default settings with documentation
- `config/config.yaml`: Example configuration ready to use
- `config/config_examples.yaml`: Additional configuration examples

```bash
# Use different configurations (remember to use blenderproc)
blenderproc run synthetic_data_generator.py --config config/config.yaml
blenderproc run synthetic_data_generator.py --config config/default_config.yaml
```

## Troubleshooting

### Common Issues

1. **Python Version**: This project requires Python 3.10. Use `python setup_pip.py` for automated setup
2. **BlenderProc Installation**: Make sure you're using Python 3.10 and the virtual environment
3. **Missing Dependencies**: Install all requirements with `pip install -r requirements.txt`
4. **Path Issues**: Use absolute paths in configuration files
5. **Memory Issues**: Reduce `render_samples` or image resolution for lower memory usage
6. **Virtual Environment**: Always activate the environment before running any scripts:
   - Linux/Mac: `source .venv/bin/activate`
   - Windows: `.venv\Scripts\activate`

### Performance Optimization

- Use SSD storage for faster I/O
- Increase `render_samples` for higher quality (slower)
- Adjust `poses_per_workspace` vs `num_images` for different workspace/pose ratios

## Results and Evaluation

After generation, you can:

1. **Visualize keypoints**: Set `visualize_keypoints: true` in config
2. **Analyze dataset statistics**: Check the generated COCO file
3. **Validate annotations**: Use the visualization images to verify quality

## Contributing

This project is part of a computer vision course assignment. The implementation focuses on:

- Phase 1: Synthetic data generation ✅
- Phase 2: Model training and prediction ✅  
- Phase 3: Domain adaptation ✅

## Citation

If you use this code in your research, please cite:

```
@misc{surgical-pose-estimation-2024,
  title={Synthetic Data Generation for Surgical Instrument 2D Pose Estimation},
  author={[Your Name]},
  year={2024},
  howpublished={\\url{https://github.com/[your-username]/[repo-name]}}
}
```
