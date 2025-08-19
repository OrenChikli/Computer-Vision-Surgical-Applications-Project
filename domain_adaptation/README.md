# Domain Adaptation for Surgical Instrument Pose Estimation

This module implements **Phase 3: Domain Adaptation** using iterative refinement and pseudo-labeling to adapt synthetic-trained models to real surgical data.

## Quick Start

### 1. Setup Configuration

```bash
# Copy example configuration
cp config_example.yaml config.yaml

# Edit configuration with your paths
nano config.yaml
```

### 2. Run Domain Adaptation

```bash
# From project root directory
python domain_adaptation/run_domain_adaptation.py --config domain_adaptation/config.yaml
```

### 3. Evaluation

**Automatic Evaluation (Recommended):**
Evaluation runs automatically when `evaluation.run_evaluation: true` in config.yaml (default behavior). This provides immediate feedback without additional commands.

**Manual Evaluation (Optional):**
```bash
# Run detailed evaluation separately (creates additional videos)
python domain_adaptation/evaluate_refinement.py domain_adaptation/config.yaml
```

**Note**: Automatic evaluation focuses on metrics to avoid video duplication, while manual evaluation creates additional annotated videos.

## Configuration Guide

### Essential Paths Configuration

```yaml
paths:
  # Phase 2 trained model (.pt file from YOLO training)
  model_path: "../training_results/surgical_pose_v1/weights/best.pt"
  
  # Phase 1 synthetic dataset (YOLO format with images/labels folders)
  synthetic_data_path: "../datasets/yolo_dataset"
  
  # Real surgical video for adaptation
  real_video_path: "../data/real_videos/surgical_video.mp4"
  
  # Output directory for all results
  output_dir: "../output/domain_adaptation_results"
```

### Key Parameters

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `refinement.iterations` | Number of adaptation cycles | 1-3 for testing, 3-5 for production |
| `pseudo_labeling.confidence_threshold` | Min confidence for pseudo-labels | 0.7-0.9 (higher = more selective) |
| `pseudo_labeling.max_pseudo_labels` | Max pseudo-labels per iteration | 300-1000 depending on video length |
| `training.epochs` | Training epochs per iteration | 5-15 (fewer for large datasets) |
| `training.batch_size` | Training batch size | 8-32 (adjust for GPU memory) |
| `evaluation.run_evaluation` | Enable automatic evaluation | true (recommended) / false |
| `evaluation.sample_rate` | Evaluation frame sampling | 5-10 (higher = faster evaluation) |

### Memory Optimization

```yaml
# For limited GPU memory
training:
  batch_size: 4          # Reduce batch size
  device: "cpu"          # Force CPU training

memory_management:
  force_cpu_inference: true    # Use CPU for inference
  chunk_size: 25              # Process fewer frames at once
  max_frames_in_memory: 50    # Reduce memory usage

video_annotation:
  enabled: false              # Disable video creation
```

### Performance Tuning

```yaml
# For faster processing
pseudo_labeling:
  confidence_threshold: 0.9   # Higher threshold = fewer but better labels
  max_pseudo_labels: 200     # Process fewer labels

evaluation:
  sample_rate: 10            # Process every 10th frame only

video_annotation:
  enabled: false             # Skip video creation
```

## Usage Examples

### Example 1: Basic Domain Adaptation

```bash
# Single iteration with default settings
python domain_adaptation/run_domain_adaptation.py --config domain_adaptation/config.yaml
```

**Expected Output:**
```
🚀 Starting domain adaptation with video annotation...
🎬 Creating baseline video annotation
🔄 Starting Iteration 1/3
🎯 Iteration 1: Extracting pseudo-labels...
💾 Iteration 1: Saving 347 pseudo-labels...
📊 Iteration 1: Creating adapted dataset...
🚀 Iteration 1: Retraining model...
🎬 Creating annotated video for iteration 1...
✅ Iteration 1 completed successfully.
🎉 Domain adaptation completed!
```

### Example 2: Multi-Iteration Refinement

```yaml
# config.yaml
refinement:
  iterations: 3
  accumulate_data: true        # Use data from all previous iterations
  save_intermediate_models: true
```

```bash
python domain_adaptation/run_domain_adaptation.py --config domain_adaptation/config.yaml
```

### Example 3: Evaluation Only (No Retraining)

```bash
# Extract pseudo-labels but skip training (for analysis)
python domain_adaptation/run_domain_adaptation.py --config domain_adaptation/config.yaml --no-retrain
```

### Example 4: Comprehensive Evaluation

```bash
# Evaluate all iterations to see progression
python domain_adaptation/evaluate_refinement.py domain_adaptation/config.yaml --evaluate-all-iterations
```

## Understanding the Output

### Directory Structure

```
output/domain_adaptation_results/
├── baseline_video_annotated.mp4          # Original model performance
├── iteration_1/
│   ├── images/                           # Training images (synthetic + pseudo)
│   ├── labels/                           # Training labels
│   ├── pseudo_labeled/                   # Raw pseudo-labeled data
│   │   ├── images/                       # Extracted frames
│   │   └── labels/                       # Generated labels
│   ├── refined_model/                    # YOLO training outputs
│   │   ├── weights/best.pt               # Best model weights
│   │   └── train/                        # Training logs
│   ├── dataset.yaml                      # Training configuration
│   ├── iteration_summary.json            # Iteration statistics
│   └── video_annotated.mp4               # Iteration performance video
├── iteration_2/                          # Same structure for iteration 2
├── iteration_3/                          # Same structure for iteration 3
├── overall_refinement_summary.json       # Overall results
└── evaluation_results/                   # Evaluation outputs
    ├── results_synthetic_only.mp4        # Original model video
    ├── results_refined.mp4               # Final refined model video
    ├── comparison_report.json             # Performance comparison
    └── detailed_frame_analysis.json      # Frame-by-frame statistics
```

### Key Files to Check

1. **`overall_refinement_summary.json`**: High-level results across all iterations
2. **`iteration_N/iteration_summary.json`**: Detailed stats for each iteration  
3. **`evaluation_results/comparison_report.json`**: Model comparison metrics
4. **Video files**: Visual comparison of model performance

### Example Results Interpretation

```json
// overall_refinement_summary.json
{
  "total_iterations": 3,
  "pseudo_labels_progression": [347, 412, 398],  // Labels per iteration
  "improvement_metrics": {
    "pseudo_labels_first_to_last": 51,           // More labels found
    "average_pseudo_labels_per_iteration": 385.7
  },
  "best_iteration": {
    "iteration_number": 2,                       // Best performing iteration
    "pseudo_labels_count": 412,
    "model_path": "iteration_2/refined_model/weights/best.pt"
  }
}
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```
❌ Failed to import VideoAnnotator: No module named 'video'
```
**Solution**: Run from project root directory, not from domain_adaptation folder

#### 2. Path Issues
```
❌ Missing required input paths: model_path: path/to/model.pt
```
**Solution**: Use absolute paths or verify relative paths in config.yaml

#### 3. GPU Memory Issues  
```
❌ CUDA out of memory
```
**Solutions**:
- Reduce `training.batch_size` to 4-8
- Enable `memory_management.force_cpu_inference: true`
- Set `training.device: "cpu"`

#### 4. No Pseudo-Labels Generated
```
⚠️ Iteration 1: No high-confidence pseudo-labels found
```
**Solutions**:
- Lower `pseudo_labeling.confidence_threshold` to 0.6-0.7
- Check video quality and ensure surgical instruments are visible
- Verify model was properly trained in Phase 2

#### 5. Video Processing Fails
```
❌ Cannot open video file
```
**Solutions**:
- Check video format (MP4 recommended)
- Verify video file is not corrupted
- Try converting video: `ffmpeg -i input.avi output.mp4`

### Performance Tips

1. **Start Small**: Test with 1 iteration before running multiple iterations
2. **Monitor Resources**: Check GPU memory usage during training
3. **Quality Check**: Review generated videos to verify improvements
4. **Iterative Tuning**: Adjust confidence threshold based on initial results

### Getting Help

1. Check the main README.md for Phase 3 documentation
2. Review configuration examples in config.yaml.example
3. Examine log files in the output directory for detailed error messages
4. Verify all dependencies are installed: `pip install ultralytics opencv-python tqdm`

## Advanced Usage

### Custom Evaluation Metrics

```python
# Custom evaluation script
from domain_adaptation import RefinementEvaluator

evaluator = RefinementEvaluator(
    original_model_path="phase2_model.pt",
    refined_model_path="iteration_3/refined_model/weights/best.pt"
)

results = evaluator.evaluate_on_video(
    video_path="surgical_video.mp4",
    output_dir="custom_evaluation",
    sample_rate=1  # Process every frame
)
```

### Batch Processing Multiple Videos

```python
# Process multiple videos
import glob
from domain_adaptation import DomainAdaptation

config_path = "domain_adaptation/config.yaml"
video_files = glob.glob("videos/*.mp4")

for video_file in video_files:
    # Update config for this video
    # Run domain adaptation
    # Save results with video-specific naming
```

This completes the comprehensive domain adaptation system for surgical instrument pose estimation!