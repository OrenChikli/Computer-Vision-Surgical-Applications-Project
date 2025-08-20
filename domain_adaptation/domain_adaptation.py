"""
Domain Adaptation with Video Annotation Integration
Saves annotated videos at each step to visualize improvements
"""

import cv2
from utils.yaml_utils import load_yaml, save_yaml
import shutil
import numpy as np
import logging
import sys
import json
import time
import gc
import torch
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
import argparse
from dataclasses import dataclass


# Add parent directory to Python path for VideoAnnotator import
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from video import VideoAnnotator
except ImportError as e:
    print(f"❌ Failed to import VideoAnnotator: {e}")
    print(f"   Make sure you're running from the project root directory")
    print(f"   Project root: {project_root}")
    raise


@dataclass
class PseudoLabel:
    """Data class for pseudo-label information - MEMORY FIXED"""
    frame_id: int
    video_path: str  # Store video path instead of full frame
    box: np.ndarray
    confidence: float
    class_id: int
    keypoints: Optional[np.ndarray] = None
    frame_shape: Tuple[int, int] = None


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        return load_yaml(config_path)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except Exception as e:
        logging.error(f"Error parsing configuration file: {e}")
        raise


class ConfigManager:
    """Manages configuration loading and validation"""

    def __init__(self, config_path: str):
        self.config = _load_config(config_path)
        self._validate_config()

    def _validate_config(self):
        """Validate required configuration keys"""
        required_sections = ['paths', 'pseudo_labeling', 'tracking', 'training', 'output']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value


class PseudoLabelExtractor:
    """Handles extraction of pseudo-labels from video using tracking - MEMORY FIXED"""

    def __init__(self, model: YOLO, config: ConfigManager):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract_from_video(self, video_path: str) -> List[PseudoLabel]:
        """Extract high-confidence pseudo-labels from real video using tracking - MEMORY FIXED"""
        self.logger.info(f"Extracting pseudo-labels from: {video_path}")

        results = self._run_tracking(video_path)
        track_history = self._process_tracking_results(results, video_path)
        pseudo_labels = self._create_pseudo_labels(track_history, video_path)

        self.logger.info(f"Extracted {len(pseudo_labels)} high-confidence pseudo-labels")
        self._cleanup_memory()
        return pseudo_labels

    def _run_tracking(self, video_path: str):
        """Run tracking on video with configured parameters"""
        tracking_params = {
            'source': video_path,
            'conf': self.config.get('pseudo_labeling.confidence_threshold'),
            'persist': self.config.get('tracking.persist'),
            'tracker': self.config.get('tracking.tracker'),
            'save': False,
            'stream': True,
            'verbose': self.config.get('tracking.verbose')
        }
        return self.model.track(**tracking_params)

    def _process_tracking_results(self, results, video_path: str) -> Dict[int, List[Dict]]:
        """Process tracking results and group by track ID - MEMORY FIXED"""
        track_history = {}
        frame_count = 0

        for result in tqdm(results, desc="Processing frames"):
            if result.boxes is not None and result.boxes.id is not None:
                detections = self._extract_frame_detections_minimal(result, frame_count, video_path)

                for detection in detections:
                    track_id = detection['track_id']
                    if track_id not in track_history:
                        track_history[track_id] = []
                    track_history[track_id].append(detection)

            frame_count += 1

            if frame_count % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return track_history

    def _extract_frame_detections_minimal(self, result, frame_count: int, video_path: str) -> List[Dict]:
        """Extract detection data from a single frame - MEMORY FIXED"""
        detections = []
        frame_shape = result.orig_img.shape[:2]

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        track_ids = result.boxes.id.cpu().numpy()
        keypoints = self._extract_keypoints(result)

        for i, track_id in enumerate(track_ids):
            detection = {
                'frame_id': frame_count,
                'video_path': video_path,
                'frame_shape': frame_shape,
                'box': boxes[i].copy(),
                'confidence': confidences[i],
                'class': classes[i],
                'track_id': track_id,
                'keypoints': keypoints[i].copy() if keypoints is not None else None
            }
            detections.append(detection)

        return detections

    def _extract_keypoints(self, result) -> Optional[np.ndarray]:
        """Extract and process keypoints from result"""
        if not hasattr(result, 'keypoints') or result.keypoints is None:
            return None

        keypoints_xy = result.keypoints.xy.cpu().numpy()

        if hasattr(result.keypoints, 'conf'):
            keypoints_conf = result.keypoints.conf.cpu().numpy()
        else:
            keypoints_conf = np.ones((len(keypoints_xy), keypoints_xy.shape[1]))

        keypoints = []
        for i in range(len(keypoints_xy)):
            kpt_combined = []
            for j in range(len(keypoints_xy[i])):
                x, y = keypoints_xy[i][j]
                conf = keypoints_conf[i][j] if j < len(keypoints_conf[i]) else 1.0
                vis = 2 if (conf > 0.5 and x > 0 and y > 0) else 0
                kpt_combined.append([x, y, vis])
            keypoints.append(kpt_combined)

        return np.array(keypoints)

    def _create_pseudo_labels(self, track_history: Dict[int, List[Dict]], video_path: str) -> List[PseudoLabel]:
        """Create pseudo-labels from filtered tracks - MEMORY FIXED"""
        pseudo_labels = []
        min_track_length = self.config.get('pseudo_labeling.track_min_length')
        confidence_threshold = self.config.get('pseudo_labeling.confidence_threshold')

        for track_id, detections in track_history.items():
            if len(detections) >= min_track_length:
                mid_idx = len(detections) // 2
                best_detection = detections[mid_idx]

                if best_detection['confidence'] >= confidence_threshold:
                    pseudo_label = PseudoLabel(
                        frame_id=best_detection['frame_id'],
                        video_path=best_detection['video_path'],
                        box=best_detection['box'],
                        confidence=best_detection['confidence'],
                        class_id=int(best_detection['class']),
                        keypoints=best_detection['keypoints'],
                        frame_shape=best_detection['frame_shape']
                    )
                    pseudo_labels.append(pseudo_label)

        max_labels = self.config.get('pseudo_labeling.max_pseudo_labels')
        if len(pseudo_labels) > max_labels:
            pseudo_labels.sort(key=lambda x: x.confidence, reverse=True)
            pseudo_labels = pseudo_labels[:max_labels]

        return pseudo_labels

    def _cleanup_memory(self):
        """Simple memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class DatasetManager:
    """Manages dataset creation and file operations - MINIMAL CHANGES"""

    def __init__(self, config: ConfigManager, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.output_dir,
            self.output_dir / self.config.get('output.images_dir'),
            self.output_dir / self.config.get('output.labels_dir'),
            self.output_dir / self.config.get('output.pseudo_dir'),
            self.output_dir / self.config.get('output.pseudo_dir') / self.config.get('output.images_dir'),
            self.output_dir / self.config.get('output.pseudo_dir') / self.config.get('output.labels_dir')
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def save_pseudo_labels(self, pseudo_labels: List[PseudoLabel]):
        """Save pseudo-labeled data in YOLO format - MEMORY FIXED"""
        self.logger.info("Saving pseudo-labeled data...")

        pseudo_images_dir = self.output_dir / self.config.get('output.pseudo_dir') / self.config.get('output.images_dir')
        pseudo_labels_dir = self.output_dir / self.config.get('output.pseudo_dir') / self.config.get('output.labels_dir')

        for i, label_data in enumerate(tqdm(pseudo_labels, desc="Saving pseudo-labels")):
            frame = self._load_frame_from_video(label_data.video_path, label_data.frame_id)

            if frame is None:
                self.logger.warning(f"Could not load frame {label_data.frame_id} from video")
                continue

            img_filename = f"pseudo_{i:06d}.jpg"
            img_path = pseudo_images_dir / img_filename
            cv2.imwrite(str(img_path), frame)

            label_filename = f"pseudo_{i:06d}.txt"
            label_path = pseudo_labels_dir / label_filename

            with open(label_path, 'w') as f:
                yolo_line = self._create_yolo_annotation(label_data, frame.shape)
                f.write(yolo_line + '\n')

            del frame

            if i % 20 == 0:
                gc.collect()

    def _load_frame_from_video(self, video_path: str, frame_id: int) -> Optional[np.ndarray]:
        """Load a specific frame from video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()

        return frame if ret else None

    def _create_yolo_annotation(self, label_data: PseudoLabel, frame_shape: Tuple[int, int, int]) -> str:
        """Create YOLO format annotation line"""
        h, w = frame_shape[:2]
        box = label_data.box

        x_center = ((box[0] + box[2]) / 2) / w
        y_center = ((box[1] + box[3]) / 2) / h
        width = (box[2] - box[0]) / w
        height = (box[3] - box[1]) / h

        line = f"{label_data.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

        if label_data.keypoints is not None:
            for kpt in label_data.keypoints:
                kpt_x = kpt[0] / w
                kpt_y = kpt[1] / h
                kpt_vis = int(kpt[2]) if len(kpt) > 2 else 2
                line += f" {kpt_x:.6f} {kpt_y:.6f} {kpt_vis}"

        return line

    def copy_synthetic_data(self, synthetic_data_path: Path):
        """Copy synthetic data to combined dataset"""
        if not synthetic_data_path.exists():
            self.logger.warning(f"Synthetic data path does not exist: {synthetic_data_path}")
            return

        self.logger.info("Copying synthetic data...")
        synthetic_images = self._find_files(synthetic_data_path, "images", self.config.get('data.image_extensions'))
        synthetic_labels = self._find_files(synthetic_data_path, "labels", [self.config.get('data.label_extension')])

        self.logger.info(f"Found {len(synthetic_images)} synthetic images and {len(synthetic_labels)} labels")
        self._copy_files(synthetic_images, self.output_dir / self.config.get('output.images_dir'))
        self._copy_files(synthetic_labels, self.output_dir / self.config.get('output.labels_dir'))

    def _find_files(self, base_path: Path, subdir: str, extensions: List[str]) -> List[Path]:
        """Find files with specified extensions in directory structure"""
        files = []
        search_dir = base_path / subdir

        if search_dir.exists():
            for subdir_name in ['train', 'val', 'test']:
                subdir_path = search_dir / subdir_name
                if subdir_path.exists() and subdir_path.is_dir():
                    files.extend([f for f in subdir_path.iterdir()
                                if f.is_file() and f.suffix.lower() in extensions])

            if not files:
                files.extend([f for f in search_dir.iterdir()
                            if f.is_file() and f.suffix.lower() in extensions])
        return files

    def _copy_files(self, files: List[Path], destination: Path):
        """Copy files to destination with error handling"""
        for file_path in tqdm(files, desc=f"Copying to {destination.name}"):
            try:
                shutil.copy2(file_path, destination / file_path.name)
            except Exception as e:
                self.logger.warning(f"Could not copy {file_path}: {e}")

    def copy_pseudo_labeled_data(self):
        """Copy pseudo-labeled data to main dataset"""
        self.logger.info("Copying pseudo-labeled data...")
        pseudo_dir = self.output_dir / self.config.get('output.pseudo_dir')
        pseudo_images = list((pseudo_dir / self.config.get('output.images_dir')).glob("*"))
        pseudo_labels = list((pseudo_dir / self.config.get('output.labels_dir')).glob("*"))

        self._copy_files(pseudo_images, self.output_dir / self.config.get('output.images_dir'))
        self._copy_files(pseudo_labels, self.output_dir / self.config.get('output.labels_dir'))
        return len(pseudo_images), len(pseudo_labels)

    def create_dataset_yaml(self, dataset_config: Dict[str, Any]):
        """Create dataset configuration file for training"""
        adapted_config = dataset_config.copy()
        adapted_config['path'] = str(self.output_dir.absolute())
        adapted_config['train'] = self.config.get('output.images_dir')
        adapted_config['val'] = self.config.get('output.images_dir')

        yaml_path = self.output_dir / self.config.get('output.dataset_config')
        save_yaml(adapted_config, yaml_path)

        self.logger.info(f"Created dataset config: {yaml_path}")
        return yaml_path


class DomainAdaptation:
    """
    Domain adaptation with video annotation at each step
    """

    def __init__(self, config_path: str):
        self._setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load configuration
        self.config = ConfigManager(config_path)

        # Initialize paths
        self.original_model_path = self.config.get('paths.model_path')
        self.model = YOLO(self.original_model_path)
        self.synthetic_data_path = Path(self.config.get('paths.synthetic_data_path'))
        self.real_video_path = self.config.get('paths.real_video_path')
        self.output_dir = Path(self.config.get('paths.output_dir'))

        # Video annotation settings
        self.video_conf_thresh = self.config.get('video_annotation.confidence_threshold', 0.5)
        self.enable_video_annotation = self.config.get('video_annotation.enabled', True)
        
        # Evaluation settings
        self.run_evaluation = self.config.get('evaluation.run_evaluation', False)
        self.evaluation_sample_rate = self.config.get('evaluation.sample_rate', 5)

        # Validate paths
        self._validate_paths()

        # Get dataset configuration
        self.dataset_config = self._get_dataset_config()

        # Initialize components
        self.pseudo_extractor = PseudoLabelExtractor(self.model, self.config)
        self.dataset_manager = DatasetManager(self.config, self.output_dir)

        self.logger.info("🎬 Domain adaptation with video annotation initialized")

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def _validate_paths(self):
        """Validate that required input paths exist and are correct format"""
        paths_to_check = {
            'model_path': self.original_model_path,
            'synthetic_data_path': self.synthetic_data_path,
            'real_video_path': self.real_video_path
        }

        missing_paths = []
        for path_name, path_value in paths_to_check.items():
            path_obj = Path(path_value)
            if not path_obj.exists():
                missing_paths.append(f"{path_name}: {path_value}")

        if missing_paths:
            error_msg = "❌ Missing required input paths:\n" + "\n".join(f"  • {p}" for p in missing_paths)
            self.logger.error(error_msg)
            self.logger.error("📝 Please update your domain_adaptation/config.yaml file with correct paths")
            raise FileNotFoundError(error_msg)

        # Validate file formats
        self._validate_file_formats()

    def _validate_file_formats(self):
        """Validate file formats and accessibility"""
        # Validate model file
        model_path = Path(self.original_model_path)
        if not model_path.suffix == '.pt':
            raise ValueError(f"❌ Model file must be a .pt file, got: {model_path.suffix}")
        
        # Validate video file
        video_path = Path(self.real_video_path)
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}
        if video_path.suffix.lower() not in video_extensions:
            self.logger.warning(f"⚠️  Video file extension {video_path.suffix} may not be supported")
        
        # Test video accessibility
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"❌ Cannot open video file: {video_path}")
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            
            if frame_count == 0:
                raise ValueError(f"❌ Video file appears to be empty: {video_path}")
                
            self.logger.info(f"✅ Video validation passed: {frame_count} frames at {fps} FPS")
            
        except Exception as e:
            raise ValueError(f"❌ Video validation failed: {e}")
            
        # Validate synthetic dataset structure
        synthetic_path = Path(self.synthetic_data_path)
        required_subdirs = ['images', 'labels']
        missing_subdirs = []
        
        for subdir in required_subdirs:
            if not (synthetic_path / subdir).exists():
                missing_subdirs.append(subdir)
                
        if missing_subdirs:
            raise ValueError(f"❌ Synthetic dataset missing required directories: {missing_subdirs}\n"
                           f"Expected YOLO format with 'images' and 'labels' folders")
                           
        # Check for dataset.yaml
        if not (synthetic_path / 'dataset.yaml').exists():
            self.logger.warning("⚠️  dataset.yaml not found in synthetic dataset - will attempt to create one")
            
        self.logger.info("✅ All input validations passed")
        
        # Validate evaluation configuration
        self._validate_evaluation_config()

    def _validate_evaluation_config(self):
        """Validate evaluation configuration settings"""
        if self.run_evaluation:
            self.logger.info("✅ Automatic evaluation enabled")
            
            # Validate sample rate
            if not isinstance(self.evaluation_sample_rate, int) or self.evaluation_sample_rate < 1:
                raise ValueError(f"❌ evaluation.sample_rate must be a positive integer, got: {self.evaluation_sample_rate}")
            
            if self.evaluation_sample_rate > 50:
                self.logger.warning(f"⚠️  High sample_rate ({self.evaluation_sample_rate}) will process very few frames")
                
            # Check for RefinementEvaluator availability
            try:
                from evaluate_refinement import RefinementEvaluator
                self.logger.info("✅ RefinementEvaluator available for automatic evaluation")
            except ImportError as e:
                raise ImportError(f"❌ Cannot import RefinementEvaluator for automatic evaluation: {e}")
                
        else:
            self.logger.info("📊 Automatic evaluation disabled - use evaluate_refinement.py manually")

    def _get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration from model"""
        original_dataset_config = _load_config(str(self.synthetic_data_path/'dataset.yaml'))
        names = original_dataset_config["names"]
        if type(names) is list:
            assert list(self.model.names.values()) == names, "Original Dataset class names do not match model names"
        else:
            assert self.model.names == names, "Original Dataset class names do not match model names"
        nc = original_dataset_config["nc"]
        assert self.model.nc == nc,"Original Dataset nc parameter does not match model's nc parameter"
        kpt_shape = original_dataset_config.get('kpt_shape', None)
        assert kpt_shape == getattr(self.model, 'kpt_shape', None), "Original Dataset 'kpt_shape' parameter does not match model's kpt_shape"

        return {
            'names': names,
            'nc': nc,
            'kpt_shape': kpt_shape,
            'flip_idx': original_dataset_config.get('flip_idx', None)
        }

    def _annotate_video_with_model(self, model_path: str, output_dir: Path, model_name: str = "model") -> Optional[str]:
        """
        Annotate video with given model and save to output directory

        Args:
            model_path: Path to the YOLO model
            output_dir: Directory to save annotated video
            model_name: Name prefix for the output video

        Returns:
            Path to annotated video or None if failed
        """
        if not self.enable_video_annotation :
            self.logger.info("🎬 Video annotation disabled")
            return None

        try:
            self.logger.info(f"🎬 Creating annotated video with {model_name}...")

            # Create video output path
            video_name = Path(self.real_video_path).stem
            output_video_path = output_dir / f"{video_name}_{model_name}_annotated.mp4"

            # Create video annotator
            annotator = VideoAnnotator(model_path, logger=self.logger)

            # Progress callback
            def progress_callback(frame_idx, total_frames, progress_pct):
                if frame_idx % 100 == 0:  # Log every 100 frames
                    self.logger.info(f"   🎬 Video annotation progress: {frame_idx}/{total_frames} ({progress_pct:.1f}%)")

            # Annotate video
            stats = annotator.annotate_video(
                video_path=self.real_video_path,
                output_path=str(output_video_path),
                conf_thresh=self.video_conf_thresh,
                save_video=True,
                display_video=False,
                progress_callback=progress_callback
            )

            self.logger.info(f"✅ Video annotation completed:")
            self.logger.info(f"   📊 Processed {stats['processed_frames']} frames")
            self.logger.info(f"   🎯 Total detections: {stats['total_detections']}")
            self.logger.info(f"   📈 Avg detections per frame: {stats['avg_detections_per_frame']:.2f}")
            self.logger.info(f"   💾 Saved to: {output_video_path}")

            return str(output_video_path)

        except ImportError as e:
            self.logger.error(f"❌ Video annotation failed - Missing dependency: {e}")
            self.logger.error("   Install required packages: pip install opencv-python ultralytics")
            return None
        except MemoryError:
            self.logger.error(f"❌ Video annotation failed - Out of memory")
            self.logger.error("   Try reducing batch_size or enabling CPU inference in config")
            return None
        except Exception as e:
            self.logger.error(f"❌ Video annotation failed: {e}")
            self.logger.error(f"   Check video file format and model compatibility")
            return None

    def run_domain_adaptation(self, retrain: bool = True) -> Optional[Dict]:
        """
        Complete iterative domain adaptation pipeline with video annotation
        """
        self.logger.info("🚀 Starting domain adaptation with video annotation...")

        # Step 0: Create annotated video with original model
        if self.enable_video_annotation:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🎬 CREATING BASELINE VIDEO ANNOTATION")
            self.logger.info(f"{'='*60}")

            baseline_video_path = self._annotate_video_with_model(
                model_path=self.original_model_path,
                output_dir=self.output_dir,
                model_name="baseline"
            )

        # Get iteration parameters
        num_iterations = self.config.get('refinement.iterations', 1)
        save_intermediate = self.config.get('refinement.save_intermediate_models', True)
        accumulate_data = self.config.get('refinement.accumulate_data', False)

        if num_iterations < 1:
            raise ValueError("Number of iterations must be at least 1")

        self.logger.info(f"Running {num_iterations} refinement iteration(s)")

        iteration_results = []
        current_model_path = self.original_model_path

        try:
            for iteration in range(1, num_iterations + 1):
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"🔄 Starting Iteration {iteration}/{num_iterations}")
                self.logger.info(f"{'='*60}")

                # Create iteration-specific output directory
                iteration_dir = self.output_dir / f"iteration_{iteration}"
                iteration_dir.mkdir(parents=True, exist_ok=True)

                # Run single iteration
                iteration_result = self._run_single_iteration(
                    iteration=iteration,
                    model_path=current_model_path,
                    iteration_dir=iteration_dir,
                    accumulate_previous_data=accumulate_data and iteration > 1,
                    retrain=retrain
                )

                if iteration_result is None:
                    self.logger.warning(f"Iteration {iteration} failed to generate pseudo-labels")
                    break


                if retrain and iteration_result.get('refined_model_path'):
                    refined_model_path = iteration_result['refined_model_path']
                    if Path(refined_model_path).exists():
                        self.logger.info(f"\n🎬 Creating annotated video for iteration {iteration}...")

                        iteration_video_path = self._annotate_video_with_model(
                            model_path=refined_model_path,
                            output_dir=iteration_dir,
                            model_name=f"iteration_{iteration}"
                        )

                        # Add video path to iteration results
                        iteration_result['annotated_video_path'] = iteration_video_path

                iteration_results.append(iteration_result)

                # Update model path for next iteration
                if retrain and iteration_result.get('refined_model_path'):
                    refined_path = Path(iteration_result['refined_model_path'])
                    if refined_path.exists():
                        current_model_path = str(refined_path)
                        self.logger.info(f"✅ Updated model path for next iteration: {current_model_path}")

                self.logger.info(f"✅ Iteration {iteration} completed successfully.")

                # Save intermediate results
                if save_intermediate:
                    self._save_iteration_summary(iteration, iteration_result, iteration_dir)

            # Create overall summary
            overall_results = self._create_overall_summary(iteration_results)

            # Add baseline video to summary
            if self.enable_video_annotation and 'baseline_video_path' in locals():
                overall_results['baseline_video_path'] = baseline_video_path

            self._save_overall_summary(overall_results)

            self.logger.info("🎉 Domain adaptation with video annotation completed!")
            self._print_video_summary(overall_results)

            # Run automatic evaluation if enabled
            if self.run_evaluation:
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"📊 RUNNING AUTOMATIC EVALUATION")
                self.logger.info(f"{'='*60}")
                
                evaluation_results = self._run_automatic_evaluation(overall_results)
                if evaluation_results:
                    overall_results['automatic_evaluation'] = evaluation_results

            return overall_results

        except Exception as e:
            self.logger.error(f"Domain adaptation failed: {e}")
            raise

    def _run_single_iteration(self, iteration: int, model_path: str, iteration_dir: Path,
                             accumulate_previous_data: bool = False, retrain: bool = True) -> Optional[Dict]:
        """Run a single iteration of domain adaptation with improved memory management"""
        self.logger.info(f"🔧 Iteration {iteration}: Using model {model_path}")

        # Initialize variables to avoid UnboundLocalError
        current_model = None
        iteration_extractor = None
        iteration_dataset_manager = None

        try:
            if hasattr(self, '_current_iteration_model'):
                del self._current_iteration_model

            current_model = YOLO(model_path)
            self._current_iteration_model = current_model

            iteration_extractor = PseudoLabelExtractor(current_model, self.config)
            iteration_dataset_manager = DatasetManager(self.config, iteration_dir)

        except Exception as e:
            self.logger.error(f"Failed to load model {model_path}: {e}")
            return None

        try:
            # Step 1: Extract pseudo-labels
            self.logger.info(f"🎯 Iteration {iteration}: Extracting pseudo-labels...")
            pseudo_labels = iteration_extractor.extract_from_video(self.real_video_path)

            if len(pseudo_labels) == 0:
                self.logger.warning(f"Iteration {iteration}: No high-confidence pseudo-labels found")
                return None

            # Clean up after extraction
            if iteration_extractor is not None:
                del iteration_extractor
                iteration_extractor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Step 2: Save pseudo-labeled data
            self.logger.info(f"💾 Iteration {iteration}: Saving {len(pseudo_labels)} pseudo-labels...")
            iteration_dataset_manager.save_pseudo_labels(pseudo_labels)

            del pseudo_labels
            gc.collect()

            # Step 3: Create adapted dataset
            self.logger.info(f"📊 Iteration {iteration}: Creating adapted dataset...")
            iteration_dataset_manager.copy_synthetic_data(self.synthetic_data_path)

            if accumulate_previous_data:
                self._accumulate_previous_data(iteration, iteration_dir)

            pseudo_count, _ = iteration_dataset_manager.copy_pseudo_labeled_data()
            iteration_dataset_manager.create_dataset_yaml(self.dataset_config)

            # Calculate statistics
            total_images = len(list((iteration_dir / self.config.get('output.images_dir')).glob("*")))
            total_labels = len(list((iteration_dir / self.config.get('output.labels_dir')).glob("*")))

            dataset_stats = {
                'total_images': total_images,
                'total_labels': total_labels,
                'pseudo_labeled_this_iteration': pseudo_count,
                'synthetic': total_images - pseudo_count if not accumulate_previous_data else 'mixed'
            }

            self.logger.info(f"📈 Iteration {iteration} dataset: {total_images} images, {pseudo_count} new pseudo-labels")

            # Clean up before training
            if current_model is not None:
                del current_model
                current_model = None

            if hasattr(self, '_current_iteration_model'):
                del self._current_iteration_model

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Step 4: Retrain model
            refined_model_path = None
            training_results = None

            if retrain:
                self.logger.info(f"🚀 Iteration {iteration}: Retraining model...")
                training_results = self._refine_model_for_iteration(iteration, iteration_dir, model_path)

                potential_refined_path = iteration_dir / self.config.get('output.refined_model_dir') / 'weights' / 'best.pt'
                if potential_refined_path.exists():
                    refined_model_path = str(potential_refined_path)
                    self.logger.info(f"🎯 Refined model saved: {refined_model_path}")
                else:
                    self.logger.warning(f"Refined model not found at expected location: {potential_refined_path}")
                    refined_model_path = None

            return {
                'iteration': iteration,
                'input_model_path': model_path,
                'refined_model_path': refined_model_path,
                'iteration_dir': str(iteration_dir),
                'pseudo_labels_count': pseudo_count,
                'dataset_stats': dataset_stats,
                'training_results': training_results,
                'retrained': retrain
            }

        except Exception as e:
            self.logger.error(f"❌ Iteration {iteration} failed: {e}")
            raise
        finally:
            # Cleanup
            try:
                if current_model is not None:
                    del current_model
            except:
                pass
            try:
                if iteration_extractor is not None:
                    del iteration_extractor
            except:
                pass
            try:
                if hasattr(self, '_current_iteration_model'):
                    del self._current_iteration_model
            except:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # [Rest of the methods remain the same as before...]
    def _accumulate_previous_data(self, current_iteration: int, current_dir: Path):
        """Accumulate pseudo-labeled data from previous iterations"""
        self.logger.info(f"📂 Accumulating data from previous iterations...")

        current_images_dir = current_dir / self.config.get('output.images_dir')
        current_labels_dir = current_dir / self.config.get('output.labels_dir')

        accumulated_count = 0

        for prev_iteration in range(1, current_iteration):
            prev_dir = self.output_dir / f"iteration_{prev_iteration}"
            prev_pseudo_dir = prev_dir / self.config.get('output.pseudo_dir')

            if prev_pseudo_dir.exists():
                prev_images = list((prev_pseudo_dir / self.config.get('output.images_dir')).glob("*"))
                prev_labels = list((prev_pseudo_dir / self.config.get('output.labels_dir')).glob("*"))

                for img_path in prev_images:
                    new_name = f"iter{prev_iteration}_{img_path.name}"
                    shutil.copy2(img_path, current_images_dir / new_name)

                for label_path in prev_labels:
                    new_name = f"iter{prev_iteration}_{label_path.name}"
                    shutil.copy2(label_path, current_labels_dir / new_name)

                accumulated_count += len(prev_images)

        if accumulated_count > 0:
            self.logger.info(f"📚 Accumulated {accumulated_count} samples from {current_iteration-1} previous iteration(s)")

    def _refine_model_for_iteration(self, iteration: int, iteration_dir: Path, base_model_path: str) -> Any:
        """Retrain model for specific iteration"""
        refined_model = None
        try:
            refined_model = YOLO(base_model_path)

            train_config = {
                'data': str(iteration_dir / self.config.get('output.dataset_config')),
                'epochs': self.config.get('training.epochs'),
                'batch': self.config.get('training.batch_size'),
                'imgsz': self.config.get('training.imgsz'),
                'device': self.config.get('training.device'),
                'project': str(iteration_dir),
                'name': self.config.get('output.refined_model_dir'),
                'exist_ok': self.config.get('model.exist_ok'),
                'pretrained': True,
                'verbose': self.config.get('training.verbose')
            }

            results = refined_model.train(**train_config)
            self.logger.info(f"🎯 Iteration {iteration}: Model refinement completed!")
            return results

        finally:
            if refined_model is not None:
                del refined_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _save_iteration_summary(self, iteration: int, iteration_result: Dict, iteration_dir: Path):
        """Save summary for individual iteration"""
        summary = {
            'iteration': iteration,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pseudo_labels_extracted': iteration_result['pseudo_labels_count'],
            'dataset_statistics': iteration_result['dataset_stats'],
            'model_paths': {
                'input_model': iteration_result['input_model_path'],
                'output_model': iteration_result['refined_model_path']
            },
            'annotated_video_path': iteration_result.get('annotated_video_path'),
            'retrained': iteration_result['retrained']
        }

        summary_path = iteration_dir / "iteration_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"📄 Iteration {iteration} summary saved: {summary_path}")

    def _create_overall_summary(self, iteration_results: List[Dict]) -> Dict:
        """Create overall summary across all iterations"""
        if not iteration_results:
            return {'iterations': 0, 'results': []}

        pseudo_labels_progression = [result['pseudo_labels_count'] for result in iteration_results]
        best_iteration_idx = max(range(len(pseudo_labels_progression)), key=lambda i: pseudo_labels_progression[i])
        best_iteration = iteration_results[best_iteration_idx]

        model_path_progression = []
        video_path_progression = []

        for result in iteration_results:
            model_path_progression.append({
                'iteration': result['iteration'],
                'input_model': result['input_model_path'],
                'output_model': result['refined_model_path']
            })


            if result.get('annotated_video_path'):
                video_path_progression.append({
                    'iteration': result['iteration'],
                    'video_path': result['annotated_video_path']
                })

        overall_summary = {
            'total_iterations': len(iteration_results),
            'pseudo_labels_progression': pseudo_labels_progression,
            'total_pseudo_labels': sum(pseudo_labels_progression),
            'model_path_progression': model_path_progression,
            'video_path_progression': video_path_progression,
            'best_iteration': {
                'iteration_number': best_iteration['iteration'],
                'pseudo_labels_count': best_iteration['pseudo_labels_count'],
                'model_path': best_iteration['refined_model_path'],
                'video_path': best_iteration.get('annotated_video_path')
            },
            'final_model_path': iteration_results[-1]['refined_model_path'],
            'final_video_path': iteration_results[-1].get('annotated_video_path'),
            'original_model_path': self.original_model_path,
            'all_iteration_results': iteration_results,
            'improvement_metrics': {
                'pseudo_labels_first_to_last': pseudo_labels_progression[-1] - pseudo_labels_progression[0] if len(pseudo_labels_progression) > 1 else 0,
                'average_pseudo_labels_per_iteration': sum(pseudo_labels_progression) / len(pseudo_labels_progression)
            }
        }

        return overall_summary

    def _save_overall_summary(self, overall_results: Dict):
        """Save overall summary across all iterations"""
        summary_path = self.output_dir / "overall_refinement_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(overall_results, f, indent=2, default=str)

        self.logger.info(f"📊 Overall refinement summary saved: {summary_path}")

    def _print_video_summary(self, overall_results: Dict):
        """Print summary of created videos"""
        print("\n" + "="*80)
        print("🎬 VIDEO ANNOTATION SUMMARY")
        print("="*80)

        # Baseline video
        if overall_results.get('baseline_video_path'):
            print(f"📹 Baseline video: {overall_results['baseline_video_path']}")

        # Iteration videos
        video_progression = overall_results.get('video_path_progression', [])
        if video_progression:
            print(f"📹 Iteration videos:")
            for video_info in video_progression:
                print(f"   • Iteration {video_info['iteration']}: {video_info['video_path']}")

        # Best iteration video
        best_video = overall_results.get('best_iteration', {}).get('video_path')
        if best_video:
            iteration_num = overall_results['best_iteration']['iteration_number']
            print(f"🏆 Best performing iteration {iteration_num} video: {best_video}")

        print("="*80)

    def _run_automatic_evaluation(self, overall_results: Dict) -> Optional[Dict]:
        """Run automatic evaluation comparing original vs final refined model"""
        try:
            from evaluate_refinement import RefinementEvaluator
            
            # Get final refined model path
            final_model_path = overall_results.get('final_model_path')
            if not final_model_path or not Path(final_model_path).exists():
                self.logger.error("❌ Final refined model not found for evaluation")
                return None
                
            self.logger.info(f"📊 Comparing original model vs final refined model...")
            self.logger.info(f"   Original: {self.original_model_path}")
            self.logger.info(f"   Refined:  {final_model_path}")
            
            # Create evaluation output directory
            eval_output_dir = self.output_dir / "automatic_evaluation_results"
            eval_output_dir.mkdir(exist_ok=True)
            
            # Create evaluator
            evaluator = RefinementEvaluator(
                original_model_path=self.original_model_path,
                refined_model_path=final_model_path
            )
            
            # Run evaluation (no video creation - domain adaptation already creates videos)
            self.logger.info(f"📹 Processing video for evaluation metrics...")
            evaluation_results = evaluator.evaluate_on_video(
                video_path=self.real_video_path,
                output_dir=str(eval_output_dir),
                sample_rate=self.evaluation_sample_rate,
                create_videos=False  # Videos already created by domain adaptation
            )
            
            # Add evaluation info to results
            evaluation_results['evaluation_config'] = {
                'sample_rate': self.evaluation_sample_rate,
                'original_model': self.original_model_path,
                'refined_model': final_model_path,
                'video_path': self.real_video_path,
                'output_dir': str(eval_output_dir)
            }
            
            self.logger.info("✅ Automatic evaluation completed!")
            self.logger.info(f"   📊 Evaluation results saved to: {eval_output_dir}")
            
            # Print quick summary
            self._print_evaluation_summary(evaluation_results)
            
            return evaluation_results
            
        except ImportError as e:
            self.logger.error(f"❌ Cannot run automatic evaluation - import failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ Automatic evaluation failed: {e}")
            self.logger.error("   You can run evaluation manually later with:")
            self.logger.error(f"   python domain_adaptation/evaluate_refinement.py domain_adaptation/config.yaml")
            return None
            
    def _print_evaluation_summary(self, evaluation_results: Dict):
        """Print quick summary of evaluation results"""
        try:
            summary = evaluation_results.get('summary', {})
            original_stats = summary.get('original_model', {}).get('stats', {})
            refined_stats = summary.get('refined_model', {}).get('stats', {})
            improvements = summary.get('improvements', {})
            
            print(f"\n📊 AUTOMATIC EVALUATION SUMMARY:")
            print(f"   Original model avg confidence: {original_stats.get('avg_confidence', 0):.3f}")
            print(f"   Refined model avg confidence:  {refined_stats.get('avg_confidence', 0):.3f}")
            print(f"   Confidence improvement:        {improvements.get('avg_confidence_improvement', 0):+.3f}")
            
            assessment = summary.get('evaluation_assessment', {}).get('overall_improvement', 'unknown')
            if assessment == 'significant':
                print(f"   ✅ Significant improvement achieved!")
            elif assessment == 'moderate':
                print(f"   🟡 Moderate improvement achieved")
            else:
                print(f"   🔴 Limited improvement observed")
                
        except Exception as e:
            self.logger.warning(f"Could not print evaluation summary: {e}")


def main():
    parser = argparse.ArgumentParser(description="Domain Adaptation with Video Annotation")
    parser.add_argument("config", default="../config.yaml", help="Path to configuration YAML file")
    parser.add_argument("--skip-videos", action="store_true", help="Skip video annotation (faster)")

    args = parser.parse_args()

    try:
        # Create domain adaptation instance
        da = DomainAdaptation(config_path=args.config)

        # Disable video annotation if requested
        if args.skip_videos:
            da.enable_video_annotation = False
            print("🎬 Video annotation disabled")

        # Run domain adaptation
        results = da.run_domain_adaptation(retrain=True)

        if results:
            print(f"\n🎯 Domain adaptation completed!")
            if results.get('final_model_path'):
                print(f"   🤖 Final model: {results['final_model_path']}")
            if results.get('final_video_path'):
                print(f"   🎬 Final video: {results['final_video_path']}")

    except FileNotFoundError as e:
        print(f"❌ Configuration or input file not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Domain adaptation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()