"""
Enhanced evaluation script for comparing model performance before and after domain adaptation
"""

import cv2
import json
from utils.yaml_utils import load_yaml
import sys
import logging
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import argparse


@dataclass
class DetectionStats:
    """Data class for detection statistics"""
    num_detections: int
    avg_confidence: float
    high_conf_detections: int  # confidence > 0.7
    max_confidence: float
    min_confidence: float


@dataclass
class FrameAnalysis:
    """Data class for frame-by-frame analysis"""
    frame_id: int
    original_stats: DetectionStats
    refined_stats: DetectionStats


class ModelEvaluator:
    """Base class for model evaluation"""

    def __init__(self, model_path: str, model_name: str):
        self.model = YOLO(model_path)
        self.model_name = model_name
        self.model_path = model_path
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{model_name}")

    def extract_detection_stats(self, result) -> DetectionStats:
        """Extract detection statistics from YOLO result"""
        if result.boxes is None or len(result.boxes) == 0:
            return DetectionStats(0, 0.0, 0, 0.0, 0.0)

        confidences = result.boxes.conf.cpu().numpy()

        return DetectionStats(
            num_detections=len(confidences),
            avg_confidence=float(np.mean(confidences)),
            high_conf_detections=int(np.sum(confidences > 0.7)),
            max_confidence=float(np.max(confidences)),
            min_confidence=float(np.min(confidences))
        )

    def run_inference(self, frame: np.ndarray):
        """Run inference on a single frame"""
        return self.model(frame, verbose=False)[0]


class RefinementEvaluator:
    """
    Enhanced evaluator for comparing models before and after domain adaptation
    """

    def __init__(self, original_model_path: str, refined_model_path: str):
        """
        Initialize evaluator with original and refined models

        Args:
            original_model_path: Path to model trained only on synthetic data (Phase 2)
            refined_model_path: Path to refined model after domain adaptation (Phase 3)
        """
        self._setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize models
        self.original_model = ModelEvaluator(original_model_path, "original")
        self.refined_model = ModelEvaluator(refined_model_path, "refined")

        self.logger.info(f"✅ Loaded original model: {original_model_path}")
        self.logger.info(f"✅ Loaded refined model: {refined_model_path}")

    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    def evaluate_on_video(self, video_path: str, output_dir: str = "evaluation_results",
                         sample_rate: int = 1, create_videos: bool = False) -> Dict:
        """
        Evaluate model performance by comparing predictions (without creating videos by default)

        Args:
            video_path: Path to real surgical video
            output_dir: Directory to save evaluation results
            sample_rate: Process every nth frame (1 = every frame)
            create_videos: Whether to create annotated videos (False by default to avoid duplication)

        Returns:
            Dictionary containing evaluation results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        self.logger.info(f"📊 Evaluating model performance on video: {video_path}")
        if not create_videos:
            self.logger.info("📹 Video creation disabled (handled by domain adaptation pipeline)")

        # Process video for metrics only (no video creation by default)
        results = self._process_video_metrics_only(video_path, sample_rate, create_videos, output_dir)

        # Generate comparison report
        comparison_report = self._generate_comparison_report(results, output_dir)

        # Save detailed results
        self._save_detailed_results(results, output_dir)

        self.logger.info(f"✅ Evaluation completed! Metrics saved in: {output_dir}")
        return comparison_report
    def evaluate_iterative_refinement(self, video_path: str, output_dir: str = "evaluation_results",
                                     sample_rate: int = 1, evaluate_all_iterations: bool = False) -> Dict:
        """
        Evaluate iterative refinement results, optionally testing all iterations

        Args:
            video_path: Path to real surgical video
            output_dir: Directory to save evaluation results
            sample_rate: Process every nth frame (1 = every frame)
            evaluate_all_iterations: Whether to evaluate all iterations or just final

        Returns:
            Dictionary containing evaluation results across iterations
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        self.logger.info(f"🎬 Evaluating iterative refinement on video: {video_path}")

        if evaluate_all_iterations:
            return self._evaluate_all_iterations(video_path, output_path, sample_rate)
        else:
            return self._evaluate_final_iteration(video_path, output_path, sample_rate)

    def _evaluate_final_iteration(self, video_path: str, output_dir: Path, sample_rate: int) -> Dict:
        """Evaluate only the final refined model vs original"""
        # This is the existing evaluation logic
        results = self._process_video(video_path, output_dir, sample_rate)
        comparison_report = self._generate_comparison_report(results, output_dir)
        self._save_detailed_results(results, output_dir)

        self.logger.info(f"✅ Final model evaluation completed! Results saved in: {output_dir}")
        return comparison_report

    def _evaluate_all_iterations(self, video_path: str, output_dir: Path, sample_rate: int) -> Dict:
        """Evaluate all iterations to show progression"""
        # Find all iteration models
        iteration_models = self._find_iteration_models()

        if not iteration_models:
            self.logger.warning("No iteration models found, falling back to final evaluation")
            return self._evaluate_final_iteration(video_path, output_dir, sample_rate)

        self.logger.info(f"🔍 Found {len(iteration_models)} iteration models to evaluate")

        # Evaluate each iteration
        iteration_results = {}
        for iteration_num, model_path in iteration_models.items():
            self.logger.info(f"📊 Evaluating iteration {iteration_num}...")

            # Create iteration-specific evaluator
            iter_evaluator = ModelEvaluator(model_path, f"iteration_{iteration_num}")

            # Process video for this iteration
            iter_results = self._process_video_for_iteration(
                video_path, iter_evaluator, iteration_num, sample_rate
            )
            iteration_results[iteration_num] = iter_results

        # Create comprehensive progression report
        progression_report = self._generate_progression_report(iteration_results, output_dir)

        # Save iteration-specific results
        self._save_progression_results(iteration_results, progression_report, output_dir)

        self.logger.info(f"✅ All iterations evaluation completed! Results saved in: {output_dir}")
        return progression_report

    def _find_iteration_models(self) -> Dict[int, str]:
        """Find all iteration models in the output directory"""
        iteration_models = {}

        # Get the base output directory from the refined model path
        refined_path = Path(self.refined_model.model_path)

        # Determine base directory based on the path structure
        if 'iteration_' in str(refined_path):
            # Path is like: /base/output/iteration_N/refined_model/weights/best.pt
            # Find the part before iteration_N
            parts = refined_path.parts
            for i, part in enumerate(parts):
                if part.startswith('iteration_'):
                    base_dir = Path(*parts[:i])
                    break
            else:
                # Fallback if iteration_ not found in parts
                base_dir = refined_path.parent.parent.parent.parent
        else:
            # Path is like: /base/output/refined_model/weights/best.pt
            # So base directory is 3 levels up
            base_dir = refined_path.parent.parent.parent

        self.logger.info(f"🔍 Searching for iteration models in: {base_dir}")

        # Find all iteration directories
        if not base_dir.exists():
            self.logger.warning(f"Base directory not found: {base_dir}")
            return iteration_models

        for iteration_dir in base_dir.glob("iteration_*"):
            if iteration_dir.is_dir():
                try:
                    iteration_num = int(iteration_dir.name.split("_")[1])
                    model_path = iteration_dir / "refined_model" / "weights" / "best.pt"

                    if model_path.exists():
                        iteration_models[iteration_num] = str(model_path)
                        self.logger.debug(f"Found iteration {iteration_num} model: {model_path}")
                except (ValueError, IndexError):
                    self.logger.warning(f"Invalid iteration directory name: {iteration_dir.name}")
                    continue

        self.logger.info(f"📊 Found {len(iteration_models)} iteration models")
        return iteration_models

    def _process_video_for_iteration(self, video_path: str, evaluator: ModelEvaluator,
                                   iteration_num: int, sample_rate: int) -> Dict:
        """Process video for a specific iteration model"""
        cap = cv2.VideoCapture(video_path)
        frame_analyses = []
        frame_count = 0
        processed_count = 0

        with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                 desc=f"Processing iteration {iteration_num}") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % sample_rate == 0:
                    # Run inference and extract stats
                    result = evaluator.run_inference(frame)
                    stats = evaluator.extract_detection_stats(result)

                    frame_analyses.append({
                        'frame_id': frame_count,
                        'stats': stats
                    })
                    processed_count += 1

                frame_count += 1
                pbar.update(1)

        cap.release()

        return {
            'iteration': iteration_num,
            'model_path': evaluator.model_path,
            'frame_analyses': frame_analyses,
            'processed_frames': processed_count
        }

    def _generate_progression_report(self, iteration_results: Dict, output_dir: Path) -> Dict:
        """Generate comprehensive progression report across all iterations"""

        # Calculate summary stats for each iteration
        iteration_summaries = {}
        for iteration_num, results in iteration_results.items():
            frame_analyses = results['frame_analyses']

            if frame_analyses:
                all_stats = [fa['stats'] for fa in frame_analyses]
                iteration_summaries[iteration_num] = {
                    'avg_detections': np.mean([s.num_detections for s in all_stats]),
                    'avg_confidence': np.mean([s.avg_confidence for s in all_stats if s.avg_confidence > 0]),
                    'avg_high_conf': np.mean([s.high_conf_detections for s in all_stats]),
                    'detection_rate': sum(1 for s in all_stats if s.num_detections > 0) / len(all_stats),
                    'total_frames': len(all_stats)
                }

        # Calculate progression metrics
        progression_metrics = self._calculate_progression_metrics(iteration_summaries)

        # Calculate improvements
        iterations = sorted(iteration_summaries.keys())
        if len(iterations) > 1:
            first_iteration_stats = iteration_summaries[iterations[0]]
            final_iteration_stats = iteration_summaries[iterations[-1]]

            overall_improvement = {
                'first_to_final_confidence': final_iteration_stats['avg_confidence'] - first_iteration_stats['avg_confidence'],
                'first_to_final_detections': final_iteration_stats['avg_detections'] - first_iteration_stats['avg_detections'],
                'best_iteration': max(iterations, key=lambda k: iteration_summaries[k]['avg_confidence'])
            }
        else:
            overall_improvement = {
                'first_to_final_confidence': 0,
                'first_to_final_detections': 0,
                'best_iteration': iterations[0] if iterations else 1
            }

        progression_report = {
            'evaluation_type': 'iterative_progression',
            'total_iterations': len(iteration_summaries),
            'original_model_path': self.original_model.model_path,
            'iteration_summaries': iteration_summaries,
            'progression_metrics': progression_metrics,
            'overall_improvement': overall_improvement
        }

        return progression_report

    def _calculate_progression_metrics(self, iteration_summaries: Dict) -> Dict:
        """Calculate progression metrics across iterations"""
        iterations = sorted(iteration_summaries.keys())

        confidence_progression = [iteration_summaries[i]['avg_confidence'] for i in iterations]
        detection_progression = [iteration_summaries[i]['avg_detections'] for i in iterations]

        return {
            'confidence_progression': confidence_progression,
            'detection_progression': detection_progression,
            'confidence_trend': 'improving' if len(confidence_progression) > 1 and confidence_progression[-1] > confidence_progression[0] else 'stable',
            'detection_trend': 'improving' if len(detection_progression) > 1 and detection_progression[-1] > detection_progression[0] else 'stable',
            'peak_performance_iteration': max(iterations, key=lambda i: iteration_summaries[i]['avg_confidence'])
        }

    def _save_progression_results(self, iteration_results: Dict, progression_report: Dict, output_dir: Path):
        """Save detailed progression results"""

        # Save progression report
        with open(output_dir / "iterative_progression_report.json", 'w') as f:
            json.dump(progression_report, f, indent=2, default=str)

        # Save detailed iteration results
        with open(output_dir / "detailed_iteration_results.json", 'w') as f:
            # Convert to serializable format
            serializable_results = {}
            for iteration_num, results in iteration_results.items():
                serializable_results[iteration_num] = {
                    'iteration': results['iteration'],
                    'model_path': results['model_path'],
                    'processed_frames': results['processed_frames'],
                    'frame_analyses': [
                        {
                            'frame_id': fa['frame_id'],
                            'detections': fa['stats'].num_detections,
                            'avg_confidence': fa['stats'].avg_confidence,
                            'high_conf_detections': fa['stats'].high_conf_detections
                        }
                        for fa in results['frame_analyses']
                    ]
                }
            json.dump(serializable_results, f, indent=2)

        # Print progression summary
        self._print_progression_evaluation_summary(progression_report)

    def _print_progression_evaluation_summary(self, progression_report: Dict):
        """Print formatted summary of iterative progression evaluation"""
        print("\n" + "="*80)
        print("📈 ITERATIVE REFINEMENT EVALUATION SUMMARY")
        print("="*80)

        summaries = progression_report['iteration_summaries']
        metrics = progression_report['progression_metrics']
        improvements = progression_report['overall_improvement']

        print(f"🔍 Evaluated {progression_report['total_iterations']} iterations")
        print(f"🎯 Best performing iteration: {improvements['best_iteration']}")
        print(f"📊 Peak performance iteration: {metrics['peak_performance_iteration']}")
        print()

        print("📈 PERFORMANCE PROGRESSION:")
        iterations = sorted(summaries.keys())
        for iteration in iterations:
            stats = summaries[iteration]
            indicator = "🥇" if iteration == improvements['best_iteration'] else "📍"
            print(f"   {indicator} Iteration {iteration}:")
            print(f"      • Avg confidence: {stats['avg_confidence']:.3f}")
            print(f"      • Avg detections/frame: {stats['avg_detections']:.2f}")
            print(f"      • Detection rate: {stats['detection_rate']:.1%}")

        print()
        print("📊 OVERALL TRENDS:")
        print(f"   • Confidence trend: {metrics['confidence_trend']} 📈" if metrics['confidence_trend'] == 'improving' else f"   • Confidence trend: {metrics['confidence_trend']} 📉")
        print(f"   • Detection trend: {metrics['detection_trend']} 📈" if metrics['detection_trend'] == 'improving' else f"   • Detection trend: {metrics['detection_trend']} 📉")

        if len(iterations) > 1:
            conf_improvement = improvements['first_to_final_confidence']
            det_improvement = improvements['first_to_final_detections']

            print()
            print("🎯 FIRST vs FINAL ITERATION:")
            print(f"   • Confidence improvement: {conf_improvement:+.3f}")
            print(f"   • Detection improvement: {det_improvement:+.2f}")

            if conf_improvement > 0.02:
                print("   ✅ Significant improvement achieved through iterations!")
            elif conf_improvement > 0.005:
                print("   🟡 Moderate improvement through iterations")
            else:
                print("   🔴 Limited improvement - consider adjusting iteration parameters")

        print("="*80)

    def _process_video_metrics_only(self, video_path: str, sample_rate: int, create_videos: bool, output_dir: Path) -> Dict:
        """Process video and collect statistics without creating videos by default"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.info(f"Video properties: {width}x{height}, {fps}fps, {total_frames} frames")

        # Create video writers only if requested
        video_writers = None
        if create_videos:
            video_writers = self._create_video_writers(output_dir, fps, width, height)
            self.logger.info("📹 Creating annotated videos...")

        # Process frames
        frame_analyses = []
        frame_count = 0
        processed_count = 0

        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample frames based on sample_rate
                if frame_count % sample_rate == 0:
                    analysis = self._analyze_frame(frame, frame_count)
                    frame_analyses.append(analysis)

                    # Write annotated frames only if video creation is enabled
                    if create_videos and video_writers:
                        self._write_annotated_frames(frame, analysis, video_writers)
                    processed_count += 1

                frame_count += 1
                pbar.update(1)

        # Cleanup
        cap.release()
        if video_writers:
            for writer in video_writers.values():
                writer.release()

        self.logger.info(f"Processed {processed_count} frames out of {total_frames}")

        return {
            'frame_analyses': frame_analyses,
            'video_info': {
                'total_frames': total_frames,
                'processed_frames': processed_count,
                'fps': fps,
                'resolution': (width, height),
                'sample_rate': sample_rate,
                'videos_created': create_videos
            }
        }

    def _process_video(self, video_path: str, output_dir: Path, sample_rate: int) -> Dict:
        """Process video and collect statistics"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.logger.info(f"Video properties: {width}x{height}, {fps}fps, {total_frames} frames")

        # Create video writers
        video_writers = self._create_video_writers(output_dir, fps, width, height)

        # Process frames
        frame_analyses = []
        frame_count = 0
        processed_count = 0

        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Sample frames based on sample_rate
                if frame_count % sample_rate == 0:
                    analysis = self._analyze_frame(frame, frame_count)
                    frame_analyses.append(analysis)

                    # Annotate and write frames
                    self._write_annotated_frames(frame, analysis, video_writers)
                    processed_count += 1

                frame_count += 1
                pbar.update(1)

        # Cleanup
        cap.release()
        for writer in video_writers.values():
            writer.release()

        self.logger.info(f"Processed {processed_count} frames out of {total_frames}")

        return {
            'frame_analyses': frame_analyses,
            'video_info': {
                'total_frames': total_frames,
                'processed_frames': processed_count,
                'fps': fps,
                'resolution': (width, height),
                'sample_rate': sample_rate
            }
        }

    def _create_video_writers(self, output_dir: Path, fps: int, width: int, height: int) -> Dict:
        """Create video writers for output videos"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        return {
            'original': cv2.VideoWriter(
                str(output_dir / "results_synthetic_only.mp4"),
                fourcc, fps, (width, height)
            ),
            'refined': cv2.VideoWriter(
                str(output_dir / "results_refined.mp4"),
                fourcc, fps, (width, height)
            )
        }

    def _analyze_frame(self, frame: np.ndarray, frame_id: int) -> FrameAnalysis:
        """Analyze a single frame with both models"""
        # Run inference with both models
        original_result = self.original_model.run_inference(frame)
        refined_result = self.refined_model.run_inference(frame)

        # Extract statistics
        original_stats = self.original_model.extract_detection_stats(original_result)
        refined_stats = self.refined_model.extract_detection_stats(refined_result)

        return FrameAnalysis(
            frame_id=frame_id,
            original_stats=original_stats,
            refined_stats=refined_stats
        )

    def _write_annotated_frames(self, frame: np.ndarray, analysis: FrameAnalysis,
                               video_writers: Dict):
        """Write annotated frames to output videos"""
        # Run inference again to get annotated frames
        original_result = self.original_model.run_inference(frame)
        refined_result = self.refined_model.run_inference(frame)

        # Get annotated frames
        original_annotated = original_result.plot()
        refined_annotated = refined_result.plot()

        # Write to video files
        video_writers['original'].write(original_annotated)
        video_writers['refined'].write(refined_annotated)

    def _calculate_summary_stats(self, frame_analyses: List[FrameAnalysis]) -> Tuple[Dict, Dict]:
        """Calculate summary statistics from frame analyses"""
        if not frame_analyses:
            empty_stats = {
                'avg_detections': 0, 'avg_confidence': 0, 'avg_high_conf': 0,
                'detection_stability': 0, 'frames_with_detections': 0, 'detection_rate': 0
            }
            return empty_stats, empty_stats

        # Extract data for original model
        original_detections = [a.original_stats.num_detections for a in frame_analyses]
        original_confidences = [a.original_stats.avg_confidence for a in frame_analyses
                               if a.original_stats.avg_confidence > 0]
        original_high_conf = [a.original_stats.high_conf_detections for a in frame_analyses]

        # Extract data for refined model
        refined_detections = [a.refined_stats.num_detections for a in frame_analyses]
        refined_confidences = [a.refined_stats.avg_confidence for a in frame_analyses
                              if a.refined_stats.avg_confidence > 0]
        refined_high_conf = [a.refined_stats.high_conf_detections for a in frame_analyses]

        # Calculate statistics
        original_stats = self._compute_stats(original_detections, original_confidences, original_high_conf)
        refined_stats = self._compute_stats(refined_detections, refined_confidences, refined_high_conf)

        return original_stats, refined_stats

    def _compute_stats(self, detections: List[int], confidences: List[float],
                      high_conf: List[int]) -> Dict:
        """Compute statistics for a set of detections"""
        return {
            'avg_detections': np.mean(detections) if detections else 0,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'avg_high_conf': np.mean(high_conf) if high_conf else 0,
            'detection_stability': np.std(detections) if detections else 0,
            'frames_with_detections': sum(1 for d in detections if d > 0),
            'detection_rate': sum(1 for d in detections if d > 0) / len(detections) if detections else 0,
            'max_detections': max(detections) if detections else 0,
            'total_frames': len(detections)
        }

    def _generate_comparison_report(self, results: Dict, output_dir: Path) -> Dict:
        """Generate a comprehensive comparison report"""
        frame_analyses = results['frame_analyses']
        video_info = results['video_info']

        # Calculate summary statistics
        original_stats, refined_stats = self._calculate_summary_stats(frame_analyses)

        # Calculate improvements
        improvements = self._calculate_improvements(original_stats, refined_stats)

        # Create comprehensive report
        report = {
            'summary': {
                'video_info': video_info,
                'original_model': {
                    'path': self.original_model.model_path,
                    'stats': original_stats
                },
                'refined_model': {
                    'path': self.refined_model.model_path,
                    'stats': refined_stats
                },
                'improvements': improvements,
                'evaluation_assessment': self._assess_improvements(improvements)
            },
            'detailed_results': {
                'frame_count': len(frame_analyses),
                'sample_rate': video_info['sample_rate']
            }
        }

        # Save report
        report_path = output_dir / "comparison_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Print summary
        self._print_comparison_summary(report)

        return report

    def _calculate_improvements(self, original_stats: Dict, refined_stats: Dict) -> Dict:
        """Calculate improvements between original and refined models"""
        return {
            'avg_detections_improvement': refined_stats['avg_detections'] - original_stats['avg_detections'],
            'avg_confidence_improvement': refined_stats['avg_confidence'] - original_stats['avg_confidence'],
            'high_conf_detection_improvement': refined_stats['avg_high_conf'] - original_stats['avg_high_conf'],
            'detection_rate_improvement': refined_stats['detection_rate'] - original_stats['detection_rate'],
            'stability_improvement': original_stats['detection_stability'] - refined_stats['detection_stability']  # Lower is better
        }

    def _assess_improvements(self, improvements: Dict) -> Dict:
        """Assess the quality of improvements"""
        confidence_improvement = improvements['avg_confidence_improvement']
        detection_improvement = improvements['avg_detections_improvement']

        assessment = {
            'overall_improvement': 'significant' if confidence_improvement > 0.05 else
                                 'moderate' if confidence_improvement > 0.01 else 'limited',
            'confidence_assessment': 'improved' if confidence_improvement > 0 else 'degraded',
            'detection_count_assessment': 'improved' if detection_improvement > 0 else 'degraded'
        }

        return assessment

    def _save_detailed_results(self, results: Dict, output_dir: Path):
        """Save detailed frame-by-frame results"""
        frame_analyses = results['frame_analyses']

        # Convert to serializable format
        detailed_data = []
        for analysis in frame_analyses:
            detailed_data.append({
                'frame_id': analysis.frame_id,
                'original': {
                    'detections': analysis.original_stats.num_detections,
                    'avg_confidence': analysis.original_stats.avg_confidence,
                    'high_conf_detections': analysis.original_stats.high_conf_detections,
                    'max_confidence': analysis.original_stats.max_confidence,
                    'min_confidence': analysis.original_stats.min_confidence
                },
                'refined': {
                    'detections': analysis.refined_stats.num_detections,
                    'avg_confidence': analysis.refined_stats.avg_confidence,
                    'high_conf_detections': analysis.refined_stats.high_conf_detections,
                    'max_confidence': analysis.refined_stats.max_confidence,
                    'min_confidence': analysis.refined_stats.min_confidence
                }
            })

        # Save detailed results
        with open(output_dir / "detailed_frame_analysis.json", 'w') as f:
            json.dump(detailed_data, f, indent=2)

    def _print_comparison_summary(self, report: Dict):
        """Print a formatted comparison summary"""
        print("\n" + "="*70)
        print("🔍 MODEL COMPARISON SUMMARY")
        print("="*70)

        summary = report['summary']
        original = summary['original_model']['stats']
        refined = summary['refined_model']['stats']
        improvements = summary['improvements']
        assessment = summary['evaluation_assessment']

        print(f"📊 Total frames analyzed: {summary['video_info']['processed_frames']}")
        print(f"📹 Video resolution: {summary['video_info']['resolution']}")
        print(f"⏱️  Sample rate: every {summary['video_info']['sample_rate']} frame(s)")
        print()

        print("📈 DETECTION PERFORMANCE:")
        print(f"  Original model (synthetic only):")
        print(f"    • Avg detections/frame: {original['avg_detections']:.2f}")
        print(f"    • Avg confidence: {original['avg_confidence']:.3f}")
        print(f"    • Detection rate: {original['detection_rate']:.1%}")
        print(f"    • High-conf detections/frame: {original['avg_high_conf']:.2f}")
        print()

        print(f"  Refined model (domain adapted):")
        print(f"    • Avg detections/frame: {refined['avg_detections']:.2f}")
        print(f"    • Avg confidence: {refined['avg_confidence']:.3f}")
        print(f"    • Detection rate: {refined['detection_rate']:.1%}")
        print(f"    • High-conf detections/frame: {refined['avg_high_conf']:.2f}")
        print()

        print("📊 IMPROVEMENTS:")
        print(f"  • Detection count: {improvements['avg_detections_improvement']:+.2f}")
        print(f"  • Confidence: {improvements['avg_confidence_improvement']:+.3f}")
        print(f"  • High-conf detections: {improvements['high_conf_detection_improvement']:+.2f}")
        print(f"  • Detection rate: {improvements['detection_rate_improvement']:+.1%}")

        # Overall assessment
        overall = assessment['overall_improvement']
        if overall == 'significant':
            print("✅ Significant improvement achieved!")
        elif overall == 'moderate':
            print("🟡 Moderate improvement achieved")
        else:
            print("🔴 Limited improvement observed")

        print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Enhanced model refinement evaluation")
    parser.add_argument("config", help="Path to configuration YAML file")

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_yaml(args.config)

        # Extract paths from config
        original_model_path = config['paths']['model_path']
        refined_model_path = str(Path(config['paths']['output_dir']) / 'refined_model' / 'weights' / 'best.pt')
        video_path = config['paths']['real_video_path']
        output_dir = config['paths'].get('evaluation_dir', 'evaluation_results')
        sample_rate = config.get('evaluation', {}).get('sample_rate', 1)

        # Validate paths
        missing_paths = []
        for name, path in [("original model", original_model_path),
                          ("refined model", refined_model_path),
                          ("video", video_path)]:
            if not Path(path).exists():
                missing_paths.append(f"{name}: {path}")

        if missing_paths:
            print("❌ Missing required files:")
            for missing in missing_paths:
                print(f"   • {missing}")
            sys.exit(1)

        # Create evaluator and run evaluation
        evaluator = RefinementEvaluator(original_model_path, refined_model_path)
        results = evaluator.evaluate_on_video(
            video_path=video_path,
            output_dir=output_dir,
            sample_rate=sample_rate,
            create_videos=True  # Enable video creation when run manually
        )

        print(f"\n🎯 Evaluation complete! Check {output_dir} for:")
        print(f"   • results_synthetic_only.mp4 - Original model predictions")
        print(f"   • results_refined.mp4 - Refined model predictions")
        print(f"   • comparison_report.json - Detailed comparison metrics")
        print(f"   • detailed_frame_analysis.json - Frame-by-frame analysis")

    except FileNotFoundError as e:
        print(f"❌ Configuration or input file not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()