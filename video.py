"""
Generic Video Annotator - Reusable video annotation functionality
Can be used as standalone script or imported as module
"""

import argparse
import os
import cv2
import torch
from ultralytics import YOLO
from typing import Optional, Callable, Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_default_device():
    """
    Automatically detect the best available PyTorch device for inference.
    
    Prioritizes CUDA GPU > Apple MPS > CPU for optimal performance.
    Essential for handling large YOLO models efficiently.
    
    Returns:
        str: Device string ('cuda:0', 'mps', or 'cpu')
    """
    return 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


class VideoAnnotator:
    """
    Reusable video annotation class for YOLO pose estimation models.
    
    Provides a clean interface for processing surgical videos with YOLO models,
    supporting both interactive display and batch processing modes.
    Designed for integration with domain adaptation pipelines.
    
    Features:
        - Automatic device detection (GPU/CPU)
        - Progress tracking and callbacks
        - Memory-efficient video processing
        - Interactive playback controls (space to pause, 'q' to quit)
        - Comprehensive statistics collection
    
    Example:
        >>> annotator = VideoAnnotator('model.pt')
        >>> stats = annotator.annotate_video('input.mp4', 'output.mp4')
        >>> print(f"Processed {stats['processed_frames']} frames")
    """

    def __init__(self, model_path: str, device: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the video annotator

        Args:
            model_path: Path to YOLO model
            device: Device to run on (auto-detected if None)
            logger: Logger instance (creates new if None)
        """
        self.device = device or get_default_device()
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # Load model
        self.logger.info(f"Loading model from '{model_path}' on device '{self.device}'")
        self.model = YOLO(model_path).eval().to(self.device)

    def annotate_video(self,
                      video_path: str,
                      output_path: str,
                      conf_thresh: float = 0.5,
                      save_video: bool = True,
                      display_video: bool = False,
                      progress_callback: Optional[Callable[[int, int, float], None]] = None,
                      frame_callback: Optional[Callable[[int, Any], None]] = None) -> Dict[str, Any]:
        """
        Annotate a video with YOLO detections

        Args:
            video_path: Input video path
            output_path: Output video path
            conf_thresh: Confidence threshold for detections
            save_video: Whether to save annotated video
            display_video: Whether to display video (for interactive use)
            progress_callback: Callback function for progress updates (frame_idx, total_frames, progress_pct)
            frame_callback: Callback function for each frame (frame_idx, results)

        Returns:
            Dictionary with annotation results and statistics
        """
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_path}")

        try:
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.logger.info(f"Processing video: {total_frames} frames at {fps} FPS ({width}x{height})")

            # Setup video writer if saving
            video_writer = None
            if save_video:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                self.logger.info(f"Will save annotated video to: {output_path}")

            # Setup display window if needed
            window_name = None
            if display_video:
                window_name = "YOLO Video Annotation"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, min(1280, width), min(720, height))

            # Processing statistics
            stats = {
                'total_frames': total_frames,
                'processed_frames': 0,
                'total_detections': 0,
                'fps': fps,
                'resolution': (width, height)
            }

            frames_processed = 0
            play = True
            last_frame = None

            while True:
                if play or last_frame is None:
                    ret, frame = cap.read()
                    if not ret:
                        self.logger.info("End of video stream reached.")
                        break

                    # Run inference
                    results = self.model(frame, conf=conf_thresh, verbose=False)[0]

                    # Count detections
                    if results.boxes is not None:
                        stats['total_detections'] += len(results.boxes)

                    # Draw annotations
                    annotated = results.plot()

                    # Save frame to video if writer is active
                    if video_writer is not None:
                        video_writer.write(annotated)

                    frames_processed += 1
                    stats['processed_frames'] = frames_processed

                    # Progress callback
                    if progress_callback and frames_processed % 30 == 0:
                        progress_pct = (frames_processed / total_frames) * 100 if total_frames > 0 else 0
                        progress_callback(frames_processed, total_frames, progress_pct)

                    # Frame callback
                    if frame_callback:
                        frame_callback(frames_processed, results)

                    last_frame = annotated
                else:
                    annotated = last_frame

                # Display if requested - provides interactive controls for review
                if display_video and window_name:
                    cv2.imshow(window_name, annotated)

                    # Handle keyboard input for playback control
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):  # 'q' key to quit
                        self.logger.info("User quit video annotation.")
                        break
                    elif key == 32:  # Space bar to pause/resume
                        play = not play
                        self.logger.info("Paused" if not play else "Resumed")

                # For batch processing mode, continue without user interaction
                elif not display_video:
                    continue

            # Final statistics
            stats['avg_detections_per_frame'] = stats['total_detections'] / max(1, stats['processed_frames'])

            return stats

        finally:
            # Cleanup
            cap.release()
            if video_writer is not None:
                video_writer.release()
                self.logger.info(f"Annotated video saved to: {output_path}")
            if display_video:
                cv2.destroyAllWindows()

    def annotate_video_simple(self, video_path: str, output_path: str, conf_thresh: float = 0.5) -> Dict[str, Any]:
        """
        Simple video annotation optimized for batch processing.
        
        Processes video without display or interactive features, making it
        perfect for domain adaptation pipelines where you need to annotate
        multiple videos automatically.
        
        Args:
            video_path: Path to input video file
            output_path: Path where annotated video will be saved
            conf_thresh: Confidence threshold for detections (0.0-1.0)
            
        Returns:
            Dictionary containing processing statistics:
            - total_frames: Total frames in video
            - processed_frames: Successfully processed frames
            - total_detections: Sum of all detections
            - avg_detections_per_frame: Average detections per frame
            - fps: Original video frame rate
            - resolution: (width, height) tuple
        """
        return self.annotate_video(
            video_path=video_path,
            output_path=output_path,
            conf_thresh=conf_thresh,
            save_video=True,
            display_video=False
        )


def annotate_video_standalone(model_path: str,
                            video_path: str,
                            output_path: str,
                            conf_thresh: float = 0.5,
                            display: bool = False) -> Dict[str, Any]:
    """
    Standalone function for video annotation without class instantiation.
    
    Convenient wrapper for one-off video processing tasks.
    Creates VideoAnnotator instance internally and handles cleanup automatically.
    
    Args:
        model_path: Path to YOLO .pt model file
        video_path: Path to input video file  
        output_path: Path for annotated output video
        conf_thresh: Detection confidence threshold (default: 0.5)
        display: Whether to show interactive video player (default: False)
        
    Returns:
        Dictionary with processing statistics and performance metrics
        
    Example:
        >>> stats = annotate_video_standalone(
        ...     'yolo11n-pose.pt', 
        ...     'surgery.mp4', 
        ...     'annotated_surgery.mp4',
        ...     conf_thresh=0.8
        ... )
    """
    annotator = VideoAnnotator(model_path)
    return annotator.annotate_video(
        video_path=video_path,
        output_path=output_path,
        conf_thresh=conf_thresh,
        save_video=True,
        display_video=display
    )


def parse_args():
    """
    Parse command line arguments for standalone script execution.
    
    Provides comprehensive CLI interface for video annotation with all
    major options accessible from command line.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Run a YOLO model on a video and display the detections live.")
    parser.add_argument("--model_path", required=True, help="Path to the YOLO .pt model file.")
    parser.add_argument("--video_path", required=True, help="Path to the input video file.")
    parser.add_argument("--conf_thresh", type=float, default=0.5, help="Confidence threshold for detections (default: 0.5).")
    parser.add_argument("--save_dir", type=str, default="./", help="Location to save frames into")
    parser.add_argument("--save_video", action="store_true", help="Save the annotated video with '_annotated' suffix.")
    parser.add_argument("--display", action="store_true", help="Display video during processing.")
    parser.add_argument("--width", type=int, default=1280, help="Width for display (default: 1280).")
    parser.add_argument("--height", type=int, default=720, help="Height for display (default: 720).")

    return parser.parse_args()


def main():
    """
    Main entry point for standalone video annotation script.
    
    Handles command line argument parsing, creates VideoAnnotator instance,
    processes video with progress tracking, and reports final statistics.
    
    Exit codes:
        0: Success
        1: Error during processing
    """
    args = parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Create annotator
    annotator = VideoAnnotator(args.model_path)

    # Generate output path
    if args.save_video:
        input_basename = os.path.basename(args.video_path)
        input_name, input_ext = os.path.splitext(input_basename)
        output_path = os.path.join(args.save_dir, f"{input_name}_annotated{input_ext}")
    else:
        output_path = None

    # Progress callback for standalone use
    def progress_callback(frame_idx, total_frames, progress_pct):
        logger.info(f"Processed {frame_idx}/{total_frames} frames ({progress_pct:.1f}%)")

    # Frame saving callback
    def frame_callback(frame_idx, results):
        # Handle 's' key press for frame saving would need more complex setup
        pass

    # Annotate video
    try:
        stats = annotator.annotate_video(
            video_path=args.video_path,
            output_path=output_path,
            conf_thresh=args.conf_thresh,
            save_video=args.save_video,
            display_video=args.display,
            progress_callback=progress_callback if not args.display else None
        )

        logger.info("Video annotation completed!")
        logger.info(f"   Processed {stats['processed_frames']} frames")
        logger.info(f"   Total detections: {stats['total_detections']}")
        logger.info(f"   Avg detections per frame: {stats['avg_detections_per_frame']:.2f}")
        if args.save_video:
            logger.info(f"   Saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error during video annotation: {e}")
        raise


if __name__ == "__main__":
    main()