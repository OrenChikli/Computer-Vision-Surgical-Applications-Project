import argparse
import cv2
import torch
from ultralytics import YOLO

def get_default_device():
    """
    Returns the default torch device: GPU/MPS if available, else CPU.
    """
    return 'cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a YOLO model on an image and display the detections."
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the YOLO .pt model file."
    )
    parser.add_argument(
        "--image_path",
        required=True,
        help="Path to the input image file."
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (default: 0.5)."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Width for display (default: 1280)."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Height for display (default: 720)."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Load model
    device = get_default_device()
    print(f"Loading model from '{args.model_path}' on device '{device}'")
    model = YOLO(args.model_path).eval().to(device)

    # Read image
    image = cv2.imread(args.image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at '{args.image_path}'")
    image = cv2.resize(image, (args.width, args.height))

    # Run inference
    print(f"Running inference with confidence >= {args.conf_thresh}")
    results = model(image, conf=args.conf_thresh, verbose=False)[0]

    # Draw detections
    annotated = results.plot()

    # Show result
    window_name = "YOLO Inference"
    cv2.imshow(window_name, annotated)
    print("Press any key in the image window to exit")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()