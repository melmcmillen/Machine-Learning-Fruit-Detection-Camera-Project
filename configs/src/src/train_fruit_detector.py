# src/train_fruit_detector.py

from ultralytics import YOLO
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 fruit detector")
    parser.add_argument(
        "--data",
        type=str,
        default="configs/fruit.yaml",
        help="Path to YOLO dataset config (.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base YOLOv8 model (n/s/m/l/x or custom .pt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save best model",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load base YOLOv8 model (nano for speed)
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project="runs/fruit_train",
        name="exp",
        exist_ok=True,
    )

    # Find best model path from results
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy best model into models/best.pt
    if best_model_path.exists():
        best_target = output_dir / "best.pt"
        best_target.write_bytes(best_model_path.read_bytes())
        print(f"[INFO] Saved best model to: {best_target}")
    else:
        print("[WARN] best.pt not found; check runs/fruit_train/exp/weights")


if __name__ == "__main__":
    main()
