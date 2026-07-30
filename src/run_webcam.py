# src/run_webcam.py

import time
import cv2
from ultralytics import YOLO
from pathlib import Path

from utils import draw_fps, FRUIT_CLASSES


def load_model(model_path: str = "models/best.pt"):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train first or update the path."
        )
    print(f"[INFO] Loading model from {model_path}")
    return YOLO(model_path)


def main():
    model = load_model()

    # Open default webcam (0). Change to 1, 2, etc. if needed.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera index or permissions.")

    print("[INFO] Press 'q' to quit.")

    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from webcam.")
            break

        # Run YOLOv8 inference
        results = model(frame, verbose=False)

        # results[0].plot() returns an annotated image (NumPy array)
        annotated_frame = results[0].plot()

        # Compute FPS
        current_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (current_time - prev_time))
        prev_time = current_time

        annotated_frame = draw_fps(annotated_frame, fps)

        # Show window
        cv2.imshow("Fruit Detector", annotated_frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam closed.")


if __name__ == "__main__":
    main()
