# src/utils.py

import cv2

FRUIT_CLASSES = ["apple", "banana", "orange", "grape", "strawberry"]


def draw_fps(frame, fps: float):
    """Draw FPS counter on the frame."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame
