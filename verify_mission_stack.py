"""Synthetic smoke test for color_detection.py and mission_planner.py.

Run this file to verify that the perception and planning layers produce the
expected outputs on simple, generated frames.
"""

from __future__ import annotations

import cv2
import numpy as np

from color_detection import detect_all_with_boxes
from mission_planner import ENTRY_MODE, RETURN_MODE, decide_flight_behavior


FRAME_HEIGHT = 480
FRAME_WIDTH = 640


def make_frame(bgr_color: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)
    cv2.rectangle(frame, (220, 150), (420, 330), bgr_color, -1)
    return frame


def run_case(name: str, frame: np.ndarray, mode: str, expected_action: str) -> None:
    detections = detect_all_with_boxes(frame)
    decision = decide_flight_behavior(detections, mode=mode)

    print(f"\nCASE: {name}")
    print("detections:", detections)
    print("decision:", decision)

    assert decision.action == expected_action, (
        f"Expected action {expected_action!r}, got {decision.action!r}"
    )


def main() -> None:
    green_frame = make_frame((0, 255, 0))
    red_frame = make_frame((0, 0, 255))
    orange_frame = make_frame((0, 165, 255))

    run_case("green / entry", green_frame, ENTRY_MODE, "follow_green")
    run_case("red / entry", red_frame, ENTRY_MODE, "avoid_red")
    run_case("orange / return", orange_frame, RETURN_MODE, "follow_orange")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()