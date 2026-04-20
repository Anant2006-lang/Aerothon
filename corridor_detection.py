from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class CorridorDetectionResult:
    corridor_error: float
    normalized_error: float
    centered: bool
    roi_shape: tuple[int, int]
    edge_pixels: int


@dataclass
class CorridorSmoother:
    alpha: float = 0.7
    previous_error: Optional[float] = None

    def update(self, current_error: float) -> float:
        if self.previous_error is None:
            self.previous_error = current_error
        else:
            self.previous_error = (
                self.alpha * self.previous_error
                + (1.0 - self.alpha) * current_error
            )
        return self.previous_error


def detect_corridor(
    frame: np.ndarray,
    *,
    canny_low: int = 70,
    canny_high: int = 180,
    deadband: int = 5,
    smoothing_state: Optional[CorridorSmoother] = None,
) -> CorridorDetectionResult:

    if frame is None or frame.size == 0:
        raise ValueError("frame must be valid")

    #  preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)

    h, w = edges.shape

    #  ROI (bottom half only)
    roi = edges[h // 2 :, :]

    # TRUE CORRIDOR BOUNDARY DETECTION
    ys, xs = np.where(roi > 0)

    if len(xs) > 100:  # ignore weak/noisy detections

        left_x = np.min(xs)
        right_x = np.max(xs)

        # ignore too narrow detections (noise)
        if (right_x - left_x) < 50:
            corridor_center = w / 2
        else:
            corridor_center = (left_x + right_x) / 2

    else:
        corridor_center = w / 2  # fallback

    #  reference center (image center)
    image_center = w / 2

    #  error
    error = corridor_center - image_center

    normalized_error = error / (w / 2)
    corridor_error = normalized_error * 100

    #  smoothing
    if smoothing_state is not None:
        corridor_error = smoothing_state.update(corridor_error)

    centered = abs(corridor_error) < deadband

    return CorridorDetectionResult(
        corridor_error=float(corridor_error),
        normalized_error=float(normalized_error),
        centered=centered,
        roi_shape=(roi.shape[0], roi.shape[1]),
        edge_pixels=int(np.count_nonzero(roi)),
    )


def detect_corridor_dict(frame: np.ndarray, **kwargs) -> Dict[str, object]:
    result = detect_corridor(frame, **kwargs)
    return {
        "corridor_error": result.corridor_error,
        "corridor_centered": result.centered,
        "corridor_normalized_error": result.normalized_error,
        "corridor_roi_shape": result.roi_shape,
        "corridor_edge_pixels": result.edge_pixels,
    }


def annotate_corridor(frame: np.ndarray, result: CorridorDetectionResult) -> np.ndarray:
    output = frame.copy()
    h, w = output.shape[:2]

    #  ROI box
    cv2.rectangle(output, (0, h // 2), (w, h), (0, 255, 255), 2)

    #  image center (white)
    cv2.line(output, (w // 2, h // 2), (w // 2, h), (255, 255, 255), 2)
    #corridor center (green)
    cv2.line(output, (int(w/2 + result.corridor_error * w/200), h//2),
         (int(w/2 + result.corridor_error * w/200), h),
         (0, 255, 0), 2)

    #  show status
    status = "CENTERED" if result.centered else "OFF-CENTER"

    text = [
        f"corridor_error: {result.corridor_error:.2f}",
        f"status: {status}",
    ]

    y = 40
    for line in text:
        cv2.putText(
            output,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30

    return output


#  TEST MODE (IMAGE)
if __name__ == "__main__":

    image = cv2.imread(r"C:\Users\ANANT BEDI\Downloads\corridorimage.jpg")

    if image is None:
        print("Error: Image not found")
        exit()

    smoother = CorridorSmoother()

    result = detect_corridor(image, smoothing_state=smoother)

    print({
        "corridor_error": result.corridor_error,
        "centered": result.centered
    })

    vis = annotate_corridor(image, result)

    vis = cv2.resize(vis, (900, 600))

    cv2.imshow("Corridor Detection (Image)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()