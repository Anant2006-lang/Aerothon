

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionResult:
	detected: bool
	error_x: Optional[int]
	area: float
	bbox: Optional[Tuple[int, int, int, int]]
	centroid: Optional[Tuple[int, int]]


GREEN_LOWER = np.array([35, 70, 70], dtype=np.uint8)
GREEN_UPPER = np.array([90, 255, 255], dtype=np.uint8)

ORANGE_LOWER = np.array([5, 100, 100], dtype=np.uint8)
ORANGE_UPPER = np.array([25, 255, 255], dtype=np.uint8)

# Red wraps around the HSV hue axis, so we use two ranges.
RED_LOWER_1 = np.array([0, 120, 70], dtype=np.uint8)
RED_UPPER_1 = np.array([10, 255, 255], dtype=np.uint8)
RED_LOWER_2 = np.array([170, 120, 70], dtype=np.uint8)
RED_UPPER_2 = np.array([180, 255, 255], dtype=np.uint8)


def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
	blurred = cv2.GaussianBlur(frame, (5, 5), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	return hsv


def _clean_mask(mask: np.ndarray) -> np.ndarray:
	kernel = np.ones((5, 5), dtype=np.uint8)
	cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
	return cleaned


def _largest_contour(mask: np.ndarray, min_area: float) -> Optional[np.ndarray]:
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return None

	contour = max(contours, key=cv2.contourArea)
	if cv2.contourArea(contour) < min_area:
		return None
	return contour


def _build_result(contour: Optional[np.ndarray], frame_width: int) -> DetectionResult:
	if contour is None:
		return DetectionResult(False, None, 0.0, None, None)

	area = float(cv2.contourArea(contour))
	x, y, w, h = cv2.boundingRect(contour)
	centroid_x = x + w // 2
	centroid_y = y + h // 2
	error_x = centroid_x - (frame_width // 2)
	return DetectionResult(True, error_x, area, (x, y, w, h), (centroid_x, centroid_y))


def _detect_by_mask(frame: np.ndarray, mask: np.ndarray, min_area: float = 800.0) -> DetectionResult:
	cleaned = _clean_mask(mask)
	contour = _largest_contour(cleaned, min_area=min_area)
	return _build_result(contour, frame_width=frame.shape[1])


def detect_green(frame: np.ndarray, min_area: float = 800.0) -> DetectionResult:
	"""Detect the green entry banner."""
	hsv = _preprocess_frame(frame)
	mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
	return _detect_by_mask(frame, mask, min_area=min_area)


def detect_orange(frame: np.ndarray, min_area: float = 800.0) -> DetectionResult:
	"""Detect the orange return path marker."""
	hsv = _preprocess_frame(frame)
	mask = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)
	return _detect_by_mask(frame, mask, min_area=min_area)


def detect_red(frame: np.ndarray, min_area: float = 700.0) -> Dict[str, object]:
	"""Detect a red restricted zone using a dual HSV range.

	Returns a dictionary because the planner typically needs both the boolean
	safety flag and a coarse left/center/right relation.
	"""
	hsv = _preprocess_frame(frame)
	mask_1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
	mask_2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
	mask = cv2.bitwise_or(mask_1, mask_2)

	result = _detect_by_mask(frame, mask, min_area=min_area)
	red_position = None
	if result.detected and result.centroid is not None:
		centroid_x = result.centroid[0]
		frame_center = frame.shape[1] // 2
		offset = centroid_x - frame_center
		if offset < -frame.shape[1] * 0.1:
			red_position = "left"
		elif offset > frame.shape[1] * 0.1:
			red_position = "right"
		else:
			red_position = "center"

	return {
		"red_detected": result.detected,
		"red_position": red_position,
		"error_x": result.error_x,
		"area": result.area,
		"bbox": result.bbox,
		"red_bbox": result.bbox,
		"centroid": result.centroid,
	}


def detect_all(frame: np.ndarray) -> Dict[str, object]:
	"""Run all color detectors and return a combined result dictionary."""
	green = detect_green(frame)
	orange = detect_orange(frame)
	red = detect_red(frame)

	return {
		"green_detected": green.detected,
		"green_error_x": green.error_x,
		"green_area": green.area,
		"orange_detected": orange.detected,
		"orange_error_x": orange.error_x,
		"orange_area": orange.area,
		**red,
	}


def annotate_frame(frame: np.ndarray, detections: Dict[str, object]) -> np.ndarray:
	"""Draw the detected boxes and labels on a frame for debugging."""
	output = frame.copy()
	height, width = output.shape[:2]
	cv2.line(output, (width // 2, 0), (width // 2, height), (255, 255, 255), 1)

	for color_name, color_bgr in (("green", (0, 255, 0)), ("orange", (0, 165, 255))):
		bbox = detections.get(f"{color_name}_bbox")
		if bbox is None:
			continue
		x, y, w, h = bbox
		cv2.rectangle(output, (x, y), (x + w, y + h), color_bgr, 2)
		cv2.putText(
			output,
			color_name,
			(x, max(20, y - 10)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			color_bgr,
			2,
			cv2.LINE_AA,
		)

	red_bbox = detections.get("red_bbox")
	if red_bbox is not None:
		x, y, w, h = red_bbox
		cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
		cv2.putText(
			output,
			"red",
			(x, max(20, y - 10)),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.6,
			(0, 0, 255),
			2,
			cv2.LINE_AA,
		)

	return output


def detect_all_with_boxes(frame: np.ndarray) -> Dict[str, object]:
	"""Run all detectors and include bounding boxes for visualization."""
	green = detect_green(frame)
	orange = detect_orange(frame)
	red = detect_red(frame)

	return {
		"green_detected": green.detected,
		"green_error_x": green.error_x,
		"green_area": green.area,
		"green_bbox": green.bbox,
		"orange_detected": orange.detected,
		"orange_error_x": orange.error_x,
		"orange_area": orange.area,
		"orange_bbox": orange.bbox,
		**red,
	}


if __name__ == "__main__":
	capture = cv2.VideoCapture(0)
	if not capture.isOpened():
		raise RuntimeError("Could not open camera")

	while True:
		success, frame = capture.read()
		if not success:
			break

		detections = detect_all_with_boxes(frame)
		annotated = annotate_frame(frame, detections)

		print(detections)
		cv2.imshow("Aerothon Color Detection", annotated)

		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

	capture.release()
	cv2.destroyAllWindows()







