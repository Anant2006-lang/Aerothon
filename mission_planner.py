

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

try:
    from pymavlink import mavutil
except ImportError:  # Optional until Pixhawk integration is enabled.
    mavutil = None


ENTRY_MODE = "ENTRY"
MISSION_MODE = "MISSION"
RETURN_MODE = "RETURN"


@dataclass(frozen=True)
class FlightDecision:
    mode: str
    action: str
    target: str
    priority: str
    lateral_error: Optional[int] = None
    red_position: Optional[str] = None
    reason: str = ""


@dataclass(frozen=True)
class VelocityCommand:
    """Body-to-NED command intent for Pixhawk control.

    Positive vx -> forward, positive vy -> right, positive vz -> down.
    Positive yaw_rate_deg_s -> clockwise yaw.
    """

    vx: float
    vy: float
    vz: float
    yaw_rate_deg_s: float


@dataclass
class AutonomousModeManager:
    """
    Rules:
    - Start in ENTRY
    - Switch to MISSION when green is stably detected and centered
    - Switch to RETURN when mission_complete is True (preferred external trigger)
    - Optional fallback: switch MISSION -> RETURN if orange is stably detected
    """

    mode: str = ENTRY_MODE
    green_lock_frames: int = 12
    orange_lock_frames: int = 15
    center_tolerance_px: int = 35
    _green_streak: int = 0
    _orange_streak: int = 0

    def update(self, detections: Dict[str, object], mission_complete: bool = False) -> str:
        self.mode = _normalize_mode(self.mode)

        if mission_complete:
            self.mode = RETURN_MODE
            return self.mode

        green_detected = bool(detections.get("green_detected", False))
        green_error = detections.get("green_error_x")
        green_centered = isinstance(green_error, int) and abs(green_error) <= self.center_tolerance_px

        orange_detected = bool(detections.get("orange_detected", False))

        if self.mode == ENTRY_MODE:
            if green_detected and green_centered:
                self._green_streak += 1
            else:
                self._green_streak = 0
            if self._green_streak >= self.green_lock_frames:
                self.mode = MISSION_MODE
                self._green_streak = 0

        elif self.mode == MISSION_MODE:
            if orange_detected:
                self._orange_streak += 1
            else:
                self._orange_streak = 0
            if self._orange_streak >= self.orange_lock_frames:
                self.mode = RETURN_MODE
                self._orange_streak = 0

        return self.mode


def _normalize_mode(mode: str) -> str:
    normalized = (mode or ENTRY_MODE).upper()
    if normalized not in {ENTRY_MODE, MISSION_MODE, RETURN_MODE}:
        return ENTRY_MODE
    return normalized


def decide_flight_behavior(
    detections: Dict[str, object],
    mode: str = ENTRY_MODE,
    mission_complete: bool = False,
) -> FlightDecision:
    """Convert perception outputs into a flight decision.

    Parameters
    ----------
    detections:
        Dictionary produced by the perception layer.
    mode:
        Current mission mode. Expected values are ENTRY, MISSION, RETURN.
    mission_complete:
        When True, the planner switches to RETURN mode.
    """
    current_mode = RETURN_MODE if mission_complete else _normalize_mode(mode)

    red_detected = bool(detections.get("red_detected", False))
    red_position = detections.get("red_position")

    if red_detected:
        return FlightDecision(
            mode=current_mode,
            action="avoid_red",
            target="red_zone",
            priority="HIGH",
            red_position=red_position if isinstance(red_position, str) else None,
            reason="Red zone detected; safety override activated.",
        )

    if current_mode == ENTRY_MODE:
        if bool(detections.get("green_detected", False)):
            return FlightDecision(
                mode=current_mode,
                action="follow_green",
                target="green_banner",
                priority="MEDIUM",
                lateral_error=detections.get("green_error_x"),
                reason="ENTRY mode: align with green banner and enter corridor.",
            )
        return FlightDecision(
            mode=current_mode,
            action="follow_corridor",
            target="corridor",
            priority="LOW",
            lateral_error=detections.get("corridor_error"),
            reason="ENTRY mode: green not visible, use corridor fallback.",
        )

    if current_mode == RETURN_MODE:
        if bool(detections.get("orange_detected", False)):
            return FlightDecision(
                mode=current_mode,
                action="follow_orange",
                target="orange_path",
                priority="MEDIUM",
                lateral_error=detections.get("orange_error_x"),
                reason="RETURN mode: align with orange return path.",
            )
        return FlightDecision(
            mode=current_mode,
            action="follow_corridor",
            target="corridor",
            priority="LOW",
            lateral_error=detections.get("corridor_error"),
            reason="RETURN mode: orange not visible, use corridor fallback.",
        )

    return FlightDecision(
        mode=current_mode,
        action="follow_corridor",
        target="corridor",
        priority="LOW",
        lateral_error=detections.get("corridor_error"),
        reason="MISSION mode: navigate corridor and perform task.",
    )


def update_mode(mode: str, mission_complete: bool = False) -> str:
    """Update the mission mode using the planned transition rules."""
    if mission_complete:
        return RETURN_MODE
    return _normalize_mode(mode)


def decision_to_dict(decision: FlightDecision) -> Dict[str, object]:
    """Convert a decision object to a plain dictionary."""
    return {
        "mode": decision.mode,
        "action": decision.action,
        "target": decision.target,
        "priority": decision.priority,
        "lateral_error": decision.lateral_error,
        "red_position": decision.red_position,
        "reason": decision.reason,
    }


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def decision_to_velocity(
    decision: FlightDecision,
    kp_lateral: float = 0.004,
    max_lateral_speed: float = 1.2,
    cruise_speed: float = 1.0,
) -> VelocityCommand:
    """Convert high-level mission decision into velocity setpoints.

    This function gives a simple first-pass controller for competition testing.
    Tune kp_lateral and speed limits on your airframe in SITL before flight.
    """
    lateral_error = float(decision.lateral_error or 0.0)
    lateral_cmd = _clamp(-kp_lateral * lateral_error, -max_lateral_speed, max_lateral_speed)

    if decision.action == "avoid_red":
        # Bias away from detected red side while slowing forward motion.
        if decision.red_position == "left":
            return VelocityCommand(vx=0.2, vy=0.7, vz=0.0, yaw_rate_deg_s=0.0)
        if decision.red_position == "right":
            return VelocityCommand(vx=0.2, vy=-0.7, vz=0.0, yaw_rate_deg_s=0.0)
        return VelocityCommand(vx=0.0, vy=0.8, vz=0.0, yaw_rate_deg_s=0.0)

    if decision.action in {"follow_green", "follow_orange", "follow_corridor"}:
        return VelocityCommand(vx=cruise_speed, vy=lateral_cmd, vz=0.0, yaw_rate_deg_s=0.0)

    # Safe fallback for unknown actions.
    return VelocityCommand(vx=0.0, vy=0.0, vz=0.0, yaw_rate_deg_s=0.0)


class PixhawkVelocitySender:
    """MAVLink velocity sender for Pixhawk using SET_POSITION_TARGET_LOCAL_NED."""

    def __init__(self, connection_string: str = "udp:127.0.0.1:14550", baud: int = 57600) -> None:
        if mavutil is None:
            raise ImportError("pymavlink is required for Pixhawk communication. Install with: pip install pymavlink")
        self.master = mavutil.mavlink_connection(connection_string, baud=baud)

    def wait_until_ready(self, timeout: int = 30) -> None:
        self.master.wait_heartbeat(timeout=timeout)

    def send_ned_velocity(self, command: VelocityCommand) -> None:
        # Ignore position/acceleration/yaw fields; only velocity and yaw-rate are active.
        type_mask = (
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
        )

        self.master.mav.set_position_target_local_ned_send(
            0,
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            type_mask,
            0.0,
            0.0,
            0.0,
            command.vx,
            command.vy,
            command.vz,
            0.0,
            0.0,
            0.0,
            0.0,
            math.radians(command.yaw_rate_deg_s),
        )


if __name__ == "__main__":
    sample = {
        "green_detected": True,
        "green_error_x": -35,
        "red_detected": False,
        "orange_detected": False,
        "corridor_error": 12,
    }

    decision = decide_flight_behavior(sample, mode=ENTRY_MODE)
    print(decision_to_dict(decision))
    print(decision_to_velocity(decision))