from __future__ import annotations

import cv2
from matplotlib import pyplot as plt

from color_detection import annotate_frame, detect_all_with_boxes
from corridor_detection import detect_corridor_dict
from mission_planner import (
    AutonomousModeManager,
    decide_flight_behavior,
)


# TEXT OVERLAY
def overlay_text(frame, lines, origin=(20, 50)):
    output = frame.copy()
    x, y = origin

    for line in lines:
        cv2.putText(
            output,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28

    return output


# ACTION EXECUTION
def execute_action(decision):
    action = decision.action

    if action == "avoid_red":
        if decision.red_position == "left":
            print("COMMAND → MOVE RIGHT")
        elif decision.red_position == "right":
            print("COMMAND → MOVE LEFT")
        else:
            print("COMMAND → STOP")

    elif action in ["follow_green", "follow_orange", "follow_corridor"]:
        error = decision.lateral_error

        if error is None:
            print("COMMAND → SEARCH")
        elif error > 30:
            print("COMMAND → MOVE LEFT")
        elif error < -30:
            print("COMMAND → MOVE RIGHT")
        else:
            print("COMMAND → MOVE FORWARD")

    else:
        print("COMMAND → HOLD")


def main():

    # IMAGE PATH
    image_path = r"C:\Users\ANANT BEDI\Downloads\corridorimage.jpg"

    frame = cv2.imread(image_path)

    if frame is None:
        print(" ERROR: Image not found. Check path.")
        return

    print(" Image loaded successfully")

    mode_manager = AutonomousModeManager()
    mission_complete = False

    # 🔹 DETECTIONS
    detections = detect_all_with_boxes(frame)
    detections.update(detect_corridor_dict(frame))

    # 🔹 DECISION
    mode = mode_manager.update(detections, mission_complete=mission_complete)

    decision = decide_flight_behavior(
        detections,
        mode=mode,
        mission_complete=mission_complete,
    )

    # 🔹 EXECUTE COMMAND
    execute_action(decision)

    print("\n--- DECISION OUTPUT ---")
    print({
        "mode": decision.mode,
        "action": decision.action,
        "target": decision.target,
        "priority": decision.priority,
        "lateral_error": decision.lateral_error,
        "red_position": decision.red_position,
    })

    print("Corridor error:", detections.get("corridor_error"))

    # DRAW BASE (color boxes)
    annotated = annotate_frame(frame, detections)

    # ADDING CORRIDOR VISUALIZATION 
    h, w = annotated.shape[:2]

    corridor_error = detections.get("corridor_error", 0)

    # convert error → pixel shift
    corridor_x = int(w/2 + corridor_error * w / 200)

    # image center (drone direction)
    cv2.line(annotated, (w//2, 0), (w//2, h), (255, 255, 255), 2)

    # corridor center
    cv2.line(annotated, (corridor_x, 0), (corridor_x, h), (0, 255, 0), 3)

    # TEXT OVERLAY
    annotated = overlay_text(
        annotated,
        [
            f"Mode: {decision.mode}",
            f"Action: {decision.action}",
            f"Target: {decision.target}",
            f"Priority: {decision.priority}",
            f"Red: {detections.get('red_detected', False)}",
            f"Corridor error: {detections.get('corridor_error', 0):.2f}",
            f"Mission Complete: {mission_complete}",
        ],
    )

    # RESIZE (SAFE SIZE)
    annotated = cv2.resize(annotated, (900, 600))

    #  SAVE OUTPUT
    cv2.imwrite("output.jpg", annotated)
    print(" Output saved as output.jpg")

    #  DISPLAY
    try:
        cv2.namedWindow("Aerothon Image Test", cv2.WINDOW_NORMAL)
        cv2.imshow("Aerothon Image Test", annotated)

        cv2.setWindowProperty("Aerothon Image Test", cv2.WND_PROP_TOPMOST, 1)

        print("\n Press any key on image window to close")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except:
        print(" OpenCV window failed — using matplotlib")

        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.title("Aerothon Output")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()