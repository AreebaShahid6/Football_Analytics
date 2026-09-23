"""
Run this once per video to find the pixel coordinates of each goal
mouth. Goal detection needs this because the camera angle/framing is
different in every video -- there's no default box that would work
for footage it's never seen.

Usage:
    python calibrate_goal_zones.py

Controls:
    - Click and drag to draw a rectangle over a goal mouth
    - Press 'l' to save the current rectangle as the LEFT goal
    - Press 'r' to save the current rectangle as the RIGHT goal
    - Press 'q' to quit and print both zones

Then copy the two printed lines into utils/config.py, replacing
GOAL_ZONE_LEFT / GOAL_ZONE_RIGHT (currently set to None there).
"""

import cv2

from utils.config import VIDEO_PATH

drawing = False
ix, iy = -1, -1
current_rect = None
left_zone = None
right_zone = None


def mouse_callback(event, x, y, flags, param):

    global drawing, ix, iy, current_rect

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_rect = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_rect = (min(ix, x), min(iy, y), max(ix, x), max(iy, y))


def main():

    global current_rect, left_zone, right_zone

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print(f"Could not open video at {VIDEO_PATH}")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Could not read a frame from {VIDEO_PATH}")
        return

    window = "Calibrate Goal Zones  |  drag a box, press l/r to save, q to finish"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, mouse_callback)

    print("Drag a rectangle over a goal mouth in the window that opened.")
    print("Press 'l' to save it as the LEFT goal zone.")
    print("Press 'r' to save it as the RIGHT goal zone.")
    print("Press 'q' when both are set (or to quit early).")

    while True:

        display = frame.copy()

        if current_rect:
            x1, y1, x2, y2 = current_rect
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

        if left_zone:
            cv2.rectangle(display, left_zone[:2], left_zone[2:], (255, 0, 0), 2)
            cv2.putText(
                display, "LEFT", left_zone[:2],
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
            )

        if right_zone:
            cv2.rectangle(display, right_zone[:2], right_zone[2:], (0, 0, 255), 2)
            cv2.putText(
                display, "RIGHT", right_zone[:2],
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )

        cv2.imshow(window, display)

        key = cv2.waitKey(20) & 0xFF

        if key == ord('l') and current_rect:
            left_zone = current_rect
            print(f"LEFT goal zone set: {left_zone}")

        elif key == ord('r') and current_rect:
            right_zone = current_rect
            print(f"RIGHT goal zone set: {right_zone}")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

    print()
    print("Paste these two lines into utils/config.py:")
    print(f"GOAL_ZONE_LEFT = {left_zone}")
    print(f"GOAL_ZONE_RIGHT = {right_zone}")


if __name__ == "__main__":
    main()
