import cv2

from utils.config import (
    PLAYER_COLOR,
    BALL_COLOR,
    TEXT_COLOR,
    BOX_THICKNESS,
    FONT_SCALE
)


def draw_corner_box(img, pt1, pt2, color, thickness=2, length=18):

    x1, y1 = pt1
    x2, y2 = pt2

    # Top Left
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)

    # Top Right
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)

    # Bottom Left
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)

    # Bottom Right
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)


def draw_player(frame, detection, color=None):

    x1, y1, x2, y2 = detection["bbox"]

    player_id = detection["id"]

    confidence = detection["confidence"]

    box_color = color if color is not None else PLAYER_COLOR

    draw_corner_box(
        frame,
        (x1, y1),
        (x2, y2),
        box_color,
        BOX_THICKNESS
    )

    label = f"Player {player_id}"

    cv2.rectangle(
        frame,
        (x1, y1 - 30),
        (x1 + 140, y1),
        box_color,
        -1
    )

    cv2.putText(
        frame,
        label,
        (x1 + 5, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        TEXT_COLOR,
        2
    )

    cv2.putText(
        frame,
        f"{confidence:.2f}",
        (x1, y2 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        box_color,
        2
    )


def draw_ball(frame, center):

    x, y = center

    cv2.circle(
        frame,
        (x, y),
        8,
        BALL_COLOR,
        -1
    )

    cv2.circle(
        frame,
        (x, y),
        16,
        BALL_COLOR,
        2
    )

    cv2.putText(
        frame,
        "BALL",
        (x + 10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        BALL_COLOR,
        2
    )


def draw_speed(frame, detection, speed):

    x1, y1, x2, y2 = detection["bbox"]

    cv2.putText(
        frame,
        f"{speed:.1f} km/h",
        (x1, y2 + 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )


def draw_fps(frame, fps):

    cv2.rectangle(
        frame,
        (15, 15),
        (180, 65),
        (35, 35, 35),
        -1
    )

    cv2.putText(
        frame,
        f"FPS : {int(fps)}",
        (30, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


def draw_team(frame, detection, team_name, color):

    x1, y1, x2, y2 = detection["bbox"]

    (text_w, text_h), _ = cv2.getTextSize(
        team_name,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        2
    )

    box_width = max(90, text_w + 10)

    cv2.rectangle(
        frame,
        (x1, y2 + 5),
        (x1 + box_width, y2 + 28),
        color,
        -1
    )

    cv2.putText(
        frame,
        team_name,
        (x1 + 5, y2 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )


def draw_possession(frame, team1_pct, team2_pct, team1_label, team2_label, team1_color, team2_color):

    box_h = 60 if len(team1_label) < 12 and len(team2_label) < 12 else 70

    cv2.rectangle(
        frame,
        (20, 80),
        (340, 80 + box_h),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        frame,
        f"{team1_label} : {team1_pct:.1f} %",
        (30, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        team1_color,
        2
    )

    cv2.putText(
        frame,
        f"{team2_label} : {team2_pct:.1f} %",
        (30, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        team2_color,
        2
    )


def draw_goal_banner(frame, team_label, team_color):

    h, w = frame.shape[:2]

    banner_h = 90

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, h // 2 - banner_h // 2),
        (w, h // 2 + banner_h // 2),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    text = f"GOAL! {team_label} team scores"

    (text_w, text_h), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        4
    )

    cv2.putText(
        frame,
        text,
        ((w - text_w) // 2, h // 2 + text_h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        team_color,
        4
    )


def draw_trail(frame, points, color=(0, 255, 255)):

    if len(points) < 2:
        return

    for i in range(1, len(points)):

        cv2.line(
            frame,
            points[i - 1],
            points[i],
            color,
            2
        )
