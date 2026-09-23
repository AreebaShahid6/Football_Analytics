import cv2
import numpy as np

from utils.config import (
    TRAIL_PANEL_BG_COLOR,
    TRAIL_PANEL_LINE_COLOR,
    PANEL_LABEL_COLOR
)


def _resize_to_fit(img, width, height):

    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def _label_panel(panel, text, accent_color=PANEL_LABEL_COLOR):

    bar_h = 34

    cv2.rectangle(panel, (0, 0), (panel.shape[1], bar_h), (20, 20, 20), -1)

    # Thin accent line under the label bar for a cleaner, more
    # "broadcast graphics" look than a flat block
    cv2.rectangle(panel, (0, bar_h), (panel.shape[1], bar_h + 3), accent_color, -1)

    cv2.putText(
        panel,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        accent_color,
        2
    )

    return panel


def _draw_legend(panel, legend):
    """
    Small color-key in the corner of the heatmap panel, e.g.
    "● Red   ● Sky Blue", so it's clear which color belongs to
    which team without having to cross-reference the main video.
    """

    if not legend:
        return panel

    x = 10
    y = panel.shape[0] - 14

    for label, color in reversed(legend):

        (text_w, _), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        cv2.circle(panel, (x + 6, y - 5), 6, color, -1)

        cv2.putText(
            panel,
            label,
            (x + 18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        x += text_w + 40

    return panel


def make_trail_panel(width, height, points):
    """
    Draws the ball's full movement history as clean connected lines on
    a blank pitch-colored canvas -- separated from the video so the
    path is easy to read (the main frame only shows a marker at the
    ball's current position, not the trail).
    """

    panel = np.full((height, width, 3), TRAIL_PANEL_BG_COLOR, dtype=np.uint8)

    # Faint grid so the trail panel reads as a pitch schematic rather
    # than a blank box
    for gx in range(0, width, 60):
        cv2.line(panel, (gx, 0), (gx, height), (30, 75, 30), 1)
    for gy in range(0, height, 60):
        cv2.line(panel, (0, gy), (width, gy), (30, 75, 30), 1)

    if len(points) < 2:
        return panel

    for i in range(1, len(points)):

        cv2.line(
            panel,
            points[i - 1],
            points[i],
            TRAIL_PANEL_LINE_COLOR,
            2,
            cv2.LINE_AA
        )

    # Mark the ball's current (most recent) position clearly
    cv2.circle(panel, points[-1], 7, (0, 0, 255), -1)
    cv2.circle(panel, points[-1], 7, (255, 255, 255), 2)

    return panel


def build_dashboard(main_frame, heatmap_frame, ball_trail_points, panel_width, legend=None):
    """
    Composites the annotated main video frame with two stacked side
    panels: a live team-colored heatmap (top) and the ball's movement
    trail (bottom). Returns the full canvas to write to the output video.

    legend: optional list of (label, bgr_color) tuples shown in the
    corner of the heatmap panel, e.g. [("Red", (30,30,200)), ...].
    """

    video_h, video_w = main_frame.shape[:2]

    top_h = video_h // 2
    bottom_h = video_h - top_h

    heatmap_panel = _resize_to_fit(heatmap_frame, panel_width, top_h)
    heatmap_panel = _label_panel(heatmap_panel, "LIVE HEATMAP")
    heatmap_panel = _draw_legend(heatmap_panel, legend)

    trail_full_res = make_trail_panel(video_w, video_h, ball_trail_points)
    trail_panel = _resize_to_fit(trail_full_res, panel_width, bottom_h)
    trail_panel = _label_panel(trail_panel, "BALL TRAIL", TRAIL_PANEL_LINE_COLOR)

    right_column = np.vstack([heatmap_panel, trail_panel])

    canvas = np.hstack([main_frame, right_column])

    return canvas
