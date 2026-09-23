import math
import time

import cv2

from utils.config import (
    VIDEO_PATH,
    OUTPUT_VIDEO,
    HEATMAP_IMAGE,
    TEAM1,
    TEAM2,
    POSSESSION_DISTANCE_THRESHOLD,
    DASHBOARD_ENABLED,
    SIDE_PANEL_WIDTH,
    HEATMAP_UPDATE_INTERVAL,
    GOAL_ZONE_LEFT,
    GOAL_ZONE_RIGHT,
    GOAL_COOLDOWN_SECONDS,
    GOAL_BANNER_SECONDS,
    GOAL_POSSESSION_LOOKBACK,
    REFEREE_COLOR
)

from collections import deque, Counter

from utils.tracker import PlayerTracker
from utils.ball_tracker import BallTracker
from utils.pose_utils import (
    PoseEstimator,
    draw_pose
)

from utils.team_classifier import TeamClassifier

from utils.speed import SpeedEstimator

from utils.heatmap import HeatmapGenerator

from utils.dashboard import build_dashboard

from utils.drawing import (
    draw_player,
    draw_ball,
    draw_speed,
    draw_team,
    draw_fps,
    draw_possession,
    draw_goal_banner
)


# ===========================================
# INITIALIZE MODELS
# ===========================================

player_tracker = PlayerTracker()

# Resolved dynamically inside PlayerTracker from the model's own
# class names -- reliable regardless of what order Roboflow/Kaggle
# exported the classes in
REFEREE_CLASS_INDEX = player_tracker.referee_class_index

ball_tracker = BallTracker()

pose_estimator = PoseEstimator()

team_classifier = TeamClassifier()

speed_estimator = SpeedEstimator()

heatmap = HeatmapGenerator()


# ===========================================
# VIDEO
# ===========================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video at {VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS) or 30

# When the dashboard is on, the output canvas is wider than the source
# video (main video + side panel column), so the writer needs to be
# sized for that, not the raw source dimensions.
output_width = width + SIDE_PANEL_WIDTH if DASHBOARD_ENABLED else width

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (output_width, height)
)


team_initialized = False

frame_counter = 0

previous_time = time.time()

# Possession is tallied in frames-with-a-clear-ball-owner, not raw
# video frames, so a few frames where the ball is lost don't skew it
team1_possession_frames = 0
team2_possession_frames = 0

background_frame = None

# Full ball movement history (unbounded) -- drawn on its own panel
# instead of over the video, so the path is easy to read
ball_trail_full = []

# The heatmap panel is expensive to regenerate every frame, so it's
# cached and refreshed only every HEATMAP_UPDATE_INTERVAL frames
cached_heatmap_panel = None

# ===========================================
# GOAL DETECTION STATE
# ===========================================

goals = []
last_goal_frame = -999999
goal_banner_frames_remaining = 0
goal_banner_label = None
goal_banner_color = (0, 255, 0)

# Recent "closest team to the ball" history, used to decide which
# team gets credit when a goal fires (the team that had the ball
# just before it crossed the line)
possession_history = deque(maxlen=GOAL_POSSESSION_LOOKBACK)

goal_detection_enabled = GOAL_ZONE_LEFT is not None and GOAL_ZONE_RIGHT is not None

if not goal_detection_enabled:
    print(
        "Goal detection is OFF: GOAL_ZONE_LEFT / GOAL_ZONE_RIGHT are not "
        "set in utils/config.py. Run calibrate_goal_zones.py to set them."
    )


# ===========================================
# MAIN LOOP
# ===========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_counter += 1

    if background_frame is None:
        background_frame = frame.copy()

    #########################################################
    # PLAYER TRACKING
    #########################################################

    detections = player_tracker.track(frame)

    #########################################################
    # TEAM INITIALIZATION
    #########################################################

    if not team_initialized:

        outfield_detections = [
            d for d in detections
            if REFEREE_CLASS_INDEX is None or d["class"] != REFEREE_CLASS_INDEX
        ]

        team_classifier.collect_samples(outfield_detections, frame)

        team_initialized = team_classifier.try_finalize_teams()

    #########################################################
    # BALL DETECTION
    #########################################################

    ball = ball_tracker.detect(frame)

    #########################################################
    # POSE ESTIMATION
    #########################################################

    poses = pose_estimator.detect(frame)

    #########################################################
    # POSSESSION COUNTERS
    #########################################################

    team1_count = 0
    team2_count = 0

    closest_team = None
    closest_distance = float("inf")

    #########################################################
    # PROCESS EACH PLAYER
    #########################################################

    for det in detections:

        is_referee = (
            REFEREE_CLASS_INDEX is not None
            and det["class"] == REFEREE_CLASS_INDEX
        )

        if is_referee:

            speed = speed_estimator.calculate_speed(det["id"], det["center"])

            heatmap.update(det["id"], det["center"])

            draw_player(frame, det, REFEREE_COLOR)
            draw_team(frame, det, "Referee", REFEREE_COLOR)
            draw_speed(frame, det, speed)

            # Referees don't belong to a team and shouldn't count
            # toward possession or the team-colored heatmap
            continue

        if team_initialized:
            team_key, team_color, team_label = team_classifier.classify(det, frame)
        else:
            team_key, team_color, team_label = TEAM1, (0, 255, 0), "Team A"

        if team_key == TEAM1:
            team1_count += 1
        else:
            team2_count += 1

        speed = speed_estimator.calculate_speed(det["id"], det["center"])

        heatmap.update(det["id"], det["center"])
        heatmap.update_team(team_key, det["center"])

        draw_player(frame, det, team_color)
        draw_team(frame, det, team_label, team_color)
        draw_speed(frame, det, speed)

        if ball is not None:

            dx = det["center"][0] - ball["center"][0]
            dy = det["center"][1] - ball["center"][1]
            distance = math.hypot(dx, dy)

            if distance < closest_distance:
                closest_distance = distance
                closest_team = team_key

    if ball is not None and ball["live"] and closest_team is not None:
        possession_history.append(closest_team)

    #########################################################
    # POSSESSION ACCUMULATION
    #########################################################

    if (
        ball is not None
        and ball["live"]
        and closest_team is not None
        and closest_distance <= POSSESSION_DISTANCE_THRESHOLD
    ):

        if closest_team == TEAM1:
            team1_possession_frames += 1
        else:
            team2_possession_frames += 1

    total_possession_frames = team1_possession_frames + team2_possession_frames

    if total_possession_frames > 0:
        team1_pct = (team1_possession_frames / total_possession_frames) * 100
        team2_pct = (team2_possession_frames / total_possession_frames) * 100
    else:
        team1_pct = 0.0
        team2_pct = 0.0

    #########################################################
    # DRAW BALL
    #########################################################

    if ball is not None:

        draw_ball(frame, ball["center"])

        if ball["live"]:
            ball_trail_full.append(ball["center"])

    #########################################################
    # GOAL DETECTION
    #########################################################

    if goal_detection_enabled and ball is not None and ball["live"]:

        cooldown_frames = GOAL_COOLDOWN_SECONDS * fps
        bx, by = ball["center"]

        in_left_zone = (
            GOAL_ZONE_LEFT[0] <= bx <= GOAL_ZONE_LEFT[2]
            and GOAL_ZONE_LEFT[1] <= by <= GOAL_ZONE_LEFT[3]
        )

        in_right_zone = (
            GOAL_ZONE_RIGHT[0] <= bx <= GOAL_ZONE_RIGHT[2]
            and GOAL_ZONE_RIGHT[1] <= by <= GOAL_ZONE_RIGHT[3]
        )

        if (
            (in_left_zone or in_right_zone)
            and (frame_counter - last_goal_frame) > cooldown_frames
            and len(possession_history) > 0
        ):
            # Credit the team that most recently had the ball -- i.e.
            # the team that just shot it in, not the goal's owner
            recent = list(possession_history)
            scoring_team = max(set(recent), key=recent.count)

            scoring_label = team_classifier.get_label(scoring_team)
            scoring_color = team_classifier.get_draw_color(scoring_team)

            goals.append({
                "team": scoring_label,
                "frame": frame_counter,
                "time_s": frame_counter / fps
            })

            last_goal_frame = frame_counter
            goal_banner_frames_remaining = int(GOAL_BANNER_SECONDS * fps)
            goal_banner_label = scoring_label
            goal_banner_color = scoring_color

            print(
                f"GOAL detected -> {scoring_label} team at "
                f"{frame_counter / fps:.1f}s (frame {frame_counter})"
            )

    if goal_banner_frames_remaining > 0:
        draw_goal_banner(frame, goal_banner_label, goal_banner_color)
        goal_banner_frames_remaining -= 1

    #########################################################
    # DRAW POSES
    #########################################################

    for pose in poses:
        draw_pose(frame, pose)

    #########################################################
    # POSSESSION + FPS OVERLAY
    #########################################################

    team1_label = team_classifier.get_label(TEAM1) if team_initialized else "Team A"
    team2_label = team_classifier.get_label(TEAM2) if team_initialized else "Team B"
    team1_draw_color = team_classifier.get_draw_color(TEAM1) if team_initialized else (0, 255, 0)
    team2_draw_color = team_classifier.get_draw_color(TEAM2) if team_initialized else (0, 128, 255)

    draw_possession(
        frame,
        team1_pct,
        team2_pct,
        team1_label,
        team2_label,
        team1_draw_color,
        team2_draw_color
    )

    current_time = time.time()
    elapsed = current_time - previous_time
    fps_display = (1 / elapsed) if elapsed > 0 else 0.0
    previous_time = current_time

    draw_fps(frame, fps_display)

    #########################################################
    # DASHBOARD COMPOSITE (heatmap panel + ball-trail panel)
    #########################################################

    if DASHBOARD_ENABLED:

        if (
            cached_heatmap_panel is None
            or frame_counter % HEATMAP_UPDATE_INTERVAL == 0
        ):
            if team_initialized:
                team_draw_colors_for_heatmap = {
                    TEAM1: team_classifier.get_draw_color(TEAM1),
                    TEAM2: team_classifier.get_draw_color(TEAM2)
                }
                cached_heatmap_panel = heatmap.generate_team_heatmap(
                    background_frame,
                    team_draw_colors_for_heatmap
                )
            else:
                cached_heatmap_panel = heatmap.generate_heatmap(background_frame)

        legend = (
            [(team1_label, team1_draw_color), (team2_label, team2_draw_color)]
            if team_initialized else None
        )

        output_frame = build_dashboard(
            frame,
            cached_heatmap_panel,
            ball_trail_full,
            SIDE_PANEL_WIDTH,
            legend=legend
        )

    else:
        output_frame = frame

    #########################################################
    # WRITE OUTPUT
    #########################################################

    writer.write(output_frame)

    if frame_counter % 30 == 0:
        print(f"Processed frame {frame_counter}")


# ===========================================
# CLEANUP
# ===========================================

cap.release()
writer.release()

print(f"Done. {frame_counter} frames processed.")
print(f"Output video saved to: {OUTPUT_VIDEO}")
print(f"Possession -> {team1_label}: {team1_pct:.1f}% | {team2_label}: {team2_pct:.1f}%")

# ===========================================
# GOAL SUMMARY
# ===========================================

print()

if not goal_detection_enabled:
    print("Goal detection was OFF (GOAL_ZONE_LEFT/RIGHT not set in config.py).")

elif not goals:
    print("No goals detected in this clip.")

else:
    print("=== GOAL SUMMARY ===")

    for g in goals:
        print(f"  {g['team']} team scored at {g['time_s']:.1f}s (frame {g['frame']})")

    tally = Counter(g["team"] for g in goals)

    print()
    print("Final tally:")

    for team_label_tally, count in tally.items():
        print(f"  {team_label_tally}: {count} goal(s)")

# ===========================================
# FINAL HEATMAP (saved as separate images)
# ===========================================

if background_frame is not None:

    heatmap_output = heatmap.generate_heatmap(background_frame)
    cv2.imwrite(str(HEATMAP_IMAGE), heatmap_output)
    print(f"Combined heatmap saved to: {HEATMAP_IMAGE}")

    if team_initialized:

        team_draw_colors = {
            TEAM1: team_classifier.get_draw_color(TEAM1),
            TEAM2: team_classifier.get_draw_color(TEAM2)
        }

        team_heatmap_output = heatmap.generate_team_heatmap(
            background_frame,
            team_draw_colors
        )

        team_heatmap_path = HEATMAP_IMAGE.with_name(
            HEATMAP_IMAGE.stem + "_by_team" + HEATMAP_IMAGE.suffix
        )

        cv2.imwrite(str(team_heatmap_path), team_heatmap_output)
        print(f"Per-team heatmap saved to: {team_heatmap_path}")
