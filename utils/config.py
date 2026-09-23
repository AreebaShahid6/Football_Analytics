from pathlib import Path

# ===========================
# PROJECT PATHS
# ===========================

ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = ROOT / "videos" / "football.mp4"
OUTPUT_VIDEO = ROOT / "videos" / "output.mp4"
HEATMAP_IMAGE = ROOT / "videos" / "heatmap.png"

# Independent toggles -- as you train each model (player first, ball
# next per your plan), flip just that one to False. There's no need
# to wait until all three are custom-trained.
USE_PRETRAINED_PLAYER = False   # your custom-trained model -- keep this, don't revert to pretrained
USE_PRETRAINED_BALL = False     # your new dedicated ball-trained model
USE_PRETRAINED_POSE = True

# YOLO26 (Ultralytics, Jan 2026) is the current generation -- faster
# on CPU than YOLO11 at the same accuracy, and specifically better at
# small/far-away objects, which is exactly the ball's problem in
# broadcast football footage.
PRETRAINED_FAMILY = "yolo26"

# Pretrained model size -- bigger = more accurate, slower, more RAM.
#   "n" (nano)   - fastest, least accurate (the original default)
#   "s" (small)  - noticeably more accurate, still runs fine on CPU
#   "m" (medium) - meaningfully more accurate, needs a decent CPU or
#                  a GPU to process a full match in reasonable time
#   "l" / "x"    - only worth it with an NVIDIA GPU
# "s" below is a real accuracy step up from nano at a small speed
# cost. If detection.py feels too slow on your machine, drop this
# back to "n"; if you have an NVIDIA GPU, try "m".
PRETRAINED_SIZE = "s"

PLAYER_MODEL = (
    f"{PRETRAINED_FAMILY}{PRETRAINED_SIZE}.pt" if USE_PRETRAINED_PLAYER
    else ROOT / "models" / "yolo26_player.pt"
)
BALL_MODEL = (
    f"{PRETRAINED_FAMILY}{PRETRAINED_SIZE}.pt" if USE_PRETRAINED_BALL
    else ROOT / "models" / "yolo26_ball.pt"
)
POSE_MODEL = (
    f"{PRETRAINED_FAMILY}{PRETRAINED_SIZE}-pose.pt" if USE_PRETRAINED_POSE
    else ROOT / "models" / "yolo26_pose.pt"
)

# Test-time augmentation for the ball model: runs inference on a
# flipped/scaled version of the frame too and merges results. This
# helped on YOLO11, but YOLO26's end-to-end architecture doesn't
# support it -- it just prints a warning and ignores the setting every
# frame. Left False to stop the log spam; no accuracy is lost since it
# wasn't actually doing anything.
BALL_TEST_TIME_AUGMENT = False

# COCO class IDs used to filter the general-purpose pretrained model
COCO_PERSON_CLASS = 0
COCO_SPORTS_BALL_CLASS = 32

# NOTE: class indices for a custom-trained model (player/goalkeeper/
# referee/ball) are NOT hardcoded here on purpose. utils/tracker.py
# and utils/ball_tracker.py read them directly from the model file
# itself (model.names) at load time instead -- Roboflow/Kaggle exports
# don't always order classes the same way, so hardcoding indices here
# was fragile and could silently drop detections if the order didn't
# match. Check the printed "Custom player model classes: {...}" line
# when you run detection.py to see exactly what your model reports.

# ===========================
# DETECTION SETTINGS
# ===========================

PLAYER_CONF = 0.45
BALL_CONF = 0.12
POSE_CONF = 0.25

IMG_SIZE = 1280

# ===========================
# TRACKER SETTINGS
# ===========================

TRACK_BUFFER = 60
MATCH_THRESHOLD = 0.80

# Ultralytics tracker config. Points at utils/bytetrack_custom.yaml,
# which is where TRACK_BUFFER/MATCH_THRESHOLD above actually take
# effect -- edit that file if you change the values here (Ultralytics
# reads the yaml directly, this path just tells it which one to use).
TRACKER_TYPE = str(ROOT / "utils" / "bytetrack_custom.yaml")

# ===========================
# DRAW SETTINGS
# ===========================

BOX_THICKNESS = 2

PLAYER_COLOR = (0, 255, 0)
BALL_COLOR = (0, 255, 255)
REFEREE_COLOR = (255, 255, 0)

TEXT_COLOR = (255, 255, 255)

FONT_SCALE = 0.6

# ===========================
# HEATMAP
# ===========================

HEATMAP_ALPHA = 0.45

# ===========================
# SPEED
# ===========================

FPS = 30
PIXELS_PER_METER = 18

# Ignore jitter smaller than this many pixels between frames when
# computing speed (stationary players otherwise show fake movement
# from box-detection noise)
MIN_MOVEMENT_PIXELS = 2

# Fastest ever recorded human sprint speed is ~44 km/h (Usain Bolt).
# Any single-frame speed reading above this is not a real movement --
# it means a track ID got swapped between two different players (an
# occlusion/ID-switch glitch), not that someone actually moved that
# fast. Cap it so a swap shows as 0 km/h instead of an absurd spike.
MAX_PLAUSIBLE_SPEED_KMH = 45

# ===========================
# TEAM COLORS
# ===========================

# These are just internal dictionary keys now -- the actual label
# shown on screen ("Red", "Sky Blue", "White", etc.) and the actual
# box color are both derived from the real detected jersey color in
# team_classifier.py, not hardcoded here.
TEAM1 = "team_1"
TEAM2 = "team_2"

# Reference palette used to turn a detected jersey RGB into a human
# readable name (nearest color by distance). Add more entries here if
# your kits use a color not well covered below.
COLOR_NAME_PALETTE = {
    "Red": (200, 30, 30),
    "Dark Red / Maroon": (120, 30, 40),
    "Blue": (30, 60, 200),
    "Sky Blue": (100, 180, 230),
    "Navy": (20, 30, 90),
    "Yellow": (230, 220, 40),
    "Orange": (230, 120, 30),
    "Green": (30, 140, 60),
    "White": (230, 230, 230),
    "Black": (25, 25, 25),
    "Gray": (140, 140, 140),
    "Purple": (120, 40, 140),
    "Pink": (230, 120, 180),
}

# ===========================
# JERSEY COLOR EXTRACTION
# ===========================

# The box around a player from a general person detector includes
# some background -- narrow the sample to the torso only (avoid the
# head/hair near the top and legs/grass near the bottom) and trim the
# left/right edges (avoid arms and background poking in at an angle).
JERSEY_CROP_Y_START = 0.15
JERSEY_CROP_Y_END = 0.50
JERSEY_CROP_X_MARGIN = 0.20

# Pitch grass is a fairly consistent green. Any pixel close to this
# reference color is excluded before clustering, so a loose box that
# catches grass behind/around the player doesn't get mistaken for the
# jersey color (this is what caused "Green"/"White" instead of the
# real kit colors).
GRASS_COLOR_RGB = (65, 140, 65)
GRASS_EXCLUSION_DISTANCE = 45

# ===========================
# TEAM COLOR ACCUMULATION
# ===========================

# A single early frame (e.g. players lined up far from camera before
# kickoff) can give a bad jersey-color read that locks in for the
# whole video. Instead, collect color samples across many frames and
# only finalize the two team colors once there's a solid pool of them.

# Only detections with a box at least this tall contribute a sample --
# small/far-away players give noisy, washed-out jersey colors.
MIN_BBOX_HEIGHT_FOR_TEAM_COLOR = 70

# How many good samples to collect (across all players, over however
# many frames it takes) before finalizing the two team colors.
TEAM_COLOR_MIN_SAMPLES = 40

# ===========================
# POSSESSION
# ===========================

# Max distance (pixels) between a player and the ball to count as
# "in possession" of it. Beyond this, no team is credited for the frame.
POSSESSION_DISTANCE_THRESHOLD = 120

# ===========================
# BALL SANITY FILTER
# ===========================

# A real football, at broadcast camera distance, is never wider than
# this many pixels. The generic pretrained model sometimes flags round
# objects in the crowd/signage as "sports ball" -- this rejects those
# obviously-too-large false positives before they get treated as the ball.
BALL_MAX_BOX_SIZE = 60

# ===========================
# DASHBOARD LAYOUT
# ===========================

# Output is a wider canvas: main video on the left (clean, no trail
# line drawn over it), a live heatmap panel top-right, and a ball
# movement-trail panel bottom-right -- so the trail is easy to read on
# its own instead of cluttering the video.
DASHBOARD_ENABLED = True

SIDE_PANEL_WIDTH = 480

# Regenerating the heatmap (Gaussian blur over the full frame) every
# single frame is expensive for no visible benefit -- refresh it every
# N frames instead. Lower = more up-to-date but slower to process.
HEATMAP_UPDATE_INTERVAL = 15

TRAIL_PANEL_BG_COLOR = (20, 60, 20)
TRAIL_PANEL_LINE_COLOR = (0, 255, 255)
PANEL_LABEL_COLOR = (0, 255, 0)

# ===========================
# GOAL DETECTION
# ===========================

# Pixel rectangles (x1, y1, x2, y2) marking each goal mouth in YOUR
# footage. These are specific to your camera angle -- there is no
# default that works for a different video. Run:
#     python calibrate_goal_zones.py
# to find them for your own footage (drag a box over each goal, press
# l/r to save, q to print the values), then paste the printed
# GOAL_ZONE_LEFT / GOAL_ZONE_RIGHT lines here.
# Leave both as None to disable goal detection.
GOAL_ZONE_LEFT = None
GOAL_ZONE_RIGHT = None

# Ignore further goal events for this many seconds after one fires, so
# the ball sitting in/near the net for several frames doesn't get
# counted as multiple goals.
GOAL_COOLDOWN_SECONDS = 5

# How many seconds the "GOAL!" banner stays on screen after it fires
GOAL_BANNER_SECONDS = 2

# How many of the most recent live-ball-possession frames to look back
# at when a goal fires, to decide which team gets credit (majority
# vote of whichever team had the ball right before it crossed the line)
GOAL_POSSESSION_LOOKBACK = 15
