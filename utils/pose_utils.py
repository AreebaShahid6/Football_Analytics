import cv2
from ultralytics import YOLO

from utils.config import (
    POSE_MODEL,
    POSE_CONF,
    IMG_SIZE
)

# COCO 17-keypoint skeleton connections (standard for YOLO pose models)
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4)
]

KEYPOINT_CONF_THRESHOLD = 0.3


class PoseEstimator:
    """
    Runs YOLO pose estimation on a frame and returns keypoints per
    detected person, along with their bounding box so callers can
    match a pose to a specific tracked player if needed.
    """

    def __init__(self):

        self.model = YOLO(str(POSE_MODEL))

    def detect(self, frame):

        results = self.model.predict(
            frame,
            conf=POSE_CONF,
            imgsz=IMG_SIZE,
            verbose=False
        )

        poses = []

        result = results[0]

        if result.keypoints is None or result.boxes is None:
            return poses

        keypoints_xy = result.keypoints.xy.cpu().numpy()

        keypoints_conf = None
        if result.keypoints.conf is not None:
            keypoints_conf = result.keypoints.conf.cpu().numpy()

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        boxes_conf = result.boxes.conf.cpu().numpy()

        for i, kpts in enumerate(keypoints_xy):

            x1, y1, x2, y2 = boxes_xyxy[i].astype(int)

            poses.append({
                "keypoints": kpts,
                "kpt_conf": keypoints_conf[i] if keypoints_conf is not None else None,
                "confidence": float(boxes_conf[i]),
                "bbox": (int(x1), int(y1), int(x2), int(y2))
            })

        return poses


def _visible(kpts, kpt_conf, idx):

    x, y = kpts[idx]

    if x <= 0 or y <= 0:
        return False

    if kpt_conf is not None and kpt_conf[idx] < KEYPOINT_CONF_THRESHOLD:
        return False

    return True


def draw_pose(frame, pose, color=(0, 255, 0), point_color=(0, 0, 255)):

    keypoints = pose["keypoints"]
    kpt_conf = pose.get("kpt_conf")

    for idx in range(len(keypoints)):

        if not _visible(keypoints, kpt_conf, idx):
            continue

        x, y = keypoints[idx]

        cv2.circle(frame, (int(x), int(y)), 3, point_color, -1)

    for a, b in SKELETON:

        if a >= len(keypoints) or b >= len(keypoints):
            continue

        if not _visible(keypoints, kpt_conf, a):
            continue

        if not _visible(keypoints, kpt_conf, b):
            continue

        xa, ya = keypoints[a]
        xb, yb = keypoints[b]

        cv2.line(
            frame,
            (int(xa), int(ya)),
            (int(xb), int(yb)),
            color,
            2
        )
