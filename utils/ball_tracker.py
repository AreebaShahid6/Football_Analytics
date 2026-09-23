from collections import deque

from ultralytics import YOLO

from utils.config import (
    BALL_MODEL,
    BALL_CONF,
    IMG_SIZE,
    BALL_MAX_BOX_SIZE,
    BALL_TEST_TIME_AUGMENT,
    USE_PRETRAINED_BALL,
    COCO_SPORTS_BALL_CLASS
)


class BallTracker:
    """
    Detects the ball with a dedicated small-object model. The ball is
    tiny and moves fast, so detections are noisy/intermittent -- this
    keeps the last known position for a few frames instead of losing
    the ball every time a single frame misses, and keeps a trail for
    drawing motion.

    When using a custom-trained model, the "ball" class index is read
    directly from the model itself (model.names) instead of being
    hardcoded in config.py -- export tools don't always order classes
    the same way, so hardcoding is fragile.
    """

    def __init__(self, trail_length=20, max_missed_frames=15):

        self.model = YOLO(str(BALL_MODEL))

        if USE_PRETRAINED_BALL:
            self.classes_filter = [COCO_SPORTS_BALL_CLASS]
        else:
            self.classes_filter = self._resolve_custom_classes()

        self.trail = deque(maxlen=trail_length)
        self.last_known = None
        self.missed_frames = 0
        self.max_missed_frames = max_missed_frames

    def _resolve_custom_classes(self):

        names = self.model.names

        print(f"Custom ball model classes (from the model file itself): {names}")

        name_to_index = {str(name).strip().lower(): idx for idx, name in names.items()}

        if "ball" in name_to_index:
            ball_idx = name_to_index["ball"]
            print(f"Tracking ball class index: {ball_idx}")
            return [ball_idx]

        print(
            "WARNING: no class named 'ball' found in the model. "
            f"Available classes: {list(name_to_index.keys())} -- "
            "detecting ALL classes so nothing is silently dropped."
        )
        return None

    def detect(self, frame):

        results = self.model.predict(
            frame,
            conf=BALL_CONF,
            imgsz=IMG_SIZE,
            classes=self.classes_filter,
            augment=BALL_TEST_TIME_AUGMENT,
            verbose=False
        )

        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return self._handle_miss()

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        # Reject any candidate whose box is bigger than a real ball
        # could ever appear at this camera distance (filters out round
        # objects in the crowd/signage the generic model mistook for it)
        valid_idxs = [
            i for i, box in enumerate(xyxy)
            if (box[2] - box[0]) <= BALL_MAX_BOX_SIZE
            and (box[3] - box[1]) <= BALL_MAX_BOX_SIZE
        ]

        if not valid_idxs:
            return self._handle_miss()

        # Among valid-sized candidates, keep the highest-confidence one
        best_idx = max(valid_idxs, key=lambda i: confs[i])

        x1, y1, x2, y2 = xyxy[best_idx].astype(int)
        conf = float(confs[best_idx])

        center = (
            int((x1 + x2) / 2),
            int((y1 + y2) / 2)
        )

        self.last_known = {
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
            "center": center,
            "confidence": conf
        }

        self.missed_frames = 0
        self.trail.append(center)

        return self._result(live=True)

    def _handle_miss(self):

        self.missed_frames += 1

        if self.missed_frames > self.max_missed_frames:
            self.last_known = None
            self.trail.clear()

        return self._result(live=False)

    def _result(self, live):

        if self.last_known is None:
            return None

        return {
            "bbox": self.last_known["bbox"],
            "center": self.last_known["center"],
            "confidence": self.last_known["confidence"],
            "trail": list(self.trail),
            # True only when the ball was actually detected this frame,
            # False when this is a carried-over position from a recent
            # miss. Callers should use this to avoid counting stale
            # positions toward frame-accurate stats like possession.
            "live": live
        }

    def reset(self):
        self.trail.clear()
        self.last_known = None
        self.missed_frames = 0
