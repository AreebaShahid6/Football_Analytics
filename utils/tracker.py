from ultralytics import YOLO

from utils.config import (
    PLAYER_MODEL,
    PLAYER_CONF,
    IMG_SIZE,
    TRACKER_TYPE,
    USE_PRETRAINED_PLAYER,
    COCO_PERSON_CLASS
)


class PlayerTracker:
    """
    Detects players/referees with a YOLO model and assigns each one a
    stable ID across frames using Ultralytics' built-in ByteTrack.

    When using a custom-trained model, class indices are read directly
    from the model itself (model.names) instead of being hardcoded in
    config.py -- Roboflow/Kaggle exports don't always order classes
    alphabetically, so hardcoding indices is fragile and silently
    drops detections if the order doesn't match what was assumed.
    """

    def __init__(self):

        self.model = YOLO(str(PLAYER_MODEL))

        self.referee_class_index = None

        if USE_PRETRAINED_PLAYER:
            self.classes_filter = [COCO_PERSON_CLASS]
        else:
            self.classes_filter = self._resolve_custom_classes()

    def _resolve_custom_classes(self):

        # model.names is a {index: class_name} dict Ultralytics embeds
        # in every model file -- this is the ground truth for what
        # index means what, straight from your trained weights
        names = self.model.names

        print(f"Custom player model classes (from the model file itself): {names}")

        name_to_index = {str(name).strip().lower(): idx for idx, name in names.items()}

        wanted = ("player", "goalkeeper", "referee")
        resolved = {}

        for label in wanted:
            if label in name_to_index:
                resolved[label] = name_to_index[label]
            else:
                print(
                    f"WARNING: no class named '{label}' found in the model. "
                    f"Available classes: {list(name_to_index.keys())}"
                )

        self.referee_class_index = resolved.get("referee")

        classes_filter = [idx for idx in resolved.values()]

        if not classes_filter:
            print(
                "WARNING: couldn't match any expected class names "
                "(player/goalkeeper/referee) in the model -- falling "
                "back to detecting ALL classes so nothing is silently "
                "dropped. Check the printed class list above."
            )
            return None

        print(f"Tracking class indices: {classes_filter} (referee = {self.referee_class_index})")

        return classes_filter

    def track(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            conf=PLAYER_CONF,
            imgsz=IMG_SIZE,
            tracker=TRACKER_TYPE,
            classes=self.classes_filter,
            verbose=False
        )

        detections = []

        boxes = results[0].boxes

        # No detections at all, or tracker hasn't assigned IDs yet
        if boxes is None or boxes.id is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        for box, track_id, conf, cls in zip(xyxy, ids, confs, classes):

            x1, y1, x2, y2 = box.astype(int)

            # Bottom-center point (approx. foot position) is far more
            # accurate than the box center for on-pitch location, and
            # is what speed/heatmap/possession calculations should use.
            center = (
                int((x1 + x2) / 2),
                int(y2)
            )

            detections.append({
                "id": int(track_id),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "confidence": float(conf),
                "class": int(cls),
                "center": center
            })

        return detections

    def reset(self):
        # Clears ByteTrack's internal track history (call this if you
        # restart tracking on a new video without recreating the object)
        self.model.predictor = None
