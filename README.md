# Football Analytics (YOLO)

## 1. Folder structure

Everything must sit exactly like this, with `detection.py` at the top
level (NOT inside `utils`) — `detection.py` imports `utils.xxx`, so it
has to be run from this root folder so Python can see `utils` as a
package next to it.

```
Football_Analytics/
├── detection.py
├── dashboard.py
├── requirements.txt
├── models/
│   ├── yolo26_player.pt
│   ├── yolo26_ball.pt
│   └── yolo26_pose.pt
├── videos/
│   ├── football.mp4        <- your input video goes here
│   ├── output.mp4           <- generated automatically
│   └── heatmap.png          <- generated automatically
└── utils/
    ├── __init__.py
    ├── config.py
    ├── tracker.py
    ├── ball_tracker.py
    ├── pose_utils.py
    ├── team_classifier.py
    ├── speed.py
    ├── heatmap.py
    └── drawing.py
```

Your models must be named exactly as in `utils/config.py`
(`yolo26_player.pt`, `yolo26_ball.pt`, `yolo26_pose.pt`) or you edit
those paths in `config.py`.

## 2. What each model needs to detect

- `yolo26_player.pt` — a model trained/fine-tuned to detect players
  (and ideally referees as a separate class). A general pretrained
  YOLO "person" detector will work but will be less accurate for
  overlapping players and won't tell players from referees.
- `yolo26_ball.pt` — a model fine-tuned specifically on the ball. The
  ball is small and fast, so a generic detector performs poorly here —
  this is the model worth training/fine-tuning most carefully for
  accuracy.
- `yolo26_pose.pt` — any YOLO pose model (e.g. Ultralytics' built-in
  `yolo11n-pose.pt` / `yolov8n-pose.pt` works out of the box, no
  fine-tuning needed, for player pose/skeleton overlays).

If you don't have custom-trained weights yet, you can start with
Ultralytics' pretrained checkpoints (`yolo11n.pt` for players,
`yolo11n-pose.pt` for pose) to get the pipeline running, then swap in
your fine-tuned `yolo26_*` weights for real accuracy on football
footage.

## 3. Setup (terminal commands)

```bash
cd Football_Analytics

python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the CUDA build of PyTorch first
(check your CUDA version at https://pytorch.org/get-started/locally/)
before `pip install -r requirements.txt`, otherwise Ultralytics will
fall back to CPU-only PyTorch and processing will be much slower.

## 4. Run

Put your input video at `videos/football.mp4`, then from the
`Football_Analytics` root folder:

```bash
python detection.py
```

Output:
- `videos/output.mp4` — annotated video (boxes, IDs, teams, speed,
  ball trail, pose skeletons, possession %, FPS)
- `videos/heatmap.png` — accumulated player movement heatmap

## 5. Accuracy tuning

All of these live in `utils/config.py`:

- `PLAYER_CONF` / `BALL_CONF` / `POSE_CONF` — raise to cut false
  positives, lower to catch more (harder) detections.
- `IMG_SIZE` — higher (e.g. 1280 → 1536) improves small/far-away
  detections (useful for the ball) at the cost of speed.
- `TRACKER_TYPE` — `bytetrack.yaml` (default, fast) vs `botsort.yaml`
  (slower, better at re-identifying a player after they're briefly
  hidden behind another player).
- `POSSESSION_DISTANCE_THRESHOLD` — pixel distance under which a
  player is considered "in possession" of the ball; tune based on
  your video's resolution/camera distance.
- `PIXELS_PER_METER` — must be calibrated against your actual camera
  view (measure a known real-world distance, e.g. the penalty box
  width, in pixels in your footage) or speed/distance numbers will be
  wrong even though detection is accurate.

## 6. Basketball / other sports

The pipeline (tracking, pose, speed, heatmap) is sport-agnostic — only
`team_classifier.py`'s jersey-color logic and `PIXELS_PER_METER`
calibration are football-specific. To reuse for basketball: swap in a
basketball-trained player/ball model, and re-measure
`PIXELS_PER_METER` against a known court dimension (e.g. the free
throw line to backboard distance).
