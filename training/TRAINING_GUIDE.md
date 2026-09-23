# Training your own player/team detector on Kaggle

This trains a model specifically on football broadcast footage, using a public
labeled dataset (player, goalkeeper, referee, ball). It fixes the two biggest
accuracy problems with the pretrained model you've been using: it can't tell a
referee from a player, and it was never trained on football footage in the
first place.

## What you need

- A free Kaggle account: https://www.kaggle.com
- A free Roboflow account: https://roboflow.com (just for downloading the
  dataset -- you're not training on their servers, Kaggle does the training)
- About 30-60 minutes of Kaggle's free GPU time

## Steps

### 1. Get the training notebook onto Kaggle

- Go to https://www.kaggle.com/code → **New Notebook**
- Click **File → Import Notebook** and upload `kaggle_train_player_model.ipynb`
  from this folder

### 2. Turn on GPU and internet

In the right-hand panel:
- **Settings → Accelerator → GPU T4 x2**
- **Settings → Internet → On**

Without these two, the notebook will fail (no internet = can't download the
dataset or install packages; no GPU = training would take many hours instead
of under an hour).

### 3. Get your dataset download code from Roboflow

- Open https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc
- Sign in (free), click **Download Dataset**
- Choose format **YOLO26** (fall back to **YOLOv11** if YOLO26 isn't offered as
  an export format yet -- the label file format barely changes between YOLO
  versions, only the `version.download(...)` string and the model file we
  train from need to match YOLO26)
- Choose **"Show download code"** (not "zip") -- this gives you a Python
  snippet with your own API key and the correct dataset version filled in
- Copy that snippet

### 4. Paste it into the notebook

In the second code cell (the one with `PASTE_YOUR_API_KEY_HERE`), replace the
placeholder code with the snippet you just copied from Roboflow.

### 5. Run all cells

**Run → Run All** (or run each cell top to bottom). Watch for:
- The class list printed after download -- should show something like
  `['ball', 'goalkeeper', 'player', 'referee']`
- Training progress -- you'll see loss numbers dropping over 50 epochs
- A final `mAP50` score after validation -- above ~0.7 is solid for this
  dataset size

### 6. Download your trained model

The last cell copies your trained weights to a file called
`yolo26_player.pt` in the notebook's **Output** panel (right sidebar).
Click it to download.

## Using it in your project

1. Move the downloaded `yolo26_player.pt` into your project's `models/`
   folder (replacing the placeholder that was there).

2. Open `utils/config.py` and change:
   ```python
   USE_PRETRAINED = False
   ```

3. Still in `config.py`, update the class filter to match what your model
   actually detects. The notebook printed the class order for you (something
   like `['ball', 'goalkeeper', 'player', 'referee']`) -- use that exact
   order and index to set:
   ```python
   # Example if the printed order was ['ball', 'goalkeeper', 'player', 'referee']:
   PLAYER_CLASSES = [1, 2, 3]   # goalkeeper, player, referee -- all people on the pitch
   ```
   (Leave `BALL_MODEL`/`BALL_CLASSES` alone for now -- you said ball training
   is next. Your new player model actually includes a `ball` class already
   from this same dataset, so once you're ready you could point `BALL_MODEL`
   at this same `yolo26_player.pt` file and set `BALL_CLASSES = [0]` to try
   it immediately, before doing separate ball-specific training.)

4. Run `python detection.py` as normal.

## A note on referees

Because this model has a separate `referee` class instead of lumping everyone
into one "person" class, you can now genuinely tell them apart -- something
the old pretrained model could never do. `tracker.py` currently keeps
`PLAYER_CLASSES` as one combined list for tracking; if you want referees drawn
differently (e.g. a distinct color instead of a team color), that's a small
follow-up change to `detection.py` -- ask if you want that wired in once your
model is trained.
