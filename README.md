# ⚽ Football Analytics — YOLO26 Computer Vision

<p align="center">
  <b>AI-Powered Football Video Analysis using YOLO, Object Tracking, Pose Estimation & Computer Vision</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/YOLO-Computer%20Vision-red?style=for-the-badge" alt="YOLO">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/Deep%20Learning-PyTorch-orange?style=for-the-badge&logo=pytorch" alt="PyTorch">
</p>

---

## 📌 Overview

**Football Analytics** is an AI-powered computer vision system designed to analyze football matches from video footage.

The system uses **YOLO-based object detection, player tracking, ball tracking, pose estimation, team classification, speed estimation, heatmaps, and possession analysis** to extract meaningful information from football videos.

The project is designed as a modular pipeline so individual components such as detection, tracking, pose estimation, and analytics can be improved independently.

---

## 🎯 Key Features

* ⚽ **Player Detection**
* 🏃 **Player Tracking with Unique IDs**
* 🥅 **Football Detection & Tracking**
* 👕 **Team Classification using Jersey Colors**
* 🦴 **Player Pose Estimation**
* 📍 **Player Movement Tracking**
* 🔥 **Movement Heatmap Generation**
* 💨 **Player Speed Estimation**
* 📊 **Ball Possession Analysis**
* 📏 **Real-World Distance Calibration**
* 🎥 **Annotated Output Video**
* 📈 **Real-Time Analytics Overlay**
* 🔧 **Configurable Detection & Tracking Parameters**

---

## 🧠 Computer Vision Pipeline

```text
                    Football Video
                          │
                          ▼
                ┌──────────────────┐
                │  YOLO Detection  │
                └────────┬─────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      Players          Ball           Pose
          │              │              │
          ▼              ▼              ▼
      Tracking       Ball Tracking   Skeleton
          │              │              │
          └───────┬──────┴──────────────┘
                  ▼
          Team Classification
                  │
                  ▼
          Speed & Distance
              Estimation
                  │
                  ▼
         Possession Analysis
                  │
                  ▼
            Heatmap Generation
                  │
                  ▼
          Annotated Video
```

---

# 📂 Project Structure

```text
Football_Analytics/
│
├── detection.py
├── calibrate_goal_zones.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   └── .gitkeep
│
├── training/
│   ├── TRAINING_GUIDE.md
│   └── kaggle_train_player_model.ipynb
│
└── utils/
    ├── __init__.py
    ├── config.py
    ├── tracker.py
    ├── ball_tracker.py
    ├── pose_utils.py
    ├── team_classifier.py
    ├── speed.py
    ├── heatmap.py
    ├── drawing.py
    └── dashboard.py
```

> **Note:** Large model weights (`.pt`, `.pth`, `.onnx`) and video files are excluded from this repository through `.gitignore`.

---

# 🔍 Detection Models

The project can use separate models for different computer vision tasks.

### 👤 Player Detection

A YOLO model trained or fine-tuned for detecting football players.

Example:

```text
models/yolo26_player.pt
```

A general person-detection model can be used for initial testing, but a football-specific model can provide better performance in crowded scenes and player overlaps.

### ⚽ Ball Detection

A model specifically trained or fine-tuned for football detection.

```text
models/yolo26_ball.pt
```

Football detection is challenging because the ball is usually:

* Small
* Fast-moving
* Frequently occluded
* Similar in appearance to other objects
* Difficult to detect at long distances

For this reason, a football-specific model is recommended.

### 🦴 Pose Estimation

A YOLO pose model can be used for player skeleton/keypoint detection.

Example:

```text
models/yolo26_pose.pt
```

A compatible pretrained YOLO pose model can also be used for initial testing.

---

# 🛠️ Technologies Used

| Technology       | Purpose                              |
| ---------------- | ------------------------------------ |
| Python           | Core programming language            |
| YOLO             | Object detection and pose estimation |
| OpenCV           | Video processing and computer vision |
| PyTorch          | Deep learning framework              |
| NumPy            | Numerical processing                 |
| Pandas           | Data processing                      |
| ByteTrack        | Object tracking                      |
| BoT-SORT         | Alternative object tracking          |
| Matplotlib       | Heatmap/visualization                |
| Jupyter Notebook | Model training and experimentation   |

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/AreebaShahid6/Football_Analytics.git
```

Move into the project:

```bash
cd Football_Analytics
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### NVIDIA GPU

For faster inference, install the appropriate CUDA-enabled PyTorch version according to your NVIDIA/CUDA configuration before installing the remaining dependencies.

---

# 🎥 Input Video

Place your football video inside the local:

```text
videos/
```

directory.

For example:

```text
videos/
└── football.mp4
```

The `videos/` directory is intentionally excluded from GitHub because video files can be very large.

---

# ▶️ Run the System

From the project root directory:

```bash
python detection.py
```

The system processes the football video and performs the configured detection and analytics tasks.

---

# 📊 Output

Depending on the configuration, the system can generate:

### 🎥 Annotated Video

```text
videos/output.mp4
```

The video can contain:

* Player bounding boxes
* Player IDs
* Team labels
* Ball tracking
* Player speed
* Possession information
* Pose skeletons
* FPS information

### 🔥 Movement Heatmap

```text
videos/heatmap.png
```

The heatmap represents accumulated player movement across the field.

---

# 📈 Analytics

## 👤 Player Tracking

Players are assigned tracking IDs so their movement can be followed across video frames.

Example:

```text
Player #1
Player #2
Player #3
...
```

Tracking helps maintain player identity even when players move throughout the scene.

---

## 👕 Team Classification

The system can classify teams using jersey-color information.

The classification logic is implemented in:

```text
utils/team_classifier.py
```

This allows the analytics pipeline to distinguish players belonging to different teams.

---

## ⚽ Ball Tracking

The ball tracker follows the detected football across frames.

Implemented in:

```text
utils/ball_tracker.py
```

Because the football is small and moves quickly, detection quality depends heavily on the training data and video resolution.

---

## 💨 Speed Estimation

Player movement can be converted into an approximate speed using pixel-to-meter calibration.

Implemented in:

```text
utils/speed.py
```

The accuracy of speed estimation depends on proper camera calibration.

---

## 🔥 Heatmap

The system records player movement and generates a visual heatmap.

Implemented in:

```text
utils/heatmap.py
```

Heatmaps can help visualize areas of the field where players spend more time.

---

## 🦴 Pose Estimation

Pose estimation identifies body keypoints and can generate player skeleton overlays.

Implemented in:

```text
utils/pose_utils.py
```

This can be useful for analyzing player movement and body positioning.

---

## 📊 Dashboard

Dashboard-related visualization functionality is located in:

```text
utils/dashboard.py
```

---

# ⚙️ Configuration

Most system parameters can be configured from:

```text
utils/config.py
```

Important parameters include:

### Detection Confidence

```python
PLAYER_CONF
BALL_CONF
POSE_CONF
```

Higher values:

```text
More strict detection
↓
Fewer false positives
```

Lower values:

```text
More sensitive detection
↓
May increase false positives
```

---

## 🖼️ Image Size

The inference image size can affect detection performance.

For example:

```python
IMG_SIZE = 1280
```

Increasing image size can help detect small objects such as the football, but it also increases computational requirements.

---

# 🎯 Object Tracking

The project can use different tracking algorithms.

### ByteTrack

```text
bytetrack.yaml
```

Advantages:

* Fast
* Lightweight
* Suitable for real-time applications

### BoT-SORT

```text
botsort.yaml
```

Can provide stronger re-identification behavior when players are temporarily occluded.

The tracker configuration can be adjusted in:

```text
utils/config.py
```

---

# 📏 Camera Calibration

Accurate speed and distance estimation requires camera calibration.

The project includes:

```text
calibrate_goal_zones.py
```

The calibration should use a known real-world distance visible in the camera.

For example:

```text
Known field distance
        ↓
Measure pixels
        ↓
Calculate pixel/meter ratio
        ↓
Use ratio for speed/distance estimation
```

The `PIXELS_PER_METER` parameter should be calibrated for the actual camera view.

> Incorrect calibration can result in incorrect speed and distance values even when player detection is accurate.

---

# 🧪 Model Training

Training resources are included in:

```text
training/
```

### Training Guide

```text
training/TRAINING_GUIDE.md
```

### Kaggle Training Notebook

```text
training/kaggle_train_player_model.ipynb
```

These resources can be used to train or fine-tune football-specific detection models.

---

# 🎯 Accuracy Improvement

The following techniques can improve performance:

### Player Detection

* Use football-specific training data
* Increase image resolution
* Add crowded-scene examples
* Include different camera angles
* Include partially occluded players

### Ball Detection

* Use high-resolution training images
* Include small-ball examples
* Include motion blur
* Include different lighting conditions
* Train on multiple camera angles

### Tracking

* Tune confidence thresholds
* Test ByteTrack and BoT-SORT
* Adjust tracker parameters
* Improve detection quality

### Speed Estimation

* Calibrate the camera correctly
* Use accurate field dimensions
* Avoid using an arbitrary pixel-to-meter ratio

---

# 🏀 Extending to Other Sports

The overall pipeline is designed to be adaptable.

It can potentially be extended to:

* 🏀 Basketball
* 🏑 Hockey
* ⚾ Baseball
* 🎾 Tennis
* 🏉 Rugby

For another sport, the main components that normally need adjustment are:

```text
Detection model
        ↓
Ball/object model
        ↓
Team classification
        ↓
Camera calibration
```

For example, basketball would require a basketball-trained player/ball detector and calibration based on known court dimensions.

---

# 🚀 Future Improvements

Planned improvements can include:

* [ ] Advanced player re-identification
* [ ] Improved football detection
* [ ] Automatic field homography
* [ ] Tactical formation detection
* [ ] Player statistics
* [ ] Pass detection
* [ ] Shot detection
* [ ] Goal detection
* [ ] Automated highlight generation
* [ ] Advanced possession analytics
* [ ] Real-time analytics dashboard
* [ ] Web-based football analytics interface

---

# 📁 Files & Model Weights

Large files are intentionally excluded from GitHub.

The `.gitignore` file excludes:

```text
venv/
videos/
*.pt
*.pth
*.onnx
runs/
outputs/
```

Therefore, users should place their required model weights locally inside:

```text
models/
```

and their input videos inside:

```text
videos/
```

before running the application.

---

# 💡 Example Workflow

```text
Football Video
      │
      ▼
YOLO Detection
      │
      ├── Players
      ├── Ball
      └── Pose
      │
      ▼
Object Tracking
      │
      ▼
Team Classification
      │
      ├── Speed
      ├── Distance
      ├── Possession
      └── Movement
      │
      ▼
Analytics
      │
      ├── Annotated Video
      └── Heatmap
```

---

# 👩‍💻 Author

### Areeba Shahid

**Computer Science Graduate | Computer Vision Engineer | Machine Learning Enthusiast**

Passionate about building intelligent systems using:

* 🤖 Artificial Intelligence
* 👁️ Computer Vision
* 🧠 Deep Learning
* 📊 Machine Learning
* 🌐 IoT

### Connect With Me

🔗 **LinkedIn:**
https://www.linkedin.com/in/areeba-shahid-1b53b231/

📧 **Email:**
[shahidareeba922@gmail.com](mailto:shahidareeba922@gmail.com)

---

# ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

<p align="center">
  <b>⚽ Turning Football Videos into Intelligent Data with Computer Vision</b>
</p>
