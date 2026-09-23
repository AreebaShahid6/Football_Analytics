import colorsys

import cv2
import numpy as np
from sklearn.cluster import KMeans

from utils.config import (
    TEAM1,
    TEAM2,
    COLOR_NAME_PALETTE,
    JERSEY_CROP_Y_START,
    JERSEY_CROP_Y_END,
    JERSEY_CROP_X_MARGIN,
    GRASS_COLOR_RGB,
    GRASS_EXCLUSION_DISTANCE,
    MIN_BBOX_HEIGHT_FOR_TEAM_COLOR,
    TEAM_COLOR_MIN_SAMPLES
)

# Below this saturation (0-1 scale), a color is genuinely neutral
# (gray/white/black) rather than a washed-out version of a real hue.
NEUTRAL_SATURATION_THRESHOLD = 0.18


def _rgb_to_hsv(rgb):

    r, g, b = [c / 255.0 for c in rgb]

    return colorsys.rgb_to_hsv(r, g, b)


def nearest_color_name(rgb):
    """
    Maps a detected jersey RGB color to the closest human-readable
    name in COLOR_NAME_PALETTE, matching by HUE rather than raw RGB
    distance. Raw RGB distance is a bad fit for compressed broadcast
    video: motion blur and compression desaturate colors, and any
    desaturated color sits numerically close to "Gray" in RGB space
    regardless of its actual hue (e.g. a true deep maroon compresses
    toward a dull brownish-gray, which raw distance mislabels as
    "Gray" every time). Matching by hue avoids that bias -- a muted
    reddish tone still reads as red/maroon, not gray, as long as it
    has any real saturation left.
    """

    h, s, v = _rgb_to_hsv(rgb)

    # Genuinely low-saturation colors really are neutral -- decide
    # between white/gray/black by brightness alone
    if s < NEUTRAL_SATURATION_THRESHOLD:
        if v > 0.75:
            return "White"
        if v < 0.25:
            return "Black"
        return "Gray"

    # Otherwise, match against whichever palette colors actually have
    # a real hue (skip comparing to neutral palette entries), using
    # circular hue distance so e.g. red (hue ~0) and hue ~0.98 are
    # recognized as close instead of far apart
    best_name = None
    best_distance = float("inf")

    for name, ref_rgb in COLOR_NAME_PALETTE.items():

        ref_h, ref_s, ref_v = _rgb_to_hsv(ref_rgb)

        if ref_s < NEUTRAL_SATURATION_THRESHOLD:
            continue

        hue_distance = min(abs(h - ref_h), 1 - abs(h - ref_h))

        if hue_distance < best_distance:
            best_distance = hue_distance
            best_name = name

    return best_name or "Gray"


def _is_grass(rgb):

    distance = np.linalg.norm(np.array(rgb) - np.array(GRASS_COLOR_RGB))

    return distance < GRASS_EXCLUSION_DISTANCE


class TeamClassifier:

    def __init__(self):

        self.team_colors = {}       # team key -> detected RGB (for classifying)
        self.team_labels = {}       # team key -> human readable name, e.g. "Sky Blue"
        self.team_draw_colors = {}  # team key -> BGR tuple (for cv2 drawing)

        # Jersey color samples collected across frames, only from
        # boxes tall enough to give a clean read -- accumulated until
        # there's enough to finalize the two team colors reliably
        self._sample_colors = []

    def extract_jersey_color(self, frame, bbox):

        x1, y1, x2, y2 = bbox

        h = y2 - y1
        w = x2 - x1

        # Torso-only crop: skips the head/hair near the top and the
        # legs/grass near the bottom, and trims the left/right edges
        # so background poking in around the arms doesn't get sampled
        x_margin = int(w * JERSEY_CROP_X_MARGIN)

        jersey = frame[
            y1 + int(h * JERSEY_CROP_Y_START): y1 + int(h * JERSEY_CROP_Y_END),
            x1 + x_margin: x2 - x_margin
        ]

        if jersey.size == 0:
            return None

        jersey = cv2.cvtColor(jersey, cv2.COLOR_BGR2RGB)

        pixels = jersey.reshape((-1, 3))

        # Drop any pixel close to pitch-grass green before clustering,
        # so a loose box that catches grass around the player doesn't
        # get mistaken for the jersey color
        non_grass_mask = np.array([not _is_grass(p) for p in pixels])
        pixels = pixels[non_grass_mask]

        if len(pixels) < 20:
            return None

        # If the sampled pixels are (near-)uniform, there's nothing
        # for a 2-cluster split to find -- just use the average color
        # directly instead of forcing KMeans into 2 clusters (which
        # triggers a ConvergenceWarning and doesn't help here anyway)
        if len(np.unique(pixels, axis=0)) < 2:
            dominant = pixels.mean(axis=0).astype(int)

            if _is_grass(dominant):
                return None

            return dominant

        kmeans = KMeans(
            n_clusters=2,
            random_state=42,
            n_init=10
        )

        kmeans.fit(pixels)

        labels = kmeans.labels_

        counts = np.bincount(labels)

        dominant = kmeans.cluster_centers_[np.argmax(counts)]

        dominant = dominant.astype(int)

        # Safety net: if even the dominant cluster still landed on
        # grass green (e.g. box was mostly background), reject it
        # rather than letting it pollute team clustering
        if _is_grass(dominant):
            return None

        return dominant

    def collect_samples(self, detections, frame):
        """
        Extracts jersey colors from any detection with a tall-enough
        box (skips small/far players that give noisy color reads) and
        adds them to the running sample pool. Call this every frame
        until try_finalize_teams() returns True.
        """

        for det in detections:

            x1, y1, x2, y2 = det["bbox"]

            if (y2 - y1) < MIN_BBOX_HEIGHT_FOR_TEAM_COLOR:
                continue

            color = self.extract_jersey_color(frame, det["bbox"])

            if color is not None:
                self._sample_colors.append(color)

    def try_finalize_teams(self):
        """
        Once enough good samples have been collected, splits them into
        two team colors. Returns True once finalized, False if there
        aren't enough samples yet (keep calling collect_samples and
        try again on later frames).
        """

        if len(self._sample_colors) < TEAM_COLOR_MIN_SAMPLES:
            return False

        colors = np.array(self._sample_colors)

        if len(np.unique(colors, axis=0)) < 2:
            # Everything collected so far came back the same color --
            # unusual, but keep collecting rather than finalizing on
            # bad data
            return False

        kmeans = KMeans(
            n_clusters=2,
            random_state=42,
            n_init=10
        )

        kmeans.fit(colors)

        for team_key, center_rgb in zip((TEAM1, TEAM2), kmeans.cluster_centers_):

            r, g, b = center_rgb.astype(int)

            self.team_colors[team_key] = center_rgb
            self.team_labels[team_key] = nearest_color_name((r, g, b))
            # cv2 draws in BGR, so flip the detected RGB back for drawing
            self.team_draw_colors[team_key] = (int(b), int(g), int(r))

        return True

    def get_label(self, team_key):

        return self.team_labels.get(team_key, team_key)

    def get_draw_color(self, team_key):

        return self.team_draw_colors.get(team_key, (0, 255, 0))

    def classify(self, detection, frame):

        if len(self.team_colors) == 0:
            return TEAM1, self.get_draw_color(TEAM1), self.get_label(TEAM1)

        color = self.extract_jersey_color(
            frame,
            detection["bbox"]
        )

        if color is None:
            return TEAM1, self.get_draw_color(TEAM1), self.get_label(TEAM1)

        d1 = np.linalg.norm(color - self.team_colors[TEAM1])
        d2 = np.linalg.norm(color - self.team_colors[TEAM2])

        team_key = TEAM1 if d1 < d2 else TEAM2

        return team_key, self.get_draw_color(team_key), self.get_label(team_key)
