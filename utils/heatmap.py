import cv2
import numpy as np

from utils.config import HEATMAP_ALPHA


class HeatmapGenerator:

    def __init__(self):

        # Store all player positions (used for the original single-color heatmap)
        self.positions = {}

        # Store positions grouped by team key, so movement can be
        # shown in each team's own detected jersey color
        self.team_positions = {}

    def update(self, player_id, center):

        if center is None:
            return

        if player_id not in self.positions:
            self.positions[player_id] = []

        self.positions[player_id].append(center)

    def update_team(self, team_key, center):

        if center is None:
            return

        if team_key not in self.team_positions:
            self.team_positions[team_key] = []

        self.team_positions[team_key].append(center)

    def get_positions(self, player_id):

        return self.positions.get(player_id, [])

    def draw_player_path(self, frame, player_id, color=(0, 255, 255)):

        points = self.get_positions(player_id)

        if len(points) < 2:
            return frame

        for i in range(1, len(points)):

            cv2.line(
                frame,
                points[i - 1],
                points[i],
                color,
                2
            )

        return frame

    def _density_map(self, height, width, points):

        heat = np.zeros((height, width), dtype=np.float32)

        for x, y in points:

            if 0 <= x < width and 0 <= y < height:

                cv2.circle(heat, (x, y), 25, 1, -1)

        if np.max(heat) == 0:
            return heat

        heat = heat / np.max(heat)

        heat = cv2.GaussianBlur(heat, (51, 51), 0)

        return heat

    def generate_heatmap(self, frame):

        height, width = frame.shape[:2]

        all_points = [p for player in self.positions.values() for p in player]

        heat = self._density_map(height, width, all_points)

        if np.max(heat) == 0:
            return frame

        heat_uint8 = np.uint8(255 * heat)

        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

        output = cv2.addWeighted(
            frame,
            1 - HEATMAP_ALPHA,
            heat_color,
            HEATMAP_ALPHA,
            0
        )

        return output

    def generate_team_heatmap(self, frame, team_draw_colors):
        """
        Same idea as generate_heatmap, but each team's density is
        tinted with that team's own detected jersey color (from
        TeamClassifier.get_draw_color) instead of one shared JET
        colormap -- so the two teams' territory is visually distinct.

        team_draw_colors: dict of {team_key: (b, g, r)} matching the
        keys used in update_team().
        """

        height, width = frame.shape[:2]

        overlay = np.zeros((height, width, 3), dtype=np.float32)

        any_data = False

        for team_key, points in self.team_positions.items():

            heat = self._density_map(height, width, points)

            if np.max(heat) == 0:
                continue

            any_data = True

            color = np.array(
                team_draw_colors.get(team_key, (0, 255, 0)),
                dtype=np.float32
            )

            # Broadcast the team color scaled by density at each pixel
            overlay += heat[..., None] * color

        if not any_data:
            return frame

        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        output = cv2.addWeighted(
            frame,
            1 - HEATMAP_ALPHA,
            overlay,
            HEATMAP_ALPHA,
            0
        )

        return output

    def reset(self):

        self.positions.clear()
        self.team_positions.clear()
