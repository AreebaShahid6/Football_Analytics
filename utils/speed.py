import math

from utils.config import (
    FPS,
    PIXELS_PER_METER,
    MIN_MOVEMENT_PIXELS,
    MAX_PLAUSIBLE_SPEED_KMH
)


class SpeedEstimator:

    def __init__(self):

        # Store previous center of each player
        self.previous_positions = {}

        # Total distance for each player
        self.total_distance = {}

    def calculate_speed(self, player_id, center):

        if player_id not in self.previous_positions:

            self.previous_positions[player_id] = center
            self.total_distance[player_id] = 0

            return 0.0

        previous = self.previous_positions[player_id]

        dx = center[0] - previous[0]
        dy = center[1] - previous[1]

        pixel_distance = math.sqrt(dx ** 2 + dy ** 2)

        # Detection-box jitter causes a few pixels of "movement" even
        # when a player is standing still -- ignore it so speed doesn't
        # falsely show a stationary player as jogging
        if pixel_distance < MIN_MOVEMENT_PIXELS:
            self.previous_positions[player_id] = center
            return 0.0

        meter_distance = pixel_distance / PIXELS_PER_METER

        # meters/second -> km/h, computed before the sanity check below
        # so we can reject implausible jumps (ID switches) up front
        speed_kmh = (meter_distance * FPS) * 3.6

        if speed_kmh > MAX_PLAUSIBLE_SPEED_KMH:
            # Almost certainly a track ID swapped between two different
            # players (e.g. after an occlusion), not a real movement --
            # sync position without counting the distance or the speed
            self.previous_positions[player_id] = center
            return 0.0

        # Update total distance
        self.total_distance[player_id] += meter_distance

        # Update last position
        self.previous_positions[player_id] = center

        return round(speed_kmh, 2)

    def get_distance(self, player_id):

        return round(
            self.total_distance.get(player_id, 0),
            2
        )

    def is_sprinting(self, speed):

        return speed > 20

    def reset(self):

        self.previous_positions.clear()
        self.total_distance.clear()
