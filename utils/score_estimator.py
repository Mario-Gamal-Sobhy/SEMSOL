import numpy as np
from collections import deque


class EngagementScorer:
    """Rule-based engagement score (0–100) + 4-level classification."""

    def __init__(self, fps=10, window_sec=8):
        self.fps = fps
        self.window = deque(maxlen=int(fps * window_sec))
        self.smoothed_score = None
        self.alpha = 0.3  # exponential smoothing factor

    def update(self, pitch, yaw):
        yaw_thresh = np.deg2rad(15)
        pitch_thresh = np.deg2rad(12)

        looking = (abs(yaw) <= yaw_thresh) and (abs(pitch) <= pitch_thresh)
        self.window.append(1 if looking else 0)

        if len(self.window) == 0:
            return 0.0, 4

        p_screen = np.mean(self.window)
        stability = 1.0 - np.std(self.window)
        raw_score = 100 * (0.7 * p_screen + 0.3 * stability)
        raw_score = np.clip(raw_score, 0, 100)

        self.smoothed_score = (
            raw_score if self.smoothed_score is None
            else self.alpha * raw_score + (1 - self.alpha) * self.smoothed_score
        )

        return self.smoothed_score, self.to_engagement_level(self.smoothed_score)

    @staticmethod
    def to_engagement_level(score):
        if score >= 75:
            return 1
        elif score >= 50:
            return 2
        elif score >= 25:
            return 3
        else:
            return 4
