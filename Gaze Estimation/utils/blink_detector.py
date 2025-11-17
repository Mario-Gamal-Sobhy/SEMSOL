import cv2
import time
import numpy as np
from collections import deque
from cvzone.FaceMeshModule import FaceMeshDetector


class BlinkDetector:
    """
    Detects blinks and calculates the blink rate based on the Eye Aspect Ratio (EAR).
    """

    def __init__(self, ear_thresh=35, ear_consec_frames=2, max_faces=1):
        self.detector = FaceMeshDetector(maxFaces=max_faces)
        self.ear_thresh = ear_thresh
        self.ear_consec_frames = ear_consec_frames
        self.counter = 0
        self.total_blinks = 0
        self.blink_timestamps = deque(maxlen=100)  # Store timestamps of recent blinks

        # Eye landmark indices for both eyes
        self.idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243,  # Left Eye
                       362, 382, 381, 380, 373, 374, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]  # Right Eye

        # Calibration attributes
        self.is_calibrated = False
        self.baseline_samples = 0
        self.ear_baseline_values = deque(maxlen=50) # Store EAR values for calibration
        self.ear_baseline_mean = 0.0
        self.ear_baseline_std = 0.0
        self.ear_threshold_multiplier = 0.7 # Multiplier for dynamic threshold

    def reset_calibration(self):
        """Resets calibration parameters."""
        self.is_calibrated = False
        self.baseline_samples = 0
        self.ear_baseline_values.clear()
        self.ear_baseline_mean = 0.0
        self.ear_baseline_std = 0.0
        self.total_blinks = 0
        self.blink_timestamps.clear()
        self.counter = 0

    def get_debug_info(self):
        """Returns debug information about the blink detector."""
        return {
            "is_calibrated": self.is_calibrated,
            "baseline_samples": self.baseline_samples,
            "ear_baseline_mean": self.ear_baseline_mean,
            "ear_baseline_std": self.ear_baseline_std,
        }

    def calculate_blink_metrics(self):
        """Calculates various blink-related metrics."""
        current_time = time.time()
        
        # Filter blink timestamps for different windows
        blinks_10s = [t for t in self.blink_timestamps if current_time - t <= 10]
        blinks_30s = [t for t in self.blink_timestamps if current_time - t <= 30]
        blinks_60s = [t for t in self.blink_timestamps if current_time - t <= 60]
        
        blink_rate_10s = len(blinks_10s) / 10.0 if len(blinks_10s) > 0 else 0.0
        blink_rate_30s = len(blinks_30s) / 30.0 if len(blinks_30s) > 0 else 0.0
        blink_rate_60s = len(blinks_60s) / 60.0 if len(blinks_60s) > 0 else 0.0
        
        # Placeholder for blink duration and eyes closed percentage (requires more sophisticated tracking)
        avg_blink_duration = 0.0 # Not implemented yet
        eyes_closed_percentage = 0.0 # Not implemented yet
        
        return {
            "blink_rate_10s": blink_rate_10s,
            "blink_rate_30s": blink_rate_30s,
            "blink_rate_60s": blink_rate_60s,
            "avg_blink_duration": avg_blink_duration,
            "eyes_closed_percentage": eyes_closed_percentage,
        }

    def assess_blink_engagement(self, blink_rate):
        """Assesses engagement based on blink rate."""
        if blink_rate < 0.1: # Very low blink rate
            return 'drowsy'
        elif blink_rate > 0.6: # Very high blink rate
            return 'stressed'
        elif blink_rate > 0.3: # Moderately high blink rate
            return 'distracted'
        else:
            return 'normal'

    def _calculate_ear(self, face, eye_indices):
        """Calculates the Eye Aspect Ratio for a single eye."""
        p1, p2, p3, p4, p5, p6 = [face[i] for i in eye_indices]

        # Vertical eye landmarks
        ver_dist, _ = self.detector.findDistance(p2, p6)
        ver_dist2, _ = self.detector.findDistance(p3, p5)

        # Horizontal eye landmarks
        hor_dist, _ = self.detector.findDistance(p1, p4)

        # Eye Aspect Ratio
        ear = ((ver_dist + ver_dist2) / (2.0 * hor_dist)) * 100
        return ear

    def update(self, frame):
        """
        Processes a single frame to detect blinks and update blink rate.

        Args:
            frame (np.ndarray): The input video frame.

        Returns:
            Tuple[float, int, bool, bool]: A tuple containing:
                - avg_ear (float): The average Eye Aspect Ratio.
                - total_blinks (int): The total number of blinks detected.
                - is_blinking (bool): True if currently blinking, False otherwise.
                - is_calibrated (bool): True if the detector is calibrated, False otherwise.
        """
        frame, faces = self.detector.findFaceMesh(frame, draw=False)
        avg_ear = None
        is_blinking = False

        if faces:
            face = faces[0]

            # Indices for left and right eyes
            left_eye_indices = [130, 159, 23, 243, 160, 158]  # H, V1, V2
            right_eye_indices = [362, 386, 374, 263, 387, 385] # H, V1, V2

            # Calculate EAR for both eyes
            left_ear = self._calculate_ear(face, left_eye_indices)
            right_ear = self._calculate_ear(face, right_eye_indices)
            avg_ear = (left_ear + right_ear) / 2.0

            # Calibration logic
            if not self.is_calibrated:
                self.ear_baseline_values.append(avg_ear)
                self.baseline_samples += 1
                if self.baseline_samples >= 50: # Collect 50 samples for calibration
                    self.ear_baseline_mean = np.mean(self.ear_baseline_values)
                    self.ear_baseline_std = np.std(self.ear_baseline_values)
                    # Set dynamic threshold based on baseline
                    self.ear_thresh = self.ear_baseline_mean - (self.ear_baseline_std * self.ear_threshold_multiplier)
                    self.is_calibrated = True
            
            if self.is_calibrated:
                if avg_ear < self.ear_thresh:
                    self.counter += 1
                else:
                    if self.counter >= self.ear_consec_frames:
                        self.total_blinks += 1
                        self.blink_timestamps.append(time.time())
                        is_blinking = True # Mark as blinking only when a blink is registered
                    self.counter = 0

        return avg_ear, self.total_blinks, is_blinking, self.is_calibrated


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    detector = BlinkDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        avg_ear, total_blinks, is_blinking, is_calibrated = detector.update(frame)

        if avg_ear is not None:
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {total_blinks}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Calibrated: {is_calibrated}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if is_blinking:
                cv2.putText(frame, "BLINK!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Blink Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()