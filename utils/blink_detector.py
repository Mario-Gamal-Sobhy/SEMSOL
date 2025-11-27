"""
utils/blink_detector.py
Optimized blink detector for student engagement monitoring.
"""

import cv2
import time
import numpy as np
from collections import deque
from cvzone.FaceMeshModule import FaceMeshDetector


class BlinkDetector:
    """
    Detects blinks and calculates blink rates with adaptive thresholding.
    Optimized for student engagement monitoring with gaze tracking integration.
    """

    def __init__(self, ear_sensitivity=0.80, ear_consec_frames=2, max_faces=1):
        """
        Args:
            ear_sensitivity: EAR threshold as % of baseline (0.80 = 80% of open-eye baseline)
            ear_consec_frames: Minimum consecutive frames for valid blink
            max_faces: Maximum number of faces to detect
        """
        self.detector = FaceMeshDetector(maxFaces=max_faces)
        self.ear_sensitivity = ear_sensitivity
        self.ear_consec_frames = ear_consec_frames

        # Blink tracking
        self.blink_counter = 0
        self.total_blinks = 0
        self.blink_timestamps = deque(maxlen=200)  # Store last 200 blinks (3+ minutes)

        # Adaptive baseline (stores OPEN-eye EAR values only)
        self.ear_baseline_history = deque(maxlen=150)
        self.dynamic_ear_thresh = 0
        self.is_calibrated = False
        
        # Current state
        self.is_currently_blinking = False
        self.last_valid_ear = None

        # Eye landmark indices (MediaPipe FaceMesh format)
        self.left_eye_indices = [33, 160, 158, 133, 153, 144]
        self.right_eye_indices = [362, 385, 387, 263, 373, 380]

        # FPS tracking
        self.last_frame_time = None
        self.fps_estimate = 30

    def _calculate_ear(self, face, eye_indices):
        """Calculate Eye Aspect Ratio for a single eye."""
        try:
            p1, p2, p3, p4, p5, p6 = [face[i] for i in eye_indices]

            # Vertical distances
            ver_dist, _ = self.detector.findDistance(p2, p6)
            ver_dist2, _ = self.detector.findDistance(p3, p5)

            # Horizontal distance
            hor_dist, _ = self.detector.findDistance(p1, p4)

            if hor_dist == 0:
                return 0.0

            ear = ((ver_dist + ver_dist2) / (2.0 * hor_dist)) * 100
            return ear
        except (IndexError, KeyError):
            return 0.0

    def _estimate_fps(self):
        """Estimate FPS dynamically from frame timings."""
        now = time.time()
        if self.last_frame_time is None:
            self.last_frame_time = now
            return self.fps_estimate

        delta = now - self.last_frame_time
        self.last_frame_time = now
        if delta > 0:
            # Smooth FPS estimate
            estimated_fps = 1.0 / delta
            self.fps_estimate = int(0.9 * self.fps_estimate + 0.1 * estimated_fps)
        return self.fps_estimate

    def _update_baseline_adaptive(self, avg_ear):
        """
        Adaptively update baseline without forced recalibration.
        Only adds clearly "open eye" values to avoid contamination.
        """
        # Initial calibration (fast)
        if not self.is_calibrated:
            # Add values during initial period, but filter outliers
            if len(self.ear_baseline_history) > 0:
                median_ear = np.median(self.ear_baseline_history)
                # Only add if within reasonable range (not a blink)
                if avg_ear > median_ear * 0.7:
                    self.ear_baseline_history.append(avg_ear)
            else:
                self.ear_baseline_history.append(avg_ear)
            
            # Complete calibration after 50 frames (~1.5 seconds at 30fps)
            if len(self.ear_baseline_history) >= 50:
                baseline_mean = np.mean(self.ear_baseline_history)
                self.dynamic_ear_thresh = baseline_mean * self.ear_sensitivity
                self.is_calibrated = True
        else:
            # Slow adaptive drift for calibrated system
            # Only update with clearly open eyes (above threshold + margin)
            if avg_ear > self.dynamic_ear_thresh * 1.25:
                baseline_mean = np.mean(self.ear_baseline_history)
                
                # Exponential moving average (1% weight to new value)
                alpha = 0.01
                new_baseline = alpha * avg_ear + (1 - alpha) * baseline_mean
                
                self.ear_baseline_history.append(new_baseline)
                self.dynamic_ear_thresh = np.mean(self.ear_baseline_history) * self.ear_sensitivity

    def _detect_blink(self, avg_ear, fps):
        """
        Detect blinks using state machine approach.
        Returns True if a blink was completed this frame.
        """
        if not self.is_calibrated or self.dynamic_ear_thresh == 0:
            return False
        
        # Adaptive grace window based on FPS
        grace_window = max(1, int(fps * 0.1))  # 100ms tolerance
        
        is_eye_closed = avg_ear < self.dynamic_ear_thresh
        
        if is_eye_closed:
            self.blink_counter += 1
            self.is_currently_blinking = True
            return False
        else:
            # Eyes opened - check if we completed a valid blink
            if 1 <= self.blink_counter <= self.ear_consec_frames + grace_window:
                # Valid blink detected
                self.blink_counter = 0
                self.is_currently_blinking = False
                return True
            else:
                # Either no blink or too long (not a blink)
                self.blink_counter = 0
                self.is_currently_blinking = False
                return False

    def calculate_blink_metrics(self):
        """
        Calculate blink rate over multiple time windows.
        Returns metrics suitable for engagement scoring.
        """
        now = time.time()
        
        # Remove timestamps older than 60 seconds
        while self.blink_timestamps and self.blink_timestamps[0] < now - 60:
            self.blink_timestamps.popleft()
        
        metrics = {
            'blink_rate_10s': 0.0,
            'blink_rate_30s': 0.0,
            'blink_rate_60s': 0.0,
            'total_blinks': self.total_blinks
        }
        
        if len(self.blink_timestamps) == 0:
            return metrics
        
        # 10-second window (immediate response)
        recent_10s = [t for t in self.blink_timestamps if t > now - 10]
        if len(recent_10s) > 0:
            metrics['blink_rate_10s'] = len(recent_10s) / 10.0
        
        # 30-second window (stable measurement)
        recent_30s = [t for t in self.blink_timestamps if t > now - 30]
        if len(recent_30s) > 0:
            metrics['blink_rate_30s'] = len(recent_30s) / 30.0
        
        # 60-second window (long-term baseline)
        if len(self.blink_timestamps) > 1:
            time_span = self.blink_timestamps[-1] - self.blink_timestamps[0]
            if time_span > 0:
                metrics['blink_rate_60s'] = len(self.blink_timestamps) / time_span
        
        return metrics

    def assess_blink_engagement(self, blink_rate_10s):
        """
        Assess engagement state based on blink patterns.
        
        Args:
            blink_rate_10s: Blinks per second over 10-second window
            
        Returns:
            str: 'normal', 'drowsy', 'stressed', or 'distracted'
        """
        if blink_rate_10s is None or blink_rate_10s == 0:
            return 'normal'
        
        blinks_per_min = blink_rate_10s * 60
        
        if blinks_per_min < 5:
            return 'drowsy'  # Zoned out, staring, hyperfocus
        elif 5 <= blinks_per_min < 10:
            return 'distracted'  # Low attention, mind wandering
        elif 10 <= blinks_per_min <= 25:
            return 'normal'  # Healthy engagement range
        elif 25 < blinks_per_min <= 35:
            return 'stressed'  # Anxious, overwhelmed
        else:
            return 'stressed'  # Very stressed, frustrated

    def update(self, frame):
        """
        Process a single frame to detect blinks and update metrics.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            tuple: (avg_ear, total_blinks, blink_rate_legacy, is_currently_blinking)
                   Note: blink_rate_legacy is for backwards compatibility (60s window)
        """
        frame, faces = self.detector.findFaceMesh(frame, draw=False)
        fps = self._estimate_fps()
        
        # Handle no face detected
        if not faces:
            self.blink_counter = 0  # Reset to avoid false blinks
            self.is_currently_blinking = False
            
            # Return last known EAR and legacy blink rate
            legacy_rate = self._calculate_legacy_blink_rate()
            return self.last_valid_ear, self.total_blinks, legacy_rate, self.is_currently_blinking
        
        # Calculate EAR
        face = faces[0]
        avg_ear = (self._calculate_ear(face, self.left_eye_indices) +
                   self._calculate_ear(face, self.right_eye_indices)) / 2.0
        
        if avg_ear == 0.0:
            # Invalid EAR calculation
            legacy_rate = self._calculate_legacy_blink_rate()
            return self.last_valid_ear, self.total_blinks, legacy_rate, self.is_currently_blinking
        
        self.last_valid_ear = avg_ear
        
        # Update adaptive baseline
        self._update_baseline_adaptive(avg_ear)
        
        # Detect blinks (only if calibrated)
        if self.is_calibrated:
            blink_detected = self._detect_blink(avg_ear, fps)
            
            if blink_detected:
                self.total_blinks += 1
                self.blink_timestamps.append(time.time())
        
        # Calculate legacy blink rate for backwards compatibility
        legacy_rate = self._calculate_legacy_blink_rate()
        
        return avg_ear, self.total_blinks, legacy_rate, self.is_currently_blinking

    def _calculate_legacy_blink_rate(self):
        """Calculate blink rate over 60s window for backwards compatibility."""
        now = time.time()
        
        # Remove old timestamps
        while self.blink_timestamps and self.blink_timestamps[0] < now - 60:
            self.blink_timestamps.popleft()
        
        if len(self.blink_timestamps) > 1:
            time_span = self.blink_timestamps[-1] - self.blink_timestamps[0]
            if time_span > 0:
                return len(self.blink_timestamps) / time_span
        
        return 0.0

    def get_debug_info(self):
        """Get detailed debug information for tuning."""
        metrics = self.calculate_blink_metrics()
        blink_state = self.assess_blink_engagement(metrics['blink_rate_10s'])
        
        return {
            'is_calibrated': self.is_calibrated,
            'dynamic_threshold': self.dynamic_ear_thresh,
            'baseline_mean': np.mean(self.ear_baseline_history) if self.ear_baseline_history else 0,
            'baseline_samples': len(self.ear_baseline_history),
            'is_blinking': self.is_currently_blinking,
            'blink_counter': self.blink_counter,
            'total_blinks': self.total_blinks,
            'blink_rate_10s': metrics['blink_rate_10s'],
            'blink_rate_30s': metrics['blink_rate_30s'],
            'blink_state': blink_state,
            'fps': self.fps_estimate
        }

    def reset_calibration(self):
        """Reset calibration (useful for new user or significant lighting change)."""
        self.ear_baseline_history.clear()
        self.dynamic_ear_thresh = 0
        self.is_calibrated = False
        self.blink_counter = 0


# Example usage
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    detector = BlinkDetector(ear_sensitivity=0.80, ear_consec_frames=2, max_faces=1)

    print("Starting blink detection... Press 'q' to quit, 'r' to reset calibration")
    print("Calibrating for first ~1.5 seconds...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        avg_ear, total_blinks, blink_rate, is_blinking = detector.update(frame)
        debug_info = detector.get_debug_info()

        if avg_ear is not None:
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Threshold: {debug_info['dynamic_threshold']:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Blinks: {total_blinks}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"State: {debug_info['blink_state'].upper()}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if not debug_info['is_calibrated']:
                cv2.putText(frame, f"CALIBRATING {debug_info['baseline_samples']}/50", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Blink Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset_calibration()
            print("Calibration reset.")

    cap.release()
    cv2.destroyAllWindows()