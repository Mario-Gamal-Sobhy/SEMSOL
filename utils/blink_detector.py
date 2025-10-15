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
            Tuple[float, int, float]: A tuple containing:
                - avg_ear (float): The average Eye Aspect Ratio.
                - total_blinks (int): The total number of blinks detected.
                - blink_rate (float): The number of blinks per second over the last minute.
        """
        frame, faces = self.detector.findFaceMesh(frame, draw=False)
        avg_ear = None
        blink_rate = 0.0

        if faces:
            face = faces[0]

            # Indices for left and right eyes
            left_eye_indices = [130, 159, 23, 243, 160, 158]  # H, V1, V2
            right_eye_indices = [362, 386, 374, 263, 387, 385] # H, V1, V2

            # Calculate EAR for both eyes
            left_ear = self._calculate_ear(face, left_eye_indices)
            right_ear = self._calculate_ear(face, right_eye_indices)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < self.ear_thresh:
                self.counter += 1
            else:
                if self.counter >= self.ear_consec_frames:
                    self.total_blinks += 1
                    self.blink_timestamps.append(time.time())
                self.counter = 0

        # Calculate blink rate (blinks per second)
        if len(self.blink_timestamps) > 1:
            time_diff = self.blink_timestamps[-1] - self.blink_timestamps[0]
            if time_diff > 0:
                blink_rate = len(self.blink_timestamps) / time_diff

        return avg_ear, self.total_blinks, blink_rate


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

        avg_ear, total_blinks, blink_rate = detector.update(frame)

        if avg_ear is not None:
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {total_blinks}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Blink Rate: {blink_rate:.2f} bps", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Blink Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()