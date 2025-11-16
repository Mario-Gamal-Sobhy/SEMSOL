import time
import cv2
import logging
import argparse
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import deque
import csv

from config import data_config
from utils.helpers import get_model, draw_bbox_gaze
from utils.blink_detector import BlinkDetector

import uniface

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')


# -------------------------
# Engagement Scorer (Rule-Based, combines gaze + blink)
# -------------------------
class EngagementScorer:
    def __init__(self, fps=10, window_sec=8, enable_blink=False):
        self.fps = fps
        self.window = deque(maxlen=int(fps * window_sec))
        self.blink_window = deque(maxlen=int(fps * window_sec))
        self.smoothed_score = None
        self.alpha = 0.3
        self.enable_blink = enable_blink

    def update(self, pitch, yaw, blink_rate=None):
        yaw_thresh = np.deg2rad(15)
        pitch_thresh = np.deg2rad(12)

        looking = (abs(yaw) <= yaw_thresh) and (abs(pitch) <= pitch_thresh)
        self.window.append(1 if looking else 0)

        if blink_rate is not None:
            self.blink_window.append(blink_rate)
        else:
            self.blink_window.append(0)

        if len(self.window) == 0:
            return 0.0, 4

        p_screen = np.mean(self.window)
        stability = 1.0 - np.std(self.window)

        # Default weights: gaze only
        if not self.enable_blink:
            raw_score = 100 * (0.7 * p_screen + 0.3 * stability)
        else:
            # Combine with blink data
            avg_blink_rate = np.mean(self.blink_window)
            blink_factor = max(0, 1.0 - min(avg_blink_rate / 0.5, 1.0))
            raw_score = 100 * (0.6 * p_screen + 0.2 * stability + 0.2 * blink_factor)

        raw_score = np.clip(raw_score, 0, 100)

        if self.smoothed_score is None:
            self.smoothed_score = raw_score
        else:
            self.smoothed_score = self.alpha * raw_score + (1 - self.alpha) * self.smoothed_score

        level = self.to_engagement_level(self.smoothed_score)
        return self.smoothed_score, level

    @staticmethod
    def to_engagement_level(score):
        if score >= 75:
            return 1  # Highly Engaged
        elif score >= 50:
            return 2  # Engaged
        elif score >= 25:
            return 3  # Partially Engaged
        else:
            return 4  # Disengaged


# -------------------------
# Arg Parser
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Gaze + Blink Engagement Estimation")
    parser.add_argument("--model", type=str, default="resnet34", help="Model name")
    parser.add_argument("--weight", type=str, default="weights/resnet34.pt", help="Model weights path")
    parser.add_argument("--view", action="store_true", default=True, help="Display video window")
    parser.add_argument("--source", type=str, default="assets/in_video.mp4", help="Video file or webcam index")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--dataset", type=str, default="gaze360", help="Dataset name")
    parser.add_argument("--enable-blink", action="store_true", help="Enable blink detection")  # 👈 new flag
    parser.add_argument("--log-file", type=str, default="metrics_log.csv", help="Path to save the metrics CSV log file.")
    return parser.parse_args()


# -------------------------
# Preprocess face crop
# -------------------------
def pre_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


# -------------------------
# Main
# -------------------------
def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running on {'GPU' if device.type == 'cuda' else 'CPU'}")

    dataset_cfg = data_config.get(params.dataset)
    if not dataset_cfg:
        raise ValueError(f"Unknown dataset: {params.dataset}")
    bins, binwidth, angle = dataset_cfg["bins"], dataset_cfg["binwidth"], dataset_cfg["angle"]
    idx_tensor = torch.arange(bins, device=device, dtype=torch.float32)

    face_detector = uniface.RetinaFace()
    scorer = EngagementScorer(fps=10, window_sec=8, enable_blink=params.enable_blink)

    # Load gaze model
    gaze_detector = get_model(params.model, bins, inference_mode=True)
    state_dict = torch.load(params.weight, map_location=device)
    gaze_detector.load_state_dict(state_dict)
    gaze_detector.to(device).eval()
    logging.info("✅ Gaze Estimation Model Loaded")

    # Initialize blink detector if needed
    blink_detector_instance = BlinkDetector() if params.enable_blink else None

    # Video capture
    cap = cv2.VideoCapture(int(params.source) if params.source.isdigit() else params.source)
    if not cap.isOpened():
        raise IOError("Cannot open source")

    # Output
    out = None
    if params.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(params.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Setup logging
    log_file = open(params.log_file, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_header = [
        "timestamp_ms", "frame", "face_det_ms", "gaze_est_ms", "blink_det_ms",
        "num_faces", "pitch_deg", "yaw_deg", "ear", "blink_rate_bps",
        "engagement_score", "engagement_level"
    ]
    log_writer.writerow(log_header)
    frame_count = 0

    prev_time = 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            log_data = {key: None for key in log_header}
            log_data["timestamp_ms"] = int(time.time() * 1000)
            log_data["frame"] = frame_count

            blink_rate = 0
            start_time = time.time()
            if params.enable_blink and blink_detector_instance:
                ear, blink_count, blink_rate = blink_detector_instance.update(frame)
                log_data["blink_det_ms"] = (time.time() - start_time) * 1000
                log_data["ear"] = ear
                log_data["blink_rate_bps"] = blink_rate
                if ear is not None:
                    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"Blinks: {blink_count}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Gaze estimation
            start_time = time.time()
            bboxes, keypoints = face_detector.detect(frame)
            log_data["face_det_ms"] = (time.time() - start_time) * 1000
            log_data["num_faces"] = len(bboxes)

            gaze_est_time_total = 0
            for i, (bbox, _) in enumerate(zip(bboxes, keypoints)):
                x_min, y_min, x_max, y_max = map(int, bbox[:4])
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(frame.shape[1], x_max), min(frame.shape[0], y_max)

                if x_max - x_min <= 0 or y_max - y_min <= 0:
                    continue

                face_crop = frame[y_min:y_max, x_min:x_max]
                image = pre_process(face_crop).to(device)

                start_time_gaze = time.time()
                pitch, yaw = gaze_detector(image)
                gaze_est_time_total += (time.time() - start_time_gaze) * 1000

                pitch_pred, yaw_pred = F.softmax(pitch, dim=1), F.softmax(yaw, dim=1)

                pitch_pred = torch.sum(pitch_pred * idx_tensor, dim=1) * binwidth - angle
                yaw_pred = torch.sum(yaw_pred * idx_tensor, dim=1) * binwidth - angle

                # Log metrics for the first detected face only to keep log simple
                if i == 0:
                    log_data["pitch_deg"] = pitch_pred.item()
                    log_data["yaw_deg"] = yaw_pred.item()

                pitch_pred = torch.deg2rad(pitch_pred)
                yaw_pred = torch.deg2rad(yaw_pred)

                # Engagement
                score, level = scorer.update(pitch_pred.cpu().item(), yaw_pred.cpu().item(), blink_rate if params.enable_blink else None)

                # Log engagement for the first face
                if i == 0:
                    log_data["engagement_score"] = score
                    log_data["engagement_level"] = level

                draw_bbox_gaze(frame, bbox, pitch_pred.cpu().numpy(), yaw_pred.cpu().numpy())

                levels = {1: "Highly Engaged", 2: "Engaged", 3: "Partially Engaged", 4: "Disengaged"}
                cv2.putText(frame, f"Engagement: {levels[level]} ({score:.1f})",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            if log_data["num_faces"] > 0:
                log_data["gaze_est_ms"] = gaze_est_time_total / log_data["num_faces"]

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            log_writer.writerow([log_data[key] for key in log_header])

            if out:
                out.write(frame)
            if params.view:
                cv2.imshow("Engagement (Gaze + Blink)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    log_file.close()
    if out:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    main(args)
