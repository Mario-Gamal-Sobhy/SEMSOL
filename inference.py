"""
inference.py
Enhanced engagement estimation combining gaze tracking and blink detection.
"""

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
# Enhanced Engagement Scorer
# -------------------------
class EngagementScorer:
    """
    Enhanced engagement scorer combining gaze tracking and blink patterns
    for robust student attention monitoring.
    """
    
    def __init__(self, fps=10, enable_blink=False):
        self.fps = fps
        self.enable_blink = enable_blink
        
        # Multi-window approach for different time scales
        self.gaze_window_short = deque(maxlen=int(fps * 10))  # 10s - immediate attention
        self.gaze_window_long = deque(maxlen=int(fps * 30))   # 30s - sustained attention
        
        # Blink windows (match detector output)
        self.blink_window_10s = deque(maxlen=int(fps * 10))
        self.blink_window_30s = deque(maxlen=int(fps * 30))
        
        # Score tracking
        self.score_history = deque(maxlen=int(fps * 60))  # 1 minute history
        self.smoothed_score = None
        self.last_level = 2
        self.alpha = 0.3  # Exponential smoothing factor
        
        # Thresholds
        self.yaw_thresh = np.deg2rad(15)
        self.pitch_thresh = np.deg2rad(12)
    
    def calculate_blink_factor(self, avg_blink_rate):
        """
        Calculate engagement factor from blink rate.
        Optimal: 10-25 blinks/min (0.17-0.42 bps)
        """
        if avg_blink_rate is None or avg_blink_rate == 0:
            return 0.7  # Neutral if no data
        
        blinks_per_min = avg_blink_rate * 60
        
        if blinks_per_min < 5:
            return 0.2  # Drowsy/zoned out - CRITICAL
        elif 5 <= blinks_per_min < 10:
            return 0.5  # Low attention
        elif 10 <= blinks_per_min <= 25:
            return 1.0  # Optimal engagement range
        elif 25 < blinks_per_min <= 35:
            return 0.6  # Stressed/distracted
        else:
            return 0.3  # Very stressed/frustrated
    
    def calculate_gaze_metrics(self, pitch, yaw):
        """Calculate gaze-based engagement metrics."""
        looking = (abs(yaw) <= self.yaw_thresh) and (abs(pitch) <= self.pitch_thresh)
        gaze_value = 1 if looking else 0
        
        self.gaze_window_short.append(gaze_value)
        self.gaze_window_long.append(gaze_value)
        
        # Immediate attention (10s)
        p_screen_short = np.mean(self.gaze_window_short)
        
        # Sustained attention (30s)
        p_screen_long = np.mean(self.gaze_window_long) if len(self.gaze_window_long) > 0 else p_screen_short
        
        # Stability (less variance = more focused)
        stability = 1.0 - np.std(self.gaze_window_short) if len(self.gaze_window_short) > 1 else 0.5
        
        return p_screen_short, p_screen_long, stability
    
    def update(self, pitch, yaw, blink_rate_10s=None, blink_rate_30s=None, 
               blink_state='normal', face_detected=True):
        """
        Update engagement score with current frame data.
        
        Args:
            pitch: Head pitch angle (radians)
            yaw: Head yaw angle (radians)
            blink_rate_10s: Blinks per second over 10s window
            blink_rate_30s: Blinks per second over 30s window
            blink_state: 'normal', 'drowsy', 'stressed', 'distracted'
            face_detected: Whether face was detected in frame
            
        Returns:
            tuple: (smoothed_score, engagement_level, details)
        """
        # Handle missing face
        if not face_detected:
            if self.smoothed_score is not None:
                return self.smoothed_score, self.last_level, {'status': 'no_face', 'trend': 'stable'}
            else:
                return 0.0, 4, {'status': 'no_face', 'trend': 'stable'}
        
        # Calculate gaze metrics
        p_screen_short, p_screen_long, stability = self.calculate_gaze_metrics(pitch, yaw)
        
        # Store blink data (handle None values properly)
        self.blink_window_10s.append(blink_rate_10s)
        self.blink_window_30s.append(blink_rate_30s)
        
        # Calculate engagement score
        if not self.enable_blink:
            # Gaze-only mode
            raw_score = 100 * (0.70 * p_screen_short + 0.30 * stability)
        else:
            # Combined mode with blink analysis
            
            # Blink factor from rate
            blink_factor_rate = self.calculate_blink_factor(blink_rate_10s)
            
            # Blink factor from state
            blink_state_map = {
                'normal': 1.0,
                'drowsy': 0.2,      # CRITICAL - student zoning out
                'stressed': 0.6,    # Anxious/overwhelmed
                'distracted': 0.5   # Attention wandering
            }
            blink_factor_state = blink_state_map.get(blink_state, 0.7)
            
            # Combined blink factor (weighted average)
            blink_factor = 0.6 * blink_factor_rate + 0.4 * blink_factor_state
            
            # Adaptive weighting based on sustained attention
            if p_screen_long > 0.7:  # Good sustained attention
                # Trust gaze more
                raw_score = 100 * (0.70 * p_screen_short + 0.20 * stability + 0.10 * blink_factor)
            else:
                # Poor gaze, rely more on blink patterns
                raw_score = 100 * (0.55 * p_screen_short + 0.20 * stability + 0.25 * blink_factor)
        
        raw_score = np.clip(raw_score, 0, 100)
        
        # Exponential smoothing
        if self.smoothed_score is None:
            self.smoothed_score = raw_score
        else:
            self.smoothed_score = self.alpha * raw_score + (1 - self.alpha) * self.smoothed_score
        
        # Store history
        self.score_history.append(self.smoothed_score)
        
        # Determine engagement level with temporal context
        level, trend = self.to_engagement_level_with_context(self.smoothed_score)
        self.last_level = level
        
        # Detailed metrics
        details = {
            'status': 'ok',
            'raw_score': raw_score,
            'p_screen_short': p_screen_short,
            'p_screen_long': p_screen_long,
            'stability': stability,
            'blink_state': blink_state,
            'blink_rate_10s': blink_rate_10s,
            'trend': trend
        }
        
        return self.smoothed_score, level, details
    
    def to_engagement_level_with_context(self, current_score):
        """
        Classify engagement level with temporal context and trend analysis.
        """
        # Calculate recent average (last 5 seconds)
        recent_window = list(self.score_history)[-int(self.fps * 5):]
        recent_avg = np.mean(recent_window) if recent_window else current_score
        
        # Detect trend
        trend = "stable"
        if len(self.score_history) >= 40:
            recent = np.mean(list(self.score_history)[-20:])
            older = np.mean(list(self.score_history)[-40:-20])
            if recent > older + 5:
                trend = "improving"
            elif recent < older - 5:
                trend = "declining"
        
        # Classification with hysteresis
        if recent_avg >= 75:
            level = 1  # Highly Engaged
        elif recent_avg >= 50:
            level = 2  # Engaged
        elif recent_avg >= 25:
            level = 3  # Partially Engaged
        else:
            level = 4  # Disengaged
        
        return level, trend
    
    @staticmethod
    def to_engagement_level(score):
        """Legacy method for backwards compatibility."""
        if score >= 75:
            return 1
        elif score >= 50:
            return 2
        elif score >= 25:
            return 3
        else:
            return 4


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
    parser.add_argument("--enable-blink", action="store_true", help="Enable blink detection for engagement")
    parser.add_argument("--log-file", type=str, default="metrics_log.csv", help="Path to save metrics CSV")
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

    # Dataset config
    dataset_cfg = data_config.get(params.dataset)
    if not dataset_cfg:
        raise ValueError(f"Unknown dataset: {params.dataset}")
    bins, binwidth, angle = dataset_cfg["bins"], dataset_cfg["binwidth"], dataset_cfg["angle"]
    idx_tensor = torch.arange(bins, device=device, dtype=torch.float32)

    # Initialize components
    face_detector = uniface.RetinaFace()
    scorer = EngagementScorer(fps=10, enable_blink=params.enable_blink)

    # Load gaze model
    gaze_detector = get_model(params.model, bins, inference_mode=True)
    state_dict = torch.load(params.weight, map_location=device)
    gaze_detector.load_state_dict(state_dict)
    gaze_detector.to(device).eval()
    logging.info("✅ Gaze Estimation Model Loaded")

    # Initialize blink detector if enabled
    blink_detector = None
    if params.enable_blink:
        blink_detector = BlinkDetector(ear_sensitivity=0.80, ear_consec_frames=2, max_faces=1)
        logging.info("✅ Blink Detector Initialized")

    # Video capture
    cap = cv2.VideoCapture(int(params.source) if params.source.isdigit() else params.source)
    if not cap.isOpened():
        raise IOError("Cannot open source")

    # Output video
    out = None
    if params.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(params.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Setup CSV logging
    log_file = open(params.log_file, 'w', newline='')
    log_writer = csv.writer(log_file)
    
    log_header = [
        "timestamp_ms", "frame", "face_det_ms", "gaze_est_ms", "blink_det_ms",
        "num_faces", "pitch_deg", "yaw_deg", "ear", "blink_rate_10s_bps", "blink_state",
        "engagement_score", "engagement_level", "engagement_trend", "is_blinking", "is_calibrated"
    ]
    log_writer.writerow(log_header)
    
    frame_count = 0
    prev_time = 0
    
    # State tracking for gaze during blinks
    last_pitch = 0.0
    last_yaw = 0.0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            log_data = {key: None for key in log_header}
            log_data["timestamp_ms"] = int(time.time() * 1000)
            log_data["frame"] = frame_count

            # --- BLINK DETECTION ---
            blink_metrics = None
            blink_state = 'normal'
            is_blinking = False
            
            if params.enable_blink and blink_detector:
                start_time = time.time()
                ear, blink_count, _, is_blinking = blink_detector.update(frame)
                blink_metrics = blink_detector.calculate_blink_metrics()
                blink_state = blink_detector.assess_blink_engagement(blink_metrics['blink_rate_10s'])
                
                log_data["blink_det_ms"] = (time.time() - start_time) * 1000
                log_data["ear"] = ear
                log_data["blink_rate_10s_bps"] = blink_metrics['blink_rate_10s']
                log_data["blink_state"] = blink_state
                log_data["is_blinking"] = is_blinking
                log_data["is_calibrated"] = blink_detector.is_calibrated
                
                # Display blink info on frame
                if ear is not None:
                    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, f"Blinks: {blink_count}", (10, 130), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    # Blink state with color coding
                    state_colors = {
                        'normal': (0, 255, 0),
                        'drowsy': (0, 0, 255),
                        'stressed': (0, 165, 255),
                        'distracted': (0, 255, 255)
                    }
                    state_color = state_colors.get(blink_state, (255, 255, 255))
                    cv2.putText(frame, f"State: {blink_state.upper()}", (10, 160), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
                    
                    # Calibration status
                    if not blink_detector.is_calibrated:
                        debug_info = blink_detector.get_debug_info()
                        cv2.putText(frame, f"CALIBRATING {debug_info['baseline_samples']}/50", 
                                   (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # --- FACE DETECTION ---
            start_time = time.time()
            bboxes, keypoints = face_detector.detect(frame)
            log_data["face_det_ms"] = (time.time() - start_time) * 1000
            
            # Robust face count
            num_faces = 0
            if bboxes is not None:
                try:
                    num_faces = len(bboxes)
                except TypeError:
                    num_faces = 0
            
            log_data["num_faces"] = num_faces
            gaze_est_time_total = 0
            face_detected_for_engagement = (num_faces > 0)

            # --- GAZE ESTIMATION ---
            if num_faces > 0:
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

                    # Log metrics for first face
                    if i == 0:
                        log_data["pitch_deg"] = pitch_pred.item()
                        log_data["yaw_deg"] = yaw_pred.item()

                    pitch_rad = torch.deg2rad(pitch_pred).cpu().item()
                    yaw_rad = torch.deg2rad(yaw_pred).cpu().item()

                    # Handle gaze during blinks - use last known position
                    if is_blinking:
                        pitch_for_engagement = last_pitch
                        yaw_for_engagement = last_yaw
                    else:
                        pitch_for_engagement = pitch_rad
                        yaw_for_engagement = yaw_rad
                        last_pitch = pitch_rad
                        last_yaw = yaw_rad

                    # --- ENGAGEMENT SCORING ---
                    if params.enable_blink and blink_metrics:
                        score, level, details = scorer.update(
                            pitch_for_engagement, 
                            yaw_for_engagement,
                            blink_rate_10s=blink_metrics['blink_rate_10s'],
                            blink_rate_30s=blink_metrics['blink_rate_30s'],
                            blink_state=blink_state,
                            face_detected=face_detected_for_engagement
                        )
                    else:
                        score, level, details = scorer.update(
                            pitch_for_engagement, 
                            yaw_for_engagement,
                            face_detected=face_detected_for_engagement
                        )

                    # Log engagement for first face
                    if i == 0:
                        log_data["engagement_score"] = score
                        log_data["engagement_level"] = level
                        log_data["engagement_trend"] = details.get('trend', 'stable')

                    # Draw gaze visualization
                    draw_bbox_gaze(frame, bbox, 
                                  torch.deg2rad(torch.tensor(log_data["pitch_deg"])).numpy(), 
                                  torch.deg2rad(torch.tensor(log_data["yaw_deg"])).numpy())

                    # Display engagement with color coding
                    levels = {
                        1: ("Highly Engaged", (0, 255, 0)),
                        2: ("Engaged", (0, 200, 255)),
                        3: ("Partially Engaged", (0, 165, 255)),
                        4: ("Disengaged", (0, 0, 255))
                    }
                    label, color = levels.get(level, ("Unknown", (255, 255, 255)))
                    
                    cv2.putText(frame, f"Engagement: {label} ({score:.1f})",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Display trend
                    trend = details.get('trend', 'stable')
                    trend_color = (0, 255, 0) if trend == "improving" else ((0, 0, 255) if trend == "declining" else (255, 255, 255))
                    cv2.putText(frame, f"Trend: {trend.upper()}", (10, 220), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, trend_color, 2)

            # Calculate average gaze estimation time
            if log_data["num_faces"] > 0:
                log_data["gaze_est_ms"] = gaze_est_time_total / log_data["num_faces"]

            # --- FPS CALCULATION ---
            curr_time = time.time()
            fps_display = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Write log entry
            log_writer.writerow([log_data[key] for key in log_header])

            # Save and display
            if out:
                out.write(frame)
            if params.view:
                cv2.imshow("Engagement Monitoring (Gaze + Blink)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r') and blink_detector:
                    # Reset blink calibration on 'r' key
                    blink_detector.reset_calibration()
                    logging.info("Blink detector calibration reset")

    # Cleanup
    cap.release()
    log_file.close()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    logging.info(f"✅ Processing complete. Metrics saved to {params.log_file}")


if __name__ == "__main__":
    args = parse_args()
    main(args)