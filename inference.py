import time
import cv2
import logging
import argparse
import warnings
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

from config import data_config
from utils.helpers import get_model, draw_bbox_gaze

import uniface
from collections import deque

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')


# -------------------------
# Engagement Scorer (Rule-Based, 4 Levels)
# -------------------------
class EngagementScorer:
    def __init__(self, fps=10, window_sec=8):
        self.fps = fps
        self.window = deque(maxlen=int(fps * window_sec))
        self.smoothed_score = None
        self.alpha = 0.3  # smoothing factor

    def update(self, pitch, yaw):
        # Thresholds (in radians)
        yaw_thresh = np.deg2rad(15)
        pitch_thresh = np.deg2rad(12)

        # looking = True if gaze is within thresholds
        looking = (abs(yaw) <= yaw_thresh) and (abs(pitch) <= pitch_thresh)
        self.window.append(1 if looking else 0)

        if len(self.window) == 0:
            return 0.0, 4  # no data yet → disengaged

        # % of time looking at screen
        p_screen = np.mean(self.window)

        # Stability proxy (lower std = more stable = more engaged)
        stability = 1.0 - np.std(self.window)

        # Weighted score (0–100)
        raw_score = 100 * (0.7 * p_screen + 0.3 * stability)
        raw_score = np.clip(raw_score, 0, 100)

        # Smooth over time
        if self.smoothed_score is None:
            self.smoothed_score = raw_score
        else:
            self.smoothed_score = (
                self.alpha * raw_score + (1 - self.alpha) * self.smoothed_score
            )

        # Convert to engagement level
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
    parser = argparse.ArgumentParser(description="Gaze estimation inference")
    parser.add_argument("--model", type=str, default="resnet34", help="Model name, default `resnet18`")
    parser.add_argument(
        "--weight",
        type=str,
        default="resnet34.pt",
        help="Path to gaze estimation model weights"
    )
    parser.add_argument("--view", action="store_true", default=True, help="Display the inference results")
    parser.add_argument("--source", type=str, default="assets/in_video.mp4",
                        help="Path to source video file or camera index")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output file")
    parser.add_argument("--dataset", type=str, default="gaze360", help="Dataset name to get dataset related configs")
    args = parser.parse_args()

    # Override default values based on selected dataset
    if args.dataset in data_config:
        dataset_config = data_config[args.dataset]
        args.bins = dataset_config["bins"]
        args.binwidth = dataset_config["binwidth"]
        args.angle = dataset_config["angle"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available options: {list(data_config.keys())}")

    return args


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
    image = transform(image)
    image_batch = image.unsqueeze(0)
    return image_batch


# -------------------------
# Main
# -------------------------
def main(params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[INFO] Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[INFO] Running on CPU")

    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)

    face_detector = uniface.RetinaFace()  # face detector
    scorer = EngagementScorer(fps=10, window_sec=8)  # engagement scorer

    try:
        gaze_detector = get_model(params.model, params.bins, inference_mode=True)
        state_dict = torch.load(params.weight, map_location=device)
        gaze_detector.load_state_dict(state_dict)
        logging.info("Gaze Estimation model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occurred while loading pre-trained weights of gaze estimation model. Exception: {e}")

    gaze_detector.to(device)
    gaze_detector.eval()

    video_source = params.source
    if video_source.isdigit() or video_source == '0':
        cap = cv2.VideoCapture(int(video_source))
    else:
        cap = cv2.VideoCapture(video_source)

    if params.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(params.output, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
    else:
        out = None

    if not cap.isOpened():
        raise IOError("Cannot open webcam or video source")

    prev_time = 0  # for FPS counter

    with torch.no_grad():
        while True:
            success, frame = cap.read()
            if not success:
                logging.info("Failed to obtain frame or EOF")
                break

            bboxes, keypoints = face_detector.detect(frame)
            for bbox, keypoint in zip(bboxes, keypoints):
                x_min, y_min, x_max, y_max = map(int, bbox[:4])

                # Clip coordinates
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(frame.shape[1], x_max)
                y_max = min(frame.shape[0], y_max)

                if x_max - x_min <= 0 or y_max - y_min <= 0:
                    continue

                image = frame[y_min:y_max, x_min:x_max]
                image = pre_process(image)
                image = image.to(device)

                pitch, yaw = gaze_detector(image)
                pitch_predicted, yaw_predicted = F.softmax(pitch, dim=1), F.softmax(yaw, dim=1)

                # Map binned outputs to angles
                pitch_predicted = torch.sum(pitch_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
                yaw_predicted = torch.sum(yaw_predicted * idx_tensor, dim=1) * params.binwidth - params.angle

                # Convert to radians
                pitch_predicted = torch.deg2rad(pitch_predicted)
                yaw_predicted = torch.deg2rad(yaw_predicted)

                # Engagement score + level
                score, level = scorer.update(
                    pitch_predicted.cpu().item(),
                    yaw_predicted.cpu().item()
                )

                # Draw gaze + engagement
                draw_bbox_gaze(frame, bbox,
                               pitch_predicted.cpu().numpy(),
                               yaw_predicted.cpu().numpy())

                levels = {
                    1: "Highly Engaged",
                    2: "Engaged",
                    3: "Partially Engaged",
                    4: "Disengaged"
                }
                cv2.putText(frame, f"Engagement: {levels[level]} ({score:.1f})",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 2)

            # FPS calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if out:
                out.write(frame)

            if params.view:
                cv2.imshow('Demo', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()

    if not args.view and not args.output:
        raise Exception("At least one of --view or --output must be provided.")

    main(args)
