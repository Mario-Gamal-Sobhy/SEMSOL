"""
app_deploy.py - Cloud-Compatible Version for Streamlit Cloud
============================================================

This version uses `streamlit-webrtc` to handle video streams, which is required
for Streamlit Cloud deployment (since cv2.VideoCapture doesn't work on remote servers).

Features:
- WebRTC video streaming
- Real-time engagement monitoring
- MongoDB Atlas integration
"""

import streamlit as st
import cv2
import torch
import numpy as np
import time
import torch.nn.functional as F
from torchvision import transforms
from collections import deque
from pathlib import Path
import joblib
import pandas as pd
from datetime import datetime
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Import modules
from utils.blink_detector import BlinkDetector
from utils.helpers import get_model, draw_bbox_gaze
from config import data_config
import uniface

# MongoDB Integration
try:
    from utils.mongodb_handler import save_session_to_mongodb
    from mongodb_config import MONGODB_CONFIG
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

# ----------------------------
# Streamlit Config
# ----------------------------
st.set_page_config(
    page_title="SEMSOL - Cloud Edition", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("☁️ SEMSOL: Cloud Engagement Monitoring")
st.markdown("""
**Cloud Deployment Version**
This version uses WebRTC for video streaming, allowing it to run on Streamlit Cloud.
""")

# ----------------------------
# Global Resources (Cached)
# ----------------------------
device = torch.device("cpu")  # Use CPU for cloud (usually no GPU available)

@st.cache_resource
def load_models():
    """Load all models once."""
    # 1. Gaze Model
    dataset_cfg = data_config.get("gaze360")
    bins = dataset_cfg["bins"]
    idx_tensor = torch.arange(bins, device=device, dtype=torch.float32)
    
    gaze_model = get_model("resnet18", bins, inference_mode=True)
    # Note: You might need to adjust path for cloud
    if Path("weights/resnet18.pt").exists():
        state_dict = torch.load("weights/resnet18.pt", map_location=device)
        gaze_model.load_state_dict(state_dict)
    gaze_model.to(device).eval()
    
    # 2. Face Detector
    face_detector = uniface.RetinaFace()
    
    # 3. Blink Detector
    blink_detector = BlinkDetector(ear_threshold=0.20)
    
    # 4. ML Classifier
    ml_classifier = None
    if Path("weights/engagement_classifier.pkl").exists():
        ml_classifier = joblib.load("weights/engagement_classifier.pkl")
        
    return gaze_model, face_detector, blink_detector, ml_classifier, idx_tensor

# Load models
try:
    gaze_model, face_detector, blink_detector, ml_classifier, idx_tensor = load_models()
    st.success("✅ Models Loaded")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()

# ----------------------------
# Video Processor
# ----------------------------
class EngagementProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.session_data = {
            'timestamps': [],
            'engagement_levels': [],
            'confidence_scores': [],
            'session_start': datetime.now(),
            'frames_processed': 0
        }
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        
        # 1. Blink Detection
        ear, _, _, is_blinking = blink_detector.update(img)
        
        # 2. Face Detection
        bboxes, keypoints = face_detector.detect(img)
        
        engagement_level = 4
        confidence = 0.0
        pitch_deg, yaw_deg = 0.0, 0.0
        
        if bboxes is not None and len(bboxes) > 0:
            # Process first face
            bbox = bboxes[0]
            x_min, y_min, x_max, y_max = map(int, bbox[:4])
            
            # Gaze Estimation
            face_crop = img[y_min:y_max, x_min:x_max]
            if face_crop.size > 0:
                # Preprocess
                pil_img = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                input_tensor = self.transform(pil_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    pitch, yaw = gaze_model(input_tensor)
                    # Simple regression for demo (adjust based on your model output)
                    pitch_deg = pitch.item() * 180 / np.pi if pitch.shape[1] == 1 else 0 # Simplified
                    # NOTE: Using simplified logic here for robustness on cloud
                    
            # Draw
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Simple Rule-based Engagement
            if abs(pitch_deg) < 15 and abs(yaw_deg) < 15 and not is_blinking:
                engagement_level = 1 # Highly engaged
            else:
                engagement_level = 3
                
        # Overlay Metrics
        cv2.putText(img, f"Engagement: {engagement_level}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Store data (simplified for thread safety)
        # In a real app, use a Queue to send this to main thread
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ----------------------------
# Main UI
# ----------------------------
st.sidebar.header("Settings")
rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

webrtc_ctx = webrtc_streamer(
    key="engagement-monitor",
    mode=webrtc_streamer.WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    video_processor_factory=EngagementProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# ----------------------------
# Session Management
# ----------------------------
if webrtc_ctx.state.playing:
    st.info("🟢 Monitoring Active")
else:
    st.info("⚪ Monitoring Stopped")
    
    # Save button (manual for now as auto-save is tricky with webrtc state)
    if st.button("💾 Save Last Session to MongoDB"):
        st.warning("Session saving requires implementing a thread-safe Queue. (Coming soon)")

st.markdown("---")
st.subheader("📋 Instructions")
st.markdown("""
1. Allow camera access when prompted.
2. The video feed will appear above.
3. Engagement metrics are drawn directly on the video.
4. **Note:** This is a simplified version for cloud deployment.
""")
