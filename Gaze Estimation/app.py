"""
app.py
Streamlit interface for real-time Engagement Monitoring
Enhanced with ML Classifier support + Rule-Based fallback
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

# Import your modules
from utils.blink_detector import BlinkDetector
from utils.helpers import get_model, draw_bbox_gaze
from config import data_config
import uniface

# ----------------------------
# Streamlit App Config
# ----------------------------
st.set_page_config(
    page_title="Engagement Monitoring System", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Real-Time Student Engagement Monitoring System")
st.markdown("""
**Multi-Modal Engagement Detection:**
- 👁️ **Blink Detection** - Detects drowsiness, stress, and attention patterns
- 👀 **Gaze Estimation** - Tracks where students are looking
- 🤖 **ML Classifier** - Predicts engagement level using trained model
""")

# ----------------------------
# Sidebar Controls
# ----------------------------
st.sidebar.header("⚙️ Configuration")

# Model selection
use_ml_classifier = st.sidebar.checkbox("Use ML Classifier", value=True, 
                                        help="Use trained ML model for classification")
classifier_path = st.sidebar.text_input(
    "Classifier Model Path", 
    "weights/resnet18.pt",
    help="Path to trained " \
    " model file"
)

# Video source
source = st.sidebar.selectbox("Video Source", ["Webcam (0)", "Video File"], index=0)
video_path = st.sidebar.text_input("Video File Path", "assets/in_video.mp4")

# Detection settings
enable_blink = st.sidebar.checkbox("Enable Blink Detection", value=True)
display_fps_target = st.sidebar.slider("Target FPS", 5, 30, 15, 
                                       help="Processing speed (higher = faster but more CPU)")

# Visual settings
show_bbox = st.sidebar.checkbox("Show Face Bounding Box", value=True)
show_gaze_arrow = st.sidebar.checkbox("Show Gaze Direction Arrow", value=True)

# Advanced settings
with st.sidebar.expander("🔧 Advanced Settings"):
    ear_sensitivity = st.slider("EAR Sensitivity", 0.70, 0.90, 0.80, 0.01,
                                help="Lower = more sensitive blink detection")
    gaze_model_name = st.selectbox("Gaze Model", ["resnet34", "resnet18"], index=1)
    gaze_weight_path = st.text_input("Gaze Model Weights", "weights/resnet18.pt")

# Start/Stop button
start_button = st.sidebar.button("🚀 Start Monitoring", type="primary")
stop_button = st.sidebar.button("⏹ Stop Monitoring", type="secondary")

# ----------------------------
# Initialize Components
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.info(f"🖥️ Device: **{device.type.upper()}**")

@st.cache_resource
def load_gaze_model():
    """Load gaze estimation model."""
    dataset_cfg = data_config.get("gaze360")
    bins, binwidth, angle = dataset_cfg["bins"], dataset_cfg["binwidth"], dataset_cfg["angle"]
    idx_tensor = torch.arange(bins, device=device, dtype=torch.float32)
    
    gaze_model = get_model(gaze_model_name, bins, inference_mode=True)
    state_dict = torch.load(gaze_weight_path, map_location=device)
    gaze_model.load_state_dict(state_dict)
    gaze_model.to(device).eval()
    
    return gaze_model, idx_tensor, binwidth, angle

@st.cache_resource
def load_ml_classifier(path):
    """Load trained ML classifier (PyTorch model)."""
    if not Path(path).exists():
        st.sidebar.warning(f"⚠️ Classifier not found: {path}")
        return None
    
    try:
        # Assuming the PyTorch model is a state_dict or a full model
        # For simplicity, let's assume it's a state_dict of a pre-defined model architecture
        # You would need to define the model architecture (e.g., ResNet18) here or import it
        
        # For now, let's just load the state_dict and return it.
        # The actual model instantiation and loading state_dict would happen in predict_with_ml
        # or a dedicated function if the model architecture is known.
        
        # Placeholder: In a real scenario, you'd instantiate your PyTorch model class
        # e.g., ml_model = MyEngagementClassifier()
        # ml_model.load_state_dict(torch.load(path, map_location=device))
        # ml_model.to(device).eval()
        # return ml_model
        
        # For now, we'll just load the state_dict.
        # This will need further refinement once the actual PyTorch model architecture is known.
        st.sidebar.success("✅ ML Classifier (PyTorch) Loaded")
        return torch.load(path, map_location=device)
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load classifier: {e}")
        return None

@st.cache_resource
def load_face_detector():
    """Load face detector."""
    return uniface.RetinaFace()

@st.cache_resource
def load_blink_detector():
    """Load blink detector."""
    return BlinkDetector(
        ear_thresh=ear_sensitivity,
        ear_consec_frames=2,
        max_faces=1
    )

# Load all components
with st.spinner("Loading models..."):
    try:
        face_detector = load_face_detector()
        gaze_model, idx_tensor, binwidth, angle = load_gaze_model()
        blink_detector = load_blink_detector() if enable_blink else None
        
        ml_classifier = None
        if use_ml_classifier:
            ml_classifier = load_ml_classifier(classifier_path)
        
        st.sidebar.success("✅ All models loaded!")
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

# Preprocessing transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess(image):
    """Preprocess face crop for gaze estimation."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return transform(image).unsqueeze(0)

def extract_features_for_ml(pitch_deg, yaw_deg, ear, blink_metrics, face_detected):
    """
    Extract features for ML classifier.
    Note: This is a simplified version for real-time inference.
    For proper aggregated features, you'd need frame history.
    """
    if ml_classifier is None:
        return None
    
    feature_names = ml_classifier.get('feature_names', [])
    
    # Build feature vector (simplified - using instantaneous values)
    # In production, you'd maintain sliding windows for proper aggregation
    features = {}
    
    # Gaze features (using current frame as proxy for mean/stats)
    features['pitch_mean'] = pitch_deg
    features['pitch_std'] = 0.0  # Would need history
    features['pitch_min'] = pitch_deg
    features['pitch_max'] = pitch_deg
    features['pitch_p25'] = pitch_deg
    features['pitch_p50'] = pitch_deg
    features['pitch_p75'] = pitch_deg
    
    features['yaw_mean'] = yaw_deg
    features['yaw_std'] = 0.0
    features['yaw_min'] = yaw_deg
    features['yaw_max'] = yaw_deg
    features['yaw_p25'] = yaw_deg
    features['yaw_p50'] = yaw_deg
    features['yaw_p75'] = yaw_deg
    
    # EAR features
    features['ear_mean'] = ear if ear else 0.0
    features['ear_std'] = 0.0
    features['ear_min'] = ear if ear else 0.0
    features['ear_max'] = ear if ear else 0.0
    features['ear_p25'] = ear if ear else 0.0
    features['ear_p50'] = ear if ear else 0.0
    features['ear_p75'] = ear if ear else 0.0
    
    # Blink features
    features['blink_count'] = blink_detector.total_blinks if blink_detector else 0
    features['blink_rate'] = blink_metrics.get('blink_rate_10s', 0.0) if blink_metrics else 0.0
    
    # Other features
    features['face_ratio'] = 1.0 if face_detected else 0.0
    features['pitch_stab'] = 1.0  # Would need history
    features['yaw_stab'] = 1.0  # Would need history
    
    # Build feature array in correct order
    feature_array = np.array([features.get(name, 0.0) for name in feature_names])
    return feature_array.reshape(1, -1)

def predict_with_ml(features):
    """Predict engagement using ML classifier."""
    if ml_classifier is None or features is None:
        return None, None
    
    try:
        model = ml_classifier['model']
        scaler = ml_classifier['scaler']
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        # Map DAiSEE labels (0-3) to engagement levels (1-4)
        # 0 (Very Low) -> 4 (Disengaged)
        # 1 (Low) -> 3 (Partially Engaged)
        # 2 (High) -> 2 (Engaged)
        # 3 (Very High) -> 1 (Highly Engaged)
        label_map = {0: 4, 1: 3, 2: 2, 3: 1}
        engagement_level = label_map.get(prediction, prediction)
        
        return engagement_level, confidence
        
    except Exception as e:
        st.sidebar.error(f"ML prediction error: {e}")
        return None, None

def rule_based_fallback(pitch_deg, yaw_deg, blink_state):
    """Simple rule-based engagement estimation (fallback)."""
    looking_at_screen = abs(pitch_deg) < 12 and abs(yaw_deg) < 15
    
    if looking_at_screen and blink_state == 'normal':
        return 1, 0.7  # Highly Engaged
    elif looking_at_screen:
        return 2, 0.6  # Engaged
    elif blink_state == 'drowsy':
        return 4, 0.8  # Disengaged
    else:
        return 3, 0.5  # Partially Engaged

# ----------------------------
# Main Monitoring Loop
# ----------------------------
if start_button:
    # Create layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Video Feed")
        stframe = st.empty()
    
    with col2:
        st.subheader("📊 Live Metrics")
        metrics_container = st.container()
        
        with metrics_container:
            fps_metric = st.empty()
            engagement_metric = st.empty()
            gaze_metric = st.empty()
            blink_metric = st.empty()
            status_metric = st.empty()
    
    # Progress bar for calibration
    calibration_progress = st.progress(0)
    calibration_text = st.empty()
    
    # Open video source
    if source.startswith("Webcam"):
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("❌ Cannot open video source!")
        st.stop()
    
    # Reset blink detector
    if blink_detector:
        blink_detector.reset_calibration()
    
    # State variables
    prev_time = time.time()
    last_pitch, last_yaw = 0.0, 0.0
    frame_count = 0
    
    # Main loop
    try:
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ End of video or camera disconnected")
                break
            
            frame_count += 1
            curr_time = time.time()
            fps_actual = 1 / (curr_time - prev_time) if prev_time else 0
            prev_time = curr_time
            
            # ----------------------------
            # Blink Detection
            # ----------------------------
            ear, blink_count, is_blinking_current_frame, is_calibrated_detector = None, 0, False, False
            blink_metrics = None
            blink_state = 'normal'
            
            if enable_blink and blink_detector:
                ear, blink_count, is_blinking_current_frame, is_calibrated_detector = blink_detector.update(frame)
                blink_metrics = blink_detector.calculate_blink_metrics()
                blink_state = blink_detector.assess_blink_engagement(
                    blink_metrics['blink_rate_10s']
                )
                
                # Update calibration progress
                if not is_calibrated_detector:
                    debug_info = blink_detector.get_debug_info()
                    progress = debug_info['baseline_samples'] / 50
                    calibration_progress.progress(min(progress, 1.0))
                    calibration_text.text(f"Calibrating blink detector: {debug_info['baseline_samples']}/50")
                else:
                    calibration_progress.empty()
                    calibration_text.empty()
            
            # ----------------------------
            # Face Detection & Gaze Estimation
            # ----------------------------
            detection_results = face_detector.detect(frame)
            
            bboxes = []
            keypoints = []
            if detection_results: # Check if the list is not empty
                for face_dict in detection_results:
                    bboxes.append(face_dict['bbox'])
                    keypoints.append(face_dict['landmarks']) # Assuming 'landmarks' is what's needed for keypoints
            
            num_faces = len(bboxes)
            face_detected = num_faces > 0
            
            pitch_deg, yaw_deg = 0.0, 0.0
            engagement_level = 4
            confidence = 0.0
            
            if face_detected:
                for i, (bbox, _) in enumerate(zip(bboxes, keypoints)):
                    x_min, y_min, x_max, y_max = map(int, bbox[:4])
                    x_min, y_min = max(0, x_min), max(0, y_min)
                    x_max = min(frame.shape[1], x_max)
                    y_max = min(frame.shape[0], y_max)
                    
                    if x_max - x_min <= 0 or y_max - y_min <= 0:
                        continue
                    
                    face_crop = frame[y_min:y_max, x_min:x_max]
                    
                    # Gaze estimation
                    with torch.no_grad():
                        image = preprocess(face_crop).to(device)
                        pitch, yaw = gaze_model(image)
                        
                        pitch_pred = F.softmax(pitch, dim=1)
                        yaw_pred = F.softmax(yaw, dim=1)
                        
                        pitch_pred = torch.sum(pitch_pred * idx_tensor, dim=1) * binwidth - angle
                        yaw_pred = torch.sum(yaw_pred * idx_tensor, dim=1) * binwidth - angle
                        
                        pitch_deg = pitch_pred.item()
                        yaw_deg = yaw_pred.item()
                    
                    # Use last known gaze during blinks
                    if is_blinking_current_frame:
                        pitch_deg = last_pitch
                        yaw_deg = last_yaw
                    else:
                        last_pitch = pitch_deg
                        last_yaw = yaw_deg
                    
                    # ----------------------------
                    # Engagement Classification
                    # ----------------------------
                    if use_ml_classifier and ml_classifier and blink_detector and is_calibrated_detector:
                        # ML-based classification
                        features = extract_features_for_ml(
                            pitch_deg, yaw_deg, ear, blink_metrics, face_detected
                        )
                        engagement_level, confidence = predict_with_ml(features)
                        
                        if engagement_level is None:
                            # Fallback to rule-based
                            engagement_level, confidence = rule_based_fallback(
                                pitch_deg, yaw_deg, blink_state
                            )
                    else:
                        # Rule-based classification
                        engagement_level, confidence = rule_based_fallback(
                            pitch_deg, yaw_deg, blink_state
                        )
                    
                    # Draw visualizations
                    if show_bbox:
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    if show_gaze_arrow:
                        draw_bbox_gaze(frame, bbox, np.deg2rad(pitch_deg), np.deg2rad(yaw_deg))
                    
                    # Only process first face
                    break
            
            # ----------------------------
            # Display Video Frame
            # ----------------------------
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add overlay text
            cv2.putText(frame_rgb, f"FPS: {fps_actual:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if is_blinking_current_frame:
                cv2.putText(frame_rgb, "BLINKING", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            stframe.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # ----------------------------
            # Update Metrics Display
            # ----------------------------
            engagement_labels = {
                1: "Highly Engaged",
                2: "Engaged",
                3: "Partially Engaged",
                4: "Disengaged"
            }
            engagement_colors = {
                1: "🟢",
                2: "🟡",
                3: "🟠",
                4: "🔴"
            }
            
            blink_state_colors = {
                'normal': '🟢',
                'drowsy': '🔴',
                'stressed': '🟠',
                'distracted': '🟡'
            }
            
            # FPS
            fps_metric.metric(
                "Frame Rate",
                f"{fps_actual:.1f} FPS",
                delta=None
            )
            
            # Engagement
            engagement_metric.markdown(f"""
            ### {engagement_colors[engagement_level]} Engagement
            **{engagement_labels[engagement_level]}**  
            Confidence: `{confidence:.1%}`  
            Method: `{'ML Classifier' if use_ml_classifier and ml_classifier else 'Rule-Based'}`
            """)
            
            # Gaze
            gaze_metric.markdown(f"""
            ### 👀 Gaze Direction
            - **Pitch:** `{pitch_deg:.1f}°`
            - **Yaw:** `{yaw_deg:.1f}°`
            - **Looking at screen:** `{abs(pitch_deg) < 12 and abs(yaw_deg) < 15}`
            """)
            
            # Blink
            if enable_blink and blink_detector:
                blink_metric.markdown(f"""
                ### 👁️ Blink Analysis
                - **EAR:** `{ear if ear else 0:.2f}`
                - **Total Blinks:** `{blink_detector.total_blinks}`
                - **Rate (10s):** `{blink_metrics['blink_rate_10s'] if blink_metrics else 0:.2f} bps`
                - **State:** {blink_state_colors.get(blink_state, '⚪')} `{blink_state.upper()}`
                - **Calibrated:** `{'✅' if is_calibrated_detector else '⏳'}`
                """)
            
            # Status
            status_metric.markdown(f"""
            ### ℹ️ Status
            - **Faces Detected:** `{num_faces}`
            - **Frame:** `{frame_count}`
            - **Device:** `{device.type.upper()}`
            """)
            
            # Control frame rate
            time.sleep(max(0, 1/display_fps_target - (time.time() - curr_time)))
    
    except KeyboardInterrupt:
        st.warning("⚠️ Monitoring interrupted by user")
    
    except Exception as e:
        st.error(f"❌ Error during monitoring: {e}")
        import traceback
        st.code(traceback.format_exc())
    
    finally:
        cap.release()
        st.success("✅ Monitoring stopped.")
        
        # Show session summary
        if blink_detector:
            st.subheader("📈 Session Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Frames", frame_count)
            with col2:
                st.metric("Total Blinks", blink_detector.total_blinks)
            with col3:
                avg_blink_rate = blink_detector.total_blinks / (frame_count / fps_actual / 60) if frame_count > 0 else 0
                st.metric("Avg Blink Rate", f"{avg_blink_rate:.1f} bpm")

else:
    st.info("👆 Configure settings in the sidebar and click **Start Monitoring** to begin!")
    
    # Show instructions
    st.markdown("""
    ## 📖 Instructions
    
    1. **Choose video source**: Webcam or video file
    2. **Enable blink detection**: For more accurate engagement estimation
    3. **Select ML classifier**: Use trained model or rule-based fallback
    4. **Adjust settings**: Sensitivity, FPS, visual options
    5. **Click Start**: Begin real-time monitoring
    
    ## 🎯 Engagement Levels
    
    - 🟢 **Highly Engaged**: Attentive, looking at screen, normal blink rate
    - 🟡 **Engaged**: Generally attentive with minor distractions
    - 🟠 **Partially Engaged**: Distracted, irregular gaze or blink patterns
    - 🔴 **Disengaged**: Not paying attention, drowsy, or looking away
    
    ## 🔍 Metrics Explained
    
    - **EAR (Eye Aspect Ratio)**: Measure of eye openness (lower = more closed)
    - **Blink Rate**: Blinks per second over 10-second window
    - **Blink State**: Classified as normal, drowsy, stressed, or distracted
    - **Pitch/Yaw**: Head orientation angles
    """)