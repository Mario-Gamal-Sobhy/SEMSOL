import streamlit as st
from src.pipeline.inference_pipeline import InferencePipeline
import io
import sounddevice as sd
import numpy as np
import webrtcvad
from scipy.io.wavfile import write
import threading
import queue
import torch
import torchaudio

# --- Page Configuration ---
st.set_page_config(
    page_title="Speech-to-Text Transcription",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Styling ---
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stRadio > div {
        flex-direction: row;
    }
</style>
""", unsafe_allow_html=True)

# --- Title and Description ---
st.title("🎙️ Speech-to-Text Transcription")
st.markdown("This application transcribes speech from an audio file or real-time recording using a deep learning model.")

# --- Model Loading ---
@st.cache_resource
def load_pipeline():
    """Loads the inference pipeline."""
    try:
        return InferencePipeline()
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
        return None

pipe = load_pipeline()

if pipe is None:
    st.stop()

# --- Main Application ---
st.sidebar.title("Options")
app_mode = st.sidebar.radio("Choose the transcription mode:", ("File Upload", "Real-time Recording"))

if app_mode == "File Upload":
    st.header("Transcribe from an Audio File")
    
    uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Transcribe"):
            with st.spinner("Transcribing..."):
                try:
                    # Read the uploaded file into a BytesIO buffer
                    buf = io.BytesIO(uploaded_file.read())
                    # Perform prediction
                    text = pipe.predict(buf)
                    st.success("Transcription Complete!")
                    st.text_area("Transcription:", value=text, height=200)
                except Exception as e:
                    st.error(f"An error occurred during transcription: {e}")

elif app_mode == "Real-time Recording":
    st.header("Transcribe in Real-time")
    st.warning("Real-time transcription is experimental and may not work perfectly in all browsers.")

    # --- Real-time Transcription Logic ---
    # NOTE: Streamlit's execution model makes real-time audio processing challenging.
    # This implementation uses a threading approach to handle audio capture and
    # transcription in the background, updating the UI with the results.

    if 'transcriber' not in st.session_state:
        st.session_state.transcriber = None
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'transcription_text' not in st.session_state:
        st.session_state.transcription_text = ""

    class RealtimeTranscriber:
        def __init__(self):
            self.vad = webrtcvad.Vad(3)
            self.pipe = pipe
            self.resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)
            self.audio_buffer = []
            self.is_speaking = False
            self.silence_frames = 0
            self.speech_frames = 0
            self.transcription_queue = queue.Queue()
            self._stop_event = threading.Event()

        def _transcribe_chunk(self, audio_data):
            try:
                text = self.pipe.predict(audio_data)
                if text:
                    self.transcription_queue.put(text)
            except Exception as e:
                st.error(f"Transcription error: {e}")

        def _process_audio(self, indata):
            audio_tensor = torch.from_numpy(indata.flatten()).float()
            resampled_audio = self.resampler(audio_tensor)
            audio_chunk_16k = (resampled_audio.numpy() * 32767).astype(np.int16)
            
            expected_samples = int(16000 * 30 / 1000)
            if len(audio_chunk_16k) != expected_samples:
                return

            is_speech = self.vad.is_speech(audio_chunk_16k.tobytes(), 16000)

            if is_speech:
                self.silence_frames = 0
                self.speech_frames += 1
                self.audio_buffer.append(audio_chunk_16k)
                if not self.is_speaking:
                    self.is_speaking = True
            else:
                self.silence_frames += 1
                if self.is_speaking and self.silence_frames > 50 and self.speech_frames > 10:
                    full_audio = np.concatenate(self.audio_buffer)
                    self.audio_buffer = []
                    self.speech_frames = 0
                    self.is_speaking = False

                    byte_io = io.BytesIO()
                    write(byte_io, 16000, full_audio)
                    byte_io.seek(0)
                    
                    threading.Thread(target=self._transcribe_chunk, args=(byte_io,)).start()

        def audio_callback(self, indata, frames, time, status):
            if status:
                print(status)
            if not self._stop_event.is_set():
                self._process_audio(indata)

        def start(self):
            self._stop_event.clear()
            self.stream = sd.InputStream(
                samplerate=44100, channels=1, dtype='float32',
                blocksize=int(44100 * 30 / 1000),
                callback=self.audio_callback
            )
            self.stream.start()

        def stop(self):
            self._stop_event.set()
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Recording", disabled=st.session_state.is_recording):
            st.session_state.is_recording = True
            st.session_state.transcriber = RealtimeTranscriber()
            st.session_state.transcriber.start()
            st.info("Recording started...")

    with col2:
        if st.button("Stop Recording", disabled=not st.session_state.is_recording):
            st.session_state.is_recording = False
            if st.session_state.transcriber:
                st.session_state.transcriber.stop()
                st.session_state.transcriber = None
            st.info("Recording stopped.")

    if st.session_state.is_recording or st.session_state.transcription_text:
        st.subheader("Live Transcription")
        text_area = st.empty()
        
        while st.session_state.is_recording:
            try:
                new_text = st.session_state.transcriber.transcription_queue.get(timeout=1)
                st.session_state.transcription_text += new_text + " "
            except queue.Empty:
                pass
            text_area.text_area("Transcription:", value=st.session_state.transcription_text, height=300)

        # Clear text after stopping
        if not st.session_state.is_recording and st.session_state.transcription_text:
             st.session_state.transcription_text = ""
