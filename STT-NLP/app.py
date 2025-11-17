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
from src import database
import pandas as pd

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
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    body {
        font-family: 'Roboto', sans-serif;
    }

    .stApp {
        background: linear-gradient(315deg, #1e1e1e 0%, #2a2a2a 74%);
        color: #f1f1f1;
    }

    .stButton>button {
        background: linear-gradient(145deg, #00bfff, #1e90ff);
        color: white;
        border-radius: 25px;
        border: none;
        padding: 12px 28px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 10px 2px;
        cursor: pointer;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 15px 0 rgba(0, 191, 255, 0.75);
    }

    .stButton>button:hover {
        background: linear-gradient(145deg, #1e90ff, #00bfff);
        box-shadow: 0 6px 20px 0 rgba(0, 191, 255, 0.95);
        transform: translateY(-2px);
    }
    
    .stButton>button:active {
        transform: translateY(1px);
        box-shadow: 0 2px 10px 0 rgba(0, 191, 255, 0.75);
    }

    .stRadio > div {
        flex-direction: row;
        background-color: #333;
        padding: 10px;
        border-radius: 25px;
    }

    .stFileUploader {
        border: 2px dashed #00bfff;
        border-radius: 15px;
        padding: 20px;
        background-color: #2a2a2a;
    }

    .stSpinner > div > div {
        border-top-color: #00bfff !important;
        border-right-color: #00bfff !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #00bfff;
        text-shadow: 2px 2px 4px #1e1e1e;
    }

    .stTextArea textarea {
        background-color: #2a2a2a;
        color: #f1f1f1;
        border-radius: 15px;
        border: 1px solid #00bfff;
    }

    .stDataFrame {
        border: 1px solid #00bfff;
        border-radius: 15px;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .stApp > div {
        animation: fadeIn 1s ease-in-out;
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

# --- NLP Model Loading ---
from src.nlp.predictor import get_nlp_predictor

@st.cache_resource
def load_nlp_predictor():
    """Loads the NLP predictor."""
    try:
        return get_nlp_predictor()
    except Exception as e:
        st.error(f"Failed to load the NLP model: {e}")
        return None

nlp_predictor = load_nlp_predictor()

def display_sentiment(text):
    if nlp_predictor and text:
        sentiment = nlp_predictor.predict(text)
        st.subheader("Sentiment Analysis")
        if sentiment == 'positive':
            st.markdown(f"### 😊 <span style='color:green;'>Positive</span>", unsafe_allow_html=True)
        elif sentiment == 'negative':
            st.markdown(f"### 😠 <span style='color:red;'>Negative</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"### 😐 <span style='color:gray;'>Neutral</span>", unsafe_allow_html=True)

# --- Main Application ---
st.sidebar.title("Options")
app_mode = st.sidebar.radio("Choose the transcription mode:", ("File Upload", "Real-time Recording", "History"))

if app_mode == "File Upload":
    st.header("Transcribe from an Audio File")
    
    uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        if st.button("Transcribe"):
            with st.spinner("Transcribing..."):
                try:
                    # Get the content of the uploaded file
                    audio_data = uploaded_file.getvalue()
                    
                    # Perform prediction
                    text = pipe.predict(io.BytesIO(audio_data))
                    
                    # Save to database
                    database.add_transcription(
                        mode="File Upload",
                        transcription=text,
                        audio_data=audio_data
                    )
                    
                    st.success("Transcription Complete!")
                    st.text_area("Transcription:", value=text, height=200, key="file_upload_transcription")
                    display_sentiment(text)
                except Exception as e:
                    st.error(f"An error occurred during transcription: {e}")

elif app_mode == "Real-time Recording":
    st.header("Transcribe in Real-time")
    st.warning("Real-time transcription is experimental and may not work perfectly in all browsers.")

    # --- Configuration from realtime_transcribe.py ---
    def get_default_input_device_samplerate() -> int:
        """Get the default input device's sample rate."""
        try:
            device_info = sd.query_devices(kind='input')
            if isinstance(device_info, dict) and 'default_samplerate' in device_info:
                return int(device_info['default_samplerate'])
            else:
                devices = sd.query_devices()
                for i, device in enumerate(devices):
                    if device['name'] == 'default':
                        return int(device['default_samplerate'])
                return 44100 # Fallback if default not found
        except Exception as e:
            print(f"Could not query devices, falling back to 44100 Hz: {e}")
            return 44100

    CAPTURE_SAMPLE_RATE = get_default_input_device_samplerate()
    TARGET_SAMPLE_RATE = 16000
    CHUNK_DURATION_MS = 30
    CHUNK_SAMPLES = int(CAPTURE_SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
    VAD_AGGRESSIVENESS = 3
    SILENCE_THRESHOLD_S = 0.5
    MIN_SPEECH_DURATION_S = 0.3

    # --- Real-time Transcription Logic ---
    if 'transcriber' not in st.session_state:
        st.session_state.transcriber = None
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'transcription_text' not in st.session_state:
        st.session_state.transcription_text = ""

    class RealtimeTranscriber:
        def __init__(self, transcription_queue):
            self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
            self.pipe = pipe
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=CAPTURE_SAMPLE_RATE, new_freq=TARGET_SAMPLE_RATE
            )
            self.audio_buffer: list[np.ndarray] = []
            self.is_speaking = False
            self.silence_frames = 0
            self.speech_frames = 0
            self.transcription_queue = transcription_queue
            self._resampled_buffer = np.array([], dtype=np.float32)
            self._stop_event = threading.Event()
            self.full_recording = []

        def _transcribe_chunk(self, audio_data: io.BytesIO):
            """Transcribes a chunk of audio data."""
            try:
                text = self.pipe.predict(audio_data)
                if text:
                    self.transcription_queue.put(text)
            except Exception as e:
                st.error(f"Transcription error: {e}")

        def _process_audio(self, indata: np.ndarray):
            """Processes a chunk of audio data from the input stream."""
            self.full_recording.append(indata.copy())
            audio_tensor = torch.from_numpy(indata.flatten()).float()
            resampled_audio = self.resampler(audio_tensor)
            
            self._resampled_buffer = np.append(self._resampled_buffer, resampled_audio.numpy())

            expected_samples = int(TARGET_SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
            while len(self._resampled_buffer) >= expected_samples:
                chunk_16k_float = self._resampled_buffer[:expected_samples]
                self._resampled_buffer = self._resampled_buffer[expected_samples:]
                
                chunk_16k_int16 = (chunk_16k_float * 32767).astype(np.int16)

                is_speech = self.vad.is_speech(chunk_16k_int16.tobytes(), TARGET_SAMPLE_RATE)

                if is_speech:
                    self.silence_frames = 0
                    self.speech_frames += 1
                    self.audio_buffer.append(chunk_16k_int16)
                    if not self.is_speaking:
                        self.is_speaking = True
                else:
                    self.silence_frames += 1
                    if self.is_speaking:
                        num_silence_chunks = (SILENCE_THRESHOLD_S * 1000) / CHUNK_DURATION_MS
                        num_speech_chunks = (MIN_SPEECH_DURATION_S * 1000) / CHUNK_DURATION_MS
                        if self.silence_frames > num_silence_chunks and self.speech_frames > num_speech_chunks:
                            full_audio = np.concatenate(self.audio_buffer)
                            self.audio_buffer = []
                            self.speech_frames = 0
                            self.is_speaking = False

                            byte_io = io.BytesIO()
                            write(byte_io, TARGET_SAMPLE_RATE, full_audio)
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
                samplerate=CAPTURE_SAMPLE_RATE, channels=1, dtype='float32',
                blocksize=CHUNK_SAMPLES,
                callback=self.audio_callback
            )
            self.stream.start()

        def stop(self):
            self._stop_event.set()
            if hasattr(self, 'stream'):
                self.stream.stop()
                self.stream.close()
            
            if self.full_recording:
                full_audio = np.concatenate(self.full_recording)
                byte_io = io.BytesIO()
                write(byte_io, CAPTURE_SAMPLE_RATE, full_audio)
                byte_io.seek(0)
                return byte_io.getvalue()
            return None

    if 'transcription_queue' not in st.session_state:
        st.session_state.transcription_queue = queue.Queue()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Recording", disabled=st.session_state.is_recording):
            st.session_state.is_recording = True
            st.session_state.transcriber = RealtimeTranscriber(st.session_state.transcription_queue)
            threading.Thread(target=st.session_state.transcriber.start).start()
            st.info("Recording started...")

    with col2:
        if st.button("Stop Recording", disabled=not st.session_state.is_recording):
            audio_data = None
            if st.session_state.transcriber:
                audio_data = st.session_state.transcriber.stop()
                st.session_state.transcriber = None
            
            if st.session_state.transcription_text:
                database.add_transcription(
                    mode="Real-time",
                    transcription=st.session_state.transcription_text,
                    audio_data=audio_data
                )
            
            st.session_state.is_recording = False
            st.info("Recording stopped.")

    if st.session_state.get('is_recording', False):
        st.subheader("Live Transcription")
        
        try:
            new_text = st.session_state.transcription_queue.get(block=False)
            st.session_state.transcription_text += new_text + " "
        except queue.Empty:
            pass

        st.text_area(
            "Transcription:", 
            st.session_state.get('transcription_text', ''), 
            height=300, 
            key="realtime_transcription"
        )

        import time
        time.sleep(0.25)
        st.rerun()
    elif st.session_state.get('transcription_text', ''):
        st.subheader("Final Transcription")
        final_text = st.session_state.get('transcription_text', '')
        st.text_area(
            "Transcription:", 
            final_text, 
            height=300, 
            key="final_transcription"
        )
        display_sentiment(final_text)
        st.session_state.transcription_text = ""

elif app_mode == "History":
    st.header("Transcription History")
    
    records = database.get_all_transcriptions()
    
    if not records:
        st.info("No transcriptions found in the history.")
    else:
        df = pd.DataFrame(records, columns=['Timestamp', 'Mode', 'Audio File', 'Transcription'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        st.dataframe(df, use_container_width=True)
        
        for index, row in df.iterrows():
            if row['Audio File']:
                st.audio(row['Audio File'])

        if st.button("Clear History"):
            database.clear_history()
            st.success("History cleared.")
            st.rerun()
