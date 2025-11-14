import sounddevice as sd
import numpy as np
import webrtcvad
from scipy.io.wavfile import write
import io
from src.pipeline.inference_pipeline import InferencePipeline
import threading
import queue
import torch
import torchaudio

# --- Configuration ---
CAPTURE_SAMPLE_RATE = 44100  # Supported capture rate
TARGET_SAMPLE_RATE = 16000  # 16kHz for the model
CHUNK_DURATION_MS = 30  # 30ms
CHUNK_SAMPLES = int(CAPTURE_SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
VAD_AGGRESSIVENESS = 3  # 0 (least aggressive) to 3 (most aggressive)
SILENCE_THRESHOLD_S = 0.5 # Seconds of silence to trigger transcription
MIN_SPEECH_DURATION_S = 0.3 # Minimum duration of speech to consider for transcription

class RealtimeTranscriber:
    def __init__(self):
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.pipe = InferencePipeline()
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=CAPTURE_SAMPLE_RATE, new_freq=TARGET_SAMPLE_RATE
        )
        self.audio_buffer = []
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.last_transcript = ""
        self.transcription_queue = queue.Queue()

    def _transcribe_chunk(self, audio_data):
        try:
            text = self.pipe.predict(audio_data)
            if text and text != self.last_transcript:
                print(f"Transcription: {text}")
                self.last_transcript = text
        except Exception as e:
            print(f"Error during transcription: {e}")

    def _process_audio(self, indata):
        # Resample the audio first
        audio_tensor = torch.from_numpy(indata.flatten()).float()
        resampled_audio = self.resampler(audio_tensor)
        
        # Convert to 16-bit PCM for VAD
        audio_chunk_16k = (resampled_audio.numpy() * 32767).astype(np.int16)
        
        # VAD expects chunks of 10, 20, or 30 ms.
        # We need to ensure our resampled chunk is the right size.
        expected_samples = int(TARGET_SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
        if len(audio_chunk_16k) != expected_samples:
            # This might happen due to resampling, pad or truncate if necessary
            # For simplicity, we'll skip this chunk if the size is wrong.
            # A more robust solution would buffer and frame the audio correctly.
            return

        is_speech = self.vad.is_speech(audio_chunk_16k.tobytes(), TARGET_SAMPLE_RATE)

        if is_speech:
            self.silence_frames = 0
            self.speech_frames += 1
            self.audio_buffer.append(audio_chunk_16k)
            if not self.is_speaking:
                print("Speaking detected...")
                self.is_speaking = True
        else:
            self.silence_frames += 1
            if self.is_speaking:
                num_silence_chunks = (SILENCE_THRESHOLD_S * 1000) / CHUNK_DURATION_MS
                num_speech_chunks = (MIN_SPEECH_DURATION_S * 1000) / CHUNK_DURATION_MS
                if self.silence_frames > num_silence_chunks and self.speech_frames > num_speech_chunks:
                    print("Silence detected, transcribing...")
                    full_audio = np.concatenate(self.audio_buffer)
                    self.audio_buffer = []
                    self.speech_frames = 0
                    self.is_speaking = False

                    # Create in-memory WAV file
                    byte_io = io.BytesIO()
                    write(byte_io, TARGET_SAMPLE_RATE, full_audio)
                    byte_io.seek(0)
                    
                    # Offload transcription to a separate thread
                    threading.Thread(target=self._transcribe_chunk, args=(byte_io,)).start()

    def start(self):
        print("Starting real-time transcription. Press Ctrl+C to stop.")
        with sd.InputStream(samplerate=CAPTURE_SAMPLE_RATE, channels=1, dtype='float32', blocksize=CHUNK_SAMPLES, callback=self.audio_callback):
            while True:
                try:
                    pass
                except KeyboardInterrupt:
                    print("\nStopping transcription.")
                    break

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status, flush=True)
        self._process_audio(indata)

if __name__ == "__main__":
    transcriber = RealtimeTranscriber()
    transcriber.start()