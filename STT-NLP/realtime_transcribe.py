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
from typing import Optional


def get_default_input_device_samplerate() -> int:
    """Get the default input device's sample rate."""
    try: # Minimum duration of speech to consider for transcription
        device_info = sd.query_devices(kind='input')
        if isinstance(device_info, dict) and 'default_samplerate' in device_info:
            return int(device_info['default_samplerate'])
        else:
           
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if device['name'] == 'default':
                    print(f"Detected default input device sample rate: {int(device['default_samplerate'])} Hz")
                    return int(device['default_samplerate'])
            print("Could not find default input device, falling back to 44100 Hz.")
            return 44100 # Fallback if default not found
    except Exception as e:
        print(f"Could not query devices, falling back to 44100 Hz: {e}")
        return 44100

# --- Configuration ---
CAPTURE_SAMPLE_RATE = get_default_input_device_samplerate()
TARGET_SAMPLE_RATE = 16000  # 16kHz for the model
CHUNK_DURATION_MS = 30  
CHUNK_SAMPLES = int(CAPTURE_SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
VAD_AGGRESSIVENESS = 3  # 0 (least aggressive) to 3 (most aggressive)
SILENCE_THRESHOLD_S = 0.5 
MIN_SPEECH_DURATION_S = 0.3 # Minimum duration of speech to consider for transcription



class RealtimeTranscriber:
    """
    A class to handle real-time audio transcription.

    This class captures audio from the microphone, detects speech using Voice
    Activity Detection (VAD), and transcribes the speech to text in real-time.
    """
    def __init__(self):
        """Initializes the RealtimeTranscriber."""
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.pipe = InferencePipeline()
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=CAPTURE_SAMPLE_RATE, new_freq=TARGET_SAMPLE_RATE
        )
        self.audio_buffer: list[np.ndarray] = []
        self.is_speaking = False
        self.silence_frames = 0
        self.speech_frames = 0
        self.transcription_queue: queue.Queue[io.BytesIO] = queue.Queue()
        self._resampled_buffer = np.array([], dtype=np.float32)
        self._stop_event: Optional[threading.Event] = None

    def _transcribe_chunk(self, audio_data: io.BytesIO):
        """Transcribes a chunk of audio data."""
        try:
            text = self.pipe.predict(audio_data)
            if text:
                print(f"Transcription: {text}")
        except Exception as e:
            print(f"Error during transcription: {e}")

    def _process_audio(self, indata: np.ndarray):
        """Processes a chunk of audio data from the input stream."""

        # Resample the audio first
        audio_tensor = torch.from_numpy(indata.flatten()).float()
        resampled_audio = self.resampler(audio_tensor)
        
        # Append to our resampled buffer
        self._resampled_buffer = np.append(self._resampled_buffer, resampled_audio.numpy())

        # Process the buffer in VAD-sized chunks
        expected_samples = int(TARGET_SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
        while len(self._resampled_buffer) >= expected_samples:
            # Get a chunk from the buffer
            chunk_16k_float = self._resampled_buffer[:expected_samples]
            self._resampled_buffer = self._resampled_buffer[expected_samples:]
            
            # Convert to 16-bit PCM for VAD
            chunk_16k_int16 = (chunk_16k_float * 32767).astype(np.int16)

            is_speech = self.vad.is_speech(chunk_16k_int16.tobytes(), TARGET_SAMPLE_RATE)

            if is_speech:
                self.silence_frames = 0
                self.speech_frames += 1
                self.audio_buffer.append(chunk_16k_int16)
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
                        
        """Starts the real-time transcription process."""
                        
        print(f"Starting real-time transcription with capture rate {CAPTURE_SAMPLE_RATE}Hz and blocksize {CHUNK_SAMPLES}.")
                        
        self._stop_event = threading.Event()
                        
        with sd.InputStream(samplerate=CAPTURE_SAMPLE_RATE, channels=1, dtype='float32', blocksize=CHUNK_SAMPLES, callback=self.audio_callback):
                        
            self._stop_event.wait()
                        

                        
    def stop(self):
                        
        """Stops the real-time transcription process."""
                        
        print("\nStopping transcription.")
                        
        if self._stop_event:
                        
            self._stop_event.set()
                        

                        
    def audio_callback(self, indata: np.ndarray, frames: int, time, status: sd.CallbackFlags):
                        
        """Callback function for the audio stream."""
                        
        if status:
                        
            print(status, flush=True)
                        
        self._process_audio(indata)
                        

                        
def main():
                        
    """Main function to run the real-time transcriber."""
                        
    transcriber = RealtimeTranscriber()
                        
    try:
                        
        transcriber.start()
                        
    except KeyboardInterrupt:
                        
        transcriber.stop()
                        

                        
if __name__ == "__main__":
                        
    main()