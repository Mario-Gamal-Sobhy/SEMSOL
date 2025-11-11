from src.config.entity import DataIngestionConfig
from src import get_logger
from src.utils.common import create_directories
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import soundfile as sf


class AudioCaptureIngestion:
    """
    Handles data ingestion for LibriSpeech dataset.
    Creates structured CSV files for train and test sets.
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.logger = get_logger("Audio Data Capture & Ingestion")

    def prepare_librispeech_data(self):
        """
        Processes LibriSpeech data, converts FLAC to WAV, and creates CSV files for train and test sets.
        """
        try:
            self.logger.info("Starting LibriSpeech data ingestion pipeline...")
            create_directories([self.config.root_dir, self.config.audio_path])

            for split_name, split_dir in [("train", "train-clean-100"), ("test", "test-clean")]:
                self.logger.info(f"Processing {split_name} split...")
                split_path = Path(self.config.data_path) / split_dir
                output_audio_dir = Path(self.config.audio_path) / split_name
                create_directories([output_audio_dir])

                audio_text_pairs = []

                self.logger.info(f"Searching for transcripts in: {split_path}")
                transcript_files = list(split_path.rglob("*.trans.txt"))
                self.logger.info(f"Found {len(transcript_files)} transcript files.")
                
                for transcript_file in tqdm(transcript_files, desc=f"Processing {split_name} transcripts"):
                    with open(transcript_file, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                utterance_id, transcript = line.strip().split(" ", 1)
                                flac_path = transcript_file.parent / f"{utterance_id}.flac"

                                if flac_path.exists():
                                    wav_path = output_audio_dir / f"{utterance_id}.wav"
                                    if not wav_path.exists():
                                        data, samplerate = sf.read(flac_path)
                                        sf.write(wav_path, data, samplerate)
                                    audio_text_pairs.append({
                                        "audio_path": str(wav_path),
                                        "transcript": transcript
                                    })
                            except ValueError:
                                self.logger.warning(f"Skipping malformed line in {transcript_file}: {line.strip()}")

                df = pd.DataFrame(audio_text_pairs)
                csv_output_path = Path(self.config.root_dir) / f"{split_name}.csv"
                df.to_csv(csv_output_path, index=False)
                self.logger.info(f"Saved {split_name} data to {csv_output_path}")

            self.logger.info("LibriSpeech ingestion completed successfully.")

        except Exception as e:
            self.logger.error(f"Error in data ingestion: {e}")
            raise e
