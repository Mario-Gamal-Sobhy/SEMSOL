from src.config.entity import DataTransformationConfig
from src import get_logger
import torchaudio
import torch
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.exceptions import AudioProcessingError

class AudioPreprocess:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.logger = get_logger("Audio Data Preprocess")

    def get_preprocess_object(self) -> torch.nn.Sequential:
        try:
            self.logger.info("Creating preprocessing object...")
            transforms = torch.nn.Sequential(
                torchaudio.transforms.Resample(orig_freq=self.config.original_sample_rate, new_freq=self.config.new_sample_rate),
                torchaudio.transforms.MelSpectrogram(sample_rate=self.config.new_sample_rate, n_mels=128, n_fft=400, hop_length=160),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
                torchaudio.transforms.TimeMasking(time_mask_param=35),
            )
            self.logger.info("Successfully created preprocessing object.")
            return transforms
        except Exception as e:
            self.logger.error(f"Error in creating preprocessing object: {e}")
            raise AudioProcessingError(f"Failed to create preprocessing object: {e}")

    def _process_audio(self, audio_path: Path, transform: torch.nn.Module) -> torch.Tensor:
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            spectrogram = transform(waveform)
            return spectrogram
        except Exception as e:
            self.logger.error(f"Error processing audio file {audio_path}: {e}")
            raise AudioProcessingError(f"Failed to process audio file {audio_path}: {e}")

    def perform_transformation(self):
        try:
            self.logger.info("Starting audio data transformation...")
            preprocessor = self.get_preprocess_object()
            torch.save(preprocessor, self.config.preprocessor_object_file)
            self.logger.info(f"Preprocessor object saved to {self.config.preprocessor_object_file}")

            for split in ["train", "test"]:
                self.logger.info(f"Processing {split} data...")
                input_df = pd.read_csv(Path(self.config.data_path) / f"{split}.csv")

                processed_data = []
                for _, row in tqdm(input_df.iterrows(), total=input_df.shape[0], desc=f"Transforming {split} data"):
                    audio_path = Path(row["audio_path"])
                    transcript = row["transcript"]
                    spectrogram = self._process_audio(audio_path, preprocessor)
                    if spectrogram is not None:
                        spectrogram_path = Path(self.config.root_dir) / split / audio_path.name.replace(".wav", ".pt")
                        spectrogram_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(spectrogram, spectrogram_path)
                        processed_data.append({"spectrogram_path": str(spectrogram_path), "transcript": transcript})

                output_df = pd.DataFrame(processed_data)
                output_csv_path = Path(self.config.root_dir) / f"{split}_processed.csv"
                output_df.to_csv(output_csv_path, index=False)
                self.logger.info(f"Saved processed {split} data to {output_csv_path}")

            self.logger.info("Audio data transformation completed.")
        except Exception as e:
            self.logger.error(f"Error in data transformation: {e}")
            raise e
