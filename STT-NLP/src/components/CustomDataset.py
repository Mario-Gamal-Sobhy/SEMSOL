import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from typing import Tuple
from src.components.FeaturesExctractor import FeaturesExtractor

class CustomDataset(Dataset):

    def __init__(self, data_path: Path, features_extractor: FeaturesExtractor):
        self.df = pd.read_csv(data_path)
        self.features_extractor = features_extractor

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:

        row = self.df.iloc[idx]
        spectrogram_path = Path(row["spectrogram_path"])
        transcript = row["transcript"]

        spectrogram = torch.load(spectrogram_path)
        spectrogram = spectrogram.squeeze(0).t()

        label = self.features_extractor.text_transform(transcript)

        input_length = spectrogram.shape[0]
        label_length = len(label)

        return spectrogram, label, input_length, label_length