import torch
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import os

class CustomDataset(Dataset):
    def __init__(self, data_path: Path, char_map_file: Path):
        self.df = pd.read_csv(data_path)
        self.char_map = self._get_char_map(char_map_file)
        # Lightweight validation only (avoid torch.load on init)
        self.df = self._filter_valid_data()

    def _filter_valid_data(self):
        valid_indices = []
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            transcript = row.get("transcript", "")
            if not isinstance(transcript, str) or not transcript.strip():
                continue

            spectrogram_path = Path(row["spectrogram_path"]) if "spectrogram_path" in row else None
            if spectrogram_path is None or not spectrogram_path.exists():
                continue
            # Fast check: file should be non-empty; avoid torch.load here
            try:
                if os.path.getsize(spectrogram_path) <= 0:
                    continue
            except OSError:
                continue
            
            valid_indices.append(i)
        
        return self.df.iloc[valid_indices].reset_index(drop=True)


    def __len__(self) -> int:
        return len(self.df)

    def _get_char_map(self, char_map_file: Path) -> Dict[str, int]:
        """Returns a character map for encoding and decoding transcripts from a file."""
        char_map = {}
        with open(char_map_file, "r", encoding="utf-8") as f:
            for line in f:
                ch, index = line.strip().split()
                char_map[ch] = int(index)
        return char_map

    def text_transform(self, text: str) -> torch.Tensor:
        """Converts a transcript into a tensor of character indices."""
        text = text.lower()
        indices = []
        for c in text:
            if c == " " and "<SPACE>" in self.char_map:
                indices.append(self.char_map["<SPACE>"])
            else:
                indices.append(self.char_map.get(c, self.char_map["'"]))
        return torch.tensor(indices, dtype=torch.long)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        row = self.df.iloc[idx]
        spectrogram_path = Path(row["spectrogram_path"])
        transcript = row["transcript"]

        spectrogram = torch.load(spectrogram_path)
        spectrogram = spectrogram.squeeze(0).t()

        label = self.text_transform(transcript)

        input_length = spectrogram.shape[0]
        label_length = len(label)

        return spectrogram, label, input_length, label_length
