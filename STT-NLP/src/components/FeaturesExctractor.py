import torch
from src.config.entity import DataTransformationConfig
from src import get_logger
from pathlib import Path
from typing import Tuple, List, Dict, Any

class FeaturesExtractor:
    def __init__(self, config: DataTransformationConfig = None, char_map_file: Path = None):
        self.config = config
        self.logger = get_logger("Features Extractor")
        if config:
            self.char_map = self._get_char_map(config.char_map_file)
        elif char_map_file:
            self.char_map = self._get_char_map(char_map_file)
        else:
            raise ValueError("Either config or char_map_file must be provided.")

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
