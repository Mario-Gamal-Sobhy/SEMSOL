from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    data_path: Path
    audio_path: Path
    source_URL: Path
    local_data_file: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    audio_path: Path
    preprocessor_object_file: Path
    train_path: Path
    test_path: Path
    processed_train_path: Path
    processed_test_path: Path
    original_sample_rate: int
    new_sample_rate: int
    char_map_file: Path

# for each model , create entity config
@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: Path
    params: Dict[str, Any]
    target_column: str
    char_map_file: Path
    processed_train_path: Path
    processed_test_path: Path
