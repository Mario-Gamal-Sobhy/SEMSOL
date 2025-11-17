import os
from src import get_logger
from box.exceptions import BoxValueError
import yaml
import json
import joblib
from typeguard import typechecked
from box import ConfigBox 
from pathlib import Path
from typing import Any

from src.exceptions import FileOperationError, EmptyFileError

logger = get_logger("common")

@typechecked
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise EmptyFileError(f"YAML file {path_to_yaml} is empty")
            logger.info(f"yaml file: {path_to_yaml.name} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise EmptyFileError(f"YAML file {path_to_yaml} is empty")
    except Exception as e:
        raise FileOperationError(e)

@typechecked
def create_directories(path_to_directories: list, verbose: bool = True):
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")

@typechecked
def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
        logger.info(f"json file saved at: {path}")

@typechecked
def load_json(path: Path) -> ConfigBox:
    with open(path, "r") as f:
        content = json.load(f)
    logger.info(f"json file loaded from: {path}")
    return ConfigBox(content)

@typechecked
def save_bin(data: Any, path: Path):
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")

@typechecked
def load_bin(path: Path) -> Any:
    data = joblib.load(filename=path)
    logger.info(f"binary file loaded from: {path}")
    return data

@typechecked
def get_nlp_config() -> ConfigBox:
    params = read_yaml(Path('params.yaml'))
    nlp_params = params.nlp_model_trainer
    
    config = ConfigBox({
        "data_path": nlp_params.data_path,
        "model_path": nlp_params.model_path,
        "vocab_path": nlp_params.vocab_path,
        "max_len": nlp_params.max_len,
        "embedding_dim": nlp_params.embedding_dim,
        "hidden_dim": nlp_params.hidden_dim,
        "output_dim": nlp_params.output_dim,
        "n_layers": nlp_params.n_layers,
        "drop_prob": nlp_params.drop_prob,
        "epochs": nlp_params.epochs,
        "batch_size": nlp_params.batch_size,
        "learning_rate": nlp_params.learning_rate,
        "label_to_int": nlp_params.label_to_int,
        "int_to_label": nlp_params.int_to_label
    })
    return config
