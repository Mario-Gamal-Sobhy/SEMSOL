from src.config.constants import *
from src.utils.common import read_yaml, create_directories
from src.config.entity import (DataIngestionConfig,
                    DataTransformationConfig,
                    ModelTrainerConfig)
from pathlib import Path


class ConfigurationManager:

    def __init__(self,config_filepath: Path = CONFIG_FILE_PATH, params_filepath: Path = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:

        config = self.config.data_ingestion

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            audio_path=Path(config.audio_path),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
        )

        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:

        config = self.config.data_transformation

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            audio_path=Path(config.audio_path),
            preprocessor_object_file=Path(config.preprocessor_object_file),
            train_path=Path(config.train_data_path),
            test_path=Path(config.test_data_path),
            processed_train_path=Path(config.processed_train_path),
            processed_test_path=Path(config.processed_test_path),
            original_sample_rate=self.params.audio_preprocessing.original_sample_rate,
            new_sample_rate=self.params.audio_preprocessing.new_sample_rate,
            char_map_file=CHAR_MAP_FILE_PATH
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:

        config = self.config.model_trainer
        parameters = self.params

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            train_data_path = Path(config.train_data_path),
            test_data_path = Path(config.test_data_path),
            model_name = Path(config.root_dir) / Path(config.model_name),
            params = parameters,  
            target_column = config.target_column,
            char_map_file = CHAR_MAP_FILE_PATH,
            processed_train_path = Path(self.config.data_transformation.processed_train_path),
            processed_test_path = Path(self.config.data_transformation.processed_test_path)
        )

        return model_trainer_config
