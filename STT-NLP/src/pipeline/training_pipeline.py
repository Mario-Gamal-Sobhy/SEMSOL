from pathlib import Path
from src import get_logger
from src.components.Audio_Ingestion import AudioCaptureIngestion
from src.components.Audio_Preprocess import AudioPreprocess
from src.components.Model_Trainer import ModelTrainer
from src.config.Configuration import ConfigurationManager
from src.utils.common import create_directories

class TrainingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.logger = get_logger("TrainingPipeline")

    def train(self):
        self.logger.info("Starting training pipeline...")

        # Get configurations
        data_ingestion_config = self.config_manager.get_data_ingestion_config()
        data_transformation_config = self.config_manager.get_data_transformation_config()
        model_trainer_config = self.config_manager.get_model_trainer_config()

        # Perform data ingestion
        if not (Path(data_ingestion_config.root_dir) / "train.csv").exists() or \
           not (Path(data_ingestion_config.root_dir) / "test.csv").exists():
            create_directories([data_ingestion_config.root_dir])
            audio_ingestion = AudioCaptureIngestion(config=data_ingestion_config)
            audio_ingestion.prepare_librispeech_data()
        else:
            self.logger.info("Data ingestion artifacts already exist. Skipping.")

        # Perform data transformation
        if not (Path(data_transformation_config.root_dir) / "train_processed.csv").exists() or \
           not (Path(data_transformation_config.root_dir) / "test_processed.csv").exists():
            create_directories([data_transformation_config.root_dir])
            audio_preprocessor = AudioPreprocess(config=data_transformation_config)
            audio_preprocessor.perform_transformation()
        else:
            self.logger.info("Data transformation artifacts already exist. Skipping.")

        # Perform model training
        if not Path(model_trainer_config.model_name).exists():
            create_directories([model_trainer_config.root_dir])
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.train()
        else:
            self.logger.info("Model already trained. Skipping.")

        self.logger.info("Training pipeline finished.")
