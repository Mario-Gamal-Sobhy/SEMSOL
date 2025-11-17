from pathlib import Path
from src import get_logger
from src.components.Audio_Ingestion import AudioCaptureIngestion
from src.components.Audio_Preprocess import AudioPreprocess
from src.components.Model_Trainer import ModelTrainer
from src.utils.common import create_directories

class TrainingPipeline:
    def __init__(self, data_ingestion_config, data_transformation_config, model_trainer_config):
        self.data_ingestion_config = data_ingestion_config
        self.data_transformation_config = data_transformation_config
        self.model_trainer_config = model_trainer_config
        self.logger = get_logger("TrainingPipeline")

    def train(self):
        self.logger.info("Starting training pipeline...")

        # Perform data ingestion
        if not Path(self.data_ingestion_config.root_dir).exists():
            create_directories([self.data_ingestion_config.root_dir])
            audio_ingestion = AudioCaptureIngestion(config=self.data_ingestion_config)
            audio_ingestion.prepare_librispeech_data()
        else:
            self.logger.info("Data ingestion artifacts already exist. Skipping.")

        # Perform data transformation
        if not Path(self.data_transformation_config.root_dir).exists():
            create_directories([self.data_transformation_config.root_dir])
            audio_preprocessor = AudioPreprocess(config=self.data_transformation_config)
            audio_preprocessor.perform_transformation()
        else:
            self.logger.info("Data transformation artifacts already exist. Skipping.")

        # Perform model training
        self.logger.info(f"Model path: {self.model_trainer_config.model_name}")
        if not Path(self.model_trainer_config.model_name).exists():
            create_directories([self.model_trainer_config.root_dir])
            model_trainer = ModelTrainer(config=self.model_trainer_config)
            model_trainer.train()
        else:
            self.logger.info("Model already trained. Skipping.")

        self.logger.info("Training pipeline finished.")
