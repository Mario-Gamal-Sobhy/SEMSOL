from src.config.Configuration import ConfigurationManager
from src.components.Model_Trainer import ModelTrainer
from src import get_logger
from src.utils.common import create_directories

if __name__ == "__main__":
    logger = get_logger("RunTrainer")
    try:
        logger.info("Starting model training...")
        config_manager = ConfigurationManager()
        model_trainer_config = config_manager.get_model_trainer_config()
        create_directories([model_trainer_config.root_dir])
        trainer = ModelTrainer(config=model_trainer_config)
        trainer.train()
        logger.info("Model training finished successfully.")
    except Exception as e:
        logger.error(f"Error in model training: {e}")
