from src.pipeline.training_pipeline import TrainingPipeline
from src.nlp.pipeline.training_pipeline import NLPTrainingPipeline
from src import get_logger
import argparse

logger = get_logger("main")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run training pipelines.")
    parser.add_argument("--pipeline", type=str, default="all", choices=["audio", "nlp", "all"],
                        help="Which pipeline to run: 'audio', 'nlp', or 'all'.")
    args = parser.parse_args()

    if args.pipeline == "audio" or args.pipeline == "all":
        try:
            logger.info("Starting audio training pipeline...")
            from src.config.Configuration import ConfigurationManager
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()
            data_transformation_config = config_manager.get_data_transformation_config()
            model_trainer_config = config_manager.get_model_trainer_config()
            audio_pipeline = TrainingPipeline(
                data_ingestion_config=data_ingestion_config,
                data_transformation_config=data_transformation_config,
                model_trainer_config=model_trainer_config
            )
            audio_pipeline.train()
            logger.info("Audio training pipeline finished.")
        except Exception as e:
            logger.exception(f"Error in audio training pipeline: {e}")
            raise e

    if args.pipeline == "nlp" or args.pipeline == "all":
        try:
            logger.info("Starting NLP training pipeline...")
            from src.utils.common import get_nlp_config
            
            config = get_nlp_config()
            
            nlp_pipeline = NLPTrainingPipeline(config=config)
            nlp_pipeline.run()
            logger.info("NLP training pipeline finished.")
        except Exception as e:
            logger.exception(f"Error in NLP training pipeline: {e}")
            raise e
