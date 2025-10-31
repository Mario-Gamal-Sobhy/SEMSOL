from src.pipeline.training_pipeline import TrainingPipeline
from src import get_logger

logger = get_logger("STT-Main")

if __name__ == "__main__":
    try:
        logger.info("Starting training pipeline...")
        TrainingPipeline().train()
        logger.info("Training pipeline finished successfully.")
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
