from src.pipeline.training_pipeline import TrainingPipeline
from src import get_logger

logger = get_logger("main")

if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.train()
    except Exception as e:
        logger.exception(e)
        raise e
