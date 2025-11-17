from src.nlp.pipeline.training_pipeline import NLPTrainingPipeline
from src.utils.common import get_nlp_config, read_yaml
import mlflow
from pathlib import Path

if __name__ == '__main__':
    config = get_nlp_config()
    mlflow_config = read_yaml(Path('config.yaml')).mlflow_config
    mlflow.set_tracking_uri(mlflow_config.tracking_uri)
    pipeline = NLPTrainingPipeline(config=config)
    pipeline.run()