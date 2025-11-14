from src.pipeline.evaluation_pipeline import EvaluationPipeline
from src.exceptions import STTException
import sys

def main():
    try:
        evaluation_pipeline = EvaluationPipeline()
        evaluation_pipeline.evaluate()
    except Exception as e:
        raise STTException(e, sys)

if __name__ == "__main__":
    main()
