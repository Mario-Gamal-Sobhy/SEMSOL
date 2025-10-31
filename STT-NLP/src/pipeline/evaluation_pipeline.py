from src.pipeline.inference_pipeline import InferencePipeline
from src.config.Configuration import ConfigurationManager
import pandas as pd
from tqdm import tqdm
import jiwer
from src import get_logger

class EvaluationPipeline:
    def __init__(self):
        self.logger = get_logger("EvaluationPipeline")
        self.config_manager = ConfigurationManager()
        self.data_transformation_config = self.config_manager.get_data_transformation_config()
        self.inference_pipeline = InferencePipeline()

    def evaluate(self):
        self.logger.info("Starting evaluation pipeline...")

        try:
            # Load test data
            test_df: pd.DataFrame = pd.read_csv(self.data_transformation_config.test_path)

            ground_truth: list[str] = []
            predictions: list[str] = []

            for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0], desc="Evaluating"):
                audio_path: str = row["audio_path"]
                transcript: str = row["transcript"]

                # Get prediction
                prediction: str = self.inference_pipeline.predict(audio_path)

                ground_truth.append(transcript)
                predictions.append(prediction)

            # Calculate WER and CER
            measures: dict = jiwer.compute_measures(ground_truth, predictions)
            wer: float = measures['wer']
            cer: float = measures['cer']

            self.logger.info(f"Word Error Rate (WER): {wer}")
            self.logger.info(f"Character Error Rate (CER): {cer}")

            self.logger.info("Evaluation pipeline finished.")

        except Exception as e:
            self.logger.error(f"Error in evaluation pipeline: {e}")
            raise e
