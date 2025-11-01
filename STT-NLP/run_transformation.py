from pathlib import Path
from src import get_logger
from src.config.Configuration import ConfigurationManager
from src.components.Audio_Ingestion import AudioCaptureIngestion
from src.components.Audio_Preprocess import AudioPreprocess
from src.utils.common import create_directories

logger = get_logger("run_transformation")

if __name__ == "__main__":
    try:
        logger.info("Starting data transformation...")
        config_manager = ConfigurationManager()

        # Ensure ingestion CSVs exist
        di_cfg = config_manager.get_data_ingestion_config()
        if not (Path(di_cfg.root_dir) / "train.csv").exists() or not (Path(di_cfg.root_dir) / "test.csv").exists():
            logger.info("Ingestion CSVs missing; running ingestion first...")
            create_directories([di_cfg.root_dir, di_cfg.audio_path])
            AudioCaptureIngestion(di_cfg).prepare_librispeech_data()

        # Run transformation (saves eval preprocessor and masked train specs)
        dt_cfg = config_manager.get_data_transformation_config()
        create_directories([dt_cfg.root_dir])
        AudioPreprocess(dt_cfg).perform_transformation()
        logger.info("Data transformation finished successfully.")
    except Exception as e:
        logger.error(f"Error in transformation: {e}")
