import pandas as pd
from sklearn.model_selection import train_test_split
from src import get_logger
from src.nlp.components.ingestion import DataIngestion
from src.nlp.components.preprocessing import TextPreprocessor
from src.nlp.components.transformation import TextTransformer
from src.nlp.components.training import ModelTrainer
from src.nlp.model.SentimentLSTM import SentimentLSTM
from src.nlp.components.evaluation import ModelEvaluator
import mlflow
import mlflow.pytorch

logger = get_logger("NLPTrainingPipeline")

class NLPTrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.label_to_int = self.config.label_to_int
        self.int_to_label = self.config.int_to_label

    def run(self):
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                "embedding_dim": self.config.embedding_dim,
                "hidden_dim": self.config.hidden_dim,
                "output_dim": self.config.output_dim,
                "n_layers": self.config.n_layers,
                "drop_prob": self.config.drop_prob,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "max_len": self.config.max_len
            })

            # Data Ingestion
            logger.info("Running Data Ingestion...")
            ingestion = DataIngestion(data_path=self.config.data_path)
            df = ingestion.get_data()
            logger.info("Data Ingestion Complete.")

            # Data Preprocessing
            logger.info("Running Data Preprocessing...")
            preprocessor = TextPreprocessor()
            preprocessed_df = preprocessor.preprocess_dataframe(df, text_column='text', sentiment_column='sentiment')
            logger.info("Data Preprocessing Complete.")

            # Data Splitting
            X = preprocessed_df['text'].values
            y = preprocessed_df['sentiment'].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Data Transformation
            logger.info("Running Data Transformation...")
            transformer = TextTransformer()
            transformer.build_vocab(X_train)
            X_train_transformed = transformer.transform(X_train, max_len=self.config.max_len)
            X_test_transformed = transformer.transform(X_test, max_len=self.config.max_len)
            transformer.save_vocab(self.config.vocab_path)
            logger.info("Data Transformation Complete.")

            # Model Training
            logger.info("Running Model Training...")
            vocab_size = len(transformer.vocab_to_int) + 1
            
            model = SentimentLSTM(
                vocab_size=vocab_size,
                embedding_dim=self.config.embedding_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.output_dim,
                n_layers=self.config.n_layers,
                drop_prob=self.config.drop_prob
            )
            model.label_to_int = self.label_to_int
            
            trainer = ModelTrainer(model=model, label_to_int=self.label_to_int)
            trained_model = trainer.train(
                X_train_transformed, 
                y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate
            )
            trainer.save_model(self.config.model_path)
            logger.info("Model Training Complete.")

            # Model Evaluation
            logger.info("Running Model Evaluation...")
            evaluator = ModelEvaluator(model=trained_model, int_to_label=self.int_to_label)
            accuracy = evaluator.evaluate(X_test_transformed, y_test)
            logger.info(f"Model Accuracy: {accuracy}")
            mlflow.log_metric("accuracy", accuracy)
            logger.info("Model Evaluation Complete.")

            # Log model
            mlflow.pytorch.log_model(trained_model, "model")

            logger.info("NLP Training Pipeline Complete.")

if __name__ == '__main__':
    from src.utils.common import get_nlp_config
    
    config = get_nlp_config()
    pipeline = NLPTrainingPipeline(config=config)
    pipeline.run()