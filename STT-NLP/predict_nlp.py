
import pandas as pd
from src.nlp.components.evaluation import ModelEvaluator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from src.utils.common import read_yaml
from pathlib import Path
from sklearn.model_selection import train_test_split

def main():
    # Load configuration
    config = read_yaml(Path('config.yaml'))
    params = read_yaml(Path('params.yaml'))
    nlp_config = params.nlp_model_trainer
    data_ingestion_config = config.data_ingestion

    # Load the trained model
    print("Loading model...")
    sentiment_model = ModelEvaluator.load_model(
        model_path=nlp_config.model_path,
        vocab_path=nlp_config.vocab_path,
        label_to_int=nlp_config.label_to_int,
        embedding_dim=nlp_config.embedding_dim,
        hidden_dim=nlp_config.hidden_dim,
        output_dim=nlp_config.output_dim,
        n_layers=nlp_config.n_layers,
        drop_prob=nlp_config.drop_prob
    )
    print("Model loaded.")

    # Load test data
    print("Loading test data...")
    df = pd.read_csv(nlp_config.data_path, encoding='latin-1')
    X = df['text']
    y = df['sentiment']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Test data loaded.")

    # Make predictions
    print("Making predictions...")
    y_pred = sentiment_model.predict(X_test, vocab_path=nlp_config.vocab_path)
    print("Predictions made.")

    # Evaluate the model
    print("Evaluating model...")
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=sentiment_model.model.label_to_int.keys()))

if __name__ == '__main__':
    main()
