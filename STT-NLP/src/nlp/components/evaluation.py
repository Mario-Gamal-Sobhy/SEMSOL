
import torch
from sklearn.metrics import accuracy_score

from src.nlp.components.preprocessing import TextPreprocessor
from src.nlp.components.transformation import TextTransformer
from src.nlp.model.SentimentLSTM import SentimentLSTM

class ModelEvaluator:

    def __init__(self, model, int_to_label):
        self.model = model
        self.int_to_label = int_to_label

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model on the test set.
        """
        self.model.eval()
        
        # Convert labels to integers
        y_test_int = [self.model.label_to_int[label] for label in y_test]
        
        test_data = torch.from_numpy(X_test).long()
        
        with torch.no_grad():
            output = self.model(test_data)
            _, predicted_idx = torch.max(output, 1)
        
        accuracy = accuracy_score(y_test_int, predicted_idx.numpy())
        return accuracy

    def predict(self, sentences, vocab_path):
        """
        Predicts the sentiment of a list of sentences.
        """
        self.model.eval()
        
        preprocessor = TextPreprocessor()
        transformer = TextTransformer.load_vocab(vocab_path)
        
        processed_sentences = [preprocessor.preprocess_text(sentence) for sentence in sentences]
        tokenized_sentences = transformer.transform(processed_sentences)
        
        tensor_sentences = torch.from_numpy(tokenized_sentences).long()
        
        with torch.no_grad():
            output = self.model(tensor_sentences)
            _, predicted_idx = torch.max(output, 1)
        
        return [self.int_to_label[idx.item()] for idx in predicted_idx]

    @classmethod
    def load_model(cls, model_path, vocab_path, label_to_int, embedding_dim=128, hidden_dim=256, output_dim=3, n_layers=2, drop_prob=0.5):
        """
        Loads a trained model and vocabulary.
        """
        transformer = TextTransformer.load_vocab(vocab_path)
        vocab_size = len(transformer.vocab_to_int) + 1
        
        model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, drop_prob)
        model.load_state_dict(torch.load(model_path))
        
        int_to_label = {v: k for k, v in label_to_int.items()}
        
        # Attach label mappings to the model object for consistency
        model.label_to_int = label_to_int
        
        return cls(model, int_to_label=int_to_label)

