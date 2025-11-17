
import torch
import torch.nn as nn
from src.nlp.model.SentimentLSTM import SentimentLSTM
from src.nlp.components.transformation import TextTransformer
from src.utils.common import read_yaml
from pathlib import Path
import numpy as np

class NLPPredictor:
    def __init__(self, model_path, vocab_path, max_len=128):
        self.transformer = TextTransformer()
        self.transformer = TextTransformer.load_vocab(vocab_path)
        
        params = read_yaml(Path('params.yaml'))
        nlp_params = params.nlp_model_trainer
        
        vocab_size = len(self.transformer.vocab_to_int) + 1
        
        self.model = SentimentLSTM(
            vocab_size=vocab_size,
            embedding_dim=nlp_params.embedding_dim,
            hidden_dim=nlp_params.hidden_dim,
            output_dim=nlp_params.output_dim,
            n_layers=nlp_params.n_layers,
            drop_prob=nlp_params.drop_prob
        )
        
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        
        self.max_len = max_len
        self.int_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}

    def predict(self, text):
        # Preprocess the text
        # This is a simplified preprocessing, for a real application, you should use the same preprocessing as in the training pipeline
        text = text.lower()
        text = ''.join([c for c in text if c not in '!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'])
        
        # Transform the text
        transformed_text = self.transformer.transform([text], max_len=self.max_len)
        
        # Convert to tensor
        tensor = torch.from_numpy(transformed_text).long()
        
        # Predict
        with torch.no_grad():
            output = self.model(tensor)
            _, predicted = torch.max(output, 1)
            
        return self.int_to_label[predicted.item()]

def get_nlp_predictor():
    params = read_yaml(Path('params.yaml'))
    nlp_params = params.nlp_model_trainer
    return NLPPredictor(
        model_path=nlp_params.model_path,
        vocab_path=nlp_params.vocab_path,
        max_len=nlp_params.max_len
    )

if __name__ == '__main__':
    predictor = get_nlp_predictor()
    text = "This is a great movie. I really enjoyed it."
    sentiment = predictor.predict(text)
    print(f"The sentiment of the text is: {sentiment}")
