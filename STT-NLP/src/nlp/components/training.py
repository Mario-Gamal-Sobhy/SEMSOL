
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

from src.nlp.model.SentimentLSTM import SentimentLSTM

class ModelTrainer:
    def __init__(self, model, label_to_int):
        self.model = model
        self.label_to_int = label_to_int

    def train(self, X_train, y_train, epochs=32, batch_size=32, learning_rate=0.001):
        """
        Trains the sentiment analysis model.
        """
        encoded_y = np.array([self.label_to_int[label] for label in y_train])

        train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(encoded_y))
        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()

        for epoch in range(epochs):
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                optimizer.zero_grad()
                output = self.model(inputs.long())
                loss = criterion(output, labels.long())
                loss.backward()
                optimizer.step()
        
        return self.model

    def save_model(self, model_path):
        """
        Saves the model state dict.
        """
        torch.save(self.model.state_dict(), model_path)

