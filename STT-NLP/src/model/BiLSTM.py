import torch.nn as nn
import torch

class BiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # out shape: (batch, seq_len, hidden_size*2)
        out = self.fc(out)
        # out shape: (batch, seq_len, num_classes)
        return out
