import torch.nn as nn
import torch
from src.model.CNN import CNN
from src.model.BiLSTM import BiLSTM
from typing import Tuple

class SpeechToText(nn.Module):
    def __init__(self, n_cnn_layers: int, n_rnn_layers: int, rnn_dim: int, n_class: int, n_feats: int, cnn_out_channels: int, stride: int = 2, dropout: float = 0.1):
        super(SpeechToText, self).__init__()
        self.n_cnn_layers = n_cnn_layers
        self.stride = stride
        
        self.cnn = CNN(1, cnn_out_channels, kernel_size=3, stride=stride, padding=1, n_cnn_layers=n_cnn_layers)
        
        # Calculate the output size of the CNN along feature axis
        rnn_input_size = cnn_out_channels * (n_feats // (stride ** n_cnn_layers))
        
        self.rnn = BiLSTM(input_size=rnn_input_size, hidden_size=rnn_dim, num_layers=n_rnn_layers, num_classes=n_class)

    def forward(self, x: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.cnn(x)

        x = x.permute(0, 3, 1, 2).contiguous()
        sizes = x.size()
        x = x.view(sizes[0], sizes[1], -1)  # (batch, seq_len, features)

        output_lengths = input_lengths
        for _ in range(self.n_cnn_layers):
            output_lengths = torch.floor((output_lengths - 1) / self.stride) + 1
        output_lengths = output_lengths.int()

        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths.cpu(), batch_first=True)

        x = self.rnn(x)
        
        return x, output_lengths
