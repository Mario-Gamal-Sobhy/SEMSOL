import torch.nn as nn
import torch

class CNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, n_cnn_layers: int):
        super(CNN, self).__init__()

        layers = []
        for _ in range(n_cnn_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)
