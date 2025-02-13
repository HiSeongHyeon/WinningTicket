import torch
import torch.nn as nn

class WDCNN(nn.Module):
    def __init__(self, first_kernel: int = 64, n_classes: int = 4):
        super(WDCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, first_kernel, stride=16, padding=24),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(16, 32, 3, stride=1, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(32, 64, 3, stride=1, padding="same"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(64, 64, 3, stride=1, padding="same"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),

            nn.Conv1d(64, 64, 3, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
        )

        with torch.no_grad():
            dummy = torch.rand(1, 1, 4096)
            dummy = self.conv_layers(dummy)
            dummy = torch.flatten(dummy, 1)
            lin_input = dummy.shape[1]

        self.linear_layers = nn.Sequential(
            nn.Linear(lin_input, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Linear(8, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.linear_layers(x)
        return x
