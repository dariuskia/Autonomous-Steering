import torch
import torch.nn as nn

class SteeringModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, 2), # 3 channels for RGB
            nn.ELU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.ELU(),
        )
        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 1 * 18, 100),
            nn.ELU(),
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Linear(10, 1)
        )
    
    def forward(self, x):
        conv = self.conv_layers(x)
        flattened = self.flatten(conv)
        out = self.linear_layers(flattened)
        return out