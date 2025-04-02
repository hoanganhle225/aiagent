import torch
import torch.nn as nn

class BehaviorCloningModel(nn.Module):
    def __init__(self, input_dim=6, output_dim=5):
        super(BehaviorCloningModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )


    def forward(self, x):
        return self.net(x)
