import torch
from torch import nn
import torch.nn.functional as F

class ChessActor(nn.Module):
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(n_obs + 1, n_actions)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.float()
        x = self.fc1(x)
        x = self.relu(x)

        return x