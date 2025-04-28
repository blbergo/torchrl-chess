from torch import nn

class ChessCritic(nn.Module):
    def __init__(self, n_obs):
        super(ChessCritic, self).__init__()
        self.fc1 = nn.Linear(n_obs + 1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float()
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        
        return x