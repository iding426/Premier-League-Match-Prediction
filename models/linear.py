import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_space = 35) -> None:
        super(MLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_space, input_space * 2),
            nn.ReLU(),
            nn.Linear(input_space * 2 ,256),
            nn.ReLU(),
            nn.Linear(256, 16),
            nn.ReLU,
            nn.Linear(16, 4)
        )

    def forward(self, team_1: np.ndarray, team_2: np.ndarray, info: np.ndarray):
        X = np.concat((info, team_1, team_2), axis=1)

        X = torch.tensor(X, dtype=torch.float32)
        out = self.net(X)

        # Softmax the first 3 outputs for win/draw/loss probabilities
        out_probs = nn.Softmax(dim=1)(out[:, :3])
        # The last output is the predicted score difference
        out_score_diff = out[:, 3].unsqueeze(1)

        return out_probs, out_score_diff