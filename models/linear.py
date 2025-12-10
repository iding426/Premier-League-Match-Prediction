import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_space=651) -> None:
        super(MLP, self).__init__()

        # Smaller, more appropriate architecture for 491 input features
        self.net = nn.Sequential(
            nn.Linear(input_space, 256),  # Reduced from 512
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased from 0.3
            nn.Linear(256, 128),  # Reduced from 256
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased from 0.3
            nn.Linear(128, 64),   # Reduced from 128
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased from 0.2
            nn.Linear(64, 4)
        )

    def forward(self, x):
        """
        Args:
            x: (B, 1288) - flattened input from LinearDataset
        Returns:
            probs: (B, 3) - win/draw/loss probabilities
            goal_diff: (B, 1) - predicted goal difference
        """
        out = self.net(x)

        # Softmax the first 3 outputs
        out_probs = torch.softmax(out[:, :3], dim=1)
        # The last output is the predicted score difference
        out_score_diff = out[:, 3].unsqueeze(1)

        return out_probs, out_score_diff
