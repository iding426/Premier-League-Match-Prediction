import torch
import torch.nn as nn
import torch.nn.functional as F


# Base Block
def build_encoder_layer(model_dim, num_heads, dropout):
    return nn.TransformerEncoderLayer(
        d_model=model_dim,
        nhead=num_heads,
        dim_feedforward=model_dim * 4,
        dropout=dropout,
        batch_first=True,
        activation="gelu"
    )

class FPLMatchPredictor(nn.Module):
    """
    Structured transformer model:
    - Team1 block (seq of tokens)
    - Team2 block (seq of tokens)
    - Match info block (1 token)
    - Transformer encoders applied separately + globally
    """

    def __init__(
        self,
        input_dim_team=32,
        input_dim_match=8,
        model_dim=128,
        num_heads=4,
        depth_team=2,
        depth_global=3,
        dropout=0.1,
    ):
        super().__init__()

        # Project Team and Match info
        self.team_proj = nn.Linear(input_dim_team, model_dim)
        self.match_proj = nn.Linear(input_dim_match, model_dim)

        # Positional embeddings
        self.team_pos_emb = nn.Parameter(torch.randn(1, 20, model_dim))
        self.match_pos_emb = nn.Parameter(torch.randn(1, 1, model_dim))

        # Team level transformer
        team_encoder_layer = build_encoder_layer(model_dim, num_heads, dropout)
        self.team_encoder = nn.TransformerEncoder(team_encoder_layer, depth_team)

        # Global transformer
        global_encoder_layer = build_encoder_layer(model_dim, num_heads, dropout)
        self.global_encoder = nn.TransformerEncoder(global_encoder_layer, depth_global)

        # Prediction heads
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 3)
        )

        # GD Head
        self.goal_diff_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, 1)
        )

    def forward(self, x):
        """
        Inputs:
            x: (B, 41, 32) - concatenated sequence from dataset
               [20 team1 tokens | 20 team2 tokens | 1 match token]
        """
        # Split: [0:20] team1, [20:40] team2, [40:41] match
        team1_tokens = x[:, :20, :]      # (B, 20, 32)
        team2_tokens = x[:, 20:40, :]    # (B, 20, 32)
        match_token = x[:, 40, :]        # (B, 32) - only first 8 dims have data

        B, T1, _ = team1_tokens.shape
        _, T2, _ = team2_tokens.shape

        # Team projection + positional embedding
        t1 = self.team_proj(team1_tokens) + self.team_pos_emb[:, :T1, :]
        t2 = self.team_proj(team2_tokens) + self.team_pos_emb[:, :T2, :]

        # Team Transformer encodings
        t1_encoded = self.team_encoder(t1)
        t2_encoded = self.team_encoder(t2)

        # Match projection (use only first 8 dims)
        m = self.match_proj(match_token[:, :8])
        m = m.unsqueeze(1)  # (B, 1, model_dim)
        m = m + self.match_pos_emb[:, :1, :]

        # Concatenate all tokens
        all_tokens = torch.cat([t1_encoded, t2_encoded, m], dim=1)

        # Global transformer
        global_encoded = self.global_encoder(all_tokens)

        # Pool (mean over tokens)
        pooled = global_encoded.mean(dim=1)

        # Heads
        probs = F.softmax(self.classifier(pooled), dim=1)
        goal_diff = self.goal_diff_head(pooled)

        return probs, goal_diff
