import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

def _ensure_columns(df, col_map=None):
    """
    Normalize Column Names
    """
    df = df.copy()
    if col_map:
        # user-supplied explicit mapping
        df = df.rename(columns=col_map)

    synonyms = {
        "home_team": ["home_team", "team_h", "team_home", "team_h_name"],
        "away_team": ["away_team", "team_a", "team_away", "team_a_name"],
        "home_goals": ["home_goals", "h_goals", "hg", "h_g"],
        "away_goals": ["away_goals", "a_goals", "ag", "a_g"],
        "date": ["date", "match_date", "kickoff"]
    }

    rename_map = {}
    for canon, alts in synonyms.items():
        if canon in df.columns:
            continue
        for alt in alts:
            if alt in df.columns:
                rename_map[alt] = canon
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure date is datetime when present
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception:
            # leave as-is; sorting may still work for strings
            pass

    return df


class TransformerDataset(Dataset):
    """
    Dataset class for Transformer match prediction.
    
    Expects a dataframe with columns:
        date, home_team, away_team, home_goals, away_goals, ...
    """

    def __init__(
        self,
        df,
        team_history_len=20,
        team_feature_dim=32,
        match_feature_dim=32,
        target_mode="probs",   # or "goal_diff"
        col_map=None,            # optional explicit column mapping
    ):
        # normalize common column names (accept understat naming)
        self.df = _ensure_columns(df, col_map=col_map)
        # sort by date (if present)
        if "date" in self.df.columns:
            self.df = self.df.sort_values("date").reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)
        self.team_history_len = team_history_len
        self.team_feature_dim = team_feature_dim
        self.match_feature_dim = match_feature_dim
        self.target_mode = target_mode

        # build dictionary: team → list of past matches (indices)
        self.team_history = self._build_team_history()

        # preprocess (optional)
        self.df["goal_diff"] = self.df["home_goals"] - self.df["away_goals"]

    # Team History
    def _build_team_history(self):
        hist = {team: [] for team in pd.unique(self.df[["home_team", "away_team"]].values.ravel())}

        for idx, row in self.df.iterrows():
            hist[row["home_team"]].append(idx)
            hist[row["away_team"]].append(idx)

        return hist

    # History Feature Block
    def _get_team_history_block(self, team_name, current_idx):
        """
        Returns a (team_history_len, team_feature_dim) tensor.
        """

        # get matches *before* the current one
        matches = [i for i in self.team_history[team_name] if i < current_idx]
        matches = matches[-self.team_history_len:]  # keep most recent N

        block = np.zeros((self.team_history_len, self.team_feature_dim), dtype=np.float32)

        for i, match_idx in enumerate(matches):
            block[-len(matches) + i] = self._team_features(team_name, match_idx)

        return block

    # Team Features
    def _team_features(self, team_name, match_idx):
        """
        Returns a vector of size team_feature_dim.
        Fill with whatever you want:
            form, goals scored, goals conceded, rolling averages, rating, etc.
        """
        row = self.df.iloc[match_idx]

        # Starter Features
        is_home = 1 if row["home_team"] == team_name else 0
        goals_for = row["home_goals"] if is_home else row["away_goals"]
        goals_against = row["away_goals"] if is_home else row["home_goals"]
        goal_diff = goals_for - goals_against

        # Basic Feature Vector
        vec = np.zeros(self.team_feature_dim, dtype=np.float32)
        vec[0] = is_home
        vec[1] = goals_for
        vec[2] = goals_against
        vec[3] = goal_diff
        
        # Advanced Features

        return vec

    # Match Block
    def _match_block(self, row):
        """
        Returns (match_feature_dim,) vector with match-specific context:
            league, season, venue, rankings, etc.
        """
        vec = np.zeros(self.match_feature_dim, dtype=np.float32)

        candidates = [
            "h_xg",
            "a_xg",
            "h_shot",
            "a_shot",
            "h_shotOnTarget",
            "a_shotOnTarget",
            "h_deep",
            "a_deep",
        ]

        for i, name in enumerate(candidates):
            if i >= self.match_feature_dim:
                break
            # use .get so missing columns default to 0.0
            try:
                vec[i] = float(row.get(name, 0.0))
            except Exception:
                vec[i] = 0.0

        idx = min(len(candidates), self.match_feature_dim)
        if idx < self.match_feature_dim:
            vec[idx] = 1.0 if row.get("neutral_venue", False) else 0.0
        if idx + 1 < self.match_feature_dim:
            vec[idx + 1] = row.get("importance", 0.0)

        return vec

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        A = row["home_team"]
        B = row["away_team"]

        # Team A history block
        A_block = self._get_team_history_block(A, idx)  # (H, F)

        # Team B history block
        B_block = self._get_team_history_block(B, idx)  # (H, F)

        # Match info block (1 token)
        M_block = self._match_block(row).reshape(1, -1)  # (1, M)

        # Concatenate into sequence:
        # [A_block tokens][B_block tokens][Match token]
        x = np.concatenate([A_block, B_block, M_block], axis=0)
        x = torch.tensor(x, dtype=torch.float32)

        # Labels
        if self.target_mode == "probs":
            # 0 = home win, 1 = draw, 2 = away win
            if row["home_goals"] > row["away_goals"]:
                y = 0
            elif row["home_goals"] < row["away_goals"]:
                y = 2
            else:
                y = 1
            y = torch.tensor(y, dtype=torch.long)

        elif self.target_mode == "goal_diff":
            y = torch.tensor([row["goal_diff"]], dtype=torch.float32)

        return x, y

    def __len__(self):
        return len(self.df)

class LinearDataset(Dataset):
    """
    Dataset that returns a single flattened feature vector per match suitable for an MLP.
    """

    def __init__(
        self,
        df,
        team_history_len=20,
        team_feature_dim=32,
        match_feature_dim=32,
        target_mode="probs",
        col_map=None,
    ):
        # normalize column names (accept understat csv columns)
        self.df = _ensure_columns(df, col_map=col_map)
        if "date" in self.df.columns:
            self.df = self.df.sort_values("date").reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)
        self.team_history_len = team_history_len
        self.team_feature_dim = team_feature_dim
        self.match_feature_dim = match_feature_dim
        self.target_mode = target_mode

        # build dictionary: team → list of past matches (indices)
        self.team_history = self._build_team_history()

        # preprocess
        self.df["goal_diff"] = self.df["home_goals"] - self.df["away_goals"]

    def _build_team_history(self):
        hist = {team: [] for team in pd.unique(self.df[["home_team", "away_team"]].values.ravel())}

        for idx, row in self.df.iterrows():
            hist[row["home_team"]].append(idx)
            hist[row["away_team"]].append(idx)

        return hist

    def _team_features(self, team_name, match_idx):
        # Keep same basic team feature construction as TransformerDataset
        row = self.df.iloc[match_idx]

        is_home = 1 if row["home_team"] == team_name else 0
        goals_for = row["home_goals"] if is_home else row["away_goals"]
        goals_against = row["away_goals"] if is_home else row["home_goals"]
        goal_diff = goals_for - goals_against

        vec = np.zeros(self.team_feature_dim, dtype=np.float32)
        vec[0] = is_home
        vec[1] = goals_for
        vec[2] = goals_against
        vec[3] = goal_diff

        return vec

    def _get_team_history_flat(self, team_name, current_idx):
        """Return flattened history vector of shape (team_history_len * team_feature_dim,)"""
        matches = [i for i in self.team_history[team_name] if i < current_idx]
        matches = matches[-self.team_history_len:]

        flat = np.zeros(self.team_history_len * self.team_feature_dim, dtype=np.float32)

        # place most-recent matches at the end of the history slot (same convention as TransformerDataset)
        for i, match_idx in enumerate(matches):
            slot = (self.team_history_len - len(matches) + i) * self.team_feature_dim
            flat[slot: slot + self.team_feature_dim] = self._team_features(team_name, match_idx)

        return flat

    def _match_vector(self, row):
        vec = np.zeros(self.match_feature_dim, dtype=np.float32)
        vec[0] = 1.0 if row.get("neutral_venue", False) else 0.0
        vec[1] = row.get("importance", 0.0)
        return vec

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        A = row["home_team"]
        B = row["away_team"]

        A_flat = self._get_team_history_flat(A, idx)
        B_flat = self._get_team_history_flat(B, idx)
        M_vec = self._match_vector(row)

        x = np.concatenate([A_flat, B_flat, M_vec], axis=0)
        x = torch.tensor(x, dtype=torch.float32)

        # Labels
        if self.target_mode == "probs":
            if row["home_goals"] > row["away_goals"]:
                y = 0
            elif row["home_goals"] < row["away_goals"]:
                y = 2
            else:
                y = 1
            y = torch.tensor(y, dtype=torch.long)

        elif self.target_mode == "goal_diff":
            y = torch.tensor([row["goal_diff"]], dtype=torch.float32)

        return x, y

    def __len__(self):
        return len(self.df)
    