import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

try:
    from .venue_info import get_venue_info, get_venue_type_encoding, should_fetch_weather
    from .weather_fetcher import get_weather_features
except ImportError:
    from venue_info import get_venue_info, get_venue_type_encoding, should_fetch_weather
    from weather_fetcher import get_weather_features

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
    
    Returns:
        x: input features
        y_result: match result (0=home win, 1=draw, 2=away win)
        y_goal_diff: goal difference (home_goals - away_goals)
    """

    def __init__(
        self,
        df,
        team_history_len=20,
        team_feature_dim=32,
        match_feature_dim=8,
        col_map=None,            # optional explicit column mapping
        fetch_weather=False,     # whether to fetch weather data (slow if no cache)
        weather_cache_path="data/weather_cache.csv",  # path to pre-fetched weather cache
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
        self.fetch_weather = fetch_weather

        # Load weather cache if it exists
        self.weather_cache = None
        if fetch_weather and os.path.exists(weather_cache_path):
            try:
                self.weather_cache = pd.read_csv(weather_cache_path)
                self.weather_cache['date'] = pd.to_datetime(self.weather_cache['date'])
                print(f"✅ Loaded weather cache from {weather_cache_path} ({len(self.weather_cache)} records)")
            except Exception as e:
                print(f"⚠️  Failed to load weather cache: {e}")
                print(f"   Will fall back to API calls if fetch_weather=True")
        elif fetch_weather:
            print(f"⚠️  Weather cache not found at {weather_cache_path}")
            print(f"   Will use slow API calls. Consider running fetch_weather.ipynb first!")

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

    def _get_weather_from_cache(self, home_team, match_date):
        """
        Lookup weather data from cache.
        Returns dict with weather features or None if not found.
        """
        if self.weather_cache is None:
            return None
        
        # Find matching record
        match = self.weather_cache[
            (self.weather_cache['home_team'] == home_team) &
            (self.weather_cache['date'] == match_date)
        ]
        
        if len(match) == 0:
            return None
        
        # Convert to dict
        record = match.iloc[0]
        return {
            'temperature_norm': record['temperature_norm'],
            'precipitation': record['precipitation'],
            'wind_speed_norm': record['wind_speed_norm'],
            'cloud_cover_norm': record['cloud_cover_norm'],
            'humidity_norm': record['humidity_norm'],
            'is_dome': record['is_dome'],
        }

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
        xG = row["h_xg"] if is_home else row["a_xg"]
        xGA = row["a_xg"] if is_home else row["h_xg"]
        goal_diff = goals_for - goals_against

        # Basic Feature Vector
        vec = np.zeros(self.team_feature_dim, dtype=np.float32)
        vec[0] = is_home
        vec[1] = goals_for
        vec[2] = goals_against
        vec[3] = goal_diff
        
        # Advanced Features
        vec[4] = xG
        vec[5] = xGA

        return vec

    def _match_block(self, row):
        """
        Returns (match_feature_dim,) vector with pre-match contextual features:
        - Venue type (open/dome/retractable)
        - Weather conditions (temperature, precipitation, wind, etc.) if fetch_weather=True
        
        All features are known BEFORE the match starts (no data leakage).
        """
        vec = np.zeros(self.match_feature_dim, dtype=np.float32)
        
        # Get home team venue info
        home_team = row.get("home_team")
        venue_info = get_venue_info(home_team)
        
        idx = 0
        
        if venue_info:
            # Venue type encoding: 0=open, 1=retractable, 2=dome
            venue_type = venue_info["venue_type"]
            if idx < self.match_feature_dim:
                vec[idx] = get_venue_type_encoding(venue_type)
                idx += 1
            
            # Weather features (if enabled and venue needs weather)
            if self.fetch_weather and should_fetch_weather(venue_type):
                match_date = row.get("date")
                if match_date and idx < self.match_feature_dim:
                    # Try cache first
                    weather = self._get_weather_from_cache(home_team, match_date)
                    
                    # Fall back to API if cache miss
                    if weather is None:
                        weather = get_weather_features(
                            venue_info["lat"],
                            venue_info["lon"],
                            match_date,
                            venue_type=venue_type,
                        )
                    
                    # Add weather features in order
                    weather_keys = [
                        "temperature_norm",
                        "precipitation",
                        "wind_speed_norm",
                        "cloud_cover_norm",
                        "humidity_norm",
                        "is_dome",
                    ]
                    
                    for key in weather_keys:
                        if idx >= self.match_feature_dim:
                            break
                        vec[idx] = weather.get(key, 0.0)
                        idx += 1
            else:
                # No weather fetch - just mark if dome
                if idx < self.match_feature_dim:
                    vec[idx] = 1.0 if venue_type == "dome" else 0.0
                    idx += 1
        
        return vec

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        A = row["home_team"]
        B = row["away_team"]

        # Team A history block
        A_block = self._get_team_history_block(A, idx)  # (20, 32)

        # Team B history block
        B_block = self._get_team_history_block(B, idx)  # (20, 32)

        # Match context block (pad to match team_feature_dim)
        M_vec = self._match_block(row)  # (8,)
        M_block = np.zeros((1, self.team_feature_dim), dtype=np.float32)  # (1, 32)
        M_block[0, :self.match_feature_dim] = M_vec  # Fill first 8 dims

        # Concatenate into sequence: [A_block tokens][B_block tokens][Match token]
        x = np.concatenate([A_block, B_block, M_block], axis=0)  # (41, 32)
        x = torch.tensor(x, dtype=torch.float32)

        # Match result (classification): 0=home win, 1=draw, 2=away win
        if row["home_goals"] > row["away_goals"]:
            y_result = 0
        elif row["home_goals"] < row["away_goals"]:
            y_result = 2
        else:
            y_result = 1
        y_result = torch.tensor(y_result, dtype=torch.long)

        # Goal difference (regression)
        y_goal_diff = torch.tensor([row["goal_diff"]], dtype=torch.float32)

        return x, y_result, y_goal_diff

    def __len__(self):
        return len(self.df)

class LinearDataset(Dataset):
    """
    Dataset that returns a single flattened feature vector per match suitable for an MLP.
    
    Returns:
        x: input features
        y_result: match result (0=home win, 1=draw, 2=away win)
        y_goal_diff: goal difference (home_goals - away_goals)
    """

    def __init__(
        self,
        df,
        team_history_len=20,
        team_feature_dim=32,
        match_feature_dim=8,
        col_map=None,
        fetch_weather=False,
        weather_cache_path="data/weather_cache.csv",
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
        self.fetch_weather = fetch_weather

        # Load weather cache if it exists
        self.weather_cache = None
        if fetch_weather and os.path.exists(weather_cache_path):
            try:
                self.weather_cache = pd.read_csv(weather_cache_path)
                self.weather_cache['date'] = pd.to_datetime(self.weather_cache['date'])
                print(f"✅ Loaded weather cache from {weather_cache_path} ({len(self.weather_cache)} records)")
            except Exception as e:
                print(f"⚠️  Failed to load weather cache: {e}")
                print(f"   Will fall back to API calls if fetch_weather=True")
        elif fetch_weather:
            print(f"⚠️  Weather cache not found at {weather_cache_path}")
            print(f"   Will use slow API calls. Consider running fetch_weather.ipynb first!")

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

    def _get_weather_from_cache(self, home_team, match_date):
        """
        Lookup weather data from cache.
        Returns dict with weather features or None if not found.
        """
        if self.weather_cache is None:
            return None
        
        # Find matching record
        match = self.weather_cache[
            (self.weather_cache['home_team'] == home_team) &
            (self.weather_cache['date'] == match_date)
        ]
        
        if len(match) == 0:
            return None
        
        # Convert to dict
        record = match.iloc[0]
        return {
            'temperature_norm': record['temperature_norm'],
            'precipitation': record['precipitation'],
            'wind_speed_norm': record['wind_speed_norm'],
            'cloud_cover_norm': record['cloud_cover_norm'],
            'humidity_norm': record['humidity_norm'],
            'is_dome': record['is_dome'],
        }

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
        """
        Returns (match_feature_dim,) vector with pre-match contextual features.
        Reuses the same logic as TransformerDataset._match_block().
        """
        vec = np.zeros(self.match_feature_dim, dtype=np.float32)
        
        home_team = row.get("home_team")
        venue_info = get_venue_info(home_team)
        
        idx = 0
        
        if venue_info:
            venue_type = venue_info["venue_type"]
            if idx < self.match_feature_dim:
                vec[idx] = get_venue_type_encoding(venue_type)
                idx += 1
            
            if self.fetch_weather and should_fetch_weather(venue_type):
                match_date = row.get("date")
                if match_date and idx < self.match_feature_dim:
                    # Try cache first
                    weather = self._get_weather_from_cache(home_team, match_date)
                    
                    # Fall back to API if cache miss
                    if weather is None:
                        weather = get_weather_features(
                            venue_info["lat"],
                            venue_info["lon"],
                            match_date,
                            venue_type=venue_type,
                        )
                    
                    weather_keys = [
                        "temperature_norm",
                        "precipitation",
                        "wind_speed_norm",
                        "cloud_cover_norm",
                        "humidity_norm",
                        "is_dome",
                    ]
                    
                    for key in weather_keys:
                        if idx >= self.match_feature_dim:
                            break
                        vec[idx] = weather.get(key, 0.0)
                        idx += 1
            else:
                if idx < self.match_feature_dim:
                    vec[idx] = 1.0 if venue_type == "dome" else 0.0
                    idx += 1
        
        return vec

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        A = row["home_team"]
        B = row["away_team"]

        A_flat = self._get_team_history_flat(A, idx)
        B_flat = self._get_team_history_flat(B, idx)

        # Match context vector (venue/weather info)
        M_vec = self._match_vector(row)

        # Concatenate: [Home_flat, Away_flat, Match_vector]
        x = np.concatenate([A_flat, B_flat, M_vec], axis=0)
        x = torch.tensor(x, dtype=torch.float32)

        # Match result (classification): 0=home win, 1=draw, 2=away win
        if row["home_goals"] > row["away_goals"]:
            y_result = 0
        elif row["home_goals"] < row["away_goals"]:
            y_result = 2
        else:
            y_result = 1
        y_result = torch.tensor(y_result, dtype=torch.long)

        # Goal difference (regression)
        y_goal_diff = torch.tensor([row["goal_diff"]], dtype=torch.float32)

        return x, y_result, y_goal_diff

    def __len__(self):
        return len(self.df)
