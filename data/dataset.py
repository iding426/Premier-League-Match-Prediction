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
        Extract features for a team from a specific match.
        These are per-match stats that become part of the team's history.
        Safe to use because we only look at matches BEFORE current_idx in _get_team_history_block.
        """
        row = self.df.iloc[match_idx]

        # Basic match outcome features
        is_home = 1 if row["home_team"] == team_name else 0
        goals_for = row["home_goals"] if is_home else row["away_goals"]
        goals_against = row["away_goals"] if is_home else row["home_goals"]
        goal_diff = goals_for - goals_against

        # xG Features
        xG = row.get("h_xg", 0) if is_home else row.get("a_xg", 0)
        xGA = row.get("a_xg", 0) if is_home else row.get("h_xg", 0)
        
        # Shot features
        shots = row.get("h_shot", 0) if is_home else row.get("a_shot", 0)
        shots_against = row.get("a_shot", 0) if is_home else row.get("h_shot", 0)
        shots_on_target = row.get("h_shotOnTarget", 0) if is_home else row.get("a_shotOnTarget", 0)
        shots_on_target_against = row.get("a_shotOnTarget", 0) if is_home else row.get("h_shotOnTarget", 0)
        
        # Deep passes
        deep = row.get("h_deep", 0) if is_home else row.get("a_deep", 0)
        deep_against = row.get("a_deep", 0) if is_home else row.get("h_deep", 0)
        
        # PPDA (Passes Per Defensive Action)
        ppda = row.get("h_ppda", 0) if is_home else row.get("a_ppda", 0)
        ppda_against = row.get("a_ppda", 0) if is_home else row.get("h_ppda", 0)

        # Feature Vector - now using 18 features instead of 10
        vec = np.zeros(self.team_feature_dim, dtype=np.float32)
        vec[0] = is_home
        vec[1] = goals_for
        vec[2] = goals_against
        vec[3] = goal_diff
        vec[4] = xG
        vec[5] = xGA
        vec[6] = shots
        vec[7] = shots_against
        vec[8] = shots_on_target
        vec[9] = shots_on_target_against
        vec[10] = deep
        vec[11] = deep_against
        vec[12] = ppda
        vec[13] = ppda_against

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

        # Match context block with team records computed from history
        M_vec = self._match_block(row)  # (base features)
        # Add home team records from matches before this one
        h_wins, h_draws, h_losses = self._get_team_record_before_match(A, idx)
        
        M_block = np.zeros((1, self.team_feature_dim), dtype=np.float32)  # (1, 32)
        M_block[0, :len(M_vec)] = M_vec  # Fill base features
        # Add team records at the end of available match_feature_dim
        if len(M_vec) + 3 <= self.team_feature_dim:
            M_block[0, len(M_vec)] = h_wins
            M_block[0, len(M_vec) + 1] = h_draws
            M_block[0, len(M_vec) + 2] = h_losses

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

    def _get_team_record_before_match(self, team_name, current_idx):
        """
        Compute team's W-D-L record from matches BEFORE current_idx.
        Returns (wins, draws, losses) tuple.
        No data leakage - only uses historical matches.
        """
        past_matches = [i for i in self.team_history[team_name] if i < current_idx]
        
        wins = 0
        draws = 0
        losses = 0
        
        for match_idx in past_matches:
            row = self.df.iloc[match_idx]
            is_home = row["home_team"] == team_name
            
            if is_home:
                if row["home_goals"] > row["away_goals"]:
                    wins += 1
                elif row["home_goals"] == row["away_goals"]:
                    draws += 1
                else:
                    losses += 1
            else:
                if row["away_goals"] > row["home_goals"]:
                    wins += 1
                elif row["away_goals"] == row["home_goals"]:
                    draws += 1
                else:
                    losses += 1
        
        return wins, draws, losses

    def _team_features(self, team_name, match_idx):
        """
        Extract features for a team from a specific match.
        These are per-match stats that become part of the team's history.
        Safe to use because we only look at matches BEFORE current_idx in history.
        """
        row = self.df.iloc[match_idx]

        is_home = 1 if row["home_team"] == team_name else 0
        goals_for = row["home_goals"] if is_home else row["away_goals"]
        goals_against = row["away_goals"] if is_home else row["home_goals"]
        goal_diff = goals_for - goals_against
        
        # xG Features
        xG = row.get("h_xg", 0) if is_home else row.get("a_xg", 0)
        xGA = row.get("a_xg", 0) if is_home else row.get("h_xg", 0)
        
        # Shot features
        shots = row.get("h_shot", 0) if is_home else row.get("a_shot", 0)
        shots_against = row.get("a_shot", 0) if is_home else row.get("h_shot", 0)
        shots_on_target = row.get("h_shotOnTarget", 0) if is_home else row.get("a_shotOnTarget", 0)
        shots_on_target_against = row.get("a_shotOnTarget", 0) if is_home else row.get("h_shotOnTarget", 0)
        
        # Deep passes
        deep = row.get("h_deep", 0) if is_home else row.get("a_deep", 0)
        deep_against = row.get("a_deep", 0) if is_home else row.get("h_deep", 0)
        
        # PPDA
        ppda = row.get("h_ppda", 0) if is_home else row.get("a_ppda", 0)
        ppda_against = row.get("a_ppda", 0) if is_home else row.get("h_ppda", 0)

        vec = np.zeros(self.team_feature_dim, dtype=np.float32)
        vec[0] = is_home
        vec[1] = goals_for
        vec[2] = goals_against
        vec[3] = goal_diff
        vec[4] = xG
        vec[5] = xGA
        vec[6] = shots
        vec[7] = shots_against
        vec[8] = shots_on_target
        vec[9] = shots_on_target_against
        vec[10] = deep
        vec[11] = deep_against
        vec[12] = ppda
        vec[13] = ppda_against

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
        
        # Add home team records computed from history
        h_wins, h_draws, h_losses = self._get_team_record_before_match(A, idx)
        M_vec_with_records = np.concatenate([M_vec, [h_wins, h_draws, h_losses]])

        # Concatenate: [Home_flat, Away_flat, Match_vector_with_records]
        x = np.concatenate([A_flat, B_flat, M_vec_with_records], axis=0)
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
