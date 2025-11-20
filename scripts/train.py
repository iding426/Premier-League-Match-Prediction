"""
Check Datasets
"""
import sys
import pathlib
import pandas as pd

# Add project root to sys.path so `from data.dataset import ...` works when
# running `python scripts/train.py` from the repository root.
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from data.dataset import LinearDataset, TransformerDataset

def main():
    path = "data/understat_match_1524.csv"
    print("Loading:", path)
    df = pd.read_csv(path, parse_dates=["date"]) 
    print("rows:", len(df))

    team_history_len = 5
    team_feature_dim = 8
    match_feature_dim = 8  # venue + weather features
    fetch_weather = False  # Set to True to fetch real weather (slow)

    print("Linear Dataset")
    ds_lin = LinearDataset(
        df,
        team_history_len=team_history_len,
        team_feature_dim=team_feature_dim,
        match_feature_dim=match_feature_dim,
        fetch_weather=fetch_weather,
    )

    print("Transformer Dataset")
    ds_trans = TransformerDataset(
        df,
        team_history_len=team_history_len,
        team_feature_dim=team_feature_dim,
        match_feature_dim=match_feature_dim,
        fetch_weather=fetch_weather,
    )

    print("LinearDataset length:", len(ds_lin))
    print("TransformerDataset length:", len(ds_trans))

    sample_indices = [0, 1, team_history_len - 1, team_history_len, 10]
    for idx in sample_indices:
        if idx >= len(df):
            break
        print(f"\n--- idx={idx} ---")
        try:
            x, y = ds_lin[idx]
            print("Linear x shape:", tuple(x.shape), "y:", y)
        except Exception as e:
            print("LinearDataset error at idx", idx, e)

        try:
            x_t, y_t = ds_trans[idx]
            print("Transformer x shape:", tuple(x_t.shape), "y:", y_t)

            # quick check: team blocks and match token
            H = team_history_len
            F = team_feature_dim
            A_block = x_t[:H]
            B_block = x_t[H: H * 2]
            M_token = x_t[H * 2]  # Match context token
            print(" A_block sum:", float(A_block.sum()), "B_block sum:", float(B_block.sum()))
            print(" M_token (venue/weather):", M_token[:match_feature_dim].numpy())
        except Exception as e:
            print("TransformerDataset error at idx", idx, e)

        # Extra verbose diagnostics: show which past matches were used for each team
        try:
            row = ds_trans.df.iloc[idx]
            A = row["home_team"]
            B = row["away_team"]
            def _show_history(team_name):
                all_matches = ds_trans.team_history.get(team_name, [])
                prior = [i for i in all_matches if i < idx]
                used = prior[-team_history_len:]
                print(f"  Team {team_name} total past matches: {len(all_matches)}, prior before idx: {len(prior)}, used (last {team_history_len}): {used}")
                for mi in used:
                    r = ds_trans.df.iloc[mi]
                    print(f"    idx={mi} date={r.get('date')} {r.get('home_team')} {r.get('home_goals')}-{r.get('away_goals')} vs {r.get('away_team')}")

            _show_history(A)
            _show_history(B)
        except Exception as e:
            print("Diagnostics error:", e)

    print("\n Done")


if __name__ == "__main__":
    main()
