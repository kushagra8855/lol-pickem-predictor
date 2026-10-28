import os
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Core pre-game metrics to aggregate vio EWMA
ROLLING_METRICS = [
    "result",          # Rolling win rate
    "gamelength_min",   # Average pace (minutes)
    "ckpm",             # Combined kills per minute (bloodiness/volatility)
    "kda_ratio",        # Combat efficiency
    "dpm",              # team damage per minute
    "earned gpm",       # post-laning gold generation speed
    "vspm",             # vision control per minute
    "dragon_share",     # % of total dragons secured
    "baron_share",      # % of total barons secured
    "tower_diff",       # Net turret differential
    "firstblood",       # Early skirmish priority
    "firsttower",       # Early map rotation priority
    "firstdragon",      # Early bot-side priority
    "golddiffat15",     # Laning phase gold advantage
    "xpdiffat15",       # Laning phase XP advantage
    "csdiffat15",       # Laning phase CS advantage
]

def engineer_per_game_rotation(team_df: pd.DataFrame) -> pd.DataFrame:
    """Computes normalized per-game ratios before rolling window aggregation."""
    df = team_df.sort_values(['date','gameid']).reset_index(drop=True).copy()

    # 1. Game length in minutes
    df["gamelength_min"] = df["gamelength"] / 60.0

    # 2. Smoothed KDA ratio
    df["kda_ratio"] = (df["kills"] + 1.0) / (df["deaths"] + 1.0)

    # 3. Objective control shares (bounded [0, 1], 0.5 when 0 total objectives spawned/taken)
    total_dragons = df["dragons"] + df["opp_dragons"]
    df["dragon_share"] = np.where(total_dragons > 0, df["dragons"] / total_dragons, 0.5)

    total_barons = df["barons"] + df["opp_barons"]
    df["baron_share"] = np.where(total_barons > 0, df["barons"] / total_barons, 0.5)

    # 4. Net structural advantage
    df["tower_diff"] = df["towers"] - df["opp_towers"]

    return df

def compute_rolling_champ_diversity(df: pd.DataFrame, window: int=15) -> tuple[pd.Series, dict[str, int]]:
    """
    Calculates how many unique champions a team has drafted over its previous `window` maps.
    Returns both the leak-free historical series (shifted by 1) and the latest unshifted count per team.
    """

    pick_cols = [c for c in ["pick1", "pick2", "pick3", "pick4", "pick5"] if c in df.columns]
    if len(pick_cols) < 5:
        return pd.Series(25.0, index=df.index), {}

    pre_match_diversity = np.full(len(df), np.nan)
    latest_diversity = {}

    # Process each team's chronological pick history using a sliding deque of length `window`
    for team, group in df.groupby("teamname", sort=False):
        recent_drafts = deque(maxlen=window)
        picks_matrix = group[pick_cols].values
        indices = group.index.values

        for i, row_idx in enumerate(indices):
            # Record PRE-MATCH unique champion count (strictly prior games)
            if len(recent_drafts) >= 3:
                unique_champs = set().union(*recent_drafts)
                # Normalize to a 15-game equivalent scale if team has between 3 and 14 games
                scale_factor = window / len(recent_drafts)
                pre_match_diversity[row_idx] = len(unique_champs) * scale_factor

            # Push current game's 5 picks into the sliding window
            valid_picks = {p for p in picks_matrix[i] if pd.notna(p) and str(p) != "nan"}
            recent_drafts.append(valid_picks)

        # Save latest unshifted diversity for future tournament simulations
        if len(recent_drafts) > 0:
            unique_champs = set().union(*recent_drafts)
            scale_factor = window / len(recent_drafts)
            latest_diversity[team] = round(len(unique_champs) * scale_factor, 2)

    return pd.Series(pre_match_diversity, index=df.index), latest_diversity

def compute_team_rolling_features(team_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes leak-free pre-match EWMA features (span=15 and span=5) per team,
    plus an unshifted latest profile snapshot for step 5 tournament simulations.
    """

    df = engineer_per_game_rotation(team_df)
    available_metrics = [m for m in ROLLING_METRICS if m in df.columns]

    # Sort strictly chronologically per team
    df = df.sort_values(["teamname", "date", "gameid"]).reset_index(drop=True)
    grouped = df.groupby("teamname", sort=False)

    pre_match_cols = {}
    latest_unshifted_cols = {}

    for col in available_metrics:
        clean_name = col.replace(" ", "_")

        # 1. Leak-free PRE-MATCH EWMA (shifted by 1 game within each team)
        shifted_series = grouped[col].shift(1)
        ewm15_pre = (
            shifted_series.groupby(df["teamname"], sort=False)
            .ewm(span=15, min_periods=3, ignore_na=True)
            .mean()
            .reset_index(level=0, drop=True)
        )
        ewm5_pre = (
            shifted_series.groupby(df["teamname"], sort=False)
            .ewm(span=5, min_periods=2, ignore_na=True)
            .mean()
            .reset_index(level=0, drop=True)
        )

        pre_match_cols[f"ewm15_{clean_name}"] = ewm15_pre
        pre_match_cols[f"ewm5_{clean_name}"] = ewm5_pre

        # 2. Unshifted POST-MATCH EWMA (used only to extract each team's latest state for Step 5)
        ewm15_latest = (
            grouped[col]
            .ewm(span=15, min_periods=3, ignore_na=True)
            .mean()
            .reset_index(level=0, drop=True)
        )
        ewm5_latest = (
            grouped[col]
            .ewm(span=5, min_periods=2, ignore_na=True)
            .mean()
            .reset_index(level=0, drop=True)
        )
        latest_unshifted_cols[f"ewm15_{clean_name}"] = ewm15_latest
        latest_unshifted_cols[f"ewm5_{clean_name}"] = ewm5_latest

    # Attach rolling features to dataframe
    features_df = pd.concat([df[["gameid", "date", "teamname", "side", "league"]], pd.DataFrame(pre_match_cols)], axis=1)

    # Compute Form Momentum (short-term 5-game winrate minus baseline 15-game winrate)
    features_df["form_momentum"] = features_df["ewm5_result"] - features_df["ewm15_result"]

    # Compute Champion Pool Diversity over last 15 games
    champ_div_series, latest_champ_div = compute_rolling_champ_diversity(df, window=15)
    features_df["champ_pool_15"] = champ_div_series

    # Flag whether the team has valid @15 timeline history (helps tree models handle LPL partial data)
    features_df["has_timeline_data"] = features_df["ewm15_golddiffat15"].notna().astype(int)

    # Build the latest team profiles snapshot table (last seen row per team)
    latest_df = pd.concat([df[["teamname", "date", "league"]], pd.DataFrame(latest_unshifted_cols)], axis=1)
    latest_df = latest_df.sort_values("date").groupby("teamname", as_index=False).last()
    latest_df["form_momentum"] = latest_df["ewm5_result"] - latest_df["ewm15_result"]
    latest_df["champ_pool_15"] = latest_df["teamname"].map(latest_champ_div).fillna(25.0)
    latest_df["has_timeline_data"] = latest_df["ewm15_golddiffat15"].notna().astype(int)

    return features_df, latest_df

def merge_matchup_features(matchups_elo_df: pd.DataFrame, team_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Joins Blue and Red pre-match rolling features onto the elo matchup table
    and calculates head-to-head differential and style-clash features.
    """
    feature_cols = [
        c for c in team_features_df.columns
        if c.startswith("ewm15_") or c.startswith("ewm5_") or c in ["form_momentum", "champ_pool_15", "has_timeline_data"]
    ]

    blue_feats = team_features_df[team_features_df["side"] == "Blue"][["gameid"] + feature_cols].copy()
    red_feats = team_features_df[team_features_df["side"] == "Red"][["gameid"] + feature_cols].copy()

    blue_feats = blue_feats.rename(columns={c: f"blue_{c}" for c in feature_cols})
    red_feats = red_feats.rename(columns={c: f"red_{c}" for c in feature_cols})

    merged = (
        matchups_elo_df
        .merge(blue_feats, on="gameid", how="left")
        .merge(red_feats, on="gameid", how="left")
    )

    # 1. Compute Direct Matchup Differentials (Blue minus Red)
    diff_dict = {}
    for c in feature_cols:
        if c != "has_timeline_data":
            diff_dict[f"diff_{c}"] = merged[f"blue_{c}"] - merged[f"red_{c}"]

    merged = pd.concat([merged, pd.DataFrame(diff_dict)], axis=1)

    # 2. Compute Stylistic Interaction Features (Crucial for Pick'ems & Crystal Ball)
    # Combined pace/bloodiness of both teams
    merged["expected_match_ckpm"] = 0.5 * (merged["blue_ewm15_ckpm"] + merged["red_ewm15_ckpm"])
    #style clash: high when one team wants chaos and the other wants slow macro
    merged["pace_mismatch_ckpm"] = (merged["blue_ewm15_ckpm"] - merged["red_ewm15_ckpm"]).abs()
    # Expected map duration in minutes
    merged["expected_gamelength_min"] = 0.5 * (
        merged["blue_ewm15_gamelength_min"] + merged["red_ewm15_gamelength_min"]
    )

    return merged.sort_values("date").reset_index(drop=True)

def run_features_pipeline(processed_dir: str = "data/processed"):
    team_path = os.path.join(processed_dir, "team_games.parquet")
    elo_path = os.path.join(processed_dir, "matchups_with_elo.parquet")
    ratings_path = os.path.join(processed_dir, "current_team_ratings.parquet")

    if not (os.path.exists(team_path) and os.path.exists(elo_path)):
        raise FileNotFoundError("Missing input Parquet files. RUN src/ingest.py and src/elo.py first.")

    print("Loading team_games.parquet and matchups_with_elo.parquet...")
    team_df = pd.read_parquet(team_path)
    matchups_elo_df = pd.read_parquet(elo_path)

    print("Computing leak-free rolling EWMA features and draft diversity...")
    team_features_df, latest_profiles_df = compute_team_rolling_features(team_df)

    print("merging blue/red features and computing stylistic differentials...")
    model_df = merge_matchup_features(matchups_elo_df, team_features_df)

    # Attach current elo ratings onto latest_team_profiles for step 5 simulation
    if os.path.exists(ratings_path):
        ratings_df = pd.read_parquet(ratings_path)
        latest_profiles_df = latest_profiles_df.merge(
            ratings_df[["teamname", "region", "elo_rating", "games_played"]],
            on="teamname",
            how="inner"
        ).sort_values("elo_rating", ascending=False)

    # Filter out eary warm-up rows where neither team has at least 3 prior games
    valid_model_df = model_df.dropna(subset=["blue_ewm15_result", "red_ewm15_result"]).reset_index(drop=True)

    out_model_path = os.path.join(processed_dir, "model_features.parquet")
    out_profiles_path = os.path.join(processed_dir, "latest_team_profiles.parquet")

    valid_model_df.to_parquet(out_model_path, index=False)
    latest_profiles_df.to_parquet(out_profiles_path, index=False)

    print("\n--- Feature Engineering Complete ---")
    print(f"Total Matches Ready for ML  : {len(valid_model_df):,}")
    print(f"Total Columns Generated     : {valid_model_df.shape[1]}")
    print(f"Active Team Profiles Saved  : {len(latest_profiles_df):,}")

    # Compare current LPL and LCK team profiles using league-level averages.
    comparison_metrics = {
        "ewm15_ckpm": "ewm15_ckpm",
        "ewm15_dragon_share": "ewm15_dragon_share",
        "ewm15_golddiffat15": "ewm15_golddiffat15",
        "ewm15_result": "ewm15_result",
        "champ_pool_15": "champ_pool_15",
    }
    league_profiles = latest_profiles_df.loc[
        latest_profiles_df["league"].isin(["LPL", "LCK"]),
        ["league", "teamname", *comparison_metrics],
    ]
    league_comparison = (
        league_profiles.groupby("league")[list(comparison_metrics)]
        .mean()
        .reindex(["LPL", "LCK"])
    )
    league_comparison.insert(
        0,
        "team_count",
        league_profiles.groupby("league")["teamname"].nunique().reindex(["LPL", "LCK"]),
    )
    print("\n--- LPL vs LCK: Average Current Team Profiles ---")
    print(league_comparison.round(3).to_string())

    plot_specs = [
        ("ewm15_dragon_share", "ewm15_golddiffat15", "Dragon Share", "Gold Differential at 15"),
        ("ewm15_result", "champ_pool_15", "Win Rate", "Champion Pool Diversity"),
    ]
    playoff_game_metadata = team_df.loc[
        team_df["playoffs"].eq(1),
        ["gameid", "series_id", "teamname", "league"],
    ]
    playoff_team_features = team_features_df.merge(
        playoff_game_metadata,
        on=["gameid", "teamname"],
        how="inner",
        suffixes=("", "_source"),
    )
    playoff_team_features["league"] = playoff_team_features["league_source"].fillna(
        playoff_team_features["league"]
    )
    playoff_series_profiles = (
        playoff_team_features.groupby(["series_id", "teamname", "league"], as_index=False)[
            list(comparison_metrics)
        ]
        .mean()
    )
    playoff_series_profiles["league"] = (
        playoff_series_profiles["league"].fillna("Unknown").astype(str)
    )
    all_leagues = sorted(playoff_series_profiles["league"].unique())
    league_cmap = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, len(plot_specs), figsize=(12, 5))
    for ax, (x_col, y_col, x_label, y_label) in zip(axes, plot_specs):
        for index, league in enumerate(all_leagues):
            league_data = playoff_series_profiles.loc[
                playoff_series_profiles["league"] == league
            ]
            ax.scatter(
                league_data[x_col],
                league_data[y_col],
                label=league,
                color=league_cmap(index % league_cmap.N),
                alpha=0.8,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{x_label} vs {y_label}")
        ax.legend(title="League")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Playoff-series team profiles (playoffs used as a Bo5 proxy)")

    fig.tight_layout()
    plot_path = os.path.join(processed_dir, "playoff_profile_comparisons.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved playoff-series comparison plots to: {plot_path}")

run_features_pipeline()