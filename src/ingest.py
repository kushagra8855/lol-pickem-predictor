import glob
import os
import pandas as pd
import numpy as np

# Tier-1 Domestic Leagues + International Events
# Includes historical and rebranded league codes
TARGET_LEAGUES = {
    "LCK", "LPL", "LEC", "LCS", "LTA N", "LTA S", "LTA",
    "PCS", "LCP", "VCS", "CBLOL", "LLA",
    "MSI", "WLDs", "WCS"
}

TEAM_CORE_COLS = [
    "gameid", "series_id", "date", "year", "split", "playoffs", "league",
    "patch", "game", "side", "teamid", "teamname", "result", "gamelength",
    "kills", "deaths", "ckpm", "firstblood", "firstdragon", "dragons",
    "opp_dragons", "elders", "opp_elders", "firstherald", "heralds",
    "void_grubs", "firstbaron", "barons", "opp_barons", "firsttower",
    "towers", "opp_towers", "turretplates", "inhibitors", "dpm",
    "earned gpm", "vspm", "golddiffat15", "xpdiffat15", "csdiffat15",
    "ban1", "ban2", "ban3", "ban4", "ban5",
    "pick1", "pick2", "pick3", "pick4", "pick5", "datacompleteness"
]

PLAYER_CORE_COLS = [
    "gameid", "series_id", "date", "year", "league", "patch", "game",
    "side", "position", "playerid", "playername", "teamid", "teamname",
    "champion", "result", "gamelength", "kills", "deaths", "assists",
    "dpm", "damageshare", "earned gpm", "earnedgoldshare", "cspm", "vspm",
    "golddiffat15", "xpdiffat15", "csdiffat15",
    "doublekills", "triplekills", "quadrakills", "pentakills",
]

def load_raw_csvs(raw_dir: str="data/raw") -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}. Download from oracleselixir.com")

    dfs = []
    for f in csv_files:
        print(f"Loading {os.path.basename(f)}...")
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total raw rows loaded: {len(combined):,}")
    return combined

def assign_series_id(team_df: pd.DataFrame) -> pd.Series:
    """
    Reconstructs a unique series_id for Bo1, Bo3, and Bo5 matches.
    Mathces between the same two teams on the same calendar day (or consecutive games)
    in the same league are grouped into a single series.
    """
    # Create an order-invariant team pair key per gameid
    game_teams = (
        team_df.groupby("gameid")
        .agg(
            date=("date", "first"),
            league=("league", "first"),
            game_num=("game", "first"),
            team_pair=("teamname", lambda x: "_vs_".join(sorted(x.astype(str))))
        )
        .sort_values("date")
        .reset_index()
    )

    # A new series starts whenever game_num == 1 OR the team_pair changes from the previous game
    is_new_series = (
        (game_teams["game_num"] == 1) |
        (game_teams["team_pair"] != game_teams["team_pair"].shift(1)) |
        (game_teams["league"] != game_teams["league"].shift(1))
    )
    game_teams["series_id"] = "S_" + is_new_series.cumsum().astype(str).str.zfill(6)

    return game_teams.set_index("gameid")["series_id"]

def build_matchups_table(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivots the 2-row-per-game team dataframe intoa 1-row-per-game matchup table
    (Blue Side vs. Red Side) sorted strictly by timestamp for Elo calculation.
    """

    blue = team_df[team_df["side"] == "Blue"].copy()
    red = team_df[team_df["side"] == "Red"].copy()

    matchups = pd.merge(
        blue[["gameid", "series_id", "date", "year", "split", "playoffs", "league", "patch", "game", "gamelength", "ckpm", "teamid", "teamname", "result", "golddiffat15"]],
        red[["gameid", "teamid", "teamname", "result"]],
        on="gameid",
        suffixes=("_blue", "_red")
    )

    # Compute series metadata (e.g., was this series a Bo1, Bo3, or Bo5?)
    series_lengths = matchups.groupby("series_id")["game"].transform("max")
    matchups["best_of"] = np.select(
        [series_lengths == 1, series_lengths < 3, series_lengths >= 3],
        [1, 3, 5],
        default=1
    )

    return matchups.sort_values("date").reset_index(drop=True)

def run_ingestion(raw_dir: str="data/raw", out_dir: str="data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    df = load_raw_csvs(raw_dir)

    # 1. Filter to Tier-1 and International leagues
    df = df[df["league"].isin(TARGET_LEAGUES)].copy()

    # 2. Parse timestamps and drop corrupted rows missing team names or results
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["gameid", "date", "teamname", "result"])

    # Ensure patch is formatted cleanly as string (e.g., "16.18")
    df["patch"] = df["patch"].astype(str)

    # 3. Separate Team summary rows (participantid 100/200) from Player rows (1-10)
    team_df = df[df["position"] == "team"].copy()
    player_df = df[df["position"] != "team"].copy()

    # Keep only valid games that have exactly 2 team rows 
    valid_games = team_df.groupby("gameid").size()
    valid_game_ids = valid_games[valid_games == 2].index
    team_df = team_df[team_df["gameid"].isin(valid_game_ids)].copy()
    player_df = player_df[player_df["gameid"].isin(valid_game_ids)].copy()

    # 4. Reconstruct series_id and map back to both tables
    series_map = assign_series_id(team_df)
    team_df["series_id"] = team_df["gameid"].map(series_map)
    player_df["series_id"] = player_df["gameid"].map(series_map)

    # 5. Select available columns safely (handles minor schema shifts across years)
    team_cols = [c for c in TEAM_CORE_COLS if c in team_df.columns]
    player_cols = [c for c in PLAYER_CORE_COLS if c in player_df.columns]

    team_df = team_df[team_cols].sort_values(["date", "gameid", "side"]).reset_index(drop=True)
    player_df = player_df[player_cols].sort_values(["date", "gameid", "side"]).reset_index(drop=True)
    matchups_df = build_matchups_table(team_df)

    # 6. Save to parquet (fast, preserves data types)
    team_df.to_parquet(os.path.join(out_dir, "team_games.parquet"), index=False)
    player_df.to_parquet(os.path.join(out_dir, "player_games.parquet"), index=False)
    matchups_df.to_parquet(os.path.join(out_dir, "matchups.parquet"), index=False)
    
    print("\n--- Ingestion Complete ---")
    print(f"Clean Games Processed : {len(matchups_df):,}")
    print(f"Unique Series Found   : {matchups_df['series_id'].nunique():,}")
    print(f"Team Rows Saved       : {len(team_df):,}")
    print(f"Player Rows Saved     : {len(player_df):,}")
    print(f"LPL Games Retained     : {len(matchups_df[matchups_df['league'] == 'LPL']):,}")

run_ingestion()