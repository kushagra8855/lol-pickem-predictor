import os
import random
import sys
import joblib
import numpy as np
import pandas as pd

if __package__:
    from .model import LolMatchupPredictor, PlattEnsembleCalibrator
else:
    from model import LolMatchupPredictor, PlattEnsembleCalibrator

# Default slot allocation by region if no manual 16-team list is passed
DEFAULT_REGIONAL_SLOTS = {
    "LCK": 4,
    "LPL": 4,
    "LEC": 3,
    "LCS": 2,
    "LTA N": 2,
    "PCS": 2,
    "LCP": 2,
    "VCS": 1,
    "CBLOL": 1,
    "LTA S": 1,
}

def select_tournament_field(profiles_df: pd.DataFrame, custom_teams: list[str] | None = None) -> pd.DataFrame:
    """
    Selects the 16 tournament teams and assigns Draw Pools (pool 1 to pool 4).
    Pass `custom_teams` (ordered from seed 1 to seed 16) once the official worlds/msi bracket is locked.
    """
    if custom_teams is not None:
        if len(custom_teams) != 16:
            raise ValueError(f"Expected exactly 16 custom teams, received {len(custom_teams)}.")
        if len(set(custom_teams)) != len(custom_teams):
            raise ValueError("Custom team names must be unique.")

        available_teams = set(profiles_df["teamname"].dropna())
        missing_teams = [team for team in custom_teams if team not in available_teams]
        if missing_teams:
            raise ValueError(
                "These custom team names were not found in latest_team_profiles.parquet: "
                f"{', '.join(missing_teams)}. Use the exact names from the profile data."
            )

        selected = profiles_df[profiles_df["teamname"].isin(custom_teams)].copy()
        # Preserve user-provided seed order
        order_map = {t: idx for idx, t in enumerate(custom_teams)}
        selected["seed_rank"] = selected["teamname"].map(order_map)
        selected = selected.sort_values("seed_rank").reset_index(drop=True)

    else:
        # Automatically pick top active teams per region based on current Elo
        picked_rows = []
        df_sorted = profiles_df.sort_values("elo_rating", ascending=False)
        for region, slots in DEFAULT_REGIONAL_SLOTS.items():
            reg_teams = df_sorted[df_sorted["region"] == region].head(slots)
            picked_rows.append(reg_teams)

        candidates = pd.concat(picked_rows, ignore_index=True).drop_duplicates(subset=["teamname"])
        if len(candidates) < 16:
            remaining = df_sorted[~df_sorted["teamname"].isin(candidates["teamname"])]
            candidates = pd.concat([candidates, remaining.head(16 - len(candidates))], ignore_index=True)

        selected = candidates.sort_values("elo_rating", ascending=False).head(16).reset_index(drop=True)

    if len(selected) != 16:
        raise ValueError(f"The tournament field requires 16 teams, but only {len(selected)} were selected.")

    # Assign Swiss Draw Pools: Pool 1(0..3), Pool 2(4..7), Pool 3(8..11), Pool 4(12..15)
    selected["team_idx"] = np.arange(len(selected))
    selected["draw_pool"] = (selected["team_idx"] // 4) + 1
    return selected

def build_probability_matrices(predictor, teams: list[str]) -> dict[int, np.ndarray]:
    """
    Pre-computes 16x16 pairwise win probability matrices for Bo1, Bo3, and Bo5.
    matrix[bo][i, j] = probability that team i defeats team j in Best-of-'bo'.
    """
    n = len(teams)
    matrices = {1: np.full((n, n), 0.5), 3: np.full((n,n), 0.5), 5: np.full((n, n), 0.5)}

    for i in range(n):
        for j in range(i+1, n):
            t_a, t_b = teams[i], teams[j]
            for bo in (1, 3, 5):
                res = predictor.predict_series(t_a, t_b, best_of=bo, playoffs=1)
                # Use neutral side probability for Swiss Bo1/Bo3, seeed for Bo5
                p_ij = res["p_a_series_seeded"] if bo == 5 else res["p_a_series_neutral"]
                matrices[bo][i, j] = p_ij
                matrices[bo][j, i] = 1.0 - p_ij

    return matrices

def pair_group_no_rematch(group: list[int], played_mask: np.ndarray, regions: list[str] | None = None) -> list[tuple[int, int]]:
    """
    Pairs teams within the same (W, L) Swiss bucket using randomized backtracking
    to avoid rematches (and avoid same-region  clashes in Round 1 when 'regions' is passed).
    """
    shuffled = group[:]
    random.shuffle(shuffled)

    def backtrack(remaining: list[int], current_pairs: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if not remaining:
            return current_pairs
        first = remaining[0]
        rest = remaining[1:]
        for idx, partner in enumerate(rest):
            # Check no-rematch constraint
            if played_mask[first, partner]:
                continue
            # Check Round 1 regional protection if provided
            if regions is not None and regions[first] == regions[partner]:
                continue

            next_remaining = rest[:idx] + rest[idx + 1:]
            result = backtrack(next_remaining, current_pairs +[(first, partner)])
            if result is not None:
                return result
        return None

    # Attempt strict constraint pairing first; fallback to relaxed pairing if bracket geometry forces a clash
    pairs = backtrack(shuffled, [])
    if pairs is None and regions is not None:
        pairs = backtrack(shuffled, []) # Drop regional constraint if needed
    if pairs is None:
        # Final fallback: simple adjacent pairing
        pairs = [(shuffled[k], shuffled[k + 1]) for k in range(0, len(shuffled), 2)]
    return pairs

def simulate_single_tournament(
        prob_matrices: dict[int, np.ndarray],
        regions: list[str],
        rng: np.random.Generator,
) -> dict:
    """
    Simulates one complete 16-team swiss stage followd by an 8-team Bo5 knockout bracket.
    """
    n_teams = 16
    wins = np.zeros(n_teams, dtype=int)
    losses = np.zeros(n_teams, dtype=int)
    played_mask = np.zeros((n_teams, n_teams), dtype=bool)

    # Round 1: Pool 1 (0..3) vs Pool 4 (12..15), Pool 2(4..7) vs Pool 3(8..11)
    p1_p4 = [0, 1, 2, 3, 12, 13, 14, 15]
    p2_p3 = [4, 5, 6, 7, 8, 9, 10, 11]

    # Force cross-pool pairings in Round 1 by temporarily marking intra-pool pairs as 'played'
    r1_mask = np.zeros((n_teams, n_teams), dtype=bool)
    for pool_slice in [range(0, 4), range(4, 8), range(8, 12), range(12, 16)]:
        for a in pool_slice:
            for b in pool_slice:
                r1_mask[a, b] = True

    r1_pairs = (
        pair_group_no_rematch(p1_p4, r1_mask, regions=regions) +
        pair_group_no_rematch(p2_p3, r1_mask, regions=regions)
    )

    for a, b in r1_pairs:
        played_mask[a, b] = played_mask[b, a] = True
        if rng.random() < prob_matrices[1][a,b]:
            wins[a] += 1
            losses[b] += 1
        else:
            wins[b] += 1
            losses[a] += 1

    # Rounds 2 through 5
    for _round_num in range(2, 6):
        # Find active teams (wins < 3 and losses < 3) grouped by (wins, losses)
        buckets: dict[tuple[int, int], list[int]] = {}
        for t in range(n_teams):
            if wins[t] < 3 and losses[t] < 3:
                rec = (int(wins[t]), int(losses[t]))
                buckets.setdefault(rec, []).append(t)

        for (w, l), group in buckets.items():
            # Advancement (w==2) or elimination (l==2) matches are Bo3; others are Bo1
            bo = 3 if (w == 2 or l == 2) else 1
            pairs = pair_group_no_rematch(group, played_mask, regions=None)
            for a, b in pairs:
                played_mask[a, b] = played_mask[b, a] = True
                if rng.random() < prob_matrices[bo][a, b]:
                    wins[a] += 1
                    losses[b] += 1
                else:
                    wins[b] += 1
                    losses[a] += 1

    # collect the 8 knockout qualifiers seeded by swiss final ercord (3-0, 3-1, 3-2)
    teams_3_0 = [t for t in range(n_teams) if wins[t] == 3 and losses[t] == 0]
    teams_3_1 = [t for t in range(n_teams) if wins[t] == 3 and losses[t] == 1]
    teams_3_2 = [t for t in range(n_teams) if wins[t] == 3 and losses[t] == 2]

    random.shuffle(teams_3_0)
    random.shuffle(teams_3_1)
    random.shuffle(teams_3_2)

    # offiial knockout seeding:
    # two 3-0 teams face two random 3-2 teams on opposite sides; remaining 3-1s and 3-2s pair off
    qf_pairs = [
        (teams_3_0[0], teams_3_2[0]),
        (teams_3_1[0], teams_3_1[1]),
        (teams_3_0[1], teams_3_2[1]),
        (teams_3_1[2], teams_3_2[2]),
    ]

    sf_teams = []
    for a, b in qf_pairs:
        winner = a if rng.random() < prob_matrices[5][a,b] else b
        sf_teams.append(winner)

    finalists = [
        sf_teams[0] if rng.random() < prob_matrices[5][sf_teams[0], sf_teams[1]] else sf_teams[1],
        sf_teams[2] if rng.random() < prob_matrices[5][sf_teams[2], sf_teams[3]] else sf_teams[3],
    ]

    champion = finalists[0] if rng.random() < prob_matrices[5][finalists[0], finalists[1]] else finalists[1]

    return {
        "wins": wins,
        "losses": losses,
        "sf_teams": sf_teams,
        "finalists": finalists,
        "champion": champion,
    }

def run_monte_carlo_bracket(
        predictor,
        field_df: pd.DataFrame,
        n_sims: int = 10000,
        seed: int = 42,
) -> pd.DataFrame:
    """Runs `n_sims` full tournament brackets and aggregates Pick'em slot probablities."""
    teams = field_df["teamname"].tolist()
    regions = field_df["region"].tolist()
    n_teams = len(teams)

    prob_matrices = build_probability_matrices(predictor, teams)
    rng = np.random.default_rng(seed)
    random.seed(seed)

    count_3_0 = np.zeros(n_teams)
    count_3_1 = np.zeros(n_teams)
    count_3_2 = np.zeros(n_teams)
    count_0_3 = np.zeros(n_teams)
    count_semis = np.zeros(n_teams)
    count_finals = np.zeros(n_teams)
    count_champ = np.zeros(n_teams)

    for _ in range(n_sims):
        sim = simulate_single_tournament(prob_matrices, regions, rng)
        w, l = sim["wins"], sim["losses"]

        for t in range(n_teams):
            if w[t] == 3 and l[t] == 0:
                count_3_0[t] += 1
            elif w[t] == 3 and l[t] == 1:
                count_3_1[t] += 1
            elif w[t] == 3 and l[t] == 2:
                count_3_2[t] += 1
            elif w[t] == 0 and l[t] == 3:
                count_0_3[t] += 1

        for t in sim["sf_teams"]:
            count_semis[t] += 1
        for t in sim["finalists"]:
            count_finals[t] += 1
        count_champ[sim["champion"]] += 1

    results = field_df[["teamname", "region", "draw_pool", "elo_rating"]].copy()
    results["prob_3_0"] = count_3_0 / n_sims
    results["prob_3_1"] = count_3_1 / n_sims
    results["prob_3_2"] = count_3_2 / n_sims
    results["prob_advance_total"] = (count_3_0 + count_3_1 + count_3_2) / n_sims
    results["prob_0_3"] = count_0_3 / n_sims
    results["prob_semifinals"] = count_semis / n_sims
    results["prob_finals"] = count_finals / n_sims
    results["prob_champion"] = count_champ / n_sims

    return results.sort_values("prob_advance_total", ascending=False).reset_index(drop=True)

def generate_crustal_ball_predictions(
        team_games_df: pd.DataFrame,
        player_games_df: pd.DataFrame,
        expected_tournament_maps: int = 80,
):
    """
    Analyses the two most recent years of matches across all teams to forecast
    Crystal ball champion presence, flex picks, and tournament event distributions.
    """
    latest_year = team_games_df["year"].max()
    first_year = latest_year - 1
    recent_team_df = team_games_df[
        team_games_df["year"].between(first_year, latest_year)
    ].copy()
    recent_player_df = player_games_df[
        player_games_df["year"].between(first_year, latest_year)
    ].copy()

    total_games = recent_team_df["gameid"].nunique()
    if total_games == 0:
        print("no recent games found for Crystal Ball aggregation.")
        return

    print(f"\nUsing all teams' games from {int(first_year)}-{int(latest_year)} ({total_games:,} games).")

    # 1. champion pick, ban, win rate, and role flex entropy
    picks_stats = (
        recent_player_df.groupby("champion")
        .agg(
            picks=("gameid", "count"),
            wins=("result", "sum"),
            roles_played=("position", lambda x: x.nunique()),
            avg_kda=('kills', lambda s: len(s)), # placeholder overwritten below
        )
    )
    picks_stats["win_rate"] = picks_stats["wins"] / picks_stats["picks"]
    picks_stats["pick_rate"] = picks_stats["picks"] / total_games

    ban_cols = [c for c in ["ban1", "ban2", "ban3", "ban4", "ban5"] if c in recent_team_df.columns]
    all_bans = pd.concat([recent_team_df[c] for c in ban_cols]).dropna()
    ban_counts = all_bans.value_counts()

    picks_stats["bans"] = picks_stats.index.map(ban_counts).fillna(0).astype(int)
    picks_stats["ban_rate"] = picks_stats["bans"] / total_games
    picks_stats["presence"] = picks_stats["pick_rate"] + picks_stats["ban_rate"]

    print("\n --- CRYSTAL BALL: CHAMPION META FORECAST ---")
    print("\n --- Top 8 Most Picked Candidates ---")
    print(
        picks_stats.sort_values("pick_rate", ascending=False)[
            ["picks", "pick_rate", "ban_rate", "win_rate", "roles_played"]
        ].head(8).round(3).to_string()
    )

    print("\n --- Top 8 Most banned candidates (High presence + Flex priority) ---")
    print(
        picks_stats.sort_values("ban_rate", ascending=False)[
            ["bans", "ban_rate", "pick_rate", "win_rate", "roles_played"]
        ].head(8).round(3).to_string()
    )

    # Highest win rate champion (Minimum 5 picks among qualified teams)
    high_vol_champs = picks_stats[picks_stats["picks"] >= 5].sort_values("win_rate", ascending=False)
    print("\n--- Top 5 Highest win-rate champions (min 5 picks) ---")
    print(high_vol_champs[["picks", "wins", "win_rate", "presence"]].head(5).round(3).to_string())

    # 2. Tournament Event Count Forecast (Poisson Simulation across ~80 maps)
    per_map_pentas = recent_player_df["pentakills"].sum() / total_games if "pentakills" in recent_player_df else 0.03
    per_map_barons = recent_team_df["barons"].sum() / total_games if "barons" in recent_team_df else 1.35
    per_map_elders = recent_team_df["elders"].sum() / total_games if "elders" in recent_team_df else 0.12
    durations_min = recent_team_df.groupby("gameid")["gamelength"].first() / 60.0

    rng = np.random.default_rng(42)
    sim_pentas = rng.poisson(lam=per_map_pentas * expected_tournament_maps, size=10000)
    sim_elders = rng.poisson(lam=per_map_elders * expected_tournament_maps, size=10000)
    sim_barons = rng.poisson(lam=per_map_barons * expected_tournament_maps, size=10000)

    # Extreme value simulation for shortest and longest map across an 80-map tournament
    sampled_tourneys = rng.choice(durations_min.dropna().values, size=(10000, expected_tournament_maps), replace=True)
    shortest_map_dist = sampled_tourneys.min(axis=1)
    longest_map_dist = sampled_tourneys.max(axis=1)

    print("\n--- Tournament Event Prop Distributions (80-Map Expectation) ---")
    print(f"Expected Total Pentakills   : Median = {int(np.median(sim_pentas))} (IQR: {int(np.percentile(sim_pentas, 25))} to {int(np.percentile(sim_pentas, 75))})")
    print(f"Expected Elder Dragons      : Median = {int(np.median(sim_elders))} (IQR: {int(np.percentile(sim_elders, 25))} to {int(np.percentile(sim_elders, 75))})")
    print(f"Expected Elder Barons       : Median = {int(np.median(sim_barons))} (IQR: {int(np.percentile(sim_barons, 25))} to {int(np.percentile(sim_barons, 75))})")
    print(f"Expected Shortest Game      : {np.median(shortest_map_dist):.1f} mins (90% CI: {np.percentile(shortest_map_dist, 5):.1f} - {np.percentile(shortest_map_dist, 95):.1f} mins)")
    print(f"Expected Longest Game       : {np.median(longest_map_dist):.1f} mins (90% CI: {np.percentile(longest_map_dist, 5):.1f} - {np.percentile(longest_map_dist, 95):.1f} mins)")

def print_optimal_pickemcard(sim_df: pd.DataFrame):
    """Translates the 10,000-run probability  table into a concrete Pick'em recommendation."""
    best_3_0 = sim_df.sort_values("prob_3_0", ascending=False).head(2)["teamname"].tolist()
    best_0_3 = sim_df.sort_values("prob_0_3", ascending=False).head(2)["teamname"].tolist()
    #best_3_1 = sim_df.sort_values("prob_3_1", ascending=False).head(2)["teamname"].tolist()
    #best_3_2 = sim_df.sort_values("prob_3_2", ascending=False).head(2)["teamname"].tolist()

    # For the 6 remaning advancing slots, pick the hightest overall advancment probability
    # excluding the team(s) already locked into the 3-0 slot
    remaining_adv = sim_df[~sim_df["teamname"].isin(best_3_0)].sort_values("prob_advance_total", ascending=False)
    best_advancing_6 = remaining_adv.head(6)["teamname"].tolist()
    champion_pick = sim_df.sort_values("prob_champion", ascending=False).iloc[0]

    print("\n========================================================")
    print("        OPTIMAL SWISS & KNOCKOUT PICK'EM SLOTS          ")
    print("========================================================")
    print(f"Recommended 3-0 Candidates      : {', '.join(best_3_0)}")
    print(f"Recommended 0-3 Candidates      : {', '.join(best_0_3)}")
    print(f"Recommended 3-1 / 3-2 Advancing : {', '.join(best_advancing_6)}")
    print(f"Highest EV Tournament Champion  : {champion_pick['teamname']} ({champion_pick["prob_champion"]*100:.1f}%)")


def run_simulaiton_pipeline(processed_dir: str = "data/processed", model_dir: str = "models"):
    predictor_path = os.path.join(model_dir, "pickem_predictor.joblib")
    profiles_path = os.path.join(processed_dir, "latest_team_profiles.parquet")
    team_games_path = os.path.join(processed_dir, "team_games.parquet")
    player_games_path = os.path.join(processed_dir, "player_games.parquet")

    if not os.path.exists(predictor_path):
        raise FileNotFoundError(f"Missing {predictor_path}. Run src/model.py first.")

    main_module = sys.modules["__main__"]
    main_module.LolMatchupPredictor = LolMatchupPredictor
    main_module.PlattEnsembleCalibrator = PlattEnsembleCalibrator
    predictor = joblib.load(predictor_path)
    profiles_df = pd.read_parquet(profiles_path)
    team_games_df = pd.read_parquet(team_games_path)
    player_games_df = pd.read_parquet(player_games_path)

    # 1. Select the 16-team tournament field (or pass custom list of 16 worlds teams here)
    worlds_teams = [
        "Gen.G", "Anyone's Legend", "G2 Esports", "Team Liquid", # Pool 1
        "Hanwha Life Esports", "Bilibili Gaming", "Team Secret Whales", "LØS", # Pool 2
        "T1", "Top Esports", "Movistar KOI", "LYON", # Pool 3
        "Dplus Kia", "Invictus Gaming", "Karmine Corp", "CTBC Flying Oyster" # Pool 4
    ]
    field_df = select_tournament_field(profiles_df, custom_teams=worlds_teams)

    print("Running 10,000 Monte Carlo Swiss + Knockout Tournament Simulations...")
    sim_results = run_monte_carlo_bracket(predictor, field_df, n_sims=10000)

    print("\n--- 10,000-Run Swiss & Knockout Probability Distribution ---")
    display_cols = [
        "teamname", "region", "draw_pool", "elo_rating",
        "prob_3_0", "prob_advance_total", "prob_0_3", "prob_semifinals", "prob_champion"
    ]
    formatted = sim_results[display_cols].copy()
    for col in ["prob_3_0", "prob_advance_total", "prob_0_3", "prob_semifinals", "prob_champion"]:
        formatted[col] = (formatted[col] * 100).round(1).astype(str) + "%"
    print(formatted.to_string(index=False))

    # 2. Output optimal Pick'em card
    print_optimal_pickemcard(sim_results)

    # 3. Run Crystal Ball meta and event Predictor
    generate_crustal_ball_predictions(
        team_games_df=team_games_df,
        player_games_df=player_games_df,
        expected_tournament_maps=80
    )

    sim_results.to_parquet(os.path.join(processed_dir, "monte_carlo_pickem_results.parquet"), index=False)


run_simulaiton_pipeline()