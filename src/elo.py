import os
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

# 1. Regional Priors
REGIONAL_PRIORS = {
    "LCK": 1600.0,
    "LPL": 1600.0,
    "LEC": 1480.0,
    "LCS": 1470.0,
    "LTA N": 1470.0,
    "LTA": 1450.0,
    "PCS": 1380.0,
    "LCP": 1390.0,
    "VCS": 1370.0,
    "CBLOL": 1310.0,
    "LTA S": 1310.0,
    "LLA": 1300.0,
}

INTERNATIONAL_LEAGUES = {"MSI", "WLDs", "WCS"}

class CrossRegionEloEngine:
    def __init__(
            self,
            base_k: float = 24.0,
            intl_k_multiplier: float = 1.45,
            playoffs_k_multiplier: float = 1.15,
            blue_side_advantage: float = 25.0,
            season_regression_weight: float = 0.25,
    ):
        self.base_k = base_k
        self.intl_k_multiplier = intl_k_multiplier
        self.playoffs_k_multiplier = playoffs_k_multiplier
        self.blue_side_advantage = blue_side_advantage
        self.season_regression_weight = season_regression_weight

        # State dictionaries: keyed by teamname
        self.ratings: dict[str, float] = {}
        self.team_region: dict[str, str] = {}
        self.games_played: dict[str, int] = {}

    def _get_prior(self, league: str) -> float:
        return REGIONAL_PRIORS.get(league, 1400.0)

    def register_team_if_new(self, team: str, league: str):
        """Initializes a team with its domestic region's prior rating."""
        if team not in self.ratings:
            prior = self._get_prior(league)
            self.ratings[team] = prior
            self.games_played[team] = 0
            if league not in INTERNATIONAL_LEAGUES:
                self.team_region[team] = league

        else:
            # Update domestic league mapping if team was first seen at an international event
            if league not in INTERNATIONAL_LEAGUES:
                self.team_region[team] = league

    def apply_new_season_regression(self):
        """
        Regresses each team's Elo toward the current mean of its domestic region
        at the start of a new calendar year to account for off-season roster swaps.
        """
        if not self.ratings:
            return

        # Compute current active mean per domestic region
        region_totals = {}
        region_counts = {}
        for team, rating in self.ratings.items():
            reg = self.team_region.get(team, "UNKNOWN")
            region_totals[reg] = region_totals.get(reg, 0.0) + rating
            region_counts[reg] = region_counts.get(reg, 0) + 1

        region_means = {
            reg: region_totals[reg] / region_counts[reg]
            for reg in region_totals
        }

        w = self.season_regression_weight
        for team, rating in self.ratings.items():
            reg = self.team_region.get(team, "UNKNOWN")
            target_mean = region_means.get(reg, self._get_prior(reg))
            self.ratings[team] = (1.0 - w) * rating + w * target_mean

    def expected_blue_prob(self, r_blue: float, r_red: float) -> float:
        """Calculates Blue side win probability including Blue side advantage."""
        elo_diff = (r_blue - r_red) + self.blue_side_advantage
        return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

    def compute_mov_multiplier(
            self,
            r_winner: float,
            r_loser: float,
            gd15_blue: float,
            gamelength_sec: float,
            blue_won: bool,
    ) -> float:
        """
        Computes an autocorrelation-adjusted Margin-of-Victory multiplier.
        Uses Gold Diff @ 15 when available; falls back to game speed for LPL partial rows.
        """
        winner_elo_diff = r_winner - r_loser

        if pd.notna(gd15_blue):
            # Signed gold lead at 15 from the winner's perspective
            winner_gd15 = gd15_blue if blue_won else -gd15_blue
            # Normalize: a +3000 gold lead at 15 when winning is a clean stomp(~2.0 units)
            dominance = max(0.0, winner_gd15) / 1500.0
        else:
            # Fallback for LPL partial rows: faster wins (under 33 mins = 1980s) signal cleaner stopms
            minutes = gamelength_sec / 60.0 if pd.notna(gamelength_sec) else 32.0
            dominance = max(0.0, (36.0 - minutes) / 5.0)

        # Autocorrelation adjustment prevents rating runaway when heavy favorites stomp underdogs
        autocorr_dampner = 2.2 / ((winner_elo_diff * 0.001) + 2.2)
        # Keep multiplier bounded in a reasonable range [0.75, 1.85]
        raw_mult = np.log(dominance + 1.7) * autocorr_dampner
        return float(np.clip(raw_mult, 0.75, 1.85))

    def process_matchups(self, matchups_df: pd.Dataframe) -> pd.Dataframe:
        df = matchups_df.sort_values("date").reset_index(drop=True).copy()

        elo_blue_pre = np.zeros(len(df))
        elo_red_pre = np.zeros(len(df))
        prob_blue_win = np.zeros(len(df))
        elo_shift_blue = np.zeros(len(df))

        current_year = None

        for idx, row in df.iterrows():
            year = row["year"]
            league = row["league"]
            playoffs = row["playoffs"]
            t_blue = row["teamname_blue"]
            t_red = row["teamname_red"]
            blue_won = bool(row["result_blue"] == 1)

            # Trigger inter-season regression when crossing into a new calendar year
            if current_year is not None and year != current_year:
                self.apply_new_season_regression()
            current_year = year

            # Ensure both teams exist in rating dictionary
            self.register_team_if_new(t_blue, league)
            self.register_team_if_new(t_red, league)

            r_blue = self.ratings[t_blue]
            r_red = self.ratings[t_red]

            # 1. Record PRE-MATCH ratings and probabilities
            p_blue = self.expected_blue_prob(r_blue, r_red)
            elo_blue_pre[idx] = r_blue
            elo_red_pre[idx] = r_red
            prob_blue_win[idx] = p_blue

            # 2. Determine dynamic K-factor
            k = self.base_k
            if league in INTERNATIONAL_LEAGUES:
                k *= self.intl_k_multiplier
            elif playoffs == 1:
                k *= self.playoffs_k_multiplier

            # 3. Calculate Margin of Victory multiplier
            r_winner = r_blue if blue_won else r_red
            r_loser = r_red if blue_won else r_blue
            mov_mult = self.compute_mov_multiplier(
                r_winner=r_winner,
                r_loser=r_loser,
                gd15_blue=row.get("golddiffat15", np.nan),
                gamelength_sec=row.get("gamelength", np.nan),
                blue_won=blue_won,
            )

            # 4. Update ratings post-match
            actual_blue = 1.0 if blue_won else 0.0
            delta_blue = k * mov_mult * (actual_blue - p_blue)

            self.ratings[t_blue] += delta_blue
            self.ratings[t_red] -= delta_blue
            self.games_played[t_blue] += 1
            self.games_played[t_red] += 1

            elo_shift_blue[idx] = delta_blue

        df["elo_blue_pre"] = elo_blue_pre
        df["elo_red_pre"] = elo_red_pre
        df["elo_diff_blue"] = df["elo_blue_pre"] - df["elo_red_pre"]
        df["elo_prob_blue"] = prob_blue_win
        df["elo_shift_blue"] = elo_shift_blue
        
        return df

    def get_leaderboard(self, min_games: int = 20) -> pd.DataFrame:
        rows = []
        for team, rating in self.ratings.items():
            if self.games_played.get(team, 0) >= min_games:
                rows.append({
                    "teamname": team,
                    "region": self.team_region.get(team, "INTL"),
                    "elo_rating": round(rating, 1),
                    "games_played": self.games_played[team],
                })
        return pd.DataFrame(rows).sort_values("elo_rating", ascending=False).reset_index(drop=True)

def evaluate_elo_baseline(df: pd.DataFrame):
    """Evaluates standalone Elo predictive performance across all games and international events."""
    y_true = df["result_blue"].astype(int)
    y_prob = df["elo_prob_blue"]
    y_pred = (y_prob >= 0.5).astype(int)

    print("\n--- Standalone Elo Baseline Metrics (All Matches) ---")
    print(f"Accuracy    : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Brier Score : {brier_score_loss(y_true, y_prob):.4f} (Random coinflip = 0.2500)")
    print(f"Log Loss    : {log_loss(y_true, y_prob):.4f} (Random coinflip = 0.6931)")

    intl_df = df[df["league"].isin(INTERNATIONAL_LEAGUES)]
    if len(intl_df) > 0:
        yi_true = intl_df["result_blue"].astype(int)
        yi_prob = intl_df["elo_prob_blue"]
        yi_pred = (yi_prob >= 0.5).astype(int)
        print("\n--- International Matches Only (MSI + Worlds) ---")
        print(f"Sample Size : {len(intl_df):,} maps")
        print(f"Accuracy    : {accuracy_score(yi_true, yi_pred):.4f}")
        print(f"Brier Score : {brier_score_loss(yi_true, yi_prob):.4f}")
        print(f"Log Loss    : {log_loss(yi_true, yi_prob):.4f}")

def run_elo_pipeline(processed_dir: str = "data/processed"):
    matchups_path = os.path.join(processed_dir, "matchups.parquet")
    if not os.path.exists(matchups_path):
        raise FileNotFoundError(f"Missing {matchups_path}. Run src/ingest.py first.")

    matchups_df = pd.read_parquet(matchups_path)

    engine = CrossRegionEloEngine(
        base_k=24.0,
        intl_k_multiplier=1.45,
        playoffs_k_multiplier=1.15,
        blue_side_advantage=25.0,
        season_regression_weight=0.25,
    )

    enriched_df = engine.process_matchups(matchups_df)
    evaluate_elo_baseline(enriched_df)

    leaderboard = engine.get_leaderboard(min_games=25)
    print("\n--- Top 15 Global Teams by Current Elo ---")
    print(leaderboard.head(15).to_string(index=False))

    # Save outputs for Step 3 & Step 5
    enriched_df.to_parquet(os.path.join(processed_dir, "matchups_with_elo.parquet"), index=False)
    leaderboard.to_parquet(os.path.join(processed_dir, "current_team_ratings.parquet"), index=False)
    print(f"\nSaved enriched matchups to {processed_dir}/matchups_with_elo.parquet")

run_elo_pipeline()