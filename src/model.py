import os
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score

# Explicit feature list matching Step 3's output schema
FEATURE_COLS = [
    # 1. Cross-Region Elo Anchors
    "elo_diff_blue",
    "elo_prob_blue",
    "elo_blue_pre",
    "elo_red_pre",
    # 2. Match Context
    "playoffs",
    "best_of",
    # 3. 15-Game Baseline Rolling Differentials (Blue - Red)
    "diff_ewm15_result",
    "diff_ewm15_gamelength_min",
    "diff_ewm15_ckpm",
    "diff_ewm15_kda_ratio",
    "diff_ewm15_dpm",
    "diff_ewm15_earned_gold",
    "diff_ewm15_vspm",
    "diff_ewm15_dragon_share",
    "diff_ewm15_baron_share",
    "diff_ewm15_tower_diff",
    "diff_ewm15_firstblood",
    "diff_ewm15_firsttower",
    "diff_ewm15_firstdragon",
    "diff_ewm15_golddiffat15",
    "diff_ewm15_xpdiffat15",
    "diff_ewm15_csdiffat15",
    # 4. 5-Game Acute Form & Momentum Differentials
    "diff_ewm5_result",
    "diff_ewm5_kda_ration",
    "diff_ewm5_dpm",
    "diff_ewm5_golddiffat15",
    "diff_form_momentum",
    # 5. Draft Depth & Stylistic Interactions
    "diff_champ_pool_15",
    "expected_match_ckpm",
    "pace_mismatch_ckpm",
    "expected_gamelength_min",
    "blue_has_timeline_data",
    "red_has_timeline_data",
]

def get_available_features(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLS if c in df.columns]

def chronological_split(df: pd.DataFrame, train_frac: float = 0.75, val_frac: float = 0.125):
    """Splits match data strictly by date into Train, Validation, and Out-of-Time Test sets."""
    df_sorted = df.sort_values("date").reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()

    print(f"Train Set   : {len(train_df):,} maps ({train_df['date'].min().date()} -> {train_df['date'].max().date()})")
    print(f"Val Set     : {len(val_df):,} maps ({val_df['date'].min().date()} -> {val_df['date'].max().date()})")
    print(f"Test Set    : {len(test_df):,} maps ({test_df['date'].min().date()} -> {test_df['date'].max().date()})")
    return train_df, val_df, test_df

class PlattEnsembleCalibrator:
    """
    Blends CatBoost raw log-odds with our Cross-Region Elo log-odds on the validation set.
    This guqrantees well-calibrated probabilities and prevents tree overconfidence in small samples.
    """

    def __init__(self):
        self.meta_lr =  LogisticRegression(C=1.0)

    @staticmethod
    def _to_logit(p: np.ndarray) -> np.ndarray:
        p_clipped = np.clip(p, 1e-4, 1.0 - 1e-4)
        return np.log(p_clipped / (1.0 - p_clipped))

    def fit(self, cb_probs: np.ndarray, elo_probs: np.ndarray, y_true: np.ndarray):
        X_meta = np.column_stack([self._to_logit(cb_probs), self._to_logit(elo_probs)])
        self.meta_lr.fit(X_meta, y_true)
        return self

    def predict_proba(self, cb_probs: np.ndarray, elo_probs: np.ndarray) -> np.ndarray:
        X_meta = np.column_stack([self._to_logit(cb_probs), self._to_logit(elo_probs)])
        return self.meta_lr.predict_proba(X_meta)[:,1]

def print_calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int=5):
    """Displays predicted probability vs. actual empirical win rate across quantile buckets."""
    df_cal = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
    df_cal["bin"] = pd.qcut(df_cal["y_prob"], q=n_bins, duplicates="drop")
    summary = df_cal.groupby("bin", observed=False).agg(
        count=("y_true", "count"),
        mean_predicted_prob=("y_prob", "mean"),
        actual_win_rate=("y_true", "mean"),
    )
    summary["calibration_error"] = (summary["mean_predicted_prob"] - summary["actual_win_rate"]).abs()
    print("\n---Test Set Probability calibration Table ---")
    print(summary.round(4).to_string())
    print(f"Mean Absolute Calibration Error (ECE): {summary['calibration_error'].mean():.4f}")

def series_win_prob_loser_picks(p_a_on_blue: float, p_a_on_red: float, best_of: int = 3, a_starts_blue: bool = True) -> float:
    """
    Exact Markov recursion for Lol Series (Bo1, Bo3, Bo5) under "Loswer Picks Next Side" Rules.
    Assumes the loser of game k selects blue side for game k+1.
    """
    if best_of  == 1:
        return p_a_on_blue if a_starts_blue else p_a_on_red

    wins_needed = (best_of // 2) + 1
    memo = {}

    def dfs(a_wins: int, b_wins: int, a_is_blue: bool) -> float:
        if a_wins == wins_needed:
            return 1.0
        if b_wins == wins_needed:
            return 0.0
        state = (a_wins, b_wins, a_is_blue)
        if state in memo:
            return memo[state]

        p_win_map = p_a_on_blue if a_is_blue else p_a_on_red

        # If team A wins this map, team B lost -> team b picks blue next map (so a_is_blue = False)
        # if team A looses this map, team A lost -> team a picks blue next map(so a_is_blue = True)
        prob = (
            p_win_map * dfs(a_wins + 1, b_wins, False) +
            (1.0 - p_win_map) * dfs(a_wins, b_wins + 1, True)
        )
        memo[state] = prob
        return prob

    return dfs(0, 0, a_starts_blue)


class LolMatchupPredictor:
    """
    Self-contained inference artifact for Step 5 tournament simulations.
    Can predict any head-to-head matchup (Bo1, Bo3, Bo5) given two team names.
    """

    def __init__(self, cb_model, calibrator, feature_cols, profiles_df, blue_side_adv: float = 25.0):
        self.cb_model = cb_model
        self.calibrator = calibrator
        self.feature_cols = feature_cols
        self.profiles = profiles_df.set_index("teamname").to_dict(orient="index")
        self.blue_side_adv = blue_side_adv

    def _build_single_map_features(self, blue_team: str, red_team: str, playoffs: int = 1, best_of: int = 3) -> tuple[pd.DataFrame, float]:
        b = self.profiles[blue_team]
        r = self.profiles[red_team]

        elo_diff = b["elo_rating"] - r["elo_rating"]
        elo_prob_blue = 1.0 / (1.0 + 10.0 ** (-(elo_diff + self.blue_side_adv) / 400.0))

        row = {
            "elo_diff_blue": elo_diff,
            "elo_prob_blue": elo_prob_blue,
            "elo_blue_pre": b["elo_rating"],
            "elo_red_pre": r["elo_rating"],
            "playoffs": playoffs,
            "best_of": best_of,
            "expected_match_ckpm": 0.5 * (b.get("ewm15_ckpm", 0.8) + r.get("ewm15_ckpm", 0.8)),
            "pace_mismatch_ckpm": abs(b.get("ewm15_ckpm", 0.8) - r.get("ewm15_ckpm", 0.8)),
            "expected_gamelength_min": 0.5 * (b.get("ewm15_gamelength_min", 32.0) + r.get("ewm15_gamelength_min", 32.0)),
            "blue_has_timeline_data": b.get("has_timeline_data", 1),
            "red_has_timeline_data": r.get("has_timeline_data", 1),
        }

        # Populate all diff_* columns dynamically from team profile keys
        for col in self.feature_cols:
            if col.startswith("diff_"):
                base_key = col.replace("diff_", "")
                b_val = b.get(base_key, np.nan)
                r_val = r.get(base_key, np.nan)
                row[col] = (b_val - r_val) if (pd.notna(b_val) and pd.notna(r_val)) else np.nan

        X_row = pd.DataFrame([row])[self.feature_cols]
        return X_row, elo_prob_blue

    def predict_map_blue_win(self, blue_team: str, red_team: str, playoffs: int = 1, best_of: int = 3) -> float:
        X_row, elo_p = self._build_single_map_features(blue_team, red_team, playoffs, best_of)
        cb_p = self.cb_model.predict_proba(X_row)[:, 1]
        cal_p = self.calibrator.predict_proba(cb_p, np.array([elo_p]))[0]
        return float(cal_p)

    def predict_series(self, team_a: str, team_b: str, best_of: int = 3, playoffs: int = 1) -> dict:
        """
        Computes map and series win probablities between team A and Team B.
        Assumes higher-elo team gets game 1 side selection (or evaluates neutral coin toss for Bo1).
        """

        p_a_on_blue = self.predict_map_blue_win(team_a, team_b, playoffs=playoffs, best_of=best_of)
        p_b_on_blue = self.predict_map_blue_win(team_b, team_a, playoffs=playoffs, best_of=best_of)
        p_a_on_red = 1.0 - p_b_on_blue

        elo_a = self.profiles[team_a]["elo_rating"]
        elo_b = self.profiles[team_b]["elo_rating"]
        a_higher_seed = elo_a >= elo_b

        if best_of == 1:
            # In swiss Bo1s where side selection isn't known before draw, average Blue & Red win probs
            p_series_neutral = 0.5 * (p_a_on_blue + p_a_on_red)
            p_series_seeded = p_a_on_blue if a_higher_seed else p_a_on_red
        else:
            p_series_seeded = series_win_prob_loser_picks(
                p_a_on_blue, p_a_on_red, best_of=best_of, a_starts_blue=a_higher_seed
            )
            p_series_neutral = 0.5 * (
                series_win_prob_loser_picks(p_a_on_blue, p_a_on_red, best_of=best_of, a_starts_blue=True) +
                series_win_prob_loser_picks(p_a_on_blue, p_a_on_red, best_of=best_of, a_starts_blue=False)
            )

        return {
            "team_a": team_a,
            "team_b": team_b,
            "best_of": best_of,
            "p_a_on_blue": round(p_a_on_blue, 4),
            "p_a_on_red": round(p_a_on_red, 4),
            "p_a_series_neutral": round(p_series_neutral, 4),
            "p_a_series_seeded": round(p_series_seeded, 4),
        }

def run_model_pipeline(processed_dir: str = "data/processed", model_dir: str = "models"):
    os.makedirs(model_dir, exist_ok=True)
    features_path = os.path.join(processed_dir, "model_features.parquet")
    profiles_path = os.path.join(processed_dir, "latest_team_profiles.parquet")

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Missing {features_path}. Run src/features.py first.")

    df = pd.read_parquet(features_path)
    profiles_df = pd.read_parquet(profiles_path)
    if "elo_rating" not in profiles_df.columns:
        ratings_path = os.path.join(processed_dir, "current_team_ratings.parquet")
        if not os.path.exists(ratings_path):
            raise FileNotFoundError(
                f"Missing Elo ratings in {profiles_path} and missing {ratings_path}. "
                "Run src/elo.py and src/features.py first."
            )
        ratings_df = pd.read_parquet(ratings_path)
        profiles_df = profiles_df.merge(
            ratings_df[["teamname", "elo_rating"]],
            on="teamname",
            how="inner",
        ).sort_values("elo_rating", ascending=False)

    feature_cols = get_available_features(df)

    print(f"Training with {len(feature_cols)} pre-match features...")
    train_df, val_df, test_df = chronological_split(df)

    X_train, y_train = train_df[feature_cols], train_df["result_blue"].astype(int).values
    X_val, y_val = val_df[feature_cols], val_df["result_blue"].astype(int).values
    X_test, y_test = test_df[feature_cols], test_df["result_blue"].astype(int).values

    # 1. Train CatBoostClassifier with strong regularization and native NaN handling
    cb_model = CatBoostClassifier(
        iterations=1200,
        learning_rate=0.025,
        depth=4,
        l2_leaf_reg=8.0,
        subsample=0.8,
        nan_mode="Min",
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=42,
        verbose=200,
    )
    cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=80, verbose=False)
    print(f"CatBoost Best Iteration: {cb_model.get_best_iteration()}")

    # 2. Fit Platt/Log-Odds Calibrator on Validation Set
    val_cb_probs = cb_model.predict_proba(X_val)[:,1]
    val_elo_probs = val_df["elo_prob_blue"].values
    calibrator = PlattEnsembleCalibrator().fit(val_cb_probs, val_elo_probs, y_val)

    # 3. Evaluate on Out-of-Time Test Set (compare standalone Elo vs. Calibrated CatBoost)
    test_elo_probs = test_df["elo_prob_blue"].values
    test_cb_raw = cb_model.predict_proba(X_test)[:,1]
    test_final_probs = calibrator.predict_proba(test_cb_raw, test_elo_probs)

    print("\n======================================================")
    print("     OUT-OF-TIME TEST SET BENCHMARK (Elo vs. ML Model)   ")
    print("======================================================")
    for name, probs in [("Step 2: Standalone Elo", test_elo_probs), ("Step 4: Calibrated CatBoost", test_final_probs)]:
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        brier = brier_score_loss(y_test, probs)
        ll = log_loss(y_test, probs)
        print(f"{name:28s} | Acc: {acc:.4f} | Brier: {brier:.4f} | LogLoss: {ll:.4f}")

    # 4. Print Calibration Diagnostics & Top 12 Feature Importances
    print_calibration_table(y_test, test_final_probs)

    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": cb_model.get_feature_importance()
    }).sort_values("importance", ascending=False)
    print("\n--- Top 12 Most Influential Features ---")
    print(importances.head(12).to_string(index=False))

    # 5. Package and save the predictor artifact for step 5
    predictor = LolMatchupPredictor(
        cb_model=cb_model,
        calibrator=calibrator,
        feature_cols=feature_cols,
        profiles_df=profiles_df,
    )
    artifact_path = os.path.join(model_dir, "pickem_predictor.joblib")
    joblib.dump(predictor, artifact_path)
    print(f"\nSaved production predictor artifact to {artifact_path}")

    # 6. Quick Demo: Predict a Top-2 Head-to-Head matchup across bo1, bo3, and bo5
    if len(profiles_df) >= 2:
        t1, t2 = profiles_df.iloc[0]["teamname"], profiles_df.iloc[1]["teamname"]
        print(f"\n--- Sample Series Forecast: {t1} vs. {t2} ---")
        for bo in [1,3,5]:
            res = predictor.predict_series(t1, t2, best_of=bo)
            print(
                f"Bo{bo} | {t1} Win Prob (Neutral): {res['p_a_series_neutral']*100:5.1f}% "
                f"| (Seeded): {res['p_a_series_seeded']*100:5.1f}% "
                f"[Blue: {res['p_a_on_blue']*100:4.1f}%, Red: {res['p_a_on_red']*100:4.1f}%]"
            )

run_model_pipeline()