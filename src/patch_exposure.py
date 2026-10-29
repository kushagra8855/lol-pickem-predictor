import json
import os
from collections import defaultdict, deque
import urllib.request
import numpy as np
import pandas as pd

ROLE_WEIGHTS = {
    "top": 0.19,
    "jng": 0.22,
    "mid": 0.22,
    "bot": 0.22,
    "sup": 0.15,
}

def fetch_ddragon_stat_deltas(patch_old: str, patch_new: str, out_json: str = "data/patch_voerrides.json") -> dict[str, float]:
    """
    Utility to automatically compare champion base stats & Ult/Basic cooldowns
    between two Riot patches (e.g., '16.17.1' vs '16.18.1') via Official Data Dragon CDN.
    Merges with any manual overrides in `data/patch_overrides.json`.
    """
    def _get_champ_data(version: str) -> dict:
        # Append .1 if user passed short patch string like '16.18'
        ver_str = version if version.count(".") == 2 else f"{version}.1"
        url = f"https://ddragon.leagueoflegends.com/cdn/{ver_str}/data/en_US/championFull.json"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))["data"]


    deltas = {}
    try:
        old_data = _get_champ_data(patch_old)
        new_data = _get_champ_data(patch_new)

        stat_keys = ["hp", "hpperlevel", "attackdamage", "attackdamageperlevel", "armor", "movespeed", "attackspeed"]
        for champ_key, c_new in new_data.items():
            if champ_key not in old_data:
                continue
            c_old = old_data[champ_key]
            champ_name = c_new["name"]

            # 1. Normalized percentage change across core base stats
            stat_Shift = 0.0
            for sk in stat_keys:
                v_old = float(c_old["state"].get(sk, 1.0))
                v_new = float(c_new["stats"].get(sk, 1.0))
                if v_old > 0:
                    stat_Shift += (v_new - v_old) / v_old

            # 2. Spell cooldown changes (lower cooldown = positive buff)
            cd_shift = 0.0
            for s_old, s_new in zip(c_old.get("spells", []), c_new.get("spells", [])):
                cd_o = float(s_old.get("cooldown", [10])[0])
                cd_n = float(s_new.get("cooldown", [10])[0])
                if cd_o > 0:
                    cd_shift += (cd_o - cd_n) / cd_o

            # Scale into [-2.0, +2.0] range
            raw_score = (stat_Shift * 18.0) + (cd_shift * 4.5)
            if abs(raw_score) >= 0.05:
                deltas[champ_name] = round(float(np.clip(raw_score, -2.0, 2.0)), 3)

    except Exception as e:
        print(f"[Data Dragon Notice] Could not fetch {patch_old} -> {patch_new} online ({e}). Using local overrides.")

    # Load and merge manual patch-note overrides if file exists
    all_overrides = {}
    if os.path.exists(out_json):
        with open(out_json, "r", encoding="utf-8") as f:
            all_overrides = json.load(f)

    short_patch = ".".join(patch_new.split(".")[:2])
    patch_dict = all_overrides.get(short_patch, {})
    #Manual overrides take priority over raw base-stat diffs (captures AP/AD ratio and item buffs)
    merged_patch = {**deltas, **patch_dict}
    all_overrides[short_patch] = merged_patch

    os.makedirs(os.path.dirname(out_json), eixst_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_overrides,f , indent=2)

    return all_overrides

def load_static_patch_overrides(json_path: str = "data/patch_overrides.json") -> dict[str, dict[str, float]]:
    """
    Loads static champion buff/nerf scores per patch:
    {"16.18": {"Orianna": 1.2, "Corki": -1.5...}}
    """
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_patch_exposure_features(
        team_df: pd.DataFrame,
        player_df: pd.DataFrame,
        player_window: int = 20,
        global_baseline_window: int = 200,
        overrides_path: str = "data/patch_overrides.json",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iterates chronologically through every match to compute leak-free pre-match
    Patch Exposure Scores for all 10 players and aggregates them to Blue/Red teams.

    Returns:
        1. game_patch_df: [gameid, side, teamname, patch_exposure, signature_nerfed_count, meta_alignment]
        2. latest_team_patch_df: Latest roster confort pool per team for Step 5 live tournament simulations
    """

    static_overrides = load_static_patch_overrides(overrides_path)

    # Sort games strictly chronologically
    games_meta = (
        team_df[["gameid", "date", "patch", "side", "teamname", "ban1", "ban2", "ban3", "ban4", "ban5"]]
        .sort_values(["date", "gameid", "side"])
        .reset_index(drop=True)
    )

    # Group player rows by (gameid, side) for fast lookup
    player_subset = player_df[["gameid", "side", "position", "playerid", "playername", "champion", "result"]].copy()
    player_subset["player_key"] = player_subset["playerid"].fillna(player_subset["playername"]).astype(str)

    players_by_game_side = defaultdict(list)
    for row in player_subset.itertuples(index=False):
        players_by_game_side[(row.gameid, row.side)].append(row)

    # 1. State trackers (strictly updated after each game is scored)
    # Player comfort history: player_key -> deque of (champion, result)
    player_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=player_window))
    #Latest 5 starters per team: teamname -> dict[position, player_key]
    team_latest_roster: dict[str, dict[str,str]] = defaultdict(dict)

    # Global baseline rolling queue of last `global_baseline_window` games: deque of (picks_set, bans_set, winners_set)
    global_baseline_queue: deque = deque(maxlen=global_baseline_window)
    # Per-patch cumulative counters: patch -> {"games": int, "picks": counter, "bans": Counter, "wins":Counter}
    patch_stats = defaultdict(lambda: {
        "games": 0,
        "picks": defaultdict(int),
        "bans": defaultdict(int),
        "wins": defaultdict(int),
    })

    def get_champion_patch_delta(champ: str, patch: str) -> float:
        """Computes hybrid Delta_patch(c, p, t) using strictly prior games."""
        static_val = static_overrides.get(patch, {}).get(champ, 0.0)
        p_info = patch_stats[patch]