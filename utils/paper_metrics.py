"""Implements the IMR / RBLM / GL / BE metrics from Kuo, Hsueh & Tsai (2023),
"Large Language Models on the Chessboard", Section 2.2, against the per-move and
per-game CSVs written by `main.run_baseline_experiment`.

`games.csv` already carries each game's final IMR/RBLM (`imr_final`/`rblm_final`) and
its BE readings (`be_move20`/`be_final`/`be_mean`), computed at record time -- the
functions here that touch `games_df` just read/aggregate those columns. The *running*
IMR(i, t) / RBLM(i, t) series (the paper's per-move curves, e.g. Figures 1-2) require
per-move data and are computed here from `moves.csv`.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd


def load_run(
    run_id: str, log_dir: str = "logs/baseline"
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    import json

    moves_df = pd.read_csv(os.path.join(log_dir, f"{run_id}_moves.csv"))
    games_df = pd.read_csv(os.path.join(log_dir, f"{run_id}_games.csv"))
    with open(os.path.join(log_dir, f"{run_id}_config.json")) as f:
        config = json.load(f)
    return moves_df, games_df, config


def llm_moves(moves_df: pd.DataFrame) -> pd.DataFrame:
    """Black's (the LLM's) rows only, sorted, with a per-game 1-indexed move_index --
    this is the paper's `t`."""
    lm = moves_df[moves_df["side"] == "black"].sort_values(["game_id", "ply_index"]).copy()
    lm["move_index"] = lm.groupby("game_id").cumcount() + 1
    return lm


def imr_series(moves_df: pd.DataFrame) -> pd.DataFrame:
    """Running IMR(i, t) = (cumulative moves with >=1 illegal attempt) / t."""
    lm = llm_moves(moves_df)
    lm["cum_moves_with_illegal"] = lm.groupby("game_id")["had_illegal"].cumsum()
    lm["imr"] = lm["cum_moves_with_illegal"] / lm["move_index"]
    return lm[["game_id", "move_index", "imr"]]


def rblm_series(moves_df: pd.DataFrame) -> pd.DataFrame:
    """Running RBLM(i, t) = (cumulative illegal attempts) / (cumulative moves with
    >=1 illegal attempt). NaN while that denominator is 0 -- the normal case for a
    game with no illegal moves yet."""
    lm = llm_moves(moves_df)
    lm["cum_illegal_attempts"] = lm.groupby("game_id")["illegal_attempts"].cumsum()
    lm["cum_moves_with_illegal"] = lm.groupby("game_id")["had_illegal"].cumsum()
    lm["rblm"] = lm["cum_illegal_attempts"] / lm["cum_moves_with_illegal"].replace(
        0, np.nan
    )
    return lm[["game_id", "move_index", "rblm"]]


def game_lengths(games_df: pd.DataFrame, unit: str = "plies") -> pd.Series:
    if unit == "plies":
        return games_df["gl_plies"]
    if unit == "fullmoves":
        return games_df["gl_fullmoves"]
    raise ValueError(f"unit must be 'plies' or 'fullmoves', got {unit!r}")


def be_at_move20(games_df: pd.DataFrame) -> pd.Series:
    """Board Evaluation at the LLM's 20th move (White's-POV centipawns), NaN for
    games that ended before reaching it."""
    return games_df["be_move20"]


def be_final(games_df: pd.DataFrame) -> pd.Series:
    return games_df["be_final"]


def be_mean_per_game(games_df: pd.DataFrame) -> pd.Series:
    return games_df["be_mean"]


def summarize_run(moves_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    """The paper-style summary table: mean IMR/RBLM/GL/BE across all games in the run."""
    n_games = len(games_df)
    be20 = be_at_move20(games_df)

    summary = {
        "n_games": n_games,
        "imr_mean": games_df["imr_final"].mean(),
        "rblm_mean": games_df["rblm_final"].mean(),
        "gl_mean_plies": games_df["gl_plies"].mean(),
        "gl_mean_fullmoves": games_df["gl_fullmoves"].mean(),
        "be_move20_mean": be20.mean(),
        "pct_games_reaching_move_20": be20.notna().mean(),
        "be_final_mean": be_final(games_df).mean(),
        "be_mean_mean": be_mean_per_game(games_df).mean(),
    }
    return pd.DataFrame([summary])
