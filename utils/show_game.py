"""Print a compact, human-readable summary of one game from a baseline moves.csv --
move number, side to move, the move played, Stockfish's eval (White's POV, in pawns),
and a flag for any illegal-move retries the LLM needed on that ply.

Usage: python utils/show_game.py <run_id> [game_id] [--log-dir logs/baseline]

If game_id is omitted, shows the first game found for that run_id.
"""

import argparse
import csv
import os
import sys


def load_moves(run_id: str, log_dir: str) -> list[dict]:
    path = os.path.join(log_dir, f"{run_id}_moves.csv")
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def format_eval(row: dict) -> str:
    if row["is_mate_score"] == "True":
        return f"#{row['mate_in']}"
    cp = row["eval_cp_before_ply"]
    if not cp:
        return "?"
    return f"{int(cp) / 100:+.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("game_id", nargs="?", default=None)
    parser.add_argument("--log-dir", default="logs/baseline")
    args = parser.parse_args()

    rows = load_moves(args.run_id, args.log_dir)
    if not rows:
        print("No moves found for that run_id.")
        sys.exit(1)

    game_id = args.game_id or rows[0]["game_id"]
    rows = [r for r in rows if r["game_id"] == game_id]

    print(f"run_id={args.run_id}  game_id={game_id}  ({len(rows)} plies)\n")

    line = ""
    for row in rows:
        move = row["move_san"] or f"[{row['move_source']}]"
        eval_str = format_eval(row)
        illegal = int(row["illegal_attempts"])
        flag = f" (!{illegal} illegal)" if illegal else ""

        if row["side"] == "white":
            line = f"{row['fullmove_number']}. {move}"
        else:
            line += f"  {move}{flag}  [eval {eval_str}]"
            print(line)
            line = ""

    if line:
        print(line)


if __name__ == "__main__":
    main()
