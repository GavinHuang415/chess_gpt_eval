import chess
import chess.engine
import os
import csv
import json
import random
import shutil
import statistics
import subprocess
import time
import platform

from typing import Optional, Tuple
from dataclasses import dataclass, field, asdict

# NOTE: LLAMA AND NANOGPT ARE EXPERIMENTAL PLAYERS that most people won't need to use
# They are commented by default to avoid unnecessary dependencies such as pytorch.
# from llama_module import BaseLlamaPlayer, LocalLlamaPlayer, LocalLoraLlamaPlayer
# from nanogpt.nanogpt_module import NanoGptPlayer
import gpt_query
from conversation import Conversation
from baseline_config import BaselineConfig


BASELINE_OPENING_MOVES = ("e4", "d4", "Nf3", "e3")

RESIGNATION_TOKENS = ("1-0", "0-1", "1/2-1/2")

MOVE_FIELDNAMES = [
    "run_id",
    "game_id",
    "ply_index",
    "fullmove_number",
    "side",
    "player_label",
    "move_source",
    "move_san",
    "move_uci",
    "illegal_attempts",
    "had_illegal",
    "api_errors",
    "rejected_responses",
    "accepted_raw_response",
    "eval_cp_before_ply",
    "is_mate_score",
    "mate_in",
    "engine_top_moves",
    "engine_chosen_index",
    "board_fen_after",
    "ply_wall_seconds",
]

GAME_FIELDNAMES = [
    "run_id",
    "game_id",
    "game_index",
    "model",
    "temperature",
    "top_p",
    "max_tokens",
    "max_illegal_retries",
    "engine_depth",
    "engine_skill_level",
    "engine_multipv",
    "opening_move",
    "result",
    "termination_reason",
    "white_score",
    "black_score",
    "llm_score",
    "gl_plies",
    "gl_fullmoves",
    "llm_move_count",
    "llm_illegal_attempts_total",
    "llm_moves_with_illegal",
    "imr_final",
    "rblm_final",
    "be_move20",
    "be_final",
    "be_mean",
    "llm_api_errors_total",
    "time_taken",
    "transcript_san",
]


def choose_opening_move(opening_moves=BASELINE_OPENING_MOVES) -> str:
    return random.choice(opening_moves)


def load_baseline_prompt(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


def parse_move_from_response(response: Optional[str]) -> Optional[str]:
    """Strict parsing matching the paper's prompt instruction to reply with only the
    move: take the first whitespace-delimited token, and strip stray wrapping
    punctuation some chat models add (backticks, quotes, asterisks, trailing periods).
    A chatty response counts as illegal, since that's what IMR is meant to measure."""
    if response is None:
        return None
    stripped = response.strip()
    if not stripped:
        return None
    first_token = stripped.split()[0]
    return first_token.strip("`*\"'.,")


def format_movetext(san_by_ply: list[str]) -> str:
    parts = []
    for i, san in enumerate(san_by_ply):
        if i % 2 == 0:
            parts.append(f"{i // 2 + 1}.")
        parts.append(san)
    return " ".join(parts)


def generate_unique_game_id() -> str:
    timestamp = int(time.time())
    random_num = random.randint(1000, 9999)
    return f"{timestamp}-{random_num}"


def _append_csv_row(path: str, fieldnames: list[str], row: dict) -> None:
    write_headers = not os.path.exists(path)
    if not write_headers:
        with open(path, "r", newline="") as f:
            existing_header = next(csv.reader(f), None)
        if existing_header is not None and existing_header != fieldnames:
            raise ValueError(
                f"{path} has a header that doesn't match the expected schema "
                f"({existing_header} != {fieldnames}). Use a new run_id or move the "
                "old file aside."
            )
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_headers:
            writer.writeheader()
        writer.writerow(row)


def _score_from_info(info: dict) -> Tuple[Optional[int], bool, Optional[int]]:
    score = info.get("score")
    if score is None:
        return None, False, None
    white_score = score.white()
    if white_score.is_mate():
        return None, True, white_score.mate()
    return white_score.score(), False, None


@dataclass
class LegalMoveResult:
    move_san: Optional[str] = None
    move_uci: Optional[chess.Move] = None
    illegal_attempts: int = 0
    api_errors: int = 0
    rejected_responses: list = field(default_factory=list)
    accepted_raw_response: Optional[str] = None
    is_resignation: bool = False
    is_illegal_move_limit: bool = False


@dataclass
class EngineDecision:
    move_san: str
    move_uci: chess.Move
    eval_cp: Optional[int]
    is_mate: bool
    mate_in: Optional[int]
    top_moves_san: list
    chosen_index: int


class Player:
    def get_move(
        self,
        board: chess.Board,
        conversation: Conversation,
        temperature: float,
        top_p: float,
    ) -> Optional[str]:
        raise NotImplementedError

    def get_config(self) -> dict:
        raise NotImplementedError


class GPTPlayer(Player):
    def __init__(
        self,
        model: str,
        max_tokens: int = 10,
        system_message: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.system_message = system_message
        self.base_url = base_url
        self.api_key = api_key

    def get_move(
        self,
        board: chess.Board,
        conversation: Conversation,
        temperature: float,
        top_p: float,
    ) -> Optional[str]:
        return gpt_query.get_chat_response(
            conversation.snapshot(),
            self.model,
            temperature,
            top_p,
            self.max_tokens,
            self.system_message,
            self.base_url,
            self.api_key,
        )

    def get_config(self) -> dict:
        return {"model": self.model}


class StockfishPlayer(Player):
    @staticmethod
    def get_stockfish_path(configured_path: Optional[str] = None) -> str:
        env_path = os.environ.get("STOCKFISH_PATH")
        for candidate in (env_path, configured_path):
            if candidate and os.path.isfile(candidate):
                return candidate

        which_path = shutil.which("stockfish") or shutil.which("stockfish.exe")
        if which_path:
            return which_path

        default_path = None
        if platform.system() == "Linux":
            default_path = "/usr/games/stockfish"
        elif platform.system() == "Darwin":
            default_path = "stockfish"

        if default_path and os.path.isfile(default_path):
            return default_path

        raise FileNotFoundError(
            "Could not locate a Stockfish executable. Set the STOCKFISH_PATH "
            "environment variable (or BaselineConfig.stockfish_path) to its full "
            "path, or add it to your system PATH. Download Stockfish from "
            "https://stockfishchess.org/download/"
        )

    def __init__(
        self,
        depth: int = 15,
        skill_level: Optional[int] = None,
        multipv: int = 3,
        stockfish_path: Optional[str] = None,
    ):
        self._depth = depth
        self._skill_level = skill_level
        self._multipv = multipv
        self._limit = chess.engine.Limit(depth=depth)

        path = StockfishPlayer.get_stockfish_path(stockfish_path)
        self._engine = chess.engine.SimpleEngine.popen_uci(path)
        self.engine_name = self._engine.id.get("name", "Stockfish")

        if self._skill_level is not None:
            self._engine.configure({"Skill Level": self._skill_level})

    def analyse_and_select(self, board: chess.Board) -> EngineDecision:
        """One analyse() call serving both the paper's requirements: the position's
        eval (from the best line) and a set of up to `multipv` candidate moves, one of
        which is sampled uniformly at random to actually be played."""
        infos = self._engine.analyse(board, self._limit, multipv=self._multipv)
        if isinstance(infos, dict):
            infos = [infos]

        eval_cp, is_mate, mate_in = _score_from_info(infos[0])

        candidates = [info["pv"][0] for info in infos if info.get("pv")]
        chosen_index = random.randrange(len(candidates))
        chosen_move = candidates[chosen_index]
        move_san = board.san(chosen_move)
        top_moves_san = [board.san(move) for move in candidates]

        return EngineDecision(
            move_san=move_san,
            move_uci=chosen_move,
            eval_cp=eval_cp,
            is_mate=is_mate,
            mate_in=mate_in,
            top_moves_san=top_moves_san,
            chosen_index=chosen_index,
        )

    def evaluate(self, board: chess.Board) -> Tuple[Optional[int], bool, Optional[int]]:
        info = self._engine.analyse(board, self._limit)
        return _score_from_info(info)

    def get_move(
        self,
        board: chess.Board,
        conversation: Conversation,
        temperature: float,
        top_p: float,
    ) -> Optional[str]:
        return self.analyse_and_select(board).move_san

    def get_config(self) -> dict:
        return {
            "engine_name": self.engine_name,
            "depth": self._depth,
            "skill_level": self._skill_level,
        }

    def close(self):
        self._engine.quit()


def play_llm_ply(
    llm_player: GPTPlayer,
    board: chess.Board,
    conversation: Conversation,
    config: BaselineConfig,
) -> LegalMoveResult:
    """Repeatedly resample the LLM's move against the SAME conversation history (no
    escalating temperature, no telling the model its move was illegal -- the paper
    resamples the turn, it doesn't give feedback) until a legal move is produced or
    config.max_illegal_retries is exhausted."""
    illegal_count = 0
    api_error_count = 0
    rejected: list[str] = []

    for _ in range(config.max_illegal_retries):
        try:
            response = llm_player.get_move(
                board, conversation, config.temperature, config.top_p
            )
        except Exception as e:
            print(f"API error while requesting move: {e}")
            api_error_count += 1
            continue

        if response is None:
            api_error_count += 1
            continue

        move_text = parse_move_from_response(response)

        if move_text in RESIGNATION_TOKENS:
            return LegalMoveResult(
                illegal_attempts=illegal_count,
                api_errors=api_error_count,
                rejected_responses=rejected,
                is_resignation=True,
            )

        move_uci = None
        if move_text:
            try:
                move_uci = board.parse_san(move_text)
            except Exception:
                move_uci = None

        if move_uci is not None and move_uci in board.legal_moves:
            return LegalMoveResult(
                move_san=board.san(move_uci),
                move_uci=move_uci,
                illegal_attempts=illegal_count,
                api_errors=api_error_count,
                rejected_responses=rejected,
                accepted_raw_response=response,
            )

        illegal_count += 1
        rejected.append(response)
        print(f"Illegal move: {move_text!r}")

    return LegalMoveResult(
        illegal_attempts=illegal_count,
        api_errors=api_error_count,
        rejected_responses=rejected,
        is_illegal_move_limit=True,
    )


def play_baseline_game(
    config: BaselineConfig,
    llm_player: GPTPlayer,
    engine_player: StockfishPlayer,
    game_index: int,
    moves_csv_path: str,
    games_csv_path: str,
) -> None:
    board = chess.Board()
    conversation = Conversation()
    game_id = generate_unique_game_id()
    game_start = time.time()

    san_by_ply: list[str] = []
    all_evals: list[int] = []
    be_by_llm_move: list[Optional[int]] = []

    illegal_attempts_total = 0
    moves_with_illegal = 0
    llm_move_count = 0
    llm_ply_attempts = 0
    api_errors_total = 0

    def log_ply(
        ply_index,
        side,
        player_label,
        move_source,
        move_san,
        move_uci,
        illegal_attempts,
        api_errors,
        rejected_responses,
        accepted_raw_response,
        eval_cp_before_ply,
        is_mate_score,
        mate_in,
        engine_top_moves,
        engine_chosen_index,
        ply_wall_seconds,
    ):
        _append_csv_row(
            moves_csv_path,
            MOVE_FIELDNAMES,
            {
                "run_id": config.run_id,
                "game_id": game_id,
                "ply_index": ply_index,
                "fullmove_number": (ply_index + 1) // 2,
                "side": side,
                "player_label": player_label,
                "move_source": move_source,
                "move_san": move_san or "",
                "move_uci": move_uci or "",
                "illegal_attempts": illegal_attempts,
                "had_illegal": 1 if illegal_attempts > 0 else 0,
                "api_errors": api_errors,
                "rejected_responses": " ||| ".join(rejected_responses),
                "accepted_raw_response": accepted_raw_response or "",
                "eval_cp_before_ply": eval_cp_before_ply,
                "is_mate_score": is_mate_score,
                "mate_in": mate_in,
                "engine_top_moves": ";".join(engine_top_moves),
                "engine_chosen_index": (
                    engine_chosen_index if engine_chosen_index is not None else ""
                ),
                "board_fen_after": board.fen(),
                "ply_wall_seconds": ply_wall_seconds,
            },
        )

    # Ply 1: White's opening move is forced to one of the paper's 4 options, not
    # played by the engine.
    opening_move_san = choose_opening_move(config.opening_moves)
    eval_before, is_mate, mate_in = engine_player.evaluate(board)
    opening_move_uci = board.parse_san(opening_move_san)
    ply_start = time.time()
    board.push(opening_move_uci)
    log_ply(
        1,
        "white",
        engine_player.engine_name,
        "random_opening",
        opening_move_san,
        opening_move_uci.uci(),
        0,
        0,
        [],
        None,
        eval_before,
        is_mate,
        mate_in,
        [],
        None,
        time.time() - ply_start,
    )
    san_by_ply.append(opening_move_san)
    if not is_mate:
        all_evals.append(eval_before)

    prompt_text = load_baseline_prompt(config.prompt_path).format(
        first_move=opening_move_san
    )
    conversation.add_user(prompt_text)

    ply_index = 1
    result = "*"
    termination_reason = "max_plies"

    while ply_index < config.max_plies and not board.is_game_over():
        # Black to move (the LLM)
        ply_index += 1
        eval_before, is_mate, mate_in = engine_player.evaluate(board)
        ply_start = time.time()
        llm_result = play_llm_ply(llm_player, board, conversation, config)
        ply_wall = time.time() - ply_start

        llm_ply_attempts += 1
        illegal_attempts_total += llm_result.illegal_attempts
        api_errors_total += llm_result.api_errors
        if llm_result.illegal_attempts > 0:
            moves_with_illegal += 1

        if llm_result.is_resignation:
            termination_reason = "llm_resigned"
            result = "1-0"
            log_ply(
                ply_index,
                "black",
                llm_player.model,
                "llm_resignation",
                None,
                None,
                llm_result.illegal_attempts,
                llm_result.api_errors,
                llm_result.rejected_responses,
                None,
                eval_before,
                is_mate,
                mate_in,
                [],
                None,
                ply_wall,
            )
            break

        if llm_result.is_illegal_move_limit:
            termination_reason = "illegal_move_limit"
            result = "*"
            log_ply(
                ply_index,
                "black",
                llm_player.model,
                "llm_illegal_limit",
                None,
                None,
                llm_result.illegal_attempts,
                llm_result.api_errors,
                llm_result.rejected_responses,
                None,
                eval_before,
                is_mate,
                mate_in,
                [],
                None,
                ply_wall,
            )
            break

        board.push(llm_result.move_uci)
        conversation.add_assistant(llm_result.accepted_raw_response)
        san_by_ply.append(llm_result.move_san)
        llm_move_count += 1

        log_ply(
            ply_index,
            "black",
            llm_player.model,
            "llm",
            llm_result.move_san,
            llm_result.move_uci.uci(),
            llm_result.illegal_attempts,
            llm_result.api_errors,
            llm_result.rejected_responses,
            llm_result.accepted_raw_response,
            eval_before,
            is_mate,
            mate_in,
            [],
            None,
            ply_wall,
        )
        if not is_mate:
            all_evals.append(eval_before)

        be_eval, be_is_mate, _ = engine_player.evaluate(board)
        be_by_llm_move.append(None if be_is_mate else be_eval)

        if board.is_game_over():
            termination_reason = board.outcome().termination.name.lower()
            result = board.result()
            break

        # White to move (the engine)
        ply_index += 1
        ply_start = time.time()
        decision = engine_player.analyse_and_select(board)
        board.push(decision.move_uci)
        conversation.add_user(decision.move_san)
        san_by_ply.append(decision.move_san)
        if not decision.is_mate:
            all_evals.append(decision.eval_cp)

        log_ply(
            ply_index,
            "white",
            engine_player.engine_name,
            "engine_top3",
            decision.move_san,
            decision.move_uci.uci(),
            0,
            0,
            [],
            None,
            decision.eval_cp,
            decision.is_mate,
            decision.mate_in,
            decision.top_moves_san,
            decision.chosen_index,
            time.time() - ply_start,
        )

        if board.is_game_over():
            termination_reason = board.outcome().termination.name.lower()
            result = board.result()
            break

    final_eval, final_is_mate, _ = engine_player.evaluate(board)
    be_final = None if final_is_mate else final_eval
    be_move20 = be_by_llm_move[19] if len(be_by_llm_move) >= 20 else None
    be_mean = statistics.mean(all_evals) if all_evals else None

    time_taken = time.time() - game_start

    if result == "1-0":
        white_score, black_score = 1, 0
    elif result == "0-1":
        white_score, black_score = 0, 1
    elif result == "1/2-1/2":
        white_score, black_score = 0.5, 0.5
    else:
        white_score, black_score = None, None
    llm_score = black_score

    imr_final = (
        moves_with_illegal / llm_ply_attempts if llm_ply_attempts else None
    )
    rblm_final = (
        illegal_attempts_total / moves_with_illegal if moves_with_illegal else None
    )

    record_game(
        games_csv_path,
        {
            "run_id": config.run_id,
            "game_id": game_id,
            "game_index": game_index,
            "model": config.model,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
            "max_illegal_retries": config.max_illegal_retries,
            "engine_depth": config.engine_depth,
            "engine_skill_level": config.engine_skill_level,
            "engine_multipv": config.engine_multipv,
            "opening_move": opening_move_san,
            "result": result,
            "termination_reason": termination_reason,
            "white_score": white_score,
            "black_score": black_score,
            "llm_score": llm_score,
            "gl_plies": len(san_by_ply),
            "gl_fullmoves": (len(san_by_ply) + 1) // 2,
            "llm_move_count": llm_move_count,
            "llm_illegal_attempts_total": illegal_attempts_total,
            "llm_moves_with_illegal": moves_with_illegal,
            "imr_final": imr_final,
            "rblm_final": rblm_final,
            "be_move20": be_move20,
            "be_final": be_final,
            "be_mean": be_mean,
            "llm_api_errors_total": api_errors_total,
            "time_taken": time_taken,
            "transcript_san": format_movetext(san_by_ply),
        },
    )

    print(
        f"Game {game_index}: result={result} termination={termination_reason} "
        f"plies={len(san_by_ply)} illegal_attempts={illegal_attempts_total}"
    )


def record_game(games_csv_path: str, row: dict) -> None:
    _append_csv_row(games_csv_path, GAME_FIELDNAMES, row)


def _git_commit_hash() -> Optional[str]:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _write_run_config(config: BaselineConfig, path: str, engine_name: str) -> None:
    payload = asdict(config)
    if payload.get("api_key"):
        payload["api_key"] = "<redacted>"
    payload["opening_moves"] = list(config.opening_moves)
    payload["stockfish_engine_name"] = engine_name
    payload["prompt_text"] = load_baseline_prompt(config.prompt_path)
    payload["git_commit"] = _git_commit_hash()
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def run_baseline_experiment(config: BaselineConfig) -> None:
    os.makedirs(config.log_dir, exist_ok=True)

    moves_csv_path = os.path.join(config.log_dir, f"{config.run_id}_moves.csv")
    games_csv_path = os.path.join(config.log_dir, f"{config.run_id}_games.csv")
    config_json_path = os.path.join(config.log_dir, f"{config.run_id}_config.json")

    llm_player = GPTPlayer(
        model=config.model,
        max_tokens=config.max_tokens,
        system_message=config.system_message,
        base_url=config.base_url,
        api_key=config.api_key,
    )
    engine_player = StockfishPlayer(
        depth=config.engine_depth,
        skill_level=config.engine_skill_level,
        multipv=config.engine_multipv,
        stockfish_path=config.stockfish_path,
    )

    try:
        _write_run_config(config, config_json_path, engine_player.engine_name)
        print(f"Run {config.run_id}: {config.num_games} games, model={config.model}")

        for game_index in range(config.num_games):
            print(f"\n=== Game {game_index + 1}/{config.num_games} ===")
            play_baseline_game(
                config, llm_player, engine_player, game_index, moves_csv_path, games_csv_path
            )
    finally:
        engine_player.close()


if __name__ == "__main__":
    # This is the single place to configure a baseline run. `model` has no default on
    # purpose -- set it to whatever chat model you want to evaluate.
    config = BaselineConfig(
        model="gpt-5.6-luna",
        base_url="https://tritonai-api.ucsd.edu/v1",
        stockfish_path=r"C:\Users\thelo\Downloads\Misc\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe",
        # gemini-3.6-flash spends part of its token budget on hidden reasoning before
        # emitting the move, so the paper's max_tokens=10 (fine for classic chat
        # models) starves it into an empty response. 300 leaves enough headroom.
        max_tokens=4000,
        num_games=1,
    )
    run_baseline_experiment(config)
