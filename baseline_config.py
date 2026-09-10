import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class BaselineConfig:
    """Every knob for a baseline-replication run, in one place. `model` has no default
    on purpose -- gpt-3.5-turbo-0301 (the paper's model) is retired, so the caller must
    make an explicit choice rather than silently inheriting a stale default."""

    model: str

    num_games: int = 10
    temperature: float = 1.0
    top_p: float = 0.9
    max_tokens: int = 10
    max_illegal_retries: int = 10

    system_message: Optional[str] = None

    # Leave both unset to use the real OpenAI API (OPENAI_API_KEY env var, then
    # gpt_inputs/api_key.txt). Set base_url to target any other OpenAI-compatible
    # endpoint instead -- e.g. Gemini's OpenAI-compatibility layer:
    #   base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    #   model="gemini-2.5-flash", api_key="<your Gemini API key>"
    base_url: Optional[str] = None
    api_key: Optional[str] = None

    stockfish_path: Optional[str] = None
    engine_depth: int = 15
    engine_skill_level: Optional[int] = None
    engine_multipv: int = 3

    opening_moves: tuple[str, ...] = ("e4", "d4", "Nf3", "e3")

    max_plies: int = 200

    log_dir: str = "logs/baseline"
    prompt_path: str = "gpt_inputs/baseline_prompt.txt"

    run_id: Optional[str] = None

    def __post_init__(self):
        if not self.model:
            raise ValueError("BaselineConfig.model must be set explicitly")
        if self.run_id is None:
            self.run_id = f"{self.model_slug()}_{int(time.time())}"

    def model_slug(self) -> str:
        return self.model.replace(".", "_").replace("/", "_")
