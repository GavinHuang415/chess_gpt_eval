from dataclasses import dataclass, field


@dataclass
class Conversation:
    """Multi-turn chat history for a single game. Owned by the game loop, never by a
    Player, since Player instances are reused across many games."""

    messages: list[dict] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def snapshot(self) -> list[dict]:
        """A shallow copy, so callers (e.g. the OpenAI client wrapper) can never mutate
        game state by appending to the list they were handed."""
        return list(self.messages)

    def to_transcript_text(self) -> str:
        lines = []
        for message in self.messages:
            lines.append(f"{message['role']}: {message['content']}")
        return "\n".join(lines)
