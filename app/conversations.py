"""Thread-safe, tokenizer-aware in-memory conversation storage."""

from copy import deepcopy
from threading import Lock


class PromptTooLong(ValueError):
    """Raised when the newest user message cannot fit the input budget."""


class ConversationStore:
    def __init__(self, tokenizer, input_budget=320):
        self.tokenizer = tokenizer
        self.input_budget = input_budget
        self._sessions = {}
        self._lock = Lock()

    def _token_count(self, messages):
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return len(self.tokenizer.encode(rendered, add_special_tokens=False))

    def build_messages(self, session_id, user_message):
        newest = [{"role": "user", "content": user_message}]
        if self._token_count(newest) > self.input_budget:
            raise PromptTooLong("The newest message exceeds the model input budget.")

        with self._lock:
            history = deepcopy(self._sessions.get(session_id, []))

        messages = history + newest
        trimmed = False
        while self._token_count(messages) > self.input_budget and len(messages) > 1:
            messages = messages[2:] if len(messages) >= 3 else newest
            trimmed = True
        return messages, trimmed

    def commit(self, session_id, user_message, assistant_message):
        turn = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
        with self._lock:
            self._sessions.setdefault(session_id, []).extend(turn)

    def snapshot(self, session_id):
        with self._lock:
            return deepcopy(self._sessions.get(session_id, []))

    def clear(self, session_id):
        with self._lock:
            return self._sessions.pop(session_id, None) is not None
