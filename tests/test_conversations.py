import pytest

from app.conversations import ConversationStore, PromptTooLong


class WordTokenizer:
    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True
    ):
        return " ".join(message["content"] for message in messages)

    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_sessions_are_isolated_and_failed_turns_are_not_precommitted():
    store = ConversationStore(WordTokenizer(), input_budget=20)

    messages, trimmed = store.build_messages("alpha", "first question")

    assert messages == [{"role": "user", "content": "first question"}]
    assert trimmed is False
    assert store.snapshot("alpha") == []

    store.commit("alpha", "first question", "first answer")
    assert store.snapshot("beta") == []
    assert store.snapshot("alpha") == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]


def test_clear_reports_whether_session_existed():
    store = ConversationStore(WordTokenizer(), input_budget=20)
    store.commit("alpha", "question", "answer")

    assert store.clear("alpha") is True
    assert store.clear("alpha") is False


def test_oldest_complete_turns_are_trimmed_first():
    store = ConversationStore(WordTokenizer(), input_budget=8)
    store.commit("alpha", "old user words", "old assistant words")
    store.commit("alpha", "new user", "new answer")

    messages, trimmed = store.build_messages("alpha", "latest question")

    assert trimmed is True
    assert messages == [
        {"role": "user", "content": "new user"},
        {"role": "assistant", "content": "new answer"},
        {"role": "user", "content": "latest question"},
    ]


def test_newest_message_is_never_silently_truncated():
    store = ConversationStore(WordTokenizer(), input_budget=3)

    with pytest.raises(PromptTooLong):
        store.build_messages("alpha", "this prompt has four words")
