from threading import Event
from pathlib import Path
import subprocess
import sys

import pytest

from app.model_runtime import (
    GenerationFailed,
    InvalidModelSelection,
    ModelBusy,
    ModelNotReady,
    ModelRuntime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TinyTokenizer:
    pad_token_id = 0

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True, **kwargs
    ):
        if tokenize:
            raise AssertionError("Tests use rendered text only")
        return " | ".join(message["content"] for message in messages)

    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_successful_load_transitions_to_ready():
    runtime = ModelRuntime(loader=lambda: (object(), TinyTokenizer()))

    assert runtime.public_status()["state"] == "unavailable"
    assert runtime.start_loading() is True
    runtime.wait_for_loading(timeout=1)

    assert runtime.public_status() == {
        "state": "ready",
        "message": "Local model ready",
        "model": "finetuned",
    }
    assert runtime.start_loading() is False


def test_duplicate_load_attempt_is_rejected_while_loading():
    release = Event()

    def delayed_loader():
        release.wait(timeout=1)
        return object(), TinyTokenizer()

    runtime = ModelRuntime(loader=delayed_loader)

    assert runtime.start_loading() is True
    assert runtime.start_loading() is False
    release.set()
    runtime.wait_for_loading(timeout=1)


def test_failed_load_exposes_safe_category_not_exception_details():
    def fail():
        raise FileNotFoundError("C:/private/cache/secret-path")

    runtime = ModelRuntime(loader=fail)
    runtime.start_loading()
    runtime.wait_for_loading(timeout=1)

    status = runtime.public_status()
    assert status["state"] == "unavailable"
    assert status["message"] == "Fine-tuned adapter is unavailable."
    assert "private" not in status["message"]


def test_chat_commits_only_after_successful_generation():
    runtime = ModelRuntime(
        loader=lambda: (object(), TinyTokenizer()),
        generator=lambda model, tokenizer, messages, max_new_tokens, model_mode: "clear answer",
    )
    runtime.start_loading()
    runtime.wait_for_loading(timeout=1)

    result = runtime.chat("alpha", "question")

    assert result == {"response": "clear answer", "context_trimmed": False}
    assert runtime.conversations.snapshot("alpha") == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "clear answer"},
    ]


def test_failed_generation_does_not_change_history():
    def failing_generator(model, tokenizer, messages, max_new_tokens, model_mode):
        raise RuntimeError("private generation detail")

    runtime = ModelRuntime(
        loader=lambda: (object(), TinyTokenizer()),
        generator=failing_generator,
    )
    runtime.start_loading()
    runtime.wait_for_loading(timeout=1)

    with pytest.raises(GenerationFailed):
        runtime.chat("alpha", "question")

    assert runtime.conversations.snapshot("alpha") == []


def test_chat_rejects_when_model_is_not_ready():
    runtime = ModelRuntime(loader=lambda: (object(), TinyTokenizer()))

    with pytest.raises(ModelNotReady):
        runtime.chat("alpha", "question")


def test_chat_fails_fast_when_inference_lock_is_held():
    runtime = ModelRuntime(loader=lambda: (object(), TinyTokenizer()))
    runtime.start_loading()
    runtime.wait_for_loading(timeout=1)
    assert runtime._inference_lock.acquire(blocking=False)

    try:
        with pytest.raises(ModelBusy):
            runtime.chat("alpha", "question")
    finally:
        runtime._inference_lock.release()


def test_model_runtime_imports_when_app_is_executed_as_a_script():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'app'); import model_runtime; "
            "print(model_runtime.ModelRuntime.__name__)",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ModelRuntime"


def test_switching_model_clears_all_conversation_context():
    runtime = ModelRuntime(
        loader=lambda: (object(), TinyTokenizer()),
        generator=lambda model, tokenizer, messages, max_new_tokens, model_mode: "answer",
    )
    runtime.start_loading()
    runtime.wait_for_loading(timeout=1)
    runtime.chat("alpha", "question")

    changed = runtime.select_model("baseline")

    assert changed is True
    assert runtime.public_status()["model"] == "baseline"
    assert runtime.conversations.snapshot("alpha") == []


def test_generation_receives_selected_model_mode():
    observed_modes = []

    def generator(model, tokenizer, messages, max_new_tokens, model_mode):
        observed_modes.append(model_mode)
        return "answer"

    runtime = ModelRuntime(
        loader=lambda: (object(), TinyTokenizer()),
        generator=generator,
    )
    runtime.start_loading()
    runtime.wait_for_loading(timeout=1)
    runtime.select_model("baseline")

    runtime.chat("alpha", "question")

    assert observed_modes == ["baseline"]


def test_select_model_rejects_unknown_name():
    runtime = ModelRuntime(loader=lambda: (object(), TinyTokenizer()))

    with pytest.raises(InvalidModelSelection):
        runtime.select_model("other")
