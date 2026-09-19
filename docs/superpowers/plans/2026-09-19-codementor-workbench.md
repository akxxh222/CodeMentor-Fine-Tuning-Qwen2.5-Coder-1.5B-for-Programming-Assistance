# CodeMentor Workbench Implementation Plan

> **Status:** Implemented. This file is retained as the historical build plan. The current application is documented in `README.md`; the repository is now a Git repository and the active adapter is V2 at `models/adapter`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a professional Flask-based CodeMentor workbench that starts independently of the model, reports model availability honestly, and supports bounded in-memory conversational inference with the fine-tuned Qwen adapter.

**Architecture:** Flask serves a local HTML/CSS/JavaScript workbench and delegates all GPU concerns to a singleton `ModelRuntime`. A thread-safe `ConversationStore` owns session history and token-budget trimming; Flask routes validate requests and map runtime failures to stable JSON responses. The Unsloth import remains lazy so backend tests run without CUDA.

**Tech Stack:** Python 3.12, Flask, PyTorch, Unsloth, Transformers tokenizer chat templates, pytest, plain HTML/CSS/JavaScript

**Spec:** `docs/superpowers/specs/2026-09-19-codementor-workbench-design.md`

## Global Constraints

- Flask must serve the interface even when CUDA, the base model, or `models/adapter` is unavailable.
- Model state values are exactly `loading`, `ready`, and `unavailable`.
- The local fine-tuned adapter path is `models/adapter`; no remote or paid inference service is introduced.
- Total model sequence length is 512 tokens; reserve 192 output tokens and enforce a 320-token rendered input budget.
- Conversation contents remain in process memory and are never written to disk.
- Only one GPU generation runs at a time; concurrent generation receives HTTP 409.
- Model output is escaped before limited fenced-code rendering; raw generated HTML is never inserted.
- Frontend assets are local and use no CDN, account, analytics, or telemetry dependency.
- The workspace is not currently a Git repository, so task-end checkpoints replace commit steps. If Git is initialized before execution, commit each completed task using the listed suggested message.

## File structure

- Create `app/__init__.py`: package marker and public app factory export.
- Create `app/conversations.py`: thread-safe session storage and tokenizer-aware context fitting.
- Create `app/model_runtime.py`: background model loading, public state, retry, inference locking, and generation.
- Replace `app/app.py`: Flask application factory, routes, validation, and local entry point.
- Create `app/templates/index.html`: semantic Mentor Workbench structure.
- Create `app/static/styles.css`: complete responsive visual system and state styles.
- Create `app/static/app.js`: sessions, polling, submissions, safe response rendering, retry, and copy behavior.
- Create `tests/test_conversations.py`: history isolation and token trimming.
- Create `tests/test_model_runtime.py`: lifecycle, retry, busy handling, and safe error categories.
- Create `tests/test_app.py`: Flask route contracts using a fake runtime.
- Create `tests/test_frontend_contract.py`: template and local-asset safety contract.
- Modify `requirements.txt`: add Flask runtime dependency.
- Modify `README.md`: launch, model-state behavior, and test instructions.

---

### Task 1: Thread-safe conversational context

**Files:**
- Create: `app/__init__.py`
- Create: `app/conversations.py`
- Test: `tests/test_conversations.py`

**Interfaces:**
- Produces: `ConversationStore(tokenizer, input_budget: int = 320)`.
- Produces: `ConversationStore.build_messages(session_id: str, user_message: str) -> tuple[list[dict[str, str]], bool]`.
- Produces: `ConversationStore.commit(session_id: str, user_message: str, assistant_message: str) -> None`.
- Produces: `ConversationStore.clear(session_id: str) -> bool`.
- Raises: `PromptTooLong` when the newest message alone exceeds the rendered input budget.

- [ ] **Step 1: Write failing isolation and commit tests**

```python
# tests/test_conversations.py
import pytest

from app.conversations import ConversationStore, PromptTooLong


class WordTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
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
    assert store.snapshot("alpha")[-1]["content"] == "first answer"


def test_clear_reports_whether_session_existed():
    store = ConversationStore(WordTokenizer(), input_budget=20)
    store.commit("alpha", "question", "answer")
    assert store.clear("alpha") is True
    assert store.clear("alpha") is False
```

- [ ] **Step 2: Run the new tests and verify the import fails**

Run: `python -m pytest tests/test_conversations.py -v`

Expected: FAIL because `app.conversations` does not exist.

- [ ] **Step 3: Write failing token-budget tests**

```python
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
```

- [ ] **Step 4: Implement minimal conversation storage and fitting**

```python
# app/conversations.py
from copy import deepcopy
from threading import Lock


class PromptTooLong(ValueError):
    pass


class ConversationStore:
    def __init__(self, tokenizer, input_budget=320):
        self.tokenizer = tokenizer
        self.input_budget = input_budget
        self._sessions = {}
        self._lock = Lock()

    def _token_count(self, messages):
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
```

Create an empty `app/__init__.py` so tests can import the package.

- [ ] **Step 5: Run conversation tests**

Run: `python -m pytest tests/test_conversations.py -v`

Expected: all tests PASS.

- [ ] **Step 6: Record the task checkpoint**

Run: `python -m pytest tests/test_conversations.py -q`

Expected: exit code 0. Suggested commit if Git is available: `feat: add bounded conversation store`.

---

### Task 2: Model lifecycle and locked inference

**Files:**
- Create: `app/model_runtime.py`
- Test: `tests/test_model_runtime.py`

**Interfaces:**
- Consumes: `ConversationStore` and `PromptTooLong` from Task 1.
- Produces: `ModelRuntime(loader=load_default_model, max_new_tokens: int = 192)`.
- Produces: `ModelRuntime.start_loading() -> bool`, `public_status() -> dict[str, str]`, `chat(session_id: str, message: str) -> dict` and `clear_session(session_id: str) -> bool`.
- Raises: `ModelNotReady`, `ModelBusy`, and `GenerationFailed` for route-level mapping.

- [ ] **Step 1: Write failing lifecycle tests with an injected loader**

```python
# tests/test_model_runtime.py
from threading import Event

import pytest

from app.model_runtime import ModelBusy, ModelNotReady, ModelRuntime


class TinyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        if tokenize:
            raise AssertionError("tests use rendered text")
        return " | ".join(message["content"] for message in messages)

    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_successful_load_transitions_to_ready():
    runtime = ModelRuntime(loader=lambda: (object(), TinyTokenizer()))
    assert runtime.public_status()["state"] == "unavailable"
    assert runtime.start_loading() is True
    runtime.wait_for_loading(timeout=1)
    assert runtime.public_status() == {"state": "ready", "message": "Local model ready"}


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
```

- [ ] **Step 2: Run lifecycle tests and verify they fail**

Run: `python -m pytest tests/test_model_runtime.py -v`

Expected: FAIL because `app.model_runtime` does not exist.

- [ ] **Step 3: Add failing generation and busy tests**

```python
def test_chat_commits_only_after_successful_generation():
    runtime = ModelRuntime(
        loader=lambda: (object(), TinyTokenizer()),
        generator=lambda model, tokenizer, messages, max_new_tokens: "clear answer",
    )
    runtime.start_loading()
    runtime.wait_for_loading(timeout=1)
    result = runtime.chat("alpha", "question")
    assert result == {"response": "clear answer", "context_trimmed": False}
    assert runtime.conversations.snapshot("alpha")[-1]["content"] == "clear answer"


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
```

- [ ] **Step 4: Implement the runtime with lazy Unsloth loading**

```python
# app/model_runtime.py
import logging
from pathlib import Path
from threading import Lock, Thread

import torch

from .conversations import ConversationStore

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = PROJECT_ROOT / "models" / "adapter"
MAX_SEQ_LENGTH = 512


class ModelNotReady(RuntimeError):
    pass


class ModelBusy(RuntimeError):
    pass


class GenerationFailed(RuntimeError):
    pass


def load_default_model():
    if not torch.cuda.is_available():
        raise RuntimeError("cuda-unavailable")
    if not (ADAPTER_PATH / "adapter_config.json").is_file():
        raise FileNotFoundError(str(ADAPTER_PATH))
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(ADAPTER_PATH),
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_default(model, tokenizer, messages, max_new_tokens):
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    inputs = {key: value.to("cuda") for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    input_length = inputs["input_ids"].shape[-1]
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()


class ModelRuntime:
    def __init__(self, loader=load_default_model, generator=generate_default, max_new_tokens=192):
        self.loader = loader
        self.generator = generator
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        self.conversations = None
        self._state = "unavailable"
        self._message = "Local model has not been loaded."
        self._state_lock = Lock()
        self._inference_lock = Lock()
        self._loading_thread = None

    def public_status(self):
        with self._state_lock:
            return {"state": self._state, "message": self._message}

    def start_loading(self):
        with self._state_lock:
            if self._state in {"loading", "ready"}:
                return False
            self._state = "loading"
            self._message = "Loading local model"
            self._loading_thread = Thread(target=self._load, daemon=True)
            self._loading_thread.start()
            return True

    def _load(self):
        try:
            model, tokenizer = self.loader()
            conversations = ConversationStore(tokenizer, input_budget=320)
        except Exception as error:
            LOGGER.exception("Model loading failed")
            if isinstance(error, FileNotFoundError):
                message = "Fine-tuned adapter is unavailable."
            elif str(error) == "cuda-unavailable":
                message = "CUDA is unavailable on this computer."
            elif "out of memory" in str(error).lower():
                message = "There is not enough GPU memory to load the model."
            else:
                message = "The local model could not be loaded."
            with self._state_lock:
                self._state, self._message = "unavailable", message
            return
        with self._state_lock:
            self.model, self.tokenizer = model, tokenizer
            self.conversations = conversations
            self._state, self._message = "ready", "Local model ready"

    def wait_for_loading(self, timeout=None):
        thread = self._loading_thread
        if thread:
            thread.join(timeout)

    def chat(self, session_id, message):
        if self.public_status()["state"] != "ready":
            raise ModelNotReady()
        if not self._inference_lock.acquire(blocking=False):
            raise ModelBusy()
        try:
            messages, trimmed = self.conversations.build_messages(session_id, message)
            try:
                response = self.generator(
                    self.model, self.tokenizer, messages, self.max_new_tokens
                )
            except Exception as error:
                LOGGER.exception("Generation failed")
                raise GenerationFailed() from error
            if not response:
                raise GenerationFailed()
            self.conversations.commit(session_id, message, response)
            return {"response": response, "context_trimmed": trimmed}
        finally:
            self._inference_lock.release()

    def clear_session(self, session_id):
        if self.conversations is None:
            return False
        return self.conversations.clear(session_id)
```

- [ ] **Step 5: Run runtime tests**

Run: `python -m pytest tests/test_model_runtime.py -v`

Expected: all tests PASS without importing Unsloth or requiring CUDA.

- [ ] **Step 6: Run the conversation and runtime suites together**

Run: `python -m pytest tests/test_conversations.py tests/test_model_runtime.py -q`

Expected: exit code 0. Suggested commit if Git is available: `feat: add asynchronous model runtime`.

---

### Task 3: Flask application and stable API contracts

**Files:**
- Replace: `app/app.py`
- Modify: `app/__init__.py`
- Create: `app/templates/index.html` (minimal route fixture; expanded in Task 4)
- Create: `tests/test_app.py`

**Interfaces:**
- Consumes: `ModelRuntime`, `ModelNotReady`, `ModelBusy`, `GenerationFailed`, and `PromptTooLong`.
- Produces: `create_app(runtime=None, start_model=True) -> flask.Flask`.
- Produces: `GET /`, `GET /api/status`, `POST /api/model/retry`, `POST /api/chat`, and `POST /api/sessions/<session_id>/clear`.

- [ ] **Step 1: Write failing status and page tests**

```python
# tests/test_app.py
from app.app import create_app


class FakeRuntime:
    def __init__(self, state="ready"):
        self.state = state
        self.started = False

    def start_loading(self):
        self.started = True
        return True

    def public_status(self):
        return {"state": self.state, "message": f"Model {self.state}"}

    def chat(self, session_id, message):
        return {"response": f"Answer: {message}", "context_trimmed": False}

    def clear_session(self, session_id):
        return session_id == "known"


def make_client(runtime=None):
    app = create_app(runtime or FakeRuntime(), start_model=False)
    app.config.update(TESTING=True)
    return app.test_client()


def test_index_is_available_independently_of_model_state():
    response = make_client(FakeRuntime("unavailable")).get("/")
    assert response.status_code == 200
    assert b"CodeMentor" in response.data


def test_status_contract():
    response = make_client(FakeRuntime("loading")).get("/api/status")
    assert response.get_json() == {"state": "loading", "message": "Model loading"}
```

- [ ] **Step 2: Run the route tests and verify they fail**

Run: `python -m pytest tests/test_app.py -v`

Expected: FAIL because `create_app` and the template do not exist.

- [ ] **Step 3: Add failing validation and error-mapping tests**

```python
from app.conversations import PromptTooLong
from app.model_runtime import GenerationFailed, ModelBusy, ModelNotReady


def test_chat_validates_json_fields():
    client = make_client()
    assert client.post("/api/chat", json={}).status_code == 400
    assert client.post("/api/chat", json={"session_id": "ok", "message": "   "}).status_code == 400
    assert client.post("/api/chat", json={"session_id": "bad space", "message": "hi"}).status_code == 400


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [(ModelNotReady(), 503), (ModelBusy(), 409), (PromptTooLong(), 400), (GenerationFailed(), 500)],
)
def test_chat_maps_runtime_errors(error, expected_status):
    runtime = FakeRuntime()
    runtime.chat = lambda session_id, message: (_ for _ in ()).throw(error)
    response = make_client(runtime).post(
        "/api/chat", json={"session_id": "session-1", "message": "hello"}
    )
    assert response.status_code == expected_status
    assert set(response.get_json()) == {"error", "code"}


def test_successful_chat_and_clear_contracts():
    client = make_client()
    response = client.post(
        "/api/chat", json={"session_id": "known", "message": "hello"}
    )
    assert response.get_json()["response"] == "Answer: hello"
    assert client.post("/api/sessions/known/clear").status_code == 204
```

- [ ] **Step 4: Implement the Flask factory and endpoints**

```python
# app/app.py
import re

from flask import Flask, jsonify, render_template, request

try:
    from .conversations import PromptTooLong
    from .model_runtime import GenerationFailed, ModelBusy, ModelNotReady, ModelRuntime
except ImportError:
    from conversations import PromptTooLong
    from model_runtime import GenerationFailed, ModelBusy, ModelNotReady, ModelRuntime

SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


def create_app(runtime=None, start_model=True):
    app = Flask(__name__)
    model_runtime = runtime or ModelRuntime()
    app.extensions["model_runtime"] = model_runtime

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/status")
    def status():
        return jsonify(model_runtime.public_status())

    @app.post("/api/model/retry")
    def retry_model():
        model_runtime.start_loading()
        return jsonify(model_runtime.public_status()), 202

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="JSON request required", code="invalid_request"), 400
        session_id = payload.get("session_id")
        message = payload.get("message")
        if not isinstance(session_id, str) or not SESSION_ID.fullmatch(session_id):
            return jsonify(error="Invalid session identifier", code="invalid_session"), 400
        if not isinstance(message, str) or not message.strip():
            return jsonify(error="Enter a programming question", code="invalid_message"), 400
        try:
            return jsonify(model_runtime.chat(session_id, message.strip()))
        except PromptTooLong:
            return jsonify(error="Your message is too long for this model.", code="prompt_too_long"), 400
        except ModelBusy:
            return jsonify(error="CodeMentor is answering another request.", code="model_busy"), 409
        except ModelNotReady:
            return jsonify(error="The local model is not ready.", code="model_unavailable"), 503
        except GenerationFailed:
            return jsonify(error="Response generation failed. Try again.", code="generation_failed"), 500

    @app.post("/api/sessions/<session_id>/clear")
    def clear_session(session_id):
        if not SESSION_ID.fullmatch(session_id):
            return jsonify(error="Invalid session identifier", code="invalid_session"), 400
        model_runtime.clear_session(session_id)
        return "", 204

    if start_model:
        model_runtime.start_loading()
    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
```

Export `create_app` from `app/__init__.py`.

Create the minimal template required to prove that Flask remains available independently of model state:

```html
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>CodeMentor</title></head>
<body><main><h1>CodeMentor</h1><p>Local programming assistant</p></main></body></html>
```

- [ ] **Step 5: Run API tests**

Run: `python -m pytest tests/test_app.py -v`

Expected: all tests PASS with the minimal `index.html`; Task 4 replaces it with the approved workbench.

- [ ] **Step 6: Run all backend tests**

Run: `python -m pytest tests/test_conversations.py tests/test_model_runtime.py tests/test_app.py -q`

Expected: exit code 0. Suggested commit if Git is available: `feat: expose CodeMentor Flask API`.

---

### Task 4: Professional Mentor Workbench markup and styling

**Files:**
- Replace: `app/templates/index.html`
- Create: `app/static/styles.css`
- Create: `tests/test_frontend_contract.py`

**Interfaces:**
- Produces stable element IDs consumed by Task 5: `model-status`, `model-status-message`, `retry-model`, `new-session`, `session-list`, `thread`, `empty-state`, `chat-form`, `prompt`, `send-button`, and `context-notice`.
- Produces CSS state hooks: `body[data-model-state]`, `.message--user`, `.message--assistant`, `.code-block`, `.is-busy`, and `.is-collapsed`.

- [ ] **Step 1: Write failing frontend contract tests**

```python
# tests/test_frontend_contract.py
from pathlib import Path

from app.app import create_app


ROOT = Path(__file__).resolve().parents[1]


class FrontendRuntime:
    def public_status(self):
        return {"state": "unavailable", "message": "Model unavailable"}

    def start_loading(self):
        return True

    def clear_session(self, session_id):
        return False


def test_workbench_contains_required_controls_and_local_assets():
    app = create_app(FrontendRuntime(), start_model=False)
    app.config.update(TESTING=True)
    html = app.test_client().get("/").get_data(as_text=True)
    for element_id in (
        "model-status", "retry-model", "new-session", "session-list",
        "thread", "chat-form", "prompt", "send-button", "context-notice",
    ):
        assert f'id="{element_id}"' in html
    assert 'href="/static/styles.css"' in html
    assert 'src="/static/app.js"' in html
    assert "http://" not in html and "https://" not in html


def test_styles_include_responsive_and_accessibility_contracts():
    css = (ROOT / "app/static/styles.css").read_text(encoding="utf-8")
    assert "@media (max-width:" in css
    assert "prefers-reduced-motion" in css
    assert ":focus-visible" in css
```

- [ ] **Step 2: Run contract tests and verify they fail**

Run: `python -m pytest tests/test_frontend_contract.py -v`

Expected: FAIL because the detailed template and stylesheet do not exist.

- [ ] **Step 3: Implement semantic workbench markup**

Build `index.html` with:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CodeMentor</title>
  <link rel="stylesheet" href="/static/styles.css">
</head>
<body data-model-state="loading">
  <div class="app-shell">
    <aside class="sidebar" aria-label="Conversation sessions">
      <a class="brand" href="/" aria-label="CodeMentor home"><span>CM</span>CodeMentor</a>
      <button id="new-session" class="new-session" type="button">New session</button>
      <p class="sidebar-label">Sessions</p>
      <nav id="session-list" class="session-list"></nav>
      <div class="sidebar-status"><i></i><span id="model-status-message">Loading local model</span></div>
    </aside>
    <main class="workspace">
      <header class="workspace-header">
        <div><h1 id="session-title">New mentoring session</h1><p>Context from this session only</p></div>
        <div id="model-status" class="model-status" role="status" aria-live="polite"><i></i><span>Loading</span></div>
      </header>
      <section id="thread" class="thread" aria-live="polite">
        <div id="empty-state" class="empty-state"><p class="eyebrow">Local programming assistant</p><h2>What are you working through?</h2><p>Ask for an explanation, paste code, or debug an error.</p></div>
      </section>
      <p id="context-notice" class="context-notice" hidden>Older messages were omitted to fit the model context.</p>
      <div class="model-unavailable" aria-live="assertive"><p>The local model is unavailable.</p><button id="retry-model" type="button">Retry loading</button></div>
      <form id="chat-form" class="composer"><label class="sr-only" for="prompt">Programming question</label><textarea id="prompt" rows="1" placeholder="Ask a follow-up or paste code…" disabled></textarea><div><span>Shift + Enter for a new line</span><button id="send-button" type="submit" disabled aria-label="Send question">↑</button></div></form>
    </main>
  </div>
  <script src="/static/app.js" defer></script>
</body>
</html>
```

- [ ] **Step 4: Implement the approved visual system**

In `styles.css`, define CSS custom properties for graphite, paper, royal blue, state colors, border, shadow, sans, and monospace. Implement the two-column shell, compact rail, readable thread width, user/assistant message hierarchy, code headers, model-state visibility, textarea growth, keyboard focus, `max-width: 760px` mobile collapse, and `prefers-reduced-motion: reduce`. Do not add gradients, glass blur, illustrations, metric cards, or marketing sections.

Start from this concrete state and layout contract, then fill every referenced class from the Task 4 markup:

```css
:root {
  --graphite: #171a1f; --paper: #f8f8f6; --surface: #ffffff;
  --ink: #181b20; --muted: #747c88; --line: #dfe1e4;
  --blue: #315bea; --green: #2fc777; --amber: #d89614; --red: #d14343;
  --sans: Inter, ui-sans-serif, system-ui, sans-serif;
  --mono: "SFMono-Regular", Consolas, monospace;
}
* { box-sizing: border-box; }
body { margin: 0; min-height: 100vh; background: var(--paper); color: var(--ink); font-family: var(--sans); }
.app-shell { min-height: 100vh; display: grid; grid-template-columns: 220px minmax(0, 1fr); }
.sidebar { display: flex; flex-direction: column; padding: 20px 14px; background: var(--graphite); color: #c5cad1; }
.workspace { min-width: 0; min-height: 100vh; display: flex; flex-direction: column; }
.thread { flex: 1; overflow-y: auto; padding: 32px max(28px, 8%); }
.message--user { width: fit-content; max-width: 70%; margin-left: auto; padding: 11px 13px; border-radius: 12px 12px 3px 12px; background: #e6ebff; }
.message--assistant { max-width: 760px; margin-top: 24px; line-height: 1.65; }
.code-block { overflow: hidden; border: 1px solid #2b3038; border-radius: 10px; background: #15181d; color: #dce5f3; font-family: var(--mono); }
body[data-model-state="ready"] .model-status i { background: var(--green); }
body[data-model-state="loading"] .model-status i { background: var(--amber); }
body[data-model-state="unavailable"] .model-status i { background: var(--red); }
body:not([data-model-state="unavailable"]) .model-unavailable { display: none; }
:focus-visible { outline: 3px solid color-mix(in srgb, var(--blue) 35%, transparent); outline-offset: 2px; }
@media (max-width: 760px) { .app-shell { grid-template-columns: 68px minmax(0, 1fr); } .sidebar-label, .session-list, .sidebar-status span { display: none; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; animation: none !important; } }
```

- [ ] **Step 5: Run frontend contract and API tests**

Run: `python -m pytest tests/test_frontend_contract.py tests/test_app.py -q`

Expected: exit code 0.

- [ ] **Step 6: Record the visual checkpoint**

Open the fake-runtime Flask app at desktop and mobile widths and compare it with the approved Mentor Workbench mockup. Suggested commit if Git is available: `feat: build Mentor Workbench interface`.

---

### Task 5: Browser state, sessions, and safe response rendering

**Files:**
- Create: `app/static/app.js`
- Modify: `tests/test_frontend_contract.py`

**Interfaces:**
- Consumes the IDs and CSS hooks from Task 4.
- Consumes JSON from `/api/status`, `/api/model/retry`, `/api/chat`, and `/api/sessions/<id>/clear`.
- Stores only `codementor.sessions` and `codementor.activeSession` metadata in `localStorage`.

- [ ] **Step 1: Add failing static safety-contract tests**

```python
def test_javascript_uses_text_content_and_never_inserts_model_html():
    script = (ROOT / "app/static/app.js").read_text(encoding="utf-8")
    assert "textContent" in script
    assert ".innerHTML" not in script
    assert "localStorage" in script
    assert "setInterval(pollStatus" in script
    assert "navigator.clipboard.writeText" in script
```

- [ ] **Step 2: Run the safety contract and verify it fails**

Run: `python -m pytest tests/test_frontend_contract.py::test_javascript_uses_text_content_and_never_inserts_model_html -v`

Expected: FAIL until the application script is implemented.

- [ ] **Step 3: Implement session metadata and model polling**

Use `crypto.randomUUID()` with a random fallback to create session IDs, store `{id, title, createdAt}` only, activate sessions through event listeners, and create one session on first visit. Implement `pollStatus()` every 1,500 ms, update `body.dataset.modelState`, enable the composer only for `ready`, and stop rapid polling once ready while checking again after any request failure.

Never persist conversation messages in local storage.

```javascript
const state = { sessions: [], activeId: null, messages: new Map(), busy: false };
const prompt = document.querySelector("#prompt");
const sendButton = document.querySelector("#send-button");

function newId() {
  return crypto.randomUUID ? crypto.randomUUID() : `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function saveSessionMetadata() {
  localStorage.setItem("codementor.sessions", JSON.stringify(state.sessions));
  localStorage.setItem("codementor.activeSession", state.activeId);
}

async function pollStatus() {
  const response = await fetch("/api/status");
  const status = await response.json();
  document.body.dataset.modelState = status.state;
  document.querySelector("#model-status-message").textContent = status.message;
  document.querySelector("#model-status span").textContent = status.state;
  const enabled = status.state === "ready" && !state.busy;
  prompt.disabled = !enabled;
  sendButton.disabled = !enabled;
}

setInterval(pollStatus, 1500);
pollStatus();
```

- [ ] **Step 4: Implement safe mixed prose/code rendering**

Split model text on triple-backtick fences. For prose segments, create paragraph nodes and assign only `textContent`. For code segments, create `<pre><code>` nodes, a language label, and a copy button whose handler calls `navigator.clipboard.writeText(codeText)`. Do not use `innerHTML`, DOM parsers, third-party Markdown, or syntax-highlighting CDNs.

```javascript
function renderResponse(text) {
  const fragment = document.createDocumentFragment();
  text.split("```").forEach((segment, index) => {
    if (index % 2 === 0) {
      segment.split(/\n{2,}/).filter(Boolean).forEach((value) => {
        const paragraph = document.createElement("p");
        paragraph.textContent = value.trim();
        fragment.append(paragraph);
      });
      return;
    }
    const [firstLine, ...rest] = segment.replace(/^\n/, "").split("\n");
    const hasLanguage = /^[A-Za-z0-9_+#.-]+$/.test(firstLine.trim());
    const codeText = (hasLanguage ? rest.join("\n") : [firstLine, ...rest].join("\n")).trim();
    const wrapper = document.createElement("div");
    wrapper.className = "code-block";
    const copy = document.createElement("button");
    copy.type = "button";
    copy.textContent = "Copy";
    copy.addEventListener("click", () => navigator.clipboard.writeText(codeText));
    const pre = document.createElement("pre");
    const code = document.createElement("code");
    code.textContent = codeText;
    pre.append(code);
    wrapper.append(copy, pre);
    fragment.append(wrapper);
  });
  return fragment;
}
```

- [ ] **Step 5: Implement submission and recovery behavior**

Handle Enter versus Shift+Enter, optimistic user bubbles, one in-flight request, a compact assistant progress row, `context_trimmed`, HTTP 400/409/503/500 messages, retryable prompt preservation, New session, session switching, textarea auto-resize, and model retry. When a browser session is revisited after Flask lost its history, show an empty thread with the stored title rather than replaying messages to the server.

```javascript
document.querySelector("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = prompt.value.trim();
  if (!message || state.busy) return;
  state.busy = true;
  prompt.disabled = sendButton.disabled = true;
  appendUserMessage(message);
  const pending = appendPendingAssistant();
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.activeId, message }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Request failed");
    pending.replaceChildren(renderResponse(payload.response));
    document.querySelector("#context-notice").hidden = !payload.context_trimmed;
    prompt.value = "";
  } catch (error) {
    pending.textContent = error.message;
    prompt.value = message;
  } finally {
    state.busy = false;
    await pollStatus();
  }
});

prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    document.querySelector("#chat-form").requestSubmit();
  }
});
```

- [ ] **Step 6: Run frontend contracts**

Run: `python -m pytest tests/test_frontend_contract.py -q`

Expected: all tests PASS.

- [ ] **Step 7: Exercise the browser behavior against a fake runtime**

Verify: unavailable state appears; Retry changes to loading; ready enables the composer; Enter sends; Shift+Enter creates a newline; a fake fenced-code response renders as text and code; copy works; New session isolates the thread; mobile sidebar collapses.

Suggested commit if Git is available: `feat: wire interactive mentoring sessions`.

---

### Task 6: Dependencies, launch documentation, and full verification

**Files:**
- Modify: `requirements.txt`
- Modify: `README.md`
- Modify: `.gitignore`

**Interfaces:**
- Produces documented commands for environment setup, Flask launch, testing, and optional real-model smoke testing.

- [ ] **Step 1: Add Flask to runtime requirements**

Add an explicit compatible Flask dependency to `requirements.txt`:

```text
Flask>=3.1,<4
```

Preserve every existing dependency line.

- [ ] **Step 2: Keep visual-companion artifacts out of source control**

Add this line to `.gitignore` without changing existing rules:

```text
.superpowers/
```

- [ ] **Step 3: Document launch and behavior**

Add a README section containing these commands:

```powershell
.\.venv\Scripts\Activate.ps1
python -m app.app
```

Document `http://127.0.0.1:5000`, the three model states, in-memory session reset on restart, adapter path `models/adapter`, CUDA requirement for actual responses, and the fact that Flask remains available when the model is unavailable.

- [ ] **Step 4: Run the complete automated suite**

Run: `python -m pytest tests -q`

Expected: all non-CUDA tests PASS; existing explicitly environment-dependent tests may remain SKIPPED with their recorded reason.

- [ ] **Step 5: Run a syntax check**

Run: `python -m compileall app src tests`

Expected: exit code 0 and no syntax errors.

- [ ] **Step 6: Run the fake-runtime browser smoke test**

Start the Flask app with an injected ready fake runtime, open `http://127.0.0.1:5000`, submit “Explain binary search”, and verify the returned prose and fenced code appear without console errors. Repeat at a narrow viewport.

- [ ] **Step 7: Run the real startup smoke test**

Run: `python -m app.app`

Expected: the workbench responds at `http://127.0.0.1:5000` immediately. The status progresses from loading to ready when CUDA and `models/adapter` are available, or to unavailable with a retry control otherwise. Stop after verifying one state transition; a full response generation is optional because it requires the local GPU.

- [ ] **Step 8: Record final evidence**

Record the pytest pass/skip counts, compile result, visible browser state, and real model-state transition in the task handoff. Suggested commit if Git is available: `docs: add CodeMentor workbench launch guide`.
