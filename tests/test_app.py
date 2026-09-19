import pytest

from app.app import create_app
from app.conversations import PromptTooLong
from app.model_runtime import (
    GenerationFailed,
    InvalidModelSelection,
    ModelBusy,
    ModelNotReady,
)


class FakeRuntime:
    def __init__(self, state="ready"):
        self.state = state
        self.started = False
        self.cleared = []
        self.selected = "finetuned"
        self.chat_error = None

    def start_loading(self):
        self.started = True
        self.state = "loading"
        return True

    def public_status(self):
        return {
            "state": self.state,
            "message": f"Model {self.state}",
            "model": self.selected,
        }

    def select_model(self, model_name):
        if model_name not in {"baseline", "finetuned"}:
            raise InvalidModelSelection()
        changed = model_name != self.selected
        self.selected = model_name
        return changed

    def chat(self, session_id, message):
        if self.chat_error:
            raise self.chat_error
        return {"response": f"Answer: {message}", "context_trimmed": False}

    def clear_session(self, session_id):
        self.cleared.append(session_id)
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

    assert response.get_json() == {
        "state": "loading",
        "message": "Model loading",
        "model": "finetuned",
    }


def test_retry_starts_loading_and_returns_public_state():
    runtime = FakeRuntime("unavailable")

    response = make_client(runtime).post("/api/model/retry")

    assert response.status_code == 202
    assert runtime.started is True
    assert response.get_json()["state"] == "loading"


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"session_id": "ok"},
        {"session_id": "bad space", "message": "hi"},
        {"session_id": "ok", "message": "   "},
    ],
)
def test_chat_validates_json_fields(payload):
    client = make_client()

    if payload is None:
        response = client.post("/api/chat", data="not-json")
    else:
        response = client.post("/api/chat", json=payload)

    assert response.status_code == 400
    assert set(response.get_json()) == {"error", "code"}


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_code"),
    [
        (ModelNotReady(), 503, "model_unavailable"),
        (ModelBusy(), 409, "model_busy"),
        (PromptTooLong(), 400, "prompt_too_long"),
        (GenerationFailed(), 500, "generation_failed"),
    ],
)
def test_chat_maps_runtime_errors(error, expected_status, expected_code):
    runtime = FakeRuntime()
    runtime.chat_error = error

    response = make_client(runtime).post(
        "/api/chat",
        json={"session_id": "session-1", "message": "hello"},
    )

    assert response.status_code == expected_status
    assert response.get_json()["code"] == expected_code


def test_successful_chat_and_clear_contracts():
    runtime = FakeRuntime()
    client = make_client(runtime)

    response = client.post(
        "/api/chat",
        json={"session_id": "known", "message": "  hello  "},
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "response": "Answer: hello",
        "context_trimmed": False,
    }
    assert client.post("/api/sessions/known/clear").status_code == 204
    assert runtime.cleared == ["known"]


def test_model_selection_returns_status_and_reset_signal():
    runtime = FakeRuntime()

    response = make_client(runtime).post(
        "/api/model/select",
        json={"model": "baseline"},
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "state": "ready",
        "message": "Model ready",
        "model": "baseline",
        "conversation_reset": True,
    }


@pytest.mark.parametrize("payload", [None, {}, {"model": "other"}])
def test_model_selection_rejects_invalid_payload(payload):
    client = make_client()
    response = (
        client.post("/api/model/select", data="not-json")
        if payload is None
        else client.post("/api/model/select", json=payload)
    )

    assert response.status_code == 400
    assert response.get_json()["code"] == "invalid_model"
