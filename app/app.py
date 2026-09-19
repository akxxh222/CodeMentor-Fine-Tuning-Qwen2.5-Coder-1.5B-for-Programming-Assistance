"""Flask entry point for the local CodeMentor workbench."""

import re

from flask import Flask, jsonify, render_template, request

try:
    from .conversations import PromptTooLong
    from .model_runtime import (
        GenerationFailed,
        InvalidModelSelection,
        ModelBusy,
        ModelNotReady,
        ModelRuntime,
    )
except ImportError:  # Support `python app/app.py` from the project root.
    from conversations import PromptTooLong
    from model_runtime import (
        GenerationFailed,
        InvalidModelSelection,
        ModelBusy,
        ModelNotReady,
        ModelRuntime,
    )


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

    @app.post("/api/model/select")
    def select_model():
        payload = request.get_json(silent=True)
        model_name = payload.get("model") if isinstance(payload, dict) else None
        if model_name not in {"baseline", "finetuned"}:
            return _error("Choose a valid model", "invalid_model", 400)

        try:
            changed = model_runtime.select_model(model_name)
        except InvalidModelSelection:
            return _error("Choose a valid model", "invalid_model", 400)
        except ModelBusy:
            return _error("Wait for the current response to finish.", "model_busy", 409)
        except ModelNotReady:
            return _error("The local model is not ready.", "model_unavailable", 503)

        response = model_runtime.public_status()
        response["conversation_reset"] = changed
        return jsonify(response)

    @app.post("/api/chat")
    def chat():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _error("JSON request required", "invalid_request", 400)

        session_id = payload.get("session_id")
        message = payload.get("message")
        if not isinstance(session_id, str) or not SESSION_ID.fullmatch(session_id):
            return _error("Invalid session identifier", "invalid_session", 400)
        if not isinstance(message, str) or not message.strip():
            return _error("Enter a programming question", "invalid_message", 400)

        try:
            return jsonify(model_runtime.chat(session_id, message.strip()))
        except PromptTooLong:
            return _error(
                "Your message is too long for this model.",
                "prompt_too_long",
                400,
            )
        except ModelBusy:
            return _error(
                "CodeMentor is answering another request.",
                "model_busy",
                409,
            )
        except ModelNotReady:
            return _error(
                "The local model is not ready.",
                "model_unavailable",
                503,
            )
        except GenerationFailed:
            return _error(
                "Response generation failed. Try again.",
                "generation_failed",
                500,
            )

    @app.post("/api/sessions/<session_id>/clear")
    def clear_session(session_id):
        if not SESSION_ID.fullmatch(session_id):
            return _error("Invalid session identifier", "invalid_session", 400)
        model_runtime.clear_session(session_id)
        return "", 204

    if start_model:
        model_runtime.start_loading()
    return app


def _error(message, code, status):
    return jsonify(error=message, code=code), status


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
