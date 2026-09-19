"""Asynchronous local model loading and serialized inference."""

import logging
from pathlib import Path
from threading import Lock, Thread

import torch

try:
    from .conversations import ConversationStore
except ImportError:  # Support `python app/app.py` from the project root.
    from conversations import ConversationStore


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = PROJECT_ROOT / "models" / "adapter"
MAX_SEQ_LENGTH = 512
INPUT_TOKEN_BUDGET = 320
MAX_NEW_TOKENS = 192


class ModelNotReady(RuntimeError):
    """Raised when inference is requested before the model is ready."""


class ModelBusy(RuntimeError):
    """Raised when another generation currently owns the GPU."""


class GenerationFailed(RuntimeError):
    """Raised when inference fails without exposing private details."""


class InvalidModelSelection(ValueError):
    """Raised when a requested comparison model is unknown."""


def load_default_model():
    """Load the local Qwen adapter lazily so Flask can start without Unsloth."""
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


def generate_default(model, tokenizer, messages, max_new_tokens, model_mode):
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {key: value.to("cuda") for key, value in inputs.items()}

    def generate():
        with torch.inference_mode():
            return model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

    if model_mode == "baseline":
        with model.disable_adapter():
            outputs = generate()
    else:
        outputs = generate()

    input_length = inputs["input_ids"].shape[-1]
    return tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True,
    ).strip()


class ModelRuntime:
    def __init__(
        self,
        loader=load_default_model,
        generator=generate_default,
        max_new_tokens=MAX_NEW_TOKENS,
    ):
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
        self._selected_model = "finetuned"

    def public_status(self):
        with self._state_lock:
            return {
                "state": self._state,
                "message": self._message,
                "model": self._selected_model,
            }

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
            conversations = ConversationStore(
                tokenizer,
                input_budget=INPUT_TOKEN_BUDGET,
            )
        except Exception as error:
            LOGGER.exception("Model loading failed")
            message = self._safe_load_error(error)
            with self._state_lock:
                self._state = "unavailable"
                self._message = message
            return

        with self._state_lock:
            self.model = model
            self.tokenizer = tokenizer
            self.conversations = conversations
            self._state = "ready"
            self._message = "Local model ready"

    @staticmethod
    def _safe_load_error(error):
        if isinstance(error, FileNotFoundError):
            return "Fine-tuned adapter is unavailable."
        if str(error) == "cuda-unavailable":
            return "CUDA is unavailable on this computer."
        if "out of memory" in str(error).lower():
            return "There is not enough GPU memory to load the model."
        return "The local model could not be loaded."

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
            messages, trimmed = self.conversations.build_messages(
                session_id,
                message,
            )
            try:
                response = self.generator(
                    self.model,
                    self.tokenizer,
                    messages,
                    self.max_new_tokens,
                    self._selected_model,
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

    def select_model(self, model_name):
        if model_name not in {"baseline", "finetuned"}:
            raise InvalidModelSelection()
        if self.public_status()["state"] != "ready":
            raise ModelNotReady()
        if not self._inference_lock.acquire(blocking=False):
            raise ModelBusy()

        try:
            with self._state_lock:
                if model_name == self._selected_model:
                    return False
                self._selected_model = model_name
                self.conversations = ConversationStore(
                    self.tokenizer,
                    input_budget=INPUT_TOKEN_BUDGET,
                )
                return True
        finally:
            self._inference_lock.release()
