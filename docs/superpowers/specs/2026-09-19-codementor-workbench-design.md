# CodeMentor Workbench design

Date: 2026-09-19
Status: Implemented; retained as the approved historical design

## Purpose

CodeMentor is a local programming assistant powered by the fine-tuned Qwen 2.5 Coder adapter in this repository. The application must demonstrate the assistant itself, not expose the dataset preparation, training, or evaluation machinery as its primary interface.

The interface should feel like a mature programming tool: restrained graphite surfaces, one blue accent, strong typography, useful density, and code-first response presentation. It must avoid decorative gradients, oversized marketing copy, generic AI imagery, and dashboard elements unrelated to asking programming questions.

## Product experience

The selected product shape is the Mentor Workbench:

- A dark session rail with a CodeMentor identity, New session action, and in-memory session list.
- A focused conversation workspace with a compact model-state header.
- User questions displayed distinctly from CodeMentor responses.
- Safe rendering of prose and fenced code blocks, including code-language labels and copy controls.
- A persistent multiline composer supporting follow-up questions and pasted code.
- Responsive behavior that collapses the session rail on narrow screens.

The interface supports current-session conversational context. A user can ask follow-up questions that refer to recent turns. Starting a new session creates an independent conversation context.

## Runtime architecture

Flask starts and serves the UI even when the model cannot load. A singleton model manager starts model loading in a background thread and exposes one of three public states:

- `loading`: model initialization is in progress;
- `ready`: the model and tokenizer are available for inference;
- `unavailable`: loading failed or the required local runtime is unavailable.

The model manager loads the fine-tuned adapter from `models/adapter` through Unsloth, using the existing Qwen 2.5 Coder base-model configuration and 4-bit loading. It enables inference mode after loading. Only one model instance may occupy GPU memory.

Model loading failures do not terminate Flask. The manager stores a short, safe error category for the UI, while the complete exception remains in the server log. A retry endpoint can initiate another loading attempt when no attempt is already running.

## Request flow

```text
Browser workbench
    -> GET /api/status
    -> status UI: loading | ready | unavailable

Browser prompt + session ID
    -> POST /api/chat
    -> validate model state and request
    -> load current in-memory session messages
    -> trim oldest complete turns to token budget
    -> run generation under the model inference lock
    -> store the successful user/assistant turn
    -> return response JSON
```

The browser never submits a prompt to an unavailable model. The backend independently enforces this rule and returns HTTP 503 unless the manager is ready.

## API

### `GET /`

Serves the Mentor Workbench HTML.

### `GET /api/status`

Returns the public model state and safe status message.

Example:

```json
{
  "state": "ready",
  "message": "Local model ready"
}
```

### `POST /api/model/retry`

Starts another background load attempt when the model is unavailable. Returns the resulting public loading state. It does not create a second concurrent load.

### `POST /api/chat`

Request:

```json
{
  "session_id": "browser-generated identifier",
  "message": "Explain why this loop never terminates."
}
```

Success response:

```json
{
  "response": "The loop condition remains true because...",
  "context_trimmed": false
}
```

The endpoint validates the JSON shape, session identifier, non-empty message, prompt size, model state, and inference availability. Only successful generations are appended to session history.

### `POST /api/sessions/<session_id>/clear`

Removes the named in-memory session and returns an empty success response. This powers both New session cleanup and explicit session clearing.

## Conversation memory and token budgeting

Conversation history is kept in Flask process memory and is not written to disk. Restarting Flask clears it.

The model retains context only within the active session. Before generation, the server renders candidate messages with the tokenizer chat template and removes the oldest complete user/assistant pairs until the prompt fits the input budget. The newest user message must always remain intact.

The total model sequence length is 512 tokens. Generation reserves 192 tokens, leaving an input budget of 320 tokens including chat-template tokens. If the newest message cannot fit within that input budget by itself, the request is rejected with HTTP 400 and actionable length guidance. When older turns are removed, the response sets `context_trimmed` to `true` and the UI shows a subtle notice.

## Concurrency

Model loading and inference use separate explicit locks:

- The loading lock prevents duplicate load attempts.
- The inference lock permits one generation at a time on the 4 GB GPU.

If generation is already in progress, another request receives a temporary busy response rather than waiting indefinitely. The UI keeps that prompt available for retry.

The in-memory conversation store also uses a lock around session reads and writes. A failed generation does not alter conversation history.

## Frontend behavior

The frontend uses local HTML, CSS, and JavaScript served by Flask. It has no CDN, account, telemetry, or paid-service dependency.

On startup, the page polls `/api/status`:

- Loading: amber status, explanatory loading state, disabled composer.
- Ready: green status, enabled composer.
- Unavailable: error summary, disabled composer, Retry loading action.

Submitting a message immediately adds the user bubble, disables duplicate submission, and shows a compact response-progress indicator. A successful response replaces the indicator with safely rendered content. A failed request preserves the user text and offers retry without inventing a response.

The renderer escapes all model output before applying limited fenced-code formatting. It does not insert raw model-generated HTML. Code blocks expose a local copy button. Enter submits; Shift+Enter inserts a newline.

The browser stores only interface-level session metadata in local storage: session identifiers, short generated titles, and the active session. Model conversation contents remain in Flask memory. After a Flask restart, stale browser sessions display as empty rather than reconstructing untrusted history client-side.

## Visual system

- Graphite session rail and near-white workspace.
- Royal blue as the only primary accent.
- Green, amber, and red reserved for model state.
- Compact sans-serif interface typography and system monospace for code.
- Fine borders and restrained shadows instead of floating glass cards.
- Maximum readable response width while allowing wider code blocks.
- Clear focus states, keyboard operation, sufficient contrast, and reduced-motion support.

## Error handling

- CUDA unavailable: `unavailable` with a concise CUDA requirement message.
- Adapter missing or invalid: `unavailable` with the expected local adapter path.
- GPU out of memory: `unavailable` during load, or a retryable generation error during chat.
- Model loading: HTTP 503 from chat with the loading state.
- Model busy: HTTP 409 with a retryable busy message.
- Invalid or oversized prompt: HTTP 400 with field-level guidance.
- Generation failure: HTTP 500 with a safe message; full details stay in server logs.

Raw Python tracebacks, local cache paths, and internal exception details are not returned to the browser.

## Project integration

The implementation will reuse the inference behavior currently expressed in `src/inference.py`, while separating reusable model loading and generation concerns from its command-line entry point. Flask-specific code remains under `app/`. Frontend templates and static assets follow Flask conventions:

```text
app/
    app.py
    templates/
        index.html
    static/
        styles.css
        app.js
```

The application is launched from the repository root so `models/adapter` resolves consistently. Flask is added to `requirements.txt` if it is not already present.

## Testing and verification

Backend tests use a fake model manager and do not require CUDA. They cover:

- loading, ready, and unavailable status responses;
- chat rejection when loading or unavailable;
- retry behavior and duplicate-load protection;
- request validation and oversized messages;
- session isolation and clearing;
- oldest-turn trimming while preserving the newest prompt;
- successful history updates and no update after failed generation;
- busy-model and generation-error responses.

Model-independent frontend verification covers responsive layout, status-state transitions, disabled/enabled composer behavior, safe response rendering, fenced code rendering, copying code, new sessions, and keyboard submission.

A local integration smoke test starts Flask with a fake model manager, opens the workbench, submits a prompt, and verifies that the answer renders. A real-model smoke test is optional because it requires CUDA and several gigabytes of GPU memory.

## Explicit non-goals

- User accounts or authentication.
- Conversation persistence across Flask restarts.
- Remote APIs or paid inference services.
- Training, evaluation, or dataset dashboards.
- Multiple simultaneous GPU generations.
- Executing user-provided or model-generated code.
- Rendering arbitrary Markdown or raw HTML from model output.
