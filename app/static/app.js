"use strict";

const elements = {
  form: document.querySelector("#chat-form"),
  prompt: document.querySelector("#prompt"),
  send: document.querySelector("#send-button"),
  thread: document.querySelector("#thread"),
  empty: document.querySelector("#empty-state"),
  status: document.querySelector("#model-status span"),
  statusMessage: document.querySelector("#model-status-message"),
  modelSelect: document.querySelector("#model-select"),
  unavailableMessage: document.querySelector("#unavailable-message"),
  retry: document.querySelector("#retry-model"),
  newSession: document.querySelector("#new-session"),
  clearSession: document.querySelector("#clear-session"),
  sessionList: document.querySelector("#session-list"),
  sessionTitle: document.querySelector("#session-title"),
  contextNotice: document.querySelector("#context-notice"),
};

const state = {
  sessions: [],
  activeId: null,
  messages: new Map(),
  busy: false,
  modelReady: false,
};

function newId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function createSession() {
  const session = {
    id: newId(),
    title: "New mentoring session",
    createdAt: Date.now(),
  };
  state.sessions.unshift(session);
  state.messages.set(session.id, []);
  state.activeId = session.id;
  renderSessionList();
  renderThread();
  elements.prompt.focus();
}

function ensureActiveSession() {
  const exists = state.sessions.some((session) => session.id === state.activeId);
  if (!exists && state.sessions.length) state.activeId = state.sessions[0].id;
  if (!state.sessions.length) createSession();
}

function activeSession() {
  return state.sessions.find((session) => session.id === state.activeId);
}

function activateSession(sessionId) {
  state.activeId = sessionId;
  if (!state.messages.has(sessionId)) state.messages.set(sessionId, []);
  renderSessionList();
  renderThread();
}

function renderSessionList() {
  elements.sessionList.replaceChildren();
  state.sessions.forEach((session) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "session-button";
    button.textContent = session.title;
    button.title = session.title;
    if (session.id === state.activeId) button.setAttribute("aria-current", "page");
    button.addEventListener("click", () => activateSession(session.id));
    elements.sessionList.append(button);
  });
}

function renderThread() {
  const messages = state.messages.get(state.activeId) || [];
  elements.thread.replaceChildren();
  elements.sessionTitle.textContent = activeSession()?.title || "New mentoring session";
  if (!messages.length) {
    elements.thread.append(elements.empty);
    bindSuggestions();
    return;
  }
  messages.forEach((message) => elements.thread.append(createMessage(message)));
  scrollThread();
}

function createMessage(message) {
  const wrapper = document.createElement("article");
  wrapper.className = `message message--${message.role}`;
  if (message.role === "assistant") {
    wrapper.append(renderResponse(message.content));
  } else {
    wrapper.textContent = message.content;
  }
  return wrapper;
}

function renderResponse(text) {
  const fragment = document.createDocumentFragment();
  text.split("```").forEach((segment, index) => {
    if (index % 2 === 0) {
      segment.split(/\n{2,}/).filter((value) => value.trim()).forEach((value) => {
        const paragraph = document.createElement("p");
        paragraph.textContent = value.trim();
        fragment.append(paragraph);
      });
      return;
    }

    const lines = segment.replace(/^\n/, "").split("\n");
    const firstLine = lines.shift() || "";
    const hasLanguage = /^[A-Za-z0-9_+#.-]+$/.test(firstLine.trim());
    const language = hasLanguage ? firstLine.trim() : "code";
    const codeText = (hasLanguage ? lines : [firstLine, ...lines]).join("\n").trim();
    fragment.append(createCodeBlock(language, codeText));
  });
  return fragment;
}

function createCodeBlock(language, codeText) {
  const wrapper = document.createElement("div");
  wrapper.className = "code-block";
  const header = document.createElement("div");
  header.className = "code-header";
  const label = document.createElement("span");
  label.textContent = language;
  const copy = document.createElement("button");
  copy.type = "button";
  copy.textContent = "Copy";
  copy.addEventListener("click", async () => {
    await navigator.clipboard.writeText(codeText);
    copy.textContent = "Copied";
    window.setTimeout(() => { copy.textContent = "Copy"; }, 1200);
  });
  header.append(label, copy);
  const pre = document.createElement("pre");
  const code = document.createElement("code");
  code.textContent = codeText;
  pre.append(code);
  wrapper.append(header, pre);
  return wrapper;
}

function appendPendingAssistant() {
  const wrapper = document.createElement("article");
  wrapper.className = "message message--assistant";
  const pending = document.createElement("div");
  pending.className = "pending";
  pending.setAttribute("aria-label", "CodeMentor is thinking");
  pending.append(document.createElement("i"), document.createElement("i"), document.createElement("i"));
  wrapper.append(pending);
  elements.thread.append(wrapper);
  scrollThread();
  return wrapper;
}

function appendMessage(role, content) {
  const messages = state.messages.get(state.activeId) || [];
  messages.push({ role, content });
  state.messages.set(state.activeId, messages);
}

function titleFrom(message) {
  const firstLine = message.split("\n")[0].trim();
  return firstLine.length > 38 ? `${firstLine.slice(0, 37)}…` : firstLine;
}

function updateSessionTitle(message) {
  const session = activeSession();
  if (!session || session.title !== "New mentoring session") return;
  session.title = titleFrom(message) || session.title;
  renderSessionList();
  elements.sessionTitle.textContent = session.title;
}

function scrollThread() {
  requestAnimationFrame(() => { elements.thread.scrollTop = elements.thread.scrollHeight; });
}

function updateComposer() {
  const enabled = state.modelReady && !state.busy;
  elements.prompt.disabled = !enabled;
  elements.send.disabled = !enabled || !elements.prompt.value.trim();
}

async function pollStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) throw new Error("Status unavailable");
    const status = await response.json();
    document.body.dataset.modelState = status.state;
    state.modelReady = status.state === "ready";
    elements.modelSelect.value = status.model;
    elements.modelSelect.disabled = status.state !== "ready" || state.busy;
    elements.status.textContent = status.state;
    elements.statusMessage.textContent = status.message;
    elements.unavailableMessage.textContent = status.message;
  } catch (error) {
    document.body.dataset.modelState = "unavailable";
    state.modelReady = false;
    elements.modelSelect.disabled = true;
    elements.status.textContent = "Offline";
    elements.statusMessage.textContent = "Flask connection unavailable";
    elements.unavailableMessage.textContent = "The CodeMentor server could not be reached.";
  }
  updateComposer();
}

function resetBrowserConversation() {
  state.sessions = [];
  state.messages = new Map();
  state.activeId = null;
  elements.contextNotice.hidden = true;
  createSession();
}

async function selectModel() {
  const previousModel = elements.modelSelect.value === "baseline" ? "finetuned" : "baseline";
  state.busy = true;
  updateComposer();
  elements.modelSelect.disabled = true;

  try {
    const response = await fetch("/api/model/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: elements.modelSelect.value }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Could not switch models");
    if (payload.conversation_reset) resetBrowserConversation();
  } catch (error) {
    elements.modelSelect.value = previousModel;
    window.alert(error.message);
  } finally {
    state.busy = false;
    await pollStatus();
  }
}

async function submitMessage(event) {
  event.preventDefault();
  const message = elements.prompt.value.trim();
  if (!message || state.busy || !state.modelReady) return;

  state.busy = true;
  updateSessionTitle(message);
  appendMessage("user", message);
  renderThread();
  const pending = appendPendingAssistant();
  elements.contextNotice.hidden = true;
  updateComposer();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.activeId, message }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Request failed");
    appendMessage("assistant", payload.response);
    elements.contextNotice.hidden = !payload.context_trimmed;
    elements.prompt.value = "";
  } catch (error) {
    appendMessage("error", error.message);
    elements.prompt.value = message;
  } finally {
    pending.remove();
    state.busy = false;
    renderThread();
    resizePrompt();
    await pollStatus();
  }
}

async function retryModel() {
  elements.retry.disabled = true;
  try {
    await fetch("/api/model/retry", { method: "POST" });
    await pollStatus();
  } finally {
    elements.retry.disabled = false;
  }
}

async function clearCurrentSession() {
  if (!state.activeId) return;
  await fetch(`/api/sessions/${encodeURIComponent(state.activeId)}/clear`, { method: "POST" });
  state.messages.set(state.activeId, []);
  elements.contextNotice.hidden = true;
  renderThread();
}

function resizePrompt() {
  elements.prompt.style.height = "auto";
  elements.prompt.style.height = `${Math.min(elements.prompt.scrollHeight, 220)}px`;
  updateComposer();
}

function bindSuggestions() {
  elements.empty.querySelectorAll("[data-suggestion]").forEach((button) => {
    button.addEventListener("click", () => {
      elements.prompt.value = button.dataset.suggestion;
      resizePrompt();
      elements.prompt.focus();
    }, { once: true });
  });
}

elements.form.addEventListener("submit", submitMessage);
elements.prompt.addEventListener("input", resizePrompt);
elements.prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});
elements.retry.addEventListener("click", retryModel);
elements.newSession.addEventListener("click", createSession);
elements.clearSession.addEventListener("click", clearCurrentSession);
elements.modelSelect.addEventListener("change", selectModel);

ensureActiveSession();
renderSessionList();
renderThread();
setInterval(pollStatus, 1500);
pollStatus();
