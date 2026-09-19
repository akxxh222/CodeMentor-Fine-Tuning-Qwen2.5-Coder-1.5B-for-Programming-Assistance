from app.app import create_app


class FrontendRuntime:
    def public_status(self):
        return {
            "state": "unavailable",
            "message": "Model unavailable",
            "model": "finetuned",
        }

    def start_loading(self):
        return True

    def clear_session(self, session_id):
        return False


def make_client():
    app = create_app(FrontendRuntime(), start_model=False)
    app.config.update(TESTING=True)
    return app.test_client()


def test_workbench_contains_required_controls_and_local_assets():
    html = make_client().get("/").get_data(as_text=True)

    for element_id in (
        "model-status",
        "model-select",
        "model-status-message",
        "retry-model",
        "new-session",
        "session-list",
        "thread",
        "chat-form",
        "prompt",
        "send-button",
        "context-notice",
    ):
        assert f'id="{element_id}"' in html
    assert 'href="/static/styles.css"' in html
    assert 'src="/static/app.js"' in html
    assert "http://" not in html
    assert "https://" not in html


def test_styles_are_responsive_and_accessible():
    response = make_client().get("/static/styles.css")
    css = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "@media (max-width:" in css
    assert "prefers-reduced-motion" in css
    assert ":focus-visible" in css
    assert 'body[data-model-state="ready"]' in css
    assert 'body[data-model-state="unavailable"]' in css


def test_javascript_uses_safe_dom_rendering_and_local_api_contracts():
    response = make_client().get("/static/app.js")
    script = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "textContent" in script
    assert ".innerHTML" not in script
    assert "localStorage" not in script
    assert "setInterval(pollStatus" in script
    assert "navigator.clipboard.writeText" in script
    assert 'fetch("/api/chat"' in script
    assert 'fetch("/api/status"' in script
    assert 'fetch("/api/model/select"' in script
