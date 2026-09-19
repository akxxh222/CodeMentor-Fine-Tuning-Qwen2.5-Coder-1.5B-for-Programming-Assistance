"""CodeMentor web application package."""


def create_app(*args, **kwargs):
    """Import the Flask factory lazily to keep utility imports lightweight."""
    from .app import create_app as factory

    return factory(*args, **kwargs)
