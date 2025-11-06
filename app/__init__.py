"""Top-level package for Founders AI application.

Ensures relative imports inside modules like `main.py` work when the app
is executed under Uvicorn in Docker (module path: `app.main:app`).
"""

__all__ = []
