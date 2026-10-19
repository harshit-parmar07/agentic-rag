import os
import sys

# Add backend directory to sys.path so module imports resolve cleanly in Vercel & AWS Lambda
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(ROOT_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from main import app, handler  # noqa: E402

__all__ = ["app", "handler"]

