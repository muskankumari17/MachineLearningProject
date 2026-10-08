"""Convenience launcher for the Flask application."""

from pathlib import Path
import runpy


APP_DIRECTORY = Path(__file__).resolve().parent / "fradulent_complaint_detection"

if __name__ == "__main__":
    runpy.run_path(APP_DIRECTORY / "app.py", run_name="__main__")
