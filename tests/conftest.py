"""Keep generated test data in a fresh, ignored workspace directory."""
from pathlib import Path
import uuid


def pytest_configure(config):
    if config.option.basetemp is None:
        path = Path(__file__).resolve().parents[1] / "artifacts/test-runs" / uuid.uuid4().hex
        if path.exists():
            raise RuntimeError("Refusing to reuse a test directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(path)
