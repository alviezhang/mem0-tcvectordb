"""Pytest-level helpers and fixtures."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TRUTHY = {"1", "true", "yes", "on"}
_INTEGRATION_SKIP_REASON = "Integration tests require --run-integration or MEM0_RUN_INTEGRATION=1."


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that hit external services.",
    )


def pytest_collection_modifyitems(config, items):
    if _should_run_integration(config):
        return
    skip_marker = pytest.mark.skip(reason=_INTEGRATION_SKIP_REASON)
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_marker)


def load_env_file(path: Optional[Path] = None, *, override: bool = False) -> None:
    """Load environment variables from a .env file for local testing."""
    env_path = path or Path(".env")
    if not env_path.is_absolute():
        env_path = Path.cwd() / env_path
    load_dotenv(dotenv_path=env_path, override=override)


def require_env(name: str) -> str:
    """Return an environment variable or raise a helpful error."""
    value = os.environ.get(name)
    if value is None or not str(value).strip():
        raise RuntimeError(f"Environment variable '{name}' is required. Set it in your shell or .env file.")
    return value


@pytest.fixture(scope="session")
def integration_env(pytestconfig) -> Dict[str, str]:
    """Load the repo-level .env file for integration tests when requested."""
    if not _should_run_integration(pytestconfig):
        pytest.skip(_INTEGRATION_SKIP_REASON)
    load_env_file(PROJECT_ROOT / ".env")
    return {
        "test_user_id": os.environ.get("MEM0_TEST_USER_ID", "demo-user"),
        "search_query": os.environ.get("MEM0_TEST_QUERY", "Who did Alice meet at GraphConf?"),
    }


def _is_truthy(value: Optional[str]) -> bool:
    return bool(value) and value.strip().lower() in _TRUTHY


def _should_run_integration(config) -> bool:
    return bool(config.getoption("--run-integration") or _is_truthy(os.environ.get("MEM0_RUN_INTEGRATION")))
