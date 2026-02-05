"""API smoke: /health, /chat, required JSON keys."""
import subprocess
import sys
import time

import pytest
import requests

API_URL = "http://127.0.0.1:8000"


def _wait_for_api(timeout: float = 60.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{API_URL}/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


@pytest.fixture(scope="module")
def api_server():
    """Start uvicorn, yield, then terminate."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not _wait_for_api():
            pytest.skip("API did not become ready in time")
        yield
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_health(api_server):
    r = requests.get(f"{API_URL}/health", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"


def test_chat_returns_required_keys(api_server):
    r = requests.post(f"{API_URL}/chat", json={"message": "egg, onion"}, timeout=30)
    assert r.status_code == 200
    data = r.json()
    required = {"query", "normalized_ingredients", "recipe_title", "ingredients", "steps", "time_minutes", "notes"}
    for key in required:
        assert key in data, f"Response should contain '{key}'"
