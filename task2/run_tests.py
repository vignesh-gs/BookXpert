"""Pytest + smoke demo (start API, 3 sample queries). Use: python run_tests.py"""
import subprocess
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8000"
SAMPLE_INPUTS = ["egg, onion", "rice, vegetables", "banana, milk, oats"]


def run_pytest():
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=PROJECT_ROOT,
    )
    return r.returncode == 0


def wait_for_api(timeout=90):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(f"{API_URL}/health", timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def smoke_demo():
    """Start API, send 3 sample requests, print responses."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_for_api():
            print("API did not become ready.")
            return False
        print("\n--- Smoke demo: 3 sample queries ---")
        for msg in SAMPLE_INPUTS:
            try:
                r = requests.post(f"{API_URL}/chat", json={"message": msg}, timeout=60)
                r.raise_for_status()
                data = r.json()
                title = data.get("recipe_title") or "(no title)"
                steps = data.get("steps") or []
                print(f"Input: {msg}")
                print(f"  Recipe: {title} | Steps: {len(steps)}")
            except Exception as e:
                print(f"Input: {msg} -> Error: {e}")
                return False
        print("--- Smoke demo passed ---\n")
        return True
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def main():
    print("Running pytest...")
    if not run_pytest():
        print("Pytest failed.")
        sys.exit(1)
    print("Running smoke demo (API + 3 queries)...")
    if not smoke_demo():
        print("Smoke demo failed.")
        sys.exit(1)
    print("All checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
