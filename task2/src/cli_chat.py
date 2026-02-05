"""CLI: type ingredients, POST /chat to API (or run infer locally if API down). Run: python -m src.cli_chat"""
import json
import sys

import requests

API_BASE = "http://127.0.0.1:8000"


def _call_api(message: str) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}/chat", json={"message": message}, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


def _display(result: dict) -> None:
    """Print title, ingredients, steps, time, tips."""
    title = result.get("recipe_title") or "Recipe"
    steps = result.get("steps") or []
    ingredients = result.get("ingredients") or []
    time_min = result.get("time_minutes")
    notes = result.get("notes", "")

    print("\n--- Recipe ---")
    print(f"Title: {title}")
    if ingredients:
        print(f"Ingredients: {', '.join(ingredients)}")
    if time_min is not None:
        print(f"Time: {time_min} minutes")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")
    if notes:
        print(f"Tips: {notes}")
    print("--------------\n")


def main():
    print("Recipe Chatbot (ingredients -> recipe). Type ingredients and press Enter. 'quit' to exit.")
    print("If the API is not running, we will try local inference.\n")

    while True:
        try:
            message = input("Ingredients: ").strip()
        except EOFError:
            break
        if not message:
            continue
        if message.lower() in ("quit", "exit", "q"):
            break

        result = _call_api(message)
        if result is None:
            print("API not reachable. Trying local inference...")
            try:
                from src.infer import run_inference
                result = run_inference(message)
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                print("Start the API with: python -m src.api")
                continue
        _display(result)


if __name__ == "__main__":
    main()
