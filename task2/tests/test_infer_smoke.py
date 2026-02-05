"""Inference smoke: 'egg, onion' returns dict with required keys (adapter optional for structure check)."""
import pytest
from src.infer import run_inference


REQUIRED_KEYS = {"query", "normalized_ingredients", "recipe_title", "ingredients", "steps", "time_minutes", "notes"}


def test_infer_returns_required_keys():
    result = run_inference("egg, onion")
    assert isinstance(result, dict), "Result should be a dict"
    for key in REQUIRED_KEYS:
        assert key in result, f"Result should contain key '{key}'"


def test_infer_egg_onion_has_recipe_like_content():
    result = run_inference("egg, onion")
    assert "recipe_title" in result
    assert "steps" in result
    assert isinstance(result["steps"], list)
    assert isinstance(result["normalized_ingredients"], list)
