"""Load base + LoRA adapter, generate recipe from ingredients. Returns JSON for API. Adapter required."""
import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    ADAPTER_DIR,
    BASE_MODEL_ID,
    get_device,
    INFERENCE_SEED,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    TOP_P,
)


def _adapter_present() -> bool:
    """Adapter dir exists and has adapter_config.json."""
    if not ADAPTER_DIR.exists():
        return False
    return (ADAPTER_DIR / "adapter_config.json").exists()


def _get_model_and_tokenizer():
    """Load base + LoRA adapter. Raises if adapter missing."""
    if not _adapter_present():
        raise FileNotFoundError(
            f"Adapter not found at {ADAPTER_DIR}. "
            "Run: python -m src.train_lora and ensure artifacts/adapter/ contains the adapter files."
        )
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        device_map="auto" if device != "cpu" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")
    try:
        model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    except Exception as e:
        raise RuntimeError(
            f"Could not load adapter from {ADAPTER_DIR}: {e}. "
            "Ensure artifacts/adapter/ contains adapter_config.json and adapter weights."
        ) from e
    return model, tokenizer, device


def _prompt_for_inference(ingredients: str) -> str:
    """Qwen chat prompt for ingredients."""
    return f"<|im_start|>user\n{ingredients}<|im_end|>\n<|im_start|>assistant\n"


def _parse_model_output(text: str, query: str) -> dict:
    """Parse Recipe/Ingredients/Steps/Time/Tips lines or JSON block into API dict."""
    normalized = _normalize_ingredients(query)
    result = {
        "query": query,
        "normalized_ingredients": normalized,
        "recipe_title": "",
        "ingredients": [],
        "steps": [],
        "time_minutes": None,
        "notes": "",
    }

    # Try JSON block first
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            result["recipe_title"] = data.get("recipe_title", data.get("name", "")) or result["recipe_title"]
            result["ingredients"] = data.get("ingredients", result["ingredients"])
            if isinstance(result["ingredients"], str):
                result["ingredients"] = [x.strip() for x in result["ingredients"].split(",") if x.strip()]
            result["steps"] = data.get("steps", result["steps"])
            if isinstance(result["steps"], str):
                result["steps"] = [s.strip() for s in result["steps"].split("|") if s.strip()]
            result["time_minutes"] = data.get("time_minutes", result["time_minutes"])
            result["notes"] = data.get("notes", data.get("tips", "")) or result["notes"]
            return result
        except json.JSONDecodeError:
            pass

    # Line-based parse (training format)
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("Recipe:"):
            result["recipe_title"] = line.replace("Recipe:", "").strip()
        elif line.startswith("Ingredients:"):
            part = line.replace("Ingredients:", "").strip()
            result["ingredients"] = [x.strip() for x in part.split(",") if x.strip()]
        elif line.startswith("Steps:"):
            part = line.replace("Steps:", "").strip()
            result["steps"] = [s.strip() for s in part.split("|") if s.strip()]
        elif line.startswith("Time:"):
            m = re.search(r"(\d+)", line)
            if m:
                result["time_minutes"] = int(m.group(1))
        elif line.startswith("Tips:"):
            result["notes"] = line.replace("Tips:", "").strip()

    if not result["recipe_title"] and text.strip():
        result["recipe_title"] = text.split("\n")[0].strip()[:80]
    if not result["steps"] and text.strip():
        result["steps"] = [t.strip() for t in text.split("\n") if t.strip()][:5]
    return result


def _normalize_ingredients(query: str) -> list:
    """Extract ingredient tokens from query (lowercase, strip phrases)."""
    s = query.lower().strip()
    for phrase in ["i have", "please suggest", "what can i cook with", "ingredients:", "recipe for"]:
        s = re.sub(re.escape(phrase), "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s)
    parts = re.split(r"[,&\s]+| and ", s)
    return [p.strip() for p in parts if len(p.strip()) > 1]


def run_inference(message: str, model=None, tokenizer=None, device=None) -> dict:
    """Generate recipe for ingredients; returns dict for API. Loads model once if not passed."""
    if model is None or tokenizer is None:
        model, tokenizer, device = _get_model_and_tokenizer()

    prompt = _prompt_for_inference(message)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if device == "cpu":
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
    else:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kw = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    try:
        torch.manual_seed(INFERENCE_SEED)
        if hasattr(torch, "generator") and device != "cpu":
            gen_kw["generator"] = torch.Generator(device=model.device).manual_seed(INFERENCE_SEED)
    except Exception:
        pass
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kw)

    decoded = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return _parse_model_output(decoded, message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", type=str, default="egg, onion", help="Ingredient list")
    args = parser.parse_args()
    result = run_inference(args.message)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
