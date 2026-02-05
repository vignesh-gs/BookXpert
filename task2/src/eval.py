"""Run inference on eval.jsonl (first N), report JSON/title/steps rates to stdout and eval_report.json."""
import json
import sys
from pathlib import Path

from src.config import ARTIFACTS_DIR, EVAL_JSONL, EVAL_REPORT_JSON
from src.dataset_format import load_jsonl
from src.infer import run_inference

DEFAULT_LIMIT = 20


def main(limit: int = DEFAULT_LIMIT):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    records = list(load_jsonl(EVAL_JSONL))[:limit]

    n_valid_json = 0
    n_has_title = 0
    n_steps_ok = 0
    total = len(records)

    for r in records:
        inp = r["input"]
        try:
            out = run_inference(inp)
        except Exception as e:
            print(f"Inference failed for '{inp[:50]}...': {e}", file=sys.stderr)
            continue
        try:
            json.dumps(out)
            n_valid_json += 1
        except (TypeError, ValueError):
            pass
        if out.get("recipe_title", "").strip():
            n_has_title += 1
        steps = out.get("steps") or []
        if len(steps) >= 3:
            n_steps_ok += 1

    report = {
        "n_eval": total,
        "json_valid_rate": n_valid_json / total if total else 0,
        "has_recipe_title_rate": n_has_title / total if total else 0,
        "steps_ge_3_rate": n_steps_ok / total if total else 0,
    }
    with open(EVAL_REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    print("Eval report:")
    print(f"  n_eval: {total}")
    print(f"  json_valid_rate: {report['json_valid_rate']:.2%}")
    print(f"  has_recipe_title_rate: {report['has_recipe_title_rate']:.2%}")
    print(f"  steps_ge_3_rate: {report['steps_ge_3_rate']:.2%}")
    print(f"Report saved to {EVAL_REPORT_JSON}")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LIMIT
    main(limit=limit)
