"""JSONL input/output format and validation."""
import json
from pathlib import Path
from typing import Iterator


def load_jsonl(path: Path) -> Iterator[dict]:
    """Yield one dict per line from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def validate_record(record: dict) -> bool:
    """Record has input and output strings only."""
    if not isinstance(record, dict):
        return False
    if set(record.keys()) != {"input", "output"}:
        return False
    if not isinstance(record.get("input"), str) or not isinstance(record.get("output"), str):
        return False
    if not record["input"].strip() or not record["output"].strip():
        return False
    return True


def count_lines(path: Path) -> int:
    """Non-empty JSONL line count."""
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n
