"""Dataset format and count (>=600 train, >=60 eval)."""
import pytest
from pathlib import Path

from src.config import DATA_DIR, EVAL_JSONL, TRAIN_JSONL
from src.dataset_format import load_jsonl, validate_record, count_lines


def test_train_and_eval_files_exist():
    assert TRAIN_JSONL.exists(), "data/train.jsonl should exist"
    assert EVAL_JSONL.exists(), "data/eval.jsonl should exist"


def test_train_count_at_least_600():
    n = count_lines(TRAIN_JSONL)
    assert n >= 600, f"train.jsonl should have >= 600 lines, got {n}"


def test_eval_count_at_least_60():
    n = count_lines(EVAL_JSONL)
    assert n >= 60, f"eval.jsonl should have >= 60 lines, got {n}"


def test_train_format_input_output_only():
    for i, record in enumerate(load_jsonl(TRAIN_JSONL)):
        assert validate_record(record), f"train record {i} should have only 'input' and 'output' keys"
        if i >= 10:
            break


def test_eval_format_input_output_only():
    for i, record in enumerate(load_jsonl(EVAL_JSONL)):
        assert validate_record(record), f"eval record {i} should have only 'input' and 'output' keys"
        if i >= 10:
            break
