"""Paths, model, device."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_JSONL = DATA_DIR / "train.jsonl"
EVAL_JSONL = DATA_DIR / "eval.jsonl"

# Artifacts
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ADAPTER_DIR = ARTIFACTS_DIR / "adapter"
TRAINING_LOG_JSON = ARTIFACTS_DIR / "training_log.json"
EVAL_REPORT_JSON = ARTIFACTS_DIR / "eval_report.json"

# Model
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# Training defaults
MAX_SEQ_LEN = 512
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
MAX_STEPS = 600
NUM_EPOCHS = 1
LEARNING_RATE = 2e-5
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Seeds (dataset generation and training)
GEN_SEED = 42
INFERENCE_SEED = 42

# Inference
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.2
TOP_P = 0.9


def get_device():
    """cuda > mps > cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
