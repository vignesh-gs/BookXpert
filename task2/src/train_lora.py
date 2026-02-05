"""LoRA SFT for Qwen2.5-1.5B-Instruct. Saves adapter to artifacts/adapter/, log to training_log.json."""
import json
import math
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from src.config import (
    ADAPTER_DIR,
    ARTIFACTS_DIR,
    BASE_MODEL_ID,
    BATCH_SIZE,
    EVAL_JSONL,
    GEN_SEED,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_SEQ_LEN,
    MAX_STEPS,
    NUM_EPOCHS,
    TRAIN_JSONL,
    TRAINING_LOG_JSON,
    get_device,
)
from src.dataset_format import load_jsonl


def _load_jsonl_to_list(path: Path) -> list:
    return list(load_jsonl(path))


def _build_chat_prompt(input_text: str) -> str:
    """Qwen chat format: user + assistant start."""
    return f"<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"


def _tokenize(examples: dict, tokenizer) -> dict:
    """Tokenize input/output; labels -100 on input, token ids on output.
    examples: batched dict with "input" and "output" lists."""
    inputs = [_build_chat_prompt(inp) for inp in examples["input"]]
    outputs = list(examples["output"])
    all_input_ids = []
    all_labels = []
    for inp, out in zip(inputs, outputs):
        inp_tok = tokenizer(inp, add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LEN - 64)
        out_tok = tokenizer(out, add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LEN - len(inp_tok["input_ids"]))
        inp_ids = inp_tok["input_ids"]
        out_ids = out_tok["input_ids"]
        full = inp_ids + out_ids
        if len(full) > MAX_SEQ_LEN:
            full = full[:MAX_SEQ_LEN]
            out_len = MAX_SEQ_LEN - len(inp_ids)
            out_ids = full[len(inp_ids):]
        labels = [-100] * len(inp_ids) + out_ids
        all_input_ids.append(full)
        all_labels.append(labels)
    return {"input_ids": all_input_ids, "labels": all_labels, "attention_mask": [[1] * len(x) for x in all_input_ids]}


def main():
    device = get_device()
    print(f"Using device: {device}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        device_map="auto" if device != "cpu" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")

    train_records = _load_jsonl_to_list(TRAIN_JSONL)
    eval_records = _load_jsonl_to_list(EVAL_JSONL)
    train_data = Dataset.from_list(
        [{"input": r["input"], "output": r["output"]} for r in train_records]
    )
    eval_data = Dataset.from_list(
        [{"input": r["input"], "output": r["output"]} for r in eval_records]
    )

    def tokenize_fn(examples):
        return _tokenize(examples, tokenizer)

    train_data = train_data.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_data.column_names,
        desc="Tokenize train",
    )
    eval_data = eval_data.map(
        tokenize_fn,
        batched=True,
        remove_columns=eval_data.column_names,
        desc="Tokenize eval",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
        label_pad_token_id=-100,
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(ADAPTER_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_steps=MAX_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=False,
        bf16=(device != "cpu"),
        logging_steps=20,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=200,
        load_best_model_at_end=True,
        seed=GEN_SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    # Training log
    log = {
        "base_model": BASE_MODEL_ID,
        "train_samples": len(train_records),
        "eval_samples": len(eval_records),
        "max_steps": MAX_STEPS,
        "max_seq_len": MAX_SEQ_LEN,
        "device": device,
        "train_loss": trainer.state.log_history[-1].get("loss", None) if trainer.state.log_history else None,
    }
    with open(TRAINING_LOG_JSON, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Training log saved to {TRAINING_LOG_JSON}")
    print(f"Adapter saved to {ADAPTER_DIR}")


if __name__ == "__main__":
    main()
