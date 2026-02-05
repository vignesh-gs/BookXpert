---
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-1.5B-Instruct
- lora
- transformers
---
# Adapter directory

The app needs the adapter files here (adapter_config.json, adapter_model.safetensors, etc.). They’re included in the repo so you don’t need to run training. To retrain: run `python -m src.train_lora` from the task2 root.

### Framework versions

- PEFT 0.18.1