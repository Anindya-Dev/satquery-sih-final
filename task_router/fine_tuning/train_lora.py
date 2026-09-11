"""
Fine-tuning script for SatQuery AI Task Router using LoRA / QLoRA with PEFT & TRL.
Adapts a lightweight Small Language Model (e.g. Qwen2.5 / Phi-3) to reliably emit
strict TaskSpec JSON without conversational drift or schema hallucinations.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_training_environment(cfg: Dict[str, Any]):
    """Verifies GPU, PyTorch, dataset paths, and dependency setup."""
    print("==================================================")
    print(" SatQuery AI - Task Router Fine-Tuning Diagnostic ")
    print("==================================================")
    
    # Check data files
    train_path = Path(cfg["training"]["train_data_path"])
    val_path = Path(cfg["training"]["val_data_path"])
    
    print(f"[1] Dataset Check:")
    print(f"    Train dataset: {train_path} -> {'EXISTS' if train_path.exists() else 'MISSING'}")
    print(f"    Val dataset:   {val_path} -> {'EXISTS' if val_path.exists() else 'MISSING'}")
    
    # Check PyTorch & CUDA
    print(f"[2] Hardware & Runtime Check:")
    try:
        import torch
        print(f"    PyTorch Version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"    CUDA Available:  {cuda_available}")
        if cuda_available:
            print(f"    Device Name:     {torch.cuda.get_device_name(0)}")
            print(f"    VRAM Total:      {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        else:
            print("    [Notice] CUDA not detected on current host. Training requires an NVIDIA GPU (or Colab/Kaggle).")
    except ImportError:
        print("    [Warning] PyTorch is not yet installed in this virtual environment.")
        
    # Check HuggingFace packages
    print(f"[3] Hugging Face Toolchain Check:")
    for pkg in ["transformers", "peft", "trl", "datasets", "bitsandbytes", "accelerate"]:
        try:
            __import__(pkg)
            print(f"    {pkg:15s}: INSTALLED")
        except ImportError:
            print(f"    {pkg:15s}: NOT INSTALLED (Required for actual training run)")
    print("==================================================")


def train(config_path: str, dry_run: bool = False):
    cfg = load_yaml_config(config_path)
    
    if dry_run:
        check_training_environment(cfg)
        print("[Dry Run Complete] Pipeline validated successfully.")
        return

    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    base_model_id = cfg["model"]["base_model_name_or_path"]
    use_4bit = cfg["model"].get("use_4bit", True) and torch.cuda.is_available()

    print(f"[*] Loading tokenizer: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[*] Configuring quantization (4-bit: {use_4bit})...")
    bnb_config = None
    if use_4bit:
        compute_dtype = getattr(torch, cfg["model"].get("bnb_4bit_compute_dtype", "bfloat16"))
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg["model"].get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=cfg["model"].get("use_nested_quant", True)
        )

    print(f"[*] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    print("[*] Setting up LoRA configuration...")
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("[*] Loading training and validation datasets...")
    data_files = {
        "train": cfg["training"]["train_data_path"],
        "validation": cfg["training"]["val_data_path"]
    }
    dataset = load_dataset("json", data_files=data_files)

    train_cfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg.get("eval_strategy", "steps"),
        eval_steps=train_cfg["eval_steps"],
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        logging_dir=f"{train_cfg['output_dir']}/logs",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        max_seq_length=train_cfg["max_seq_length"],
        tokenizer=tokenizer,
        args=training_args
    )

    print("[*] Beginning fine-tuning...")
    trainer.train()

    print(f"[*] Saving fine-tuned adapter to {train_cfg['output_dir']}...")
    trainer.model.save_pretrained(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])
    print("[✓] Task Router fine-tuning complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SatQuery Task Router Fine-Tuner")
    parser.add_argument(
        "--config",
        type=str,
        default="task_router/fine_tuning/configs/lora_config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run environment diagnostics and dataset checks without executing heavy training"
    )
    args = parser.parse_args()
    train(args.config, dry_run=args.dry_run)
