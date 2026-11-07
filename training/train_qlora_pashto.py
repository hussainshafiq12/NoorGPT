#!/usr/bin/env python3
"""
train_qlora_pashto.py
=====================
QLoRA (4-bit NF4) Instruction Fine-Tuning for Pashto Dual-Dialect LLM
Model: google/gemma-2-9b-it (with unsloth/gemma-2-9b-it-bnb-4bit support)
Hardware: 1x NVIDIA RTX 4090 (24GB VRAM)
Methodology: Atlas-Chat / Paper 1 (Response-Only Loss Masking)
"""

import os
import sys

# Prevent CUDA memory fragmentation for 256k vocab models (Gemma-2)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Self-healing environment bootstrap: re-exec in unsloth_env if torch is missing
try:
    import torch
except ImportError:
    venv_py = "/home/zohaib/Desktop/Pahto LLM V2/Pashto-LLM-V2/unsloth_env/bin/python3"
    if os.path.exists(venv_py):
        print(f"[BOOTSTRAP] System python missing torch. Automatically switching to: {venv_py}")
        os.execv(venv_py, [venv_py] + sys.argv)
    else:
        raise

import json
import argparse
import time
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

class PashtoInstructionDataset(Dataset):
    """
    Dataset loader for Pashto instruction pairs with strict response-only loss masking.
    User prompt tokens are masked with label -100 so cross-entropy loss is computed
    strictly on the assistant response tokens.
    """
    def __init__(self, data_path, tokenizer, max_seq_length=2048):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = []

        print(f"Loading Pashto SFT dataset from: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        print(f"Loaded {len(self.samples):,} verified instruction pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        messages = item["messages"]
        user_content = messages[0]["content"].strip()
        asst_content = messages[1]["content"].strip()

        # Gemma-2 Turn Formatting
        # <start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n{assistant}<end_of_turn>
        prompt_text = f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"
        full_text = prompt_text + f"{asst_content}<end_of_turn>\n"

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=True)

        # Truncate if exceeds max_seq_length
        if len(full_ids) > self.max_seq_length:
            full_ids = full_ids[:self.max_seq_length]

        prompt_len = min(len(prompt_ids), len(full_ids))

        # Labels: Mask prompt with -100
        labels = list(full_ids)
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": full_ids,
            "labels": labels
        }

class PashtoDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids = []
        labels = []
        attention_mask = []

        for x in batch:
            ids = x["input_ids"]
            lbl = x["labels"]
            pad_len = max_len - len(ids)

            padded_ids = ids + [self.pad_token_id] * pad_len
            padded_lbl = lbl + [-100] * pad_len
            padded_mask = [1] * len(ids) + [0] * pad_len

            input_ids.append(padded_ids)
            labels.append(padded_lbl)
            attention_mask.append(padded_mask)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }

def resolve_model_path(model_id):
    if os.path.isdir(model_id) and (Path(model_id) / "model.safetensors").exists():
        return model_id
    
    # 1. Check exact model_id cache
    hub_name = "models--" + model_id.replace("/", "--")
    cache_hub_dir = Path.home() / ".cache/huggingface/hub" / hub_name / "snapshots"
    if cache_hub_dir.exists():
        for snap in cache_hub_dir.glob("*"):
            if (snap / "model.safetensors").exists():
                return str(snap)
                
    # 2. Check verified 4-bit mirror cache
    alt_hub_name = "models--unsloth--gemma-2-9b-it-bnb-4bit"
    alt_cache_dir = Path.home() / ".cache/huggingface/hub" / alt_hub_name / "snapshots"
    if alt_cache_dir.exists():
        for snap in alt_cache_dir.glob("*"):
            if (snap / "model.safetensors").exists():
                return str(snap)
                
    return model_id

def find_train_file(train_arg):
    candidates = [
        Path(train_arg),
        Path("NoorGPT") / train_arg,
        Path.cwd() / train_arg,
        Path.cwd() / "NoorGPT" / train_arg,
        Path(__file__).resolve().parent.parent / train_arg
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    raise FileNotFoundError(f"Could not locate training dataset file: {train_arg}")

def main():
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning for Pashto Dual-Dialect LLM")
    parser.add_argument("--model_id", type=str, default="google/gemma-2-9b-it", help="Base model Hugging Face ID")
    parser.add_argument("--train_file", type=str, default="data/processed/train_balanced_75k.jsonl", help="Training JSONL file")
    parser.add_argument("--output_dir", type=str, default="checkpoints/gemma2_pashto_dual_75k", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size (1 for 256k vocab Gemma-2 on 24GB)")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps (effective batch 8)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    args = parser.parse_args()

    train_path = find_train_file(args.train_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("NOORGPT: PASHTO DUAL-DIALECT QLORA TRAINING (ATLAS-CHAT METHODOLOGY)")
    print("=" * 80)
    print(f"Base Model ID:      {args.model_id}")
    print(f"Training Dataset:   {train_path}")
    print(f"Output Checkpoints: {output_dir}")
    print(f"Context Length:     {args.max_seq_length} tokens")
    print(f"Batch Configuration: Batch {args.batch_size} x Accum {args.grad_accum} = Effective Batch {args.batch_size * args.grad_accum}")
    print(f"LoRA Rank / Alpha:  r={args.lora_r}, alpha={args.lora_alpha} (Scale = {args.lora_alpha / args.lora_r})")
    print(f"Learning Rate:      {args.lr} (Cosine decay, 3% warmup)")
    print(f"Loss Masking:       Response-Only (<start_of_turn>model\\n)")
    print(f"Target GPU:         {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB VRAM)")
    print("=" * 80)

    # Resolve model ID to local offline cache path
    chosen_model_id = resolve_model_path(args.model_id)
    print(f"Resolved Model Path: {chosen_model_id}")

    # 1. Load Tokenizer
    print(f"\nLoading Tokenizer: {chosen_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(chosen_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"✓ Tokenizer loaded. Vocabulary Size: {len(tokenizer):,}")

    # 2. Load 4-bit Quantized Base Model
    print(f"\nLoading 4-bit NF4 Base Model onto RTX 4090: {chosen_model_id}...")
    if "bnb-4bit" in chosen_model_id:
        model = AutoModelForCausalLM.from_pretrained(
            chosen_model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            chosen_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )

    # 3. Prepare Model for k-bit training & Enable Gradient Checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    # 4. LoRA Adapter Configuration
    print("\nConfiguring Cross-Dialect LoRA Adapter...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Dataset Preparation
    print("\nPreparing Tokenized SFT Dataset with Response-Only Masking...")
    dataset = PashtoInstructionDataset(train_path, tokenizer, max_seq_length=args.max_seq_length)
    collator = PashtoDataCollator(pad_token_id=tokenizer.pad_token_id)

    # 6. Training Arguments
    effective_batch = args.batch_size * args.grad_accum
    total_steps = math.ceil(len(dataset) * args.epochs / effective_batch)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        fp16=False,
        bf16=True,
        dataloader_num_workers=4,
        report_to="none",
        logging_first_step=True
    )

    print(f"\nTraining Plan Summary:")
    print(f"  • Total Epochs:            {args.epochs}")
    print(f"  • Total Training Pairs:    {len(dataset):,}")
    print(f"  • Total Calculated Steps:  ~{total_steps:,}")
    print(f"  • Save Interval:           Every 500 steps")
    print(f"  • Precision:               bfloat16")
    print(f"  • Optimizer:               paged_adamw_8bit")
    print("=" * 80)

    # 7. Execute Training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator
    )

    print("\n🚀 Starting QLoRA Fine-Tuning Execution...")
    train_result = trainer.train()

    print("\n" + "=" * 80)
    print("TRAINING RUN COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Final Training Loss: {train_result.training_loss:.4f}")
    
    # Save final model adapter and tokenizer
    print(f"Saving final adapter to: {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("All artifacts saved successfully.")

if __name__ == "__main__":
    main()
