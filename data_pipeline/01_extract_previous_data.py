#!/usr/bin/env python3
"""
01_extract_previous_data.py
===========================
Extracts, cleans, and dialect-routes raw pairs from existing workspace archives:
- Pashto-LLM-V2/data/combined_augmented_train.json
- Pashto-LLM-V2/data/Peshawari_train.json
- Pashto-LLM-V2/data/Afghani_train.json
- Pashto-LLM-V2/data/conversational_pashto.json
- fatima/dataset/fatima-pashto.json

Strictly adheres to Atlas-Chat schema:
{
  "dialect": "afghan" | "peshawari",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
NO "domain", "topic", or "id" fields allowed.
"""

import os
import sys
import json
import re
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = BASE_DIR.parent

SOURCES = [
    {
        "path": WORKSPACE_DIR / "Pashto-LLM-V2/data/combined_augmented_train.json",
        "format": "json_list",
        "default_dialect": None,
        "input_key": "input",
        "output_key": "output",
        "dialect_key": "dialect"
    },
    {
        "path": WORKSPACE_DIR / "Pashto-LLM-V2/data/Peshawari_train.json",
        "format": "jsonl",
        "default_dialect": "peshawari",
        "input_key": "input",
        "output_key": "output",
        "dialect_key": "dialect"
    },
    {
        "path": WORKSPACE_DIR / "Pashto-LLM-V2/data/Afghani_train.json",
        "format": "json_list",
        "default_dialect": "afghan",
        "input_key": "input",
        "output_key": "output",
        "dialect_key": "dialect"
    },
    {
        "path": WORKSPACE_DIR / "Pashto-LLM-V2/data/conversational_pashto.json",
        "format": "json_list",
        "default_dialect": None,
        "input_key": "input",
        "output_key": "output",
        "dialect_key": "dialect"
    },
    {
        "path": WORKSPACE_DIR / "Pashto-LLM-V2/data/pashto_merged_dataset.json",
        "format": "json_list",
        "default_dialect": "afghan",
        "input_key": "input",
        "output_key": "output",
        "dialect_key": "dialect"
    },
    {
        "path": WORKSPACE_DIR / "fatima/dataset/fatima-pashto.json",
        "format": "json_list",
        "default_dialect": None,
        "input_key": "instruction",
        "output_key": "response",
        "dialect_key": "dialect"
    }
]

RAW_OUT_DIR = BASE_DIR / "data/raw"
AFGHAN_OUT = RAW_OUT_DIR / "afghan_raw/extracted_afghan.jsonl"
PESHAWARI_OUT = RAW_OUT_DIR / "peshawari_raw/extracted_peshawari.jsonl"
LOG_FILE = BASE_DIR / "logs/dataset_generation.log"

# Lexical markers
AFGHAN_MARKERS = ["موږ", "روغتون", "پوهنتون", "ناروغه", "ښوونځی", "کندهار", "کابل", "محصل", "خوراک", "ګړندی"]
PESHAWARI_MARKERS = ["مونږ", "کښې", "دے", "پېښور", "حجره", "ډاک", "ډوډۍ", "ښایسته", "لالا"]
SEVERE_URDU_INTRUSIONS = ["لڑکا", "کتابیں", "گاڑی", "شکریه", "طالب علم", "میز", "کرسی", "سکول", "ہم نے"]

# Pashto script regex (Arabic/Pashto block)
PASHTO_CHAR_REGEX = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')

def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def is_valid_pashto(text, min_len=10, min_script_ratio=0.70):
    if not text or len(text.strip()) < min_len:
        return False
    # Check script ratio
    clean = re.sub(r'\s+', '', text)
    if not clean:
        return False
    pashto_chars = len(PASHTO_CHAR_REGEX.findall(clean))
    ratio = pashto_chars / len(clean)
    return ratio >= min_script_ratio

def has_severe_urdu(text):
    return any(u in text for u in SEVERE_URDU_INTRUSIONS)

def classify_dialect(text, current_dialect=None):
    if current_dialect:
        d = current_dialect.strip().lower()
        if "afg" in d or "awan" in d:
            return "afghan"
        if "pes" in d or "shaur" in d:
            return "peshawari"
    
    # Lexical score
    afg_score = sum(1 for m in AFGHAN_MARKERS if m in text)
    pes_score = sum(1 for m in PESHAWARI_MARKERS if m in text)
    
    if afg_score > pes_score:
        return "afghan"
    elif pes_score > afg_score:
        return "peshawari"
    
    return "afghan" if current_dialect and "afg" in current_dialect else "peshawari"

def extract_and_route():
    os.makedirs(AFGHAN_OUT.parent, exist_ok=True)
    os.makedirs(PESHAWARI_OUT.parent, exist_ok=True)
    os.makedirs(LOG_FILE.parent, exist_ok=True)

    log("=" * 70)
    log("TASK 2 / SCRIPT 01: Extracting and Standardizing Previous Datasets")
    log("=" * 70)

    afghan_count = 0
    peshawari_count = 0
    skipped_count = 0
    total_processed = 0

    seen_pairs = set()

    with open(AFGHAN_OUT, "w", encoding="utf-8") as f_afg, \
         open(PESHAWARI_OUT, "w", encoding="utf-8") as f_pes:

        for src in SOURCES:
            path = src["path"]
            if not path.exists():
                log(f"[WARNING] File not found: {path}")
                continue

            log(f"Processing: {path.name} (Format: {src['format']})")
            file_items = []

            if src["format"] == "json_list":
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        file_items = json.load(f)
                except Exception as e:
                    log(f"Error loading {path}: {e}")
                    continue
            elif src["format"] == "jsonl":
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                file_items.append(json.loads(line))
                except Exception as e:
                    log(f"Error reading JSONL {path}: {e}")
                    continue

            log(f"  Loaded {len(file_items):,} records.")

            for item in file_items:
                total_processed += 1
                u_text = (item.get(src["input_key"]) or "").strip()
                a_text = (item.get(src["output_key"]) or "").strip()

                if not is_valid_pashto(u_text, min_len=5) or not is_valid_pashto(a_text, min_len=10):
                    skipped_count += 1
                    continue

                if has_severe_urdu(u_text) or has_severe_urdu(a_text):
                    skipped_count += 1
                    continue

                pair_hash = hash((u_text, a_text))
                if pair_hash in seen_pairs:
                    skipped_count += 1
                    continue
                seen_pairs.add(pair_hash)

                raw_dialect = item.get(src["dialect_key"]) or src["default_dialect"]
                dialect = classify_dialect(u_text + " " + a_text, raw_dialect)

                record = {
                    "dialect": dialect,
                    "messages": [
                        {"role": "user", "content": u_text},
                        {"role": "assistant", "content": a_text}
                    ]
                }

                line_str = json.dumps(record, ensure_ascii=False) + "\n"
                if dialect == "afghan":
                    f_afg.write(line_str)
                    afghan_count += 1
                else:
                    f_pes.write(line_str)
                    peshawari_count += 1

    log("\nExtraction Summary:")
    log(f"  Total Processed:  {total_processed:,}")
    log(f"  Valid Afghan:     {afghan_count:,} -> {AFGHAN_OUT}")
    log(f"  Valid Peshawari:  {peshawari_count:,} -> {PESHAWARI_OUT}")
    log(f"  Skipped/Filtered: {skipped_count:,}")
    log(f"  Retained Total:   {afghan_count + peshawari_count:,}")
    log("=" * 70)

if __name__ == "__main__":
    extract_and_route()
