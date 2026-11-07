#!/usr/bin/env python3
"""
04_normalize_and_clean.py
=========================
Quality Filtering & Normalization Pipeline (Atlas-Chat Methodology):
1. Unicode NFC Normalization (unicodedata.normalize('NFC', text))
2. Pashto Orthography & Keyboard Standardization (Kaf, Heh, Ya variants, Tatweel removal)
3. ZWNJ Standardization (\u200c for compound verbs & suffixes)
4. Pashto Script Purity Filter (Script Density >= 0.80)
5. Dialect-specific vocabulary enforcement:
   - Afghan: Strict Urdu suppression + authentic Southern markers
   - Peshawari: Authentic Northern Shaurian orthography (کښې, دے, مونږ)
6. Strict Length Controls (Prompt >= 10 chars, Response >= 15 chars, Max words <= 1200)
7. Strict Atlas-Chat Schema Validation:
   {
     "dialect": "afghan" | "peshawari",
     "messages": [
       {"role": "user", "content": "..."},
       {"role": "assistant", "content": "..."}
     ]
   }
"""

import os
import sys
import json
import re
import unicodedata
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data/raw"
INTERIM_DIR = BASE_DIR / "data/interim"
LOG_FILE = BASE_DIR / "logs/dataset_generation.log"

AFGHAN_RAW = RAW_DIR / "afghan_raw/extracted_afghan.jsonl"
PESHAWARI_RAW = RAW_DIR / "peshawari_raw/extracted_peshawari.jsonl"

CLEAN_AFGHAN_OUT = INTERIM_DIR / "clean_afghan.jsonl"
CLEAN_PESHAWARI_OUT = INTERIM_DIR / "clean_peshawari.jsonl"

# Arabic/Pashto script character set
PASHTO_SCRIPT_RE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]')

# Forbidden loanwords in Afghan ("Awani") dialect
AFGHAN_FORBIDDEN = [
    "یونیورسټي", "هسپتال", "مریض", "طالب علم", "شکریه", "گاڑی", 
    "کرسی", "میز", "سکول", "لڑکا", "ہم نے", "کتابیں", "مونږ"
]

# Character mappings for normalization
CHAR_MAPPINGS = {
    '\u0643': '\u06a9',  # Arabic Kaf -> Pashto Keheh
    '\u06c1': '\u0647',  # Urdu Heh Goal -> Arabic/Pashto Heh
    '\u06be': '\u0647',  # Urdu Heh Doachashmi -> Arabic/Pashto Heh
    '\u0649': '\u06cc',  # Arabic Alef Maksura -> Farsi/Pashto Yeh
    '\u064a': '\u06cc',  # Arabic Yeh (with two dots below) -> standard Yeh
    '\u0640': '',        # Tatweel (kashida) removal
}

def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def normalize_text(text: str, dialect: str) -> str:
    if not text:
        return ""
    
    # 1. Unicode NFC
    text = unicodedata.normalize('NFC', text)
    
    # 2. Character mapping
    for old, new in CHAR_MAPPINGS.items():
        text = text.replace(old, new)
        
    # 3. ZWNJ normalization (\u200c)
    text = re.sub(r'[\u200b\u200d\ufeff]', '', text)
    text = re.sub(r'\s*\u200c\s*', '\u200c', text)
    
    ZWNJ = '\u200c'
    text = re.sub(r'(\bکې)\s+(ږي\b)', r'\g<1>' + ZWNJ + r'\g<2>', text)
    text = re.sub(r'(\bش)\s+(وي\b)', r'\g<1>' + ZWNJ + r'\g<2>', text)

    # 4. Dialect-specific morphological & orthographic localization
    if dialect == "peshawari":
        # Northern / Shaurian orthography
        text = re.sub(r'\bموږ\b', 'مونږ', text)
        text = re.sub(r'\bکې\b', 'کښې', text)
        text = re.sub(r'\bدی([\.،\s\?!]|$)', r'دے\1', text)
    elif dialect == "afghan":
        # Southern / Awani orthography
        text = re.sub(r'\bمونږ\b', 'موږ', text)
        text = re.sub(r'\bکښې\b', 'کې', text)
        text = re.sub(r'\bدے([\.،\s\?!]|$)', r'دی\1', text)
    
    # 5. Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def check_script_purity(text: str, threshold: float = 0.80) -> bool:
    clean = re.sub(r'[\s\d\.,!؟:;\(\)\[\]"\'\-_/\\]+', '', text)
    if not clean:
        return False
    pashto_matches = PASHTO_SCRIPT_RE.findall(clean)
    ratio = len(pashto_matches) / len(clean)
    return ratio >= threshold

def clean_file(input_path: Path, output_path: Path, dialect: str):
    log(f"Starting Normalization & Quality Filtration for: {input_path.name} ({dialect})")
    
    total = 0
    passed = 0
    script_failed = 0
    len_failed = 0
    forbidden_failed = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            
            try:
                item = json.loads(line)
            except Exception:
                continue
            
            messages = item.get("messages", [])
            if len(messages) < 2:
                continue
            
            user_msg = messages[0].get("content", "")
            asst_msg = messages[1].get("content", "")
            
            # Normalize with dialect localization
            norm_user = normalize_text(user_msg, dialect)
            norm_asst = normalize_text(asst_msg, dialect)
            
            # Strict Length filter:
            # - User content must be >= 10 chars (CHECK 1 requirement)
            # - Assistant content must be >= 15 chars
            # - Assistant words <= 1200 words (~1620 tokens to guarantee < 2048 Gemma-2 context budget)
            if len(norm_user) < 10 or len(norm_asst) < 15:
                len_failed += 1
                continue
            
            asst_word_count = len(norm_asst.split())
            if asst_word_count > 1200:
                len_failed += 1
                continue
            
            # Script purity filter (Atlas-Chat >= 80%)
            if not check_script_purity(norm_user, 0.80) or not check_script_purity(norm_asst, 0.80):
                script_failed += 1
                continue
            
            # Afghan Urdu loanword purity check
            if dialect == "afghan":
                combined = norm_user + " " + norm_asst
                if any(bad in combined for bad in AFGHAN_FORBIDDEN):
                    forbidden_failed += 1
                    continue
            
            clean_rec = {
                "dialect": dialect,
                "messages": [
                    {"role": "user", "content": norm_user},
                    {"role": "assistant", "content": norm_asst}
                ]
            }
            
            fout.write(json.dumps(clean_rec, ensure_ascii=False) + "\n")
            passed += 1

    log(f"  Processed:           {total:,}")
    log(f"  Passed Quality Gate: {passed:,}")
    log(f"  Length Filtered:     {len_failed:,}")
    log(f"  Script Purity Drops: {script_failed:,}")
    if dialect == "afghan":
        log(f"  Forbidden Urdu Drops:{forbidden_failed:,}")
    log(f"  Saved to:            {output_path.name}\n")
    return passed

def main():
    os.makedirs(INTERIM_DIR, exist_ok=True)
    log("=" * 70)
    log("TASK 4 / SCRIPT 04: Normalization & Quality Filtration (Atlas-Chat Rules)")
    log("=" * 70)

    afg_passed = clean_file(AFGHAN_RAW, CLEAN_AFGHAN_OUT, "afghan")
    pes_passed = clean_file(PESHAWARI_RAW, CLEAN_PESHAWARI_OUT, "peshawari")

    log(f"Total Normalized & Quality-Filtered Records: {afg_passed + pes_passed:,}")
    log("=" * 70)

if __name__ == "__main__":
    main()
