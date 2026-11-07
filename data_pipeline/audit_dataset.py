#!/usr/bin/env python3
"""
audit_dataset.py
================
Comprehensive 7-Dimension Forensic Audit for NoorGPT Multi-Dialect Dataset
Strictly following Atlas-Chat / Paper 1 Quality Standards.
"""

import os
import sys
import json
import re
import random
import numpy as np
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data/processed"
BENCHMARK_DIR = BASE_DIR / "data/benchmark"

FILES = {
    "train_balanced": PROCESSED_DIR / "train_balanced_75k.jsonl",
    "train_afghan": PROCESSED_DIR / "train_afghan_35k.jsonl",
    "train_peshawari": PROCESSED_DIR / "train_peshawari_40k.jsonl",
    "test_afghan": BENCHMARK_DIR / "test_afghan_1000.jsonl",
    "test_peshawari": BENCHMARK_DIR / "test_peshawari_1000.jsonl"
}

AFGHAN_MARKERS = ["موږ", "پوهنتون", "روغتون", "ناروغه", "ښوونځی", "دی"]
PESHAWARI_MARKERS = ["مونږ", "کښې", "دے", "پېښور"]
FORBIDDEN_URDU = ["یونیورسټي", "هسپتال", "مریض", "طالب علم", "شکریه", "گاڑی"]

def get_word_set(text):
    return set(re.findall(r'\b\w+\b', text))

def jaccard_word_sim(s1, s2):
    if not s1 or not s2:
        return 0.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union else 0.0

def run_audit():
    print("=" * 80)
    print("STARTING FORENSIC DATASET AUDIT (7 DIMENSIONS)")
    print("=" * 80)

    # 1. Check file existence
    for k, p in FILES.items():
        if not p.exists():
            print(f"[ERROR] Missing file: {p}")
            sys.exit(1)

    # Storage
    records = {k: [] for k in FILES}
    corrupted_lines = {k: [] for k in FILES}
    schema_errors = {k: [] for k in FILES}
    empty_short_errors = {k: [] for k in FILES}
    domain_field_detected = {k: 0 for k in FILES}

    total_lines_scanned = 0

    # CHECK 1: JSONL Syntax & Schema Integrity
    for k, p in FILES.items():
        with open(p, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                total_lines_scanned += 1
                line_str = line.strip()
                if not line_str:
                    empty_short_errors[k].append((line_no, "Empty line"))
                    continue
                try:
                    obj = json.loads(line_str)
                except Exception as e:
                    corrupted_lines[k].append((line_no, str(e)))
                    continue

                # Check schema
                if "domain" in obj:
                    domain_field_detected[k] += 1

                dialect = obj.get("dialect")
                if dialect not in ("afghan", "peshawari"):
                    schema_errors[k].append((line_no, f"Invalid dialect: {dialect}"))

                messages = obj.get("messages")
                if not isinstance(messages, list) or len(messages) < 2:
                    schema_errors[k].append((line_no, "Messages not list or len < 2"))
                    continue

                # Role check
                roles = [m.get("role") for m in messages]
                valid_roles = True
                for i, r in enumerate(roles):
                    expected = "user" if i % 2 == 0 else "assistant"
                    if r != expected:
                        valid_roles = False
                        break
                if not valid_roles:
                    schema_errors[k].append((line_no, f"Invalid alternating roles: {roles}"))

                # Content length check (< 10 characters)
                has_short = False
                for m in messages:
                    c = (m.get("content") or "").strip()
                    if len(c) < 10:
                        has_short = True
                        break
                if has_short:
                    empty_short_errors[k].append((line_no, "Content < 10 chars"))

                records[k].append(obj)

    # CHECK 2: Dataset Counts & Dialect Proportions
    train_bal_len = len(records["train_balanced"])
    train_afg_len = len(records["train_afghan"])
    train_pes_len = len(records["train_peshawari"])
    test_afg_len = len(records["test_afghan"])
    test_pes_len = len(records["test_peshawari"])

    afg_in_bal = sum(1 for r in records["train_balanced"] if r["dialect"] == "afghan")
    pes_in_bal = sum(1 for r in records["train_balanced"] if r["dialect"] == "peshawari")

    # CHECK 3: Linguistic Purity & Dialect Segregation
    # In Afghan train + test:
    afg_records_all = records["train_afghan"] + records["test_afghan"]
    pes_records_all = records["train_peshawari"] + records["test_peshawari"]

    afg_marker_count = 0
    afg_urdu_intrusion_samples = []

    for idx, r in enumerate(afg_records_all):
        u = r["messages"][0]["content"]
        a = r["messages"][1]["content"]
        txt = u + " " + a
        if any(m in txt for m in AFGHAN_MARKERS):
            afg_marker_count += 1
        found_urdu = [w for w in FORBIDDEN_URDU if w in txt]
        if found_urdu:
            afg_urdu_intrusion_samples.append((idx, found_urdu, u[:40]))

    pes_marker_count = 0
    for r in pes_records_all:
        u = r["messages"][0]["content"]
        a = r["messages"][1]["content"]
        txt = u + " " + a
        if any(m in txt for m in PESHAWARI_MARKERS):
            pes_marker_count += 1

    afg_marker_pct = (afg_marker_count / len(afg_records_all)) * 100 if afg_records_all else 0
    pes_marker_pct = (pes_marker_count / len(pes_records_all)) * 100 if pes_records_all else 0

    # CHECK 4: Data Leakage & Contamination (Zero-Tolerance)
    train_user_prompts = [r["messages"][0]["content"].strip() for r in records["train_balanced"]]
    train_prompt_set = set(train_user_prompts)

    test_afg_prompts = [r["messages"][0]["content"].strip() for r in records["test_afghan"]]
    test_pes_prompts = [r["messages"][0]["content"].strip() for r in records["test_peshawari"]]
    test_prompts_all = test_afg_prompts + test_pes_prompts

    exact_leakage = [p for p in test_prompts_all if p in train_prompt_set]

    # Near-duplicate check (sample or indexed word check)
    train_word_sets = [get_word_set(p) for p in train_user_prompts]
    test_word_sets = [get_word_set(p) for p in test_prompts_all]

    # Inverted index for fast candidate checking
    inv_index = {}
    for idx, ws in enumerate(train_word_sets):
        for w in ws:
            if len(w) > 3: # meaningful words
                inv_index.setdefault(w, []).append(idx)

    near_leakage = []
    for t_idx, tws in enumerate(test_word_sets):
        candidate_indices = set()
        for w in tws:
            if w in inv_index:
                candidate_indices.update(inv_index[w])
        for c_idx in candidate_indices:
            sim = jaccard_word_sim(tws, train_word_sets[c_idx])
            if sim > 0.70:
                near_leakage.append((t_idx, c_idx, sim, test_prompts_all[t_idx][:50]))
                break

    # CHECK 5: Intra-Dataset Deduplication
    unique_prompts_count = len(train_prompt_set)
    unique_prompt_pct = (unique_prompts_count / train_bal_len) * 100 if train_bal_len else 0
    dup_prompt_count = train_bal_len - unique_prompts_count

    # CHECK 6: Sequence Length & Context Budget
    prompt_words = [len(r["messages"][0]["content"].split()) for r in records["train_balanced"]]
    response_words = [len(r["messages"][1]["content"].split()) for r in records["train_balanced"]]
    # Est tokens (~1.35x words for Pashto subwords)
    response_tokens = [int(w * 1.35) for w in response_words]

    avg_p_words = np.mean(prompt_words)
    avg_r_words = np.mean(response_words)
    max_p_words = np.max(prompt_words)
    max_r_words = np.max(response_words)
    p95_r_tokens = np.percentile(response_tokens, 95)
    max_r_tokens = np.max(response_tokens)

    # CHECK 7: Spot-Check Samples
    random.seed(42)
    sample_afg = random.sample(records["train_afghan"], 3)
    sample_pes = random.sample(records["train_peshawari"], 3)

    # Compilation of audit metrics
    total_corrupted = sum(len(v) for v in corrupted_lines.values())
    total_empty_short = sum(len(v) for v in empty_short_errors.values())
    total_schema_errs = sum(len(v) for v in schema_errors.values())
    total_domain_keys = sum(domain_field_detected.values())

    results = {
        "total_lines_scanned": int(total_lines_scanned),
        "total_corrupted": int(total_corrupted),
        "total_empty_short": int(total_empty_short),
        "total_schema_errs": int(total_schema_errs),
        "total_domain_keys": int(total_domain_keys),
        "train_bal_len": int(train_bal_len),
        "train_afg_len": int(train_afg_len),
        "train_pes_len": int(train_pes_len),
        "test_afg_len": int(test_afg_len),
        "test_pes_len": int(test_pes_len),
        "afg_pct": float((afg_in_bal / train_bal_len) * 100),
        "pes_pct": float((pes_in_bal / train_bal_len) * 100),
        "afg_marker_pct": float(afg_marker_pct),
        "pes_marker_pct": float(pes_marker_pct),
        "afg_urdu_intrusions": int(len(afg_urdu_intrusion_samples)),
        "exact_leakage": int(len(exact_leakage)),
        "near_leakage": int(len(near_leakage)),
        "unique_prompts_count": int(unique_prompts_count),
        "unique_prompt_pct": float(unique_prompt_pct),
        "dup_prompt_count": int(dup_prompt_count),
        "avg_p_words": float(avg_p_words),
        "avg_r_words": float(avg_r_words),
        "max_p_words": int(max_p_words),
        "max_r_words": int(max_r_words),
        "p95_r_tokens": float(p95_r_tokens),
        "max_r_tokens": int(max_r_tokens),
        "sample_afg": sample_afg,
        "sample_pes": sample_pes
    }

    # Print Formatted Report to Console
    print("\n" + "=" * 50)
    print("PASHTO DATASET FORENSIC AUDIT REPORT")
    print("=" * 50)
    print("1. SYNTAX & INTEGRITY:")
    print(f"   • Total Lines Scanned:    {results['total_lines_scanned']:,}")
    print(f"   • Corrupted JSON Lines:   {results['total_corrupted']} - Must be 0")
    print(f"   • Empty / Broken Texts:   {results['total_empty_short']} - Must be 0")
    print(f"   • Role Ordering Errors:   {results['total_schema_errs']} - Must be 0")
    print(f"   • 'domain' Field Found:   {results['total_domain_keys']} - Must be 0")

    print("\n2. VOLUME & PROPORTIONS:")
    print(f"   • Total Training Pairs:   {results['train_bal_len']:,}")
    print(f"   • Afghan Split:           {results['train_afg_len']:,} ({results['afg_pct']:.1f}%)")
    print(f"   • Peshawari Split:        {results['train_pes_len']:,} ({results['pes_pct']:.1f}%)")
    print(f"   • Held-Out Benchmark:     {results['test_afg_len'] + results['test_pes_len']:,} pairs ({results['test_afg_len']:,} Afghan + {results['test_pes_len']:,} Peshawari)")

    print("\n3. DIALECT PURITY:")
    print(f"   • Afghan Marker Density:  {results['afg_marker_pct']:.1f}%")
    print(f"   • Peshawari Marker Rate:  {results['pes_marker_pct']:.1f}%")
    print(f"   • Urdu Intrusion in Afghan: {results['afg_urdu_intrusions']} (Target: 0)")

    print("\n4. CONTAMINATION & LEAKAGE:")
    print(f"   • Exact Train-Test Overlap: {results['exact_leakage']} (MUST BE 0)")
    print(f"   • Near-Duplicate Leakage:   {results['near_leakage']} (Jaccard > 0.70)")
    print(f"   • Unique Prompts:           {results['unique_prompts_count']:,} ({results['unique_prompt_pct']:.2f}%)")
    print(f"   • Duplicate Prompts:        {results['dup_prompt_count']:,}")

    print("\n5. LENGTH PROFILE:")
    print(f"   • Avg Prompt Length:      {results['avg_p_words']:.1f} words")
    print(f"   • Avg Response Length:    {results['avg_r_words']:.1f} words")
    print(f"   • Max Prompt Length:      {results['max_p_words']:,} words")
    print(f"   • Max Response Length:    {results['max_r_words']:,} words (~{results['max_r_tokens']:,} est tokens)")
    print(f"   • 95th Pct Response Len:  ~{int(results['p95_r_tokens']):,} est tokens")

    print("\n6. SPOT-CHECK SAMPLES:")
    print("\n--- [AFGHAN SAMPLE 1] ---")
    print("User:", results['sample_afg'][0]['messages'][0]['content'])
    print("Assistant:", results['sample_afg'][0]['messages'][1]['content'][:250] + "...")
    print("\n--- [PESHAWARI SAMPLE 1] ---")
    print("User:", results['sample_pes'][0]['messages'][0]['content'])
    print("Assistant:", results['sample_pes'][0]['messages'][1]['content'][:250] + "...")

    is_green = (
        results['total_corrupted'] == 0 and
        results['total_empty_short'] == 0 and
        results['total_schema_errs'] == 0 and
        results['total_domain_keys'] == 0 and
        results['exact_leakage'] == 0 and
        results['near_leakage'] == 0 and
        results['afg_urdu_intrusions'] == 0 and
        results['unique_prompt_pct'] >= 98.5
    )

    print("\n" + "=" * 50)
    verdict = "GREEN LIGHT: READY TO TRAIN" if is_green else "RED LIGHT: FIX REQUIRED"
    print(f"FINAL VERDICT: [{verdict}]")
    print("=" * 50)

    with open(BASE_DIR / "logs/audit_results.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in results.items() if not k.startswith("sample")}, f, indent=2)

    return results

if __name__ == "__main__":
    run_audit()
