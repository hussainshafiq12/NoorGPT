#!/usr/bin/env python3
"""
05_minhash_deduplication.py
===========================
Vectorized MinHash LSH Deduplication Pipeline (Atlas-Chat Methodology):
1. 128 permutation hashes per document via vectorized NumPy
2. Character 3-gram shingling
3. Intra-pool pruning: Jaccard similarity >= 0.80 dropped
4. Strict Atlas-Chat Schema:
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
import numpy as np
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INTERIM_DIR = BASE_DIR / "data/interim"
DEDUP_DIR = INTERIM_DIR / "dedup"
LOG_FILE = BASE_DIR / "logs/dataset_generation.log"

AFGHAN_IN = INTERIM_DIR / "clean_afghan.jsonl"
PESHAWARI_IN = INTERIM_DIR / "clean_peshawari.jsonl"

DEDUP_AFGHAN_OUT = DEDUP_DIR / "dedup_afghan.jsonl"
DEDUP_PESHAWARI_OUT = DEDUP_DIR / "dedup_peshawari.jsonl"

NUM_PERM = 128
LARGE_PRIME = 2147483647  # 2^31 - 1
BANDS = 16
ROWS = NUM_PERM // BANDS  # 8

np.random.seed(42)
A = np.random.randint(1, LARGE_PRIME - 1, size=(NUM_PERM, 1), dtype=np.int64)
B = np.random.randint(0, LARGE_PRIME - 1, size=(NUM_PERM, 1), dtype=np.int64)

def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def get_shingles(text: str, k: int = 3):
    compact = re.sub(r'\s+', '', text)
    if len(compact) < k:
        return {compact}
    return {compact[i:i+k] for i in range(len(compact) - k + 1)}

def compute_signature_numpy(shingles: set) -> np.ndarray:
    if not shingles:
        return np.zeros(NUM_PERM, dtype=np.int64)
    # 31-bit positive hashes
    sh_arr = np.array([abs(hash(s)) % LARGE_PRIME for s in shingles], dtype=np.int64).reshape(1, -1)
    # Vectorized: (128, 1) * (1, N) + (128, 1) % LARGE_PRIME -> (128, N)
    all_hashes = (A * sh_arr + B) % LARGE_PRIME
    return np.min(all_hashes, axis=1)

def deduplicate_dataset(input_file: Path, output_file: Path, dialect: str, jaccard_threshold: float = 0.80):
    log(f"Starting Vectorized MinHash LSH (128 hashes) for: {input_file.name} ({dialect})")
    os.makedirs(output_file.parent, exist_ok=True)
    
    lsh_buckets = defaultdict(list)
    signatures = []
    kept_records = []
    
    total = 0
    duplicates_dropped = 0
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            
            try:
                rec = json.loads(line)
            except Exception:
                continue
            
            full_text = rec["messages"][0]["content"] + " " + rec["messages"][1]["content"]
            shingles = get_shingles(full_text, k=3)
            sig = compute_signature_numpy(shingles)
            
            # Check collisions
            is_dup = False
            candidate_ids = set()
            
            for band_idx in range(BANDS):
                band_start = band_idx * ROWS
                band_tuple = tuple(sig[band_start : band_start + ROWS])
                bucket_key = (band_idx, band_tuple)
                if bucket_key in lsh_buckets:
                    candidate_ids.update(lsh_buckets[bucket_key])
            
            for cand_id in candidate_ids:
                cand_sig = signatures[cand_id]
                sim = np.mean(sig == cand_sig)
                if sim >= jaccard_threshold:
                    is_dup = True
                    break
            
            if is_dup:
                duplicates_dropped += 1
                continue
            
            current_id = len(signatures)
            signatures.append(sig)
            kept_records.append(rec)
            
            for band_idx in range(BANDS):
                band_start = band_idx * ROWS
                band_tuple = tuple(sig[band_start : band_start + ROWS])
                lsh_buckets[(band_idx, band_tuple)].append(current_id)

            if total % 10000 == 0:
                print(f"  Processed {total:,} | Kept {len(kept_records):,} | Duplicates {duplicates_dropped:,}")

    log(f"  Total Processed:     {total:,}")
    log(f"  Duplicates Pruned:   {duplicates_dropped:,} (Jaccard >= {jaccard_threshold:.2f})")
    log(f"  Unique Retained:     {len(kept_records):,}")
    
    with open(output_file, "w", encoding="utf-8") as fout:
        for rec in kept_records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"  Written to:          {output_file.name}\n")
    return len(kept_records)

def main():
    os.makedirs(DEDUP_DIR, exist_ok=True)
    log("=" * 70)
    log("TASK 4 / SCRIPT 05: MinHash LSH Deduplication (128 Permutations)")
    log("=" * 70)

    afg_kept = deduplicate_dataset(AFGHAN_IN, DEDUP_AFGHAN_OUT, "afghan", jaccard_threshold=0.80)
    pes_kept = deduplicate_dataset(PESHAWARI_IN, DEDUP_PESHAWARI_OUT, "peshawari", jaccard_threshold=0.80)

    log(f"Total Unique Deduplicated Records: {afg_kept + pes_kept:,}")
    log("=" * 70)

if __name__ == "__main__":
    main()
