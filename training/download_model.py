#!/usr/bin/env python3
"""
download_model.py
=================
Pre-downloads google/gemma-2-9b-it tokenizer and model weights
from Hugging Face into the local HF cache with resume capability.
"""

import sys
import time
from huggingface_hub import snapshot_download, HfApi

MODEL_ID = "google/gemma-2-9b-it"

def main():
    print("=" * 70)
    print(f"DOWNLOADING BASE MODEL: {MODEL_ID}")
    print("=" * 70)

    api = HfApi()
    try:
        user = api.whoami()
        print(f"✓ Authenticated as Hugging Face user: {user.get('name')}")
    except Exception as e:
        print(f"✗ HF Authentication warning: {e}")

    start_time = time.time()
    print(f"Starting snapshot download for {MODEL_ID}...")
    local_dir = snapshot_download(
        repo_id=MODEL_ID,
        resume_download=True
    )
    elapsed = time.time() - start_time
    print(f"\n✓ Download completed successfully in {elapsed:.1f} seconds!")
    print(f"Cached at: {local_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()

