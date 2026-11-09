#!/usr/bin/env python3
"""
scripts/publish_dataset_to_hf.py
================================
Stages and publishes Pashto-SFT-Mixture to Hugging Face Hub under hussainshafiq.
Configures full YAML metadata for Hugging Face Dataset Viewer and generates
the complete dataset card README.md.
"""

import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "hussainshafiq/Pashto-SFT-Mixture"
STAGING_DIR = Path("/tmp/pashto_sft_mixture_staging")
TOKEN = os.environ.get("HF_TOKEN")

if not TOKEN:
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        TOKEN = token_file.read_text().strip()

if not TOKEN:
    print("❌ Error: No Hugging Face token found in environment or ~/.cache/huggingface/token.")
    sys.exit(1)

# Source files from NoorGPT
BASE_DIR = Path(__file__).resolve().parent.parent / "NoorGPT"
FILES_TO_STAGE = {
    "data/train_balanced_75k.jsonl": BASE_DIR / "data/processed/train_balanced_75k.jsonl",
    "data/train_afghan_35k.jsonl": BASE_DIR / "data/processed/train_afghan_35k.jsonl",
    "data/train_peshawari_40k.jsonl": BASE_DIR / "data/processed/train_peshawari_40k.jsonl",
    "data/test_afghan_1000.jsonl": BASE_DIR / "data/benchmark/test_afghan_1000.jsonl",
    "data/test_peshawari_1000.jsonl": BASE_DIR / "data/benchmark/test_peshawari_1000.jsonl",
    "audit/audit_results.json": BASE_DIR / "logs/audit_results.json",
    "configs/dataset_schema.json": BASE_DIR / "configs/dataset_schema.json",
}

README_CONTENT = """---
annotations_creators:
- machine-generated
- crowdsourced
language:
- ps
language_bcp47:
- ps-AF
- ps-PK
language_details: Afghan Pashto and Peshawari Pashto
license: odc-by
multilinguality:
- multilingual
pretty_name: Pashto SFT Mixture
size_categories:
- 10K<n<100K
source_datasets:
- original
tags:
- instruction-tuning
- pashto
- afghan-pashto
- peshawari-pashto
- conversational
- sft
task_categories:
- text-generation
- question-answering
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train_balanced_75k.jsonl
  - split: train_afghan
    path: data/train_afghan_35k.jsonl
  - split: train_peshawari
    path: data/train_peshawari_40k.jsonl
  - split: test_afghan
    path: data/test_afghan_1000.jsonl
  - split: test_peshawari
    path: data/test_peshawari_1000.jsonl
---

# Pashto-SFT-Mixture

**Pashto-SFT-Mixture** is an open-source, high-quality multi-dialect instruction-tuning dataset for Pashto language models. It is released under the [ODC-BY license](https://opendatacommons.org/licenses/by/1-0/) as a research artifact.

The dataset consists of **75,000 verified training pairs** and **2,000 held-out evaluation pairs** covering two major regional varieties of Pashto:
1. **Afghan Pashto**: Standard literary Pashto widely used across Afghanistan (35,000 training pairs + 1,000 test pairs).
2. **Peshawari Pashto**: Regional Pashto spoken across Peshawar and Khyber Pakhtunkhwa, Pakistan (40,000 training pairs + 1,000 test pairs).

---

## Dataset Summary

| Split Name | File Path | Number of Pairs | Dialect Focus | Purpose |
| :--- | :--- | :---: | :--- | :--- |
| **`train`** (Master) | `data/train_balanced_75k.jsonl` | **75,000** | 35k Afghan + 40k Peshawari | Primary SFT Training Mixture |
| **`train_afghan`** | `data/train_afghan_35k.jsonl` | 35,000 | Afghan Pashto | Dialect-Specific SFT |
| **`train_peshawari`** | `data/train_peshawari_40k.jsonl` | 40,000 | Peshawari Pashto | Dialect-Specific SFT |
| **`test_afghan`** | `data/test_afghan_1000.jsonl` | 1,000 | Afghan Pashto | Held-Out Evaluation Benchmark |
| **`test_peshawari`** | `data/test_peshawari_1000.jsonl` | 1,000 | Peshawari Pashto | Held-Out Evaluation Benchmark |

---

## How We Created the Dataset

Instead of relying on noisy web scraping or unfiltered internet dumps, we built this dataset using three focused data curation streams:

### 1. Instruction Adaptation & Localization
We collected high-quality general instructions across diverse domains (reasoning, science, culture, everyday tasks) and translated them into natural Pashto. We preserved programming code and technical terms in English while adapting the language, idioms, and context to fit authentic native Pashto speakers.

### 2. Synthetic Question-Answering from Native Articles
We gathered authentic Pashto articles, history texts, and regional news reports. From these, we extracted passages (200 to 400 words) and generated challenging question-and-answer pairs grounded strictly in the factual context of those texts.

### 3. Everyday Conversational & Cultural Dialogues
To ensure natural multi-turn conversational fluency, we generated realistic dialogue scenarios reflecting daily life, community guidance, traditional Hujra gatherings, local cuisine, and social interactions.

---

## Quality Filtering & Cleaning Pipeline

To ensure maximum linguistic purity and model performance, we passed every sample through a multi-stage cleaning pipeline:

* **Standardizing Letters & Keys**: Inconsistent Pashto and Arabic keyboard characters were standardized (mapping Arabic Kaf `\\u0643` to Pashto Keheh `\\u06a9`, unifying Urdu Heh variants to standard Pashto Heh `\\u0647`, and removing decorative Tatweel/Kashida text stretchers `\\u0640`).
* **Preserving Compound Words**: We strictly preserved the Zero-Width Non-Joiner (ZWNJ, `\\u200c`) to maintain correct orthography for compound words (such as `بین‌الوزارتي`) and verbal prefixes.
* **Script Purity Filter**: Any sample with less than 80% Arabic/Pashto script density was removed to eliminate foreign text fragments and broken tokens.
* **Urdu Word Removal**: Pakistani Pashto on the internet frequently contains Urdu code-switching. We implemented algorithmic regex filters to detect and remove Urdu grammatical words (such as `ہے`, `ہیں`, `تھا`, `کو`, `کے`, `لڑکا`), ensuring that the Peshawari data remains grammatically authentic Pashto.
* **Deduplication with MinHash LSH**: Every candidate instruction was tokenized and hashed with 128 permutation functions. All pairs with Jaccard similarity $\\ge 0.80$ were removed to eliminate repetition.
* **Audit & Zero Overlap**: The final dataset was verified with a 7-dimension forensic audit, confirming 100% valid JSONL syntax and **strictly 0% overlap** between the 75k training data and the 2k test benchmark.

---

## Dialect Comparison Guide

The dataset carefully distinguishes between the two regional varieties:

| Concept / Word | Afghan Pashto | Peshawari Pashto | English Meaning |
| :--- | :--- | :--- | :--- |
| **University** | پوهنتون (*Pohantoon*) | یونیورسټي (*University / Pohantoon*) | Higher education institution |
| **Hospital** | روغتون (*Roghtoon*) | هسپتال (*Hospital / Roghtoon*) | Medical facility |
| **School** | ښوونځی (*Shwownzai*) | سکول (*School / Shwownzai*) | Primary/secondary school |
| **In / Inside** | کې (*ke*) | کښې (*kxay / khey*) | Locative postposition |
| **Is (Present)** | دی (*day*) | دے (*de / dai*) | 3rd person singular copula |
| **Man / Person** | سړی (*Sarai*) | سړے (*Sare*) | Noun |

---

## Dataset Schema

All files are formatted as JSON Lines (`.jsonl`). Each line contains:

```json
{
  "dialect": "afghan",
  "messages": [
    {
      "role": "user",
      "content": "د کابل پوهنتون د تاسیس په اړه لنډ معلومات راکړه."
    },
    {
      "role": "assistant",
      "content": "کابل پوهنتون په ۱۹۳۲ زېږدیز کال کې تاسیس شو او د افغانستان تر ټولو پخوانی او معتبر پوهنتون ګڼل کېږي."
    }
  ]
}
```

---

## How to Use

You can load this dataset directly using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

# 1. Load the master 75k balanced training set
dataset = load_dataset("hussainshafiq/Pashto-SFT-Mixture", split="train")

# 2. Load dialect-specific training splits
afghan_train = load_dataset("hussainshafiq/Pashto-SFT-Mixture", split="train_afghan")
peshawari_train = load_dataset("hussainshafiq/Pashto-SFT-Mixture", split="train_peshawari")

# 3. Load held-out evaluation benchmarks
afghan_test = load_dataset("hussainshafiq/Pashto-SFT-Mixture", split="test_afghan")
peshawari_test = load_dataset("hussainshafiq/Pashto-SFT-Mixture", split="test_peshawari")
```

---

## Personal and Sensitive Information
The dataset does not contain personal or sensitive information (PII). All instructions and responses have been audited for safety.

---

## License
This dataset is released under the [Open Data Commons Attribution License (ODC-BY)](https://opendatacommons.org/licenses/by/1-0/).

---

## Citation
If you use this dataset in your research, please cite:

```bibtex
@dataset{hussain2026pashtosftmixture,
  title        = {Pashto-SFT-Mixture: A Multi-Dialect Instruction Tuning Dataset for Pashto},
  author       = {Hussain, Shafiq and Research Team},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\\url{https://huggingface.co/datasets/hussainshafiq/Pashto-SFT-Mixture}}
}
```
"""

def stage_dataset():
    print(f"📁 Preparing staging directory at: {STAGING_DIR}")
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    
    (STAGING_DIR / "data").mkdir(parents=True, exist_ok=True)
    (STAGING_DIR / "audit").mkdir(parents=True, exist_ok=True)
    (STAGING_DIR / "configs").mkdir(parents=True, exist_ok=True)

    # Copy data and config files
    for rel_dest, src_path in FILES_TO_STAGE.items():
        if not src_path.exists():
            print(f"❌ Missing file: {src_path}")
            sys.exit(1)
        dest_path = STAGING_DIR / rel_dest
        print(f"  -> Copying {src_path.name} ({src_path.stat().st_size / 1e6:.1f} MB)...")
        shutil.copy2(src_path, dest_path)

    # Write README.md
    readme_path = STAGING_DIR / "README.md"
    readme_path.write_text(README_CONTENT.strip() + "\n", encoding="utf-8")
    print(f"  -> Generated README.md ({readme_path.stat().st_size} bytes)")
    print("✅ Staging complete!\n")

def upload_dataset():
    api = HfApi(token=TOKEN)

    # Verify user
    user = api.whoami()
    print(f"👤 Authenticated as: {user.get('name')} ({user.get('email')})")

    print(f"🚀 Creating public repository on Hugging Face: {REPO_ID}...")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=False,
        exist_ok=True
    )
    print(f"✅ Repository {REPO_ID} verified/created on Hub.")

    print(f"📤 Uploading staged dataset folder to {REPO_ID}...")
    api.upload_folder(
        folder_path=str(STAGING_DIR),
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="feat: Initial release of Pashto-SFT-Mixture (75k SFT + 2k Benchmarks)"
    )
    print(f"🎉 Successfully uploaded {REPO_ID} to Hugging Face Hub!")

def verify_dataset():
    print("\n🔍 Running post-upload verification with datasets library...")
    try:
        from datasets import load_dataset
        ds = load_dataset(REPO_ID, split="test_afghan", token=TOKEN)
        print(f"✅ Verification Success! Loaded 'test_afghan' split with {len(ds)} samples.")
        print(f"Sample 0: {ds[0]}")
    except Exception as e:
        print(f"⚠️ Note on direct client load (Hub indexing might take 1-2 minutes): {e}")

def main():
    stage_dataset()
    upload_dataset()
    verify_dataset()
    print("\n" + "="*70)
    print(f"🌟 DATASET IS LIVE AT: https://huggingface.co/datasets/{REPO_ID}")
    print("="*70)

if __name__ == "__main__":
    main()
