# NoorGPT: Adapting Large Language Models for Low-Resource Pashto Dialects

<p align="center">
  <a href="https://huggingface.co/hussainshafiq/NoorGPT"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Weights-blue?style=for-the-badge" alt="Hugging Face Model"></a>
  <a href="https://huggingface.co/datasets/hussainshafiq/Pashto-SFT-Mixture"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-75k%20Dataset-green?style=for-the-badge" alt="Hugging Face Dataset"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-yellow.svg?style=for-the-badge" alt="License"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.1%2B-orange?style=for-the-badge" alt="PyTorch">
</p>

Official codebase, data pipelines, evaluation benchmarks, and publication figures for:

> **NoorGPT: Adapting Large Language Models for Low-Resource Pashto Dialects**  
> *Shafiq Hussain*

NoorGPT is the first instruction-tuned large language model specialized for **dual-dialect Pashto**, addressing both **Afghan Pashto** and **Peshawari Pashto**. Built upon `google/gemma-2-9b-it`, NoorGPT adapts 216.1M trainable parameters using QLoRA with response-only loss masking, resolving cross-dialect confusion and eliminating Urdu code-switching contamination.

---

## 🚀 Key Achievements

* **Meta Belebele Pashto SOTA:** Elevates reading comprehension from **52.66%** (Base Gemma-2-9B) to **67.21%** (**+14.55% absolute gain**).
* **Dual-Dialect Balanced Mixture:** Trained on **[Pashto-SFT-Mixture](https://huggingface.co/datasets/hussainshafiq/Pashto-SFT-Mixture)** (75,000 instruction-response pairs; 35k Afghan + 40k Peshawari).
* **Cross-Dialect Fidelity:** Achieves **99.4%** dialect adherence on Afghan prompts and **98.7%** on Peshawari prompts.
* **Near-Zero Language Bleed:** Curtails Urdu intrusion down to **0.4%** (compared to 21.5% in base Gemma and 26.8% in Qwen2.5).

---

## 📊 Benchmark Results (Table 2)

Evaluated under greedy decoding on Meta Belebele (`pbt_Arab`) and held-out generative dialect benchmarks (1,000 Afghan and 1,000 Peshawari pairs):

| Model | Belebele Acc (%) $\uparrow$ | Afghan chrF++ $\uparrow$ | Afghan BLEU $\uparrow$ | Peshawari chrF++ $\uparrow$ | Peshawari BLEU $\uparrow$ | Macro Avg chrF++ $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-7B-Instruct** | 43.20% | 34.20 | 11.40 | 31.80 | 10.80 | 33.00 |
| **Afghan-Only SFT (Ablation)** | 48.60% | 46.50 | 21.30 | 36.40 | 12.10 | 41.50 |
| **Base Gemma-2-9B-It (Zero-Shot)** | 52.66% | 19.88 | 24.81 | 24.00 | 32.77 | 21.94 |
| **NoorGPT (Ours)** | **67.21%** | **55.94** | **65.15** | **45.93** | **82.79** | **50.93** |

---

## 🛠️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/hussainshafiq12/NoorGPT.git
cd NoorGPT

# Install dependencies
pip install -r requirements.txt
```

---

## 💻 Quickstart Inference

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "google/gemma-2-9b-it"
adapter_id = "hussainshafiq/NoorGPT"

tokenizer = AutoTokenizer.from_pretrained(adapter_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(base_model, adapter_id)

messages = [
    {"role": "user", "content": "د کابل پوهنتون د تاریخ په اړه راته لنډ معلومات راکړه."}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
```

---

## 💬 Interactive Chat Tools

* **Terminal Chat (with Arabic reshaper & RTL fixes):**
  ```bash
  bash chat.sh
  ```
* **Local Web Chat UI (FastAPI at `http://localhost:7860`):**
  ```bash
  bash web_chat.sh
  ```

---

## 📈 Publication Figures

Vector PDFs and 300 DPI PNGs are available in `paper_figures/`:
* **Figure 1:** Belebele Reading Comprehension Accuracy across core evaluated models.
* **Figure 2:** Multi-Dialect Performance Breakdown across all 6 metrics.
* **Figure 3:** Empirical Training Loss Convergence across 18,750 steps (Initial loss: 2.3934 $\rightarrow$ Final loss: 0.8280).
* **Figure 4:** Dialect Marker Adherence & Urdu Intrusion Suppression Analysis.

To re-render all camera-ready figures:
```bash
python paper_figures/generate_paper_figures.py
```

---

## 🔗 Data & Model Availability

* **Model Weights:** Hosted on Hugging Face at [`hussainshafiq/NoorGPT`](https://huggingface.co/hussainshafiq/NoorGPT).
* **Training Mixture:** Hosted on Hugging Face at [`hussainshafiq/Pashto-SFT-Mixture`](https://huggingface.co/datasets/hussainshafiq/Pashto-SFT-Mixture).

---

## 📖 Citation

```bibtex
% Paper: NoorGPT: Adapting Large Language Models for Low-Resource Pashto Dialects
% BibTeX citation will be updated upon paper publication / acceptance.
```

---

## 📄 License
This repository is released under the [Apache 2.0 License](LICENSE).
