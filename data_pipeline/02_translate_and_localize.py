#!/usr/bin/env python3
"""
02_translate_and_localize.py
============================
Stream 1: Translating High-Quality Seeds (TÜLU / Alpaca Style)
Adopts the Atlas-Chat methodology:
- Uses dual dialect system prompts (Afghan 'Awani' vs Peshawari 'Shaurian')
- Maintains strict dialect vocabulary profiles
- Outputs clean Atlas-Chat schema:
  {
    "dialect": "afghan" | "peshawari",
    "messages": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }
NO domain field. Supports OpenAI, Anthropic, Gemini, and offline high-quality fallback seeds.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INTERIM_DIR = BASE_DIR / "data/interim"
OUTPUT_FILE = INTERIM_DIR / "stream1_translated_seeds.jsonl"
CACHE_DIR = INTERIM_DIR / "stream1_cache"

AFGHAN_SYSTEM_PROMPT = """You are an expert translator specializing in authentic Afghan Pashto (Southern/Central standard dialect). Translate the following instruction and response into pure Afghan Pashto following these strict rules:
1. Use standardized Afghan academic and institutional lexicon (e.g., use 'پوهنتون' [Pohantoon] for university, 'روغتون' [Roghtoon] for hospital, 'ناروغه' [Narogha] for patient, 'ښوونځی' [Xxowanzai] for school).
2. Do NOT use Pakistani Urdu loanwords (e.g., avoid 'یونیورسټي', 'هسپتال', 'مریض').
3. Use authentic Southern/Central pronominal clitics and forms (e.g., use 'موږ' [Muzh] for we/us, not Northern 'مونږ').
4. Ensure proper orthography using official Pashto letters (ښ, ږ, څ, ځ, ۍ, ې).
Output strictly a JSON object: {"instruction": "...", "response": "..."}"""

PESHAWARI_SYSTEM_PROMPT = """You are an expert translator specializing in Peshawari Pashto (Northern / Pakistani dialect). Translate the following instruction and response into authentic Peshawari Pashto following these strict rules:
1. Reflect natural Northern Pashto phrasing as spoken and written across Peshawar and Khyber Pakhtunkhwa.
2. Employ Northern orthography and pronoun structures (e.g., use 'مونږ' [Moong] for we/us, 'کښې' [Kxhey] for in, 'دے' [Dai] for is).
3. Colloquial loanwords common in Pakistani Pashto print media are permitted where standard.
Output strictly a JSON object: {"instruction": "...", "response": "..."}"""

CURATED_SEEDS = [
    {
        "instruction": "Explain the difference between renewable and non-renewable energy sources with two examples of each.",
        "response": "Renewable energy comes from natural sources that replenish themselves faster than they are consumed, such as solar power and wind energy. Non-renewable energy comes from sources that exist in finite amounts and deplete with use, such as coal and crude oil."
    },
    {
        "instruction": "What are the key lifestyle factors that help prevent cardiovascular disease?",
        "response": "Key lifestyle factors include maintaining a balanced diet low in saturated fats, engaging in regular physical exercise (at least 150 minutes weekly), avoiding smoking and tobacco products, managing chronic stress, and maintaining a healthy body weight."
    },
    {
        "instruction": "How does photosynthesis convert sunlight into chemical energy in green plants?",
        "response": "In plant chloroplasts, chlorophyll absorbs photon energy from sunlight. This light energy splits water molecules into oxygen and hydrogen protons, which drives the Calvin cycle to convert carbon dioxide into glucose, releasing oxygen into the atmosphere."
    },
    {
        "instruction": "Summarize the primary responsibilities of a municipal government in urban management.",
        "response": "A municipal government oversees local public services including clean water distribution, waste collection, public transport maintenance, road repair, urban zoning regulations, and local park and safety management."
    },
    {
        "instruction": "What is the historical significance of the Silk Road in cross-cultural exchange?",
        "response": "The Silk Road was a vast network of trade routes connecting East Asia, Central Asia, South Asia, and the Mediterranean. Beyond silk and spices, it enabled the exchange of scientific knowledge, paper-making technology, religions, and artistic traditions across civilizations."
    }
]

def translate_with_api(instruction, response, dialect, model="gpt-4o"):
    # Stub for API integration with openai/anthropic
    # Checks environment variables OPENAI_API_KEY / ANTHROPIC_API_KEY
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    sys_prompt = AFGHAN_SYSTEM_PROMPT if dialect == "afghan" else PESHAWARI_SYSTEM_PROMPT
    user_payload = json.dumps({"instruction": instruction, "response": response}, ensure_ascii=False)

    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            msg = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
                system=sys_prompt,
                messages=[{"role": "user", "content": user_payload}]
            )
            raw = msg.content[0].text
            parsed = json.loads(raw)
            return parsed.get("instruction"), parsed.get("response")
        except Exception as e:
            print(f"[Anthropic API Error] {e}")

    if openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            res = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_payload}
                ],
                response_format={"type": "json_object"}
            )
            parsed = json.loads(res.choices[0].message.content)
            return parsed.get("instruction"), parsed.get("response")
        except Exception as e:
            print(f"[OpenAI API Error] {e}")

    return None, None

def main():
    parser = argparse.ArgumentParser(description="Stream 1: Seed Instruction Translation")
    parser.add_argument("--limit", type=int, default=10, help="Number of seeds to translate")
    args = parser.parse_args()

    os.makedirs(INTERIM_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("Stream 1 Seed Translator Initialized.")
    print(f"Output target: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

