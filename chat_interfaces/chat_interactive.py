#!/usr/bin/env python3
"""
scripts/chat_interactive.py
===========================
Interactive Multi-Turn Pashto Dialect Terminal Chat Tool
Model: Google Gemma-2-9B-It QLoRA Dual-Dialect (checkpoints/gemma2_pashto_dual_75k)
Hardware: NVIDIA GeForce RTX 4090 (24GB VRAM)

Features:
- Fixed Pashto RTL / Arabic cursive rendering for LTR terminals
- Multi-turn conversation context retention
- Instant dialect switching (/afghan, /peshawari, /auto)
- Terminal rendering mode switching (/render bidi | reshaped | raw)
- Command-line single-shot test mode (--prompt "...")
"""

import os
import sys

# Prevent CUDA memory fragmentation for 256k vocab models
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Self-healing environment bootstrap: re-exec in unsloth_env if dependencies missing
try:
    import torch
    import transformers
    import peft
    import arabic_reshaper
    from bidi.algorithm import get_display
except ImportError:
    venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Pashto-LLM-V2", "unsloth_env", "bin", "python3")
    if os.path.exists(venv_py):
        print(f"[BOOTSTRAP] Activating dedicated virtual environment: {venv_py}")
        os.execv(venv_py, [venv_py] + sys.argv)
    else:
        raise

import argparse
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

AFGHAN_STEERING = "لارښوونه: خپل ځواب یوازې په سوچه، معیاري او ادبي افغاني پښتو کې ولیکه."
PESHAWARI_STEERING = "لارښوونه: خپل ځواب په روانه او عام فهمه پېښورۍ پښتو کښې ولیکه."

# Configure Arabic/Pashto reshaper to preserve all Pashto extended characters
RESHAPER = arabic_reshaper.ArabicReshaper(configuration={
    'delete_harakat': False,
    'support_ligatures': True,
})

def render_pashto(text: str, mode: str = "bidi") -> str:
    """
    Renders Pashto text properly for terminals.
    
    Modes:
      'bidi'     : Reshapes letters into cursive connected forms AND reverses RTL runs
                   so that standard LTR terminals (VS Code, Linux bash) display the text
                   in correct Right-To-Left reading order with connected letters. (DEFAULT)
      'reshaped' : Reshapes letters into cursive connected forms without reversing
                   (for terminals with native BiDi support).
      'raw'      : Original raw Unicode text without any transformations.
    """
    if not text or mode == "raw":
        return text

    lines = text.split("\n")
    rendered_lines = []
    for line in lines:
        if not line.strip():
            rendered_lines.append(line)
            continue

        has_arabic = any(
            '\u0600' <= c <= '\u06ff' or 
            '\u0750' <= c <= '\u077f' or 
            '\ufb50' <= c <= '\ufdff' or 
            '\ufe70' <= c <= '\ufeff' 
            for c in line
        )
        if not has_arabic:
            rendered_lines.append(line)
            continue

        try:
            reshaped = RESHAPER.reshape(line)
            if mode == "bidi":
                rendered_lines.append(get_display(reshaped))
            else:
                rendered_lines.append(reshaped)
        except Exception:
            rendered_lines.append(line)

    return "\n".join(rendered_lines)

def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Pashto Dual-Dialect Chat")
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="checkpoints/gemma2_pashto_dual_75k",
        help="Path to the fine-tuned LoRA checkpoint directory"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="/home/zohaib/.cache/huggingface/hub/models--unsloth--gemma-2-9b-it-bnb-4bit/snapshots/27b027bcbb6b1861b02551d5c699d5d07f29610a",
        help="Path or HuggingFace ID of the base 4-bit model"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single-shot prompt mode. If specified, generates a response and exits."
    )
    parser.add_argument(
        "--dialect",
        type=str,
        default="auto",
        choices=["auto", "afghan", "peshawari"],
        help="Initial dialect mode (auto, afghan, peshawari)"
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        default="bidi",
        choices=["bidi", "reshaped", "raw"],
        help="Terminal Pashto rendering mode: 'bidi' (default, fixes LTR reverse), 'reshaped', or 'raw'"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (use 0.0 for deterministic greedy decoding)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum generation length in tokens"
    )
    return parser.parse_args()

def print_banner(dialect_mode, render_mode, temp, top_p):
    print("\n" + "=" * 72)
    print(" 🌟  PASHTO DUAL-DIALECT INTERACTIVE CHAT (GEMMA-2-9B-IT)  🌟")
    print("=" * 72)
    print(f" Current Dialect Mode : [{dialect_mode.upper()}]")
    print(f" Terminal Render Mode : [{render_mode.upper()}] (Fixes backwards/separated Pashto)")
    print(f" Decoding Settings    : Temperature={temp}, Top-p={top_p}")
    print("-" * 72)
    print(" Dialect Commands:")
    print("   /afghan     -> Lock mode to Pure Afghan Pashto ('Awani')")
    print("   /peshawari  -> Lock mode to Peshawari Pashto ('Shaurian')")
    print("   /auto       -> Auto-detect dialect from your input")
    print(" Rendering & Control Commands:")
    print("   /render     -> Cycle rendering mode (bidi -> reshaped -> raw)")
    print("   /clear      -> Reset multi-turn conversation memory")
    print("   /temp <val> -> Change temperature (e.g. /temp 0.2 or /temp 0.8)")
    print("   /exit       -> Quit the chat session")
    print("=" * 72 + "\n")

def main():
    args = parse_args()

    # Verify adapter path
    if not os.path.exists(args.adapter_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        alt_path = os.path.join(repo_root, args.adapter_path)
        if os.path.exists(alt_path):
            args.adapter_path = alt_path
        else:
            print(f"❌ Error: Adapter path '{args.adapter_path}' does not exist.")
            sys.exit(1)

    print(f"🚀 Loading base model: {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    print(f"🔗 Attaching Pashto LoRA adapter: {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    print("✅ Model loaded successfully and ready for dialogue!\n")

    render_mode = args.render_mode

    # Single-shot execution if --prompt is provided
    if args.prompt:
        user_content = args.prompt.strip()
        if args.dialect == "afghan":
            user_content = f"{AFGHAN_STEERING}\n\n{user_content}"
        elif args.dialect == "peshawari":
            user_content = f"{PESHAWARI_STEERING}\n\n{user_content}"

        msgs = [{"role": "user", "content": user_content}]
        prompt_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

        print(f"User ❯ {render_pashto(args.prompt, mode=render_mode)}")
        print("Model ❯ ", end="", flush=True)
        
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id
        }
        if args.temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        rendered_response = render_pashto(raw_response, mode=render_mode)
        print(rendered_response)
        return

    # Interactive Multi-Turn Loop
    dialect_mode = args.dialect
    temperature = args.temperature
    top_p = args.top_p
    history = []

    print_banner(dialect_mode, render_mode, temperature, top_p)

    while True:
        try:
            prompt_indicator = f"[{dialect_mode.capitalize()}] User ❯ "
            user_input = input(prompt_indicator).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye / په مخه مو ښه! Exiting session.")
            break

        if not user_input:
            continue

        # Process Slash Commands
        cmd = user_input.lower()
        if cmd in ["/exit", "/quit", ":q"]:
            print("\n👋 Goodbye / په مخه مو ښه! Exiting session.")
            break
        elif cmd in ["/clear", "/reset"]:
            history = []
            print("🧹 Conversation memory cleared! Starting fresh session.\n")
            continue
        elif cmd == "/afghan":
            dialect_mode = "afghan"
            print("🇦🇫 Dialect mode set to: AFGHAN PASHTO ('Awani')\n")
            continue
        elif cmd == "/peshawari":
            dialect_mode = "peshawari"
            print("🇵🇰 Dialect mode set to: PESHAWARI PASHTO ('Shaurian')\n")
            continue
        elif cmd == "/auto":
            dialect_mode = "auto"
            print("🌐 Dialect mode set to: AUTO-DETECT (responds in the user's dialect)\n")
            continue
        elif cmd == "/render" or cmd.startswith("/render "):
            parts = cmd.split()
            if len(parts) == 2 and parts[1] in ["bidi", "reshaped", "raw"]:
                render_mode = parts[1]
            else:
                # Cycle modes
                cycle = {"bidi": "reshaped", "reshaped": "raw", "raw": "bidi"}
                render_mode = cycle[render_mode]
            print(f"🔤 Terminal rendering mode set to: [{render_mode.upper()}]")
            if render_mode == "bidi":
                print("   (Cursive ligatures + Right-to-Left reordering for standard LTR terminals)\n")
            elif render_mode == "reshaped":
                print("   (Cursive ligatures without reversing, for terminals with native BiDi)\n")
            else:
                print("   (Raw Unicode output)\n")
            continue
        elif cmd.startswith("/temp"):
            parts = cmd.split()
            if len(parts) == 2:
                try:
                    new_temp = float(parts[1])
                    temperature = max(0.0, min(2.0, new_temp))
                    print(f"⚙️ Temperature set to: {temperature}\n")
                except ValueError:
                    print("⚠️ Invalid temperature value. Example: /temp 0.7\n")
            else:
                print(f"Current temperature: {temperature}\n")
            continue
        elif cmd in ["/help", "/?"]:
            print_banner(dialect_mode, render_mode, temperature, top_p)
            continue

        # Build prompt turn with steering if requested
        turn_content = user_input
        if dialect_mode == "afghan":
            turn_content = f"{AFGHAN_STEERING}\n\n{user_input}"
        elif dialect_mode == "peshawari":
            turn_content = f"{PESHAWARI_STEERING}\n\n{user_input}"

        history.append({"role": "user", "content": turn_content})

        # Apply chat template
        prompt_text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

        # Echo formatted input to let user visually verify their prompt
        print(f"❯ Query: {render_pashto(user_input, mode=render_mode)}")
        print("🤖 Model ❯ ", end="", flush=True)

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id
        }
        if temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # Extract generated response text to record in history
        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        assistant_response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        history.append({"role": "assistant", "content": assistant_response})
        
        # Render the response with cursive ligatures and proper reading order
        rendered_output = render_pashto(assistant_response, mode=render_mode)
        print(rendered_output)
        print()  # newline separator

if __name__ == "__main__":
    main()
