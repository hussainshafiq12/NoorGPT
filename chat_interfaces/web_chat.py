#!/usr/bin/env python3
"""
scripts/web_chat.py
===================
Modern Native RTL Browser Chat Interface for Pashto Dual-Dialect LLM
Model: Google Gemma-2-9B-It QLoRA Pashto Dual-Dialect (checkpoints/gemma2_pashto_dual_75k)
Hardware: NVIDIA GeForce RTX 4090 (24GB VRAM)

Features:
- 100% Native Web RTL & Arabic/Pashto Font Shaping (Zero terminal glitches)
- Multi-turn conversation retention
- Instant Afghan / Peshawari / Auto dialect switching
- Beautiful responsive UI on http://localhost:7860
"""

import os
import sys

# Prevent CUDA memory fragmentation for 256k vocab models
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Self-healing environment bootstrap
try:
    import torch
    import transformers
    import peft
    import fastapi
    import uvicorn
except ImportError:
    venv_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Pashto-LLM-V2", "unsloth_env", "bin", "python3")
    if os.path.exists(venv_py):
        print(f"[BOOTSTRAP] Activating dedicated virtual environment: {venv_py}")
        os.execv(venv_py, [venv_py] + sys.argv)
    else:
        raise

import argparse
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

AFGHAN_STEERING = "لارښوونه: خپل ځواب یوازې په سوچه، معیاري او ادبي افغاني پښتو کې ولیکه."
PESHAWARI_STEERING = "لارښوونه: خپل ځواب په روانه او عام فهمه پېښورۍ پښتو کښې ولیکه."

app = FastAPI(title="Pashto Dual-Dialect LLM Web Chat")

# Global state
MODEL = None
TOKENIZER = None

HTML_PAGE = """<!DOCTYPE html>
<html lang="ps" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>د پښتو دوه ګوني ژبني ماډل سره خبرې (Gemma-2-9B Pashto)</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --chat-bg: #1e293b;
            --user-bubble: #2563eb;
            --bot-bubble: #334155;
            --text-color: #f8fafc;
            --accent: #38bdf8;
            --border: #475569;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', Tahoma, 'Noto Sans Pashto', 'Noto Naskh Arabic', sans-serif; }
        body { background: var(--bg-color); color: var(--text-color); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
        header { background: #0b1120; padding: 14px 20px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
        header h1 { font-size: 1.15rem; font-weight: 600; color: var(--accent); }
        .controls { display: flex; gap: 8px; align-items: center; }
        .dialect-btn { background: #1e293b; border: 1px solid var(--border); color: #cbd5e1; padding: 6px 14px; border-radius: 20px; cursor: pointer; font-size: 0.85rem; transition: all 0.2s; }
        .dialect-btn.active { background: var(--accent); color: #0b1120; font-weight: 600; border-color: var(--accent); }
        .clear-btn { background: #ef4444; color: white; border: none; padding: 6px 14px; border-radius: 20px; cursor: pointer; font-size: 0.85rem; }
        #chat-window { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
        .msg { max-width: 80%; padding: 14px 18px; border-radius: 16px; line-height: 1.8; font-size: 1.05rem; word-break: break-word; }
        .msg.user { align-self: flex-start; background: var(--user-bubble); color: white; border-bottom-right-radius: 4px; }
        .msg.bot { align-self: flex-end; background: var(--bot-bubble); color: #f1f5f9; border-bottom-left-radius: 4px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2); }
        .meta-tag { font-size: 0.72rem; opacity: 0.75; margin-bottom: 4px; font-weight: bold; }
        footer { padding: 16px 24px; background: #0b1120; border-top: 1px solid var(--border); }
        form { display: flex; gap: 12px; }
        input[type="text"] { flex: 1; background: #1e293b; border: 1px solid var(--border); border-radius: 28px; padding: 14px 22px; color: white; font-size: 1.05rem; outline: none; }
        input[type="text"]:focus { border-color: var(--accent); }
        button[type="submit"] { background: var(--accent); color: #0b1120; border: none; border-radius: 28px; padding: 0 28px; font-weight: bold; font-size: 1rem; cursor: pointer; transition: opacity 0.2s; }
        button[type="submit"]:disabled { opacity: 0.5; cursor: not-allowed; }
        .typing { display: inline-block; animation: blink 1s infinite; }
        @keyframes blink { 0%, 100% { opacity: 0.2; } 50% { opacity: 1; } }
    </style>
</head>
<body>
    <header>
        <h1>🌟 پښتو دوه ګونی ژبنی ماډل (Pashto Dual-Dialect LLM)</h1>
        <div class="controls">
            <button class="dialect-btn active" data-dialect="auto" onclick="setDialect('auto')">🌐 اتومات (Auto)</button>
            <button class="dialect-btn" data-dialect="afghan" onclick="setDialect('afghan')">🇦🇫 افغاني (Afghan)</button>
            <button class="dialect-btn" data-dialect="peshawari" onclick="setDialect('peshawari')">🇵🇰 پېښورۍ (Peshawari)</button>
            <button class="clear-btn" onclick="clearChat()">🧹 پاکول</button>
        </div>
    </header>

    <div id="chat-window">
        <div class="msg bot">
            <div class="meta-tag">🤖 ماډل (Gemma-2-9B Pashto)</div>
            سلامونه او نېکې هیلې! زه د پښتو دوه ګوني ماډل یم. تاسو کولی شئ په افغاني يا پېښورۍ پښتو کې هره پوښتنه وپوښتئ.
        </div>
    </div>

    <footer>
        <form id="chat-form" onsubmit="sendMessage(event)">
            <input type="text" id="user-input" placeholder="خپله پوښتنه دلته ولیکئ..." autocomplete="off" autofocus />
            <button type="submit" id="send-btn">لېږل ❯</button>
        </form>
    </footer>

    <script>
        let currentDialect = 'auto';
        let history = [];

        function setDialect(d) {
            currentDialect = d;
            document.querySelectorAll('.dialect-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.dialect === d);
            });
        }

        function clearChat() {
            history = [];
            document.getElementById('chat-window').innerHTML = `
                <div class="msg bot">
                    <div class="meta-tag">🤖 ماډل</div>
                    د خبرو حافظه پاکه شوه. اوس یوه نوې موضوع پیل کولی شئ!
                </div>
            `;
        }

        async function sendMessage(e) {
            e.preventDefault();
            const input = document.getElementById('user-input');
            const text = input.value.trim();
            if (!text) return;

            const chatWin = document.getElementById('chat-window');
            const sendBtn = document.getElementById('send-btn');

            // Render user bubble
            const userMsgDiv = document.createElement('div');
            userMsgDiv.className = 'msg user';
            userMsgDiv.innerHTML = `<div class="meta-tag">👤 تاسو (${currentDialect.toUpperCase()})</div>${text}`;
            chatWin.appendChild(userMsgDiv);

            input.value = '';
            input.disabled = true;
            sendBtn.disabled = true;

            // Render bot loading bubble
            const botMsgDiv = document.createElement('div');
            botMsgDiv.className = 'msg bot';
            botMsgDiv.innerHTML = `<div class="meta-tag">🤖 ماډل</div><span class="typing">د ځواب چمتو کول...</span>`;
            chatWin.appendChild(botMsgDiv);
            chatWin.scrollTop = chatWin.scrollHeight;

            try {
                const res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: text,
                        dialect: currentDialect,
                        history: history
                    })
                });
                const data = await res.json();
                botMsgDiv.innerHTML = `<div class="meta-tag">🤖 ماډل</div>${data.response.replace(/\\n/g, '<br>')}`;
                history = data.history;
            } catch (err) {
                botMsgDiv.innerHTML = `<div class="meta-tag" style="color:#ef4444">⚠️ تېروتنه</div>تېروتنه رامنځته شوه، مهرباني وکړئ بیا هڅه وکړئ.`;
            } finally {
                input.disabled = false;
                sendBtn.disabled = false;
                input.focus();
                chatWin.scrollTop = chatWin.scrollHeight;
            }
        }
    </script>
</body>
</html>
"""

class ChatRequest(BaseModel):
    message: str
    dialect: str = "auto"
    history: List[Dict[str, str]] = []

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return HTML_PAGE

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    global MODEL, TOKENIZER

    user_text = req.message.strip()
    turn_content = user_text
    if req.dialect == "afghan":
        turn_content = f"{AFGHAN_STEERING}\n\n{user_text}"
    elif req.dialect == "peshawari":
        turn_content = f"{PESHAWARI_STEERING}\n\n{user_text}"

    # Build conversation history
    full_history = list(req.history)
    full_history.append({"role": "user", "content": turn_content})

    prompt_text = TOKENIZER.apply_chat_template(full_history, tokenize=False, add_generation_prompt=True)
    inputs = TOKENIZER(prompt_text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = MODEL.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=TOKENIZER.pad_token_id
        )

    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response_text = TOKENIZER.decode(gen_tokens, skip_special_tokens=True).strip()

    # Append response to history
    full_history.append({"role": "assistant", "content": response_text})

    return JSONResponse({
        "response": response_text,
        "history": full_history
    })

def main():
    parser = argparse.ArgumentParser(description="Web Chat for Pashto Dual-Dialect LLM")
    parser.add_argument("--port", type=int, default=7860, help="Port to run web server on")
    parser.add_argument("--adapter_path", type=str, default="checkpoints/gemma2_pashto_dual_75k")
    parser.add_argument("--base_model", type=str, default="/home/zohaib/.cache/huggingface/hub/models--unsloth--gemma-2-9b-it-bnb-4bit/snapshots/27b027bcbb6b1861b02551d5c699d5d07f29610a")
    args = parser.parse_args()

    global MODEL, TOKENIZER
    print(f"🚀 Loading base model: {args.base_model}...")
    TOKENIZER = AutoTokenizer.from_pretrained(args.base_model)
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token
    TOKENIZER.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    print(f"🔗 Attaching Pashto LoRA adapter: {args.adapter_path}...")
    MODEL = PeftModel.from_pretrained(base_model, args.adapter_path)
    MODEL.eval()

    print(f"\n🌟 Web Chat Server running at: http://localhost:{args.port}\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()

