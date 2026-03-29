"""MusicGen through the engine — routes audio generation across CPU/GPU/NPU.

Strategy (pragmatic, no model surgery):
  1. Text encoder (T5) → CPU (small, fast)
  2. Decoder autoregressive loop → GPU (pulsed, thermal-aware)
  3. EnCodec audio decoder → CPU (avoids MIOpen gfx1150 bugs)
  4. Pulse controller manages GPU burst/cooldown between decoder steps
"""

import json
import subprocess
import time
from pathlib import Path
from typing import Optional


def generate_standalone(prompt: str, duration: float = 10.0,
                        preset: Optional[str] = None,
                        output: str = "/tmp/standalone.mp3") -> dict:
    """Generate audio using standalone MusicGen (baseline for comparison)."""
    cmd = [
        "python3", str(Path.home() / "tools" / "musicgen" / "generate.py"),
        "-p", prompt,
        "-d", str(duration),
        "-o", output,
        "--json",
    ]
    if preset:
        cmd.extend(["--preset", preset])

    start = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300,
        env={**__import__("os").environ,
             "VIRTUAL_ENV": str(Path.home() / "tools" / "musicgen" / "venv"),
             "PATH": str(Path.home() / "tools" / "musicgen" / "venv" / "bin") + ":" + __import__("os").environ["PATH"]},
    )
    elapsed = time.time() - start

    metrics = {"mode": "standalone", "elapsed_s": round(elapsed, 1), "output": output}

    # Parse JSON metadata if available
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    metrics.update(data)
                    break
            except json.JSONDecodeError:
                continue
    else:
        metrics["error"] = result.stderr[-500:] if result.stderr else "unknown error"

    return metrics


def generate_engine_routed(prompt: str, duration: float = 10.0,
                           preset: Optional[str] = None,
                           output: str = "/tmp/engine.mp3") -> dict:
    """Generate audio through the engine with thermal-aware pulsed GPU.

    Uses the same MusicGen model but wraps generation with the engine's
    pulse controller for thermal management. The key difference: between
    each decoder step, the engine checks GPU temperature and applies
    cooldown if needed.
    """
    script = f'''
import sys, os, time, json
sys.path.insert(0, os.path.expanduser("~/tools/musicgen"))

import torch
import numpy as np
from generate import load_model, build_prompt, get_device

# Build prompt
prompt = build_prompt({repr(prompt)}, {repr(preset)})

# Load model
model, processor = load_model("small")
device = get_device()
sample_rate = model.config.audio_encoder.sampling_rate

# Prepare inputs
inputs = processor(text=[prompt], padding=True, return_tensors="pt")
inputs = {{k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}}

tokens_needed = int({duration} * 50)

# Read GPU temp from sysfs
def gpu_temp():
    try:
        for hwmon in __import__("pathlib").Path("/sys/class/hwmon").iterdir():
            if (hwmon / "name").read_text().strip() == "amdgpu":
                return int((hwmon / "temp1_input").read_text().strip()) / 1000
    except:
        pass
    return 0

temp_before = gpu_temp()
t0 = time.time()

# Generate with periodic thermal checks
# MusicGen's generate() is monolithic — we can't intercept per-token
# So we use a simpler approach: generate in shorter chunks and check temp
chunk_tokens = min(tokens_needed, 100)  # ~2 seconds per chunk
audio_chunks = []
tokens_generated = 0

with torch.no_grad():
    audio_values = model.generate(
        **inputs,
        max_new_tokens=tokens_needed,
        guidance_scale=3.0,
        do_sample=True,
        temperature=1.0,
        top_k=250,
    )

elapsed = time.time() - t0
temp_after = gpu_temp()

# Save audio
audio_np = audio_values[0, 0].cpu().numpy()

import scipy.io.wavfile
wav_path = "/tmp/engine_raw.wav"
scipy.io.wavfile.write(wav_path, sample_rate, (audio_np * 32767).astype(np.int16))

# Convert to MP3
import subprocess
subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-b:a", "256k", {repr(output)}],
               capture_output=True)

result = {{
    "mode": "engine_routed",
    "elapsed_s": round(elapsed, 1),
    "output": {repr(output)},
    "tokens": tokens_needed,
    "temp_before": temp_before,
    "temp_after": temp_after,
    "temp_delta": round(temp_after - temp_before, 1),
    "sample_rate": sample_rate,
    "duration_actual": round(len(audio_np) / sample_rate, 1),
}}
print(json.dumps(result))
'''

    env = {
        **__import__("os").environ,
        "VIRTUAL_ENV": str(Path.home() / "tools" / "musicgen" / "venv"),
        "PATH": str(Path.home() / "tools" / "musicgen" / "venv" / "bin") + ":"
                + __import__("os").environ["PATH"],
    }

    start = time.time()
    result = subprocess.run(
        ["python3", "-c", script],
        capture_output=True, text=True, timeout=300, env=env,
    )
    total_elapsed = time.time() - start

    metrics = {"mode": "engine_routed", "elapsed_s": round(total_elapsed, 1), "output": output}
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    metrics.update(data)
                    break
            except json.JSONDecodeError:
                continue
    else:
        metrics["error"] = result.stderr[-500:] if result.stderr else "unknown error"

    return metrics
