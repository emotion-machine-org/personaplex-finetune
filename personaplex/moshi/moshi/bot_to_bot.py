# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Bot-to-bot full-duplex conversation using multi-process architecture.

Each bot runs in its own subprocess with CUDA_VISIBLE_DEVICES restricting it
to a single physical GPU (seen as cuda:0).  This lets CUDA graphs work
correctly — they break across multiple GPUs in one process.

The main (orchestrator) process cross-feeds PCM frames between the two
workers via multiprocessing queues with a 1-frame (80ms) natural delay.
Both workers process each frame in parallel.

Output: stereo WAV (Bot A = left, Bot B = right) + per-bot text transcripts.
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from typing import List, Optional

import numpy as np
import sentencepiece
import sphn

from .client_utils import make_log
from .models import loaders
from .offline import _get_voice_prompt_dir


def _log(level: str, msg: str):
    print(make_log(level, msg))


def _worker(
    gpu_id: int,
    config: dict,
    in_q: mp.Queue,
    out_q: mp.Queue,
):
    """Bot worker process.

    Sets CUDA_VISIBLE_DEVICES so only one physical GPU is visible (as cuda:0),
    loads the full model stack, then enters a frame-processing loop.
    """
    # Must be set before any CUDA initialization.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Ensure CUDA graphs are enabled (undo any leftover env from testing).
    os.environ.pop("NO_CUDA_GRAPH", None)

    import torch
    from .client_utils import make_log as _make_log
    from .models import LMGen, loaders as _loaders
    from .offline import seed_all, warmup, wrap_with_system_tags

    name = config["name"]
    device = "cuda:0"

    def log(level, msg):
        print(_make_log(level, msg))

    if config.get("seed") not in (None, -1):
        seed_all(config["seed"])

    try:
        with torch.no_grad():
            # --- Mimi ---
            log("info", f"[Bot {name}] loading mimi (GPU {gpu_id})")
            mimi_weight = config["mimi_weight"]
            if mimi_weight is None:
                from huggingface_hub import hf_hub_download
                mimi_weight = hf_hub_download(config["hf_repo"], _loaders.MIMI_NAME)
            mimi = _loaders.get_mimi(mimi_weight, device)
            other_mimi = _loaders.get_mimi(mimi_weight, device)

            # --- Moshi LM ---
            log("info", f"[Bot {name}] loading moshi (GPU {gpu_id})")
            moshi_weight = config["moshi_weight"]
            if moshi_weight is None:
                from huggingface_hub import hf_hub_download
                moshi_weight = hf_hub_download(config["hf_repo"], _loaders.MOSHI_NAME)
            lm = _loaders.get_moshi_lm(
                moshi_weight, device=device, cpu_offload=config["cpu_offload"]
            )
            lm.eval()

            # --- LMGen ---
            frame_size = int(mimi.sample_rate / mimi.frame_rate)
            lm_gen = LMGen(
                lm,
                audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
                sample_rate=mimi.sample_rate,
                device=device,
                frame_rate=mimi.frame_rate,
                save_voice_prompt_embeddings=False,
                use_sampling=not config["greedy"],
                temp=config["temp_audio"],
                temp_text=config["temp_text"],
                top_k=config["topk_audio"],
                top_k_text=config["topk_text"],
                rep_penalty=config.get("rep_penalty", 0.0),
                rep_penalty_window=config.get("rep_penalty_window", 30),
            )

            # --- Streaming mode ---
            mimi.streaming_forever(1)
            other_mimi.streaming_forever(1)
            lm_gen.streaming_forever(1)

            # --- Warmup (primes CUDA graphs on cuda:0) ---
            log("info", f"[Bot {name}] warming up")
            warmup(mimi, other_mimi, lm_gen, device, frame_size)

            # --- Voice prompt ---
            vp = config["voice_prompt_path"]
            if vp.endswith(".pt"):
                lm_gen.load_voice_prompt_embeddings(vp)
            else:
                lm_gen.load_voice_prompt(vp)

            # --- Text prompt ---
            tokenizer = sentencepiece.SentencePieceProcessor(config["tokenizer_path"])
            tp = config["text_prompt"]
            lm_gen.text_prompt_tokens = (
                tokenizer.encode(wrap_with_system_tags(tp)) if tp else None
            )

            # --- System prompt phases (voice -> silence -> text -> silence) ---
            mimi.reset_streaming()
            other_mimi.reset_streaming()
            lm_gen.reset_streaming()
            lm_gen.step_system_prompts(mimi)
            mimi.reset_streaming()

            # --- Greeting (optional) ---
            greeting = config.get("greeting")
            if greeting:
                lm_gen.greeting_tokens = tokenizer.encode(greeting)
                lm_gen.prepare_greeting()

            # --- Screech guard (inside worker, on LMGen directly) ---
            if config.get("screech_guard"):
                lm_gen.screech_guard = True
                lm_gen.screech_hf_threshold = config["screech_hf_threshold"]
                lm_gen.screech_consec_frames = config["screech_confirm_frames"]
                lm_gen.screech_cooldown_frames = int(config["screech_cooldown"] * mimi.frame_rate)
                lm_gen.screech_recovery_token_ids = tokenizer.encode("OK")
                lm_gen._screech_mimi = mimi  # for PCM-level spectral analysis
                log("info", f"[Bot {name}] screech guard enabled (hf>{config['screech_hf_threshold']}, "
                    f"consec>={config['screech_confirm_frames']})")

            log("info", f"[Bot {name}] ready (GPU {gpu_id})")
            out_q.put(("ready", frame_size, float(mimi.frame_rate), int(mimi.sample_rate)))

            # --- Frame processing loop ---
            while True:
                msg = in_q.get()
                if msg is None:
                    break

                # Handle structured messages: context or nudge + pcm
                if isinstance(msg, dict) and "context" in msg:
                    lm_gen.inject_context(msg["context"], tokenizer)
                    pcm_in = torch.from_numpy(msg["pcm"]).to(device)
                elif isinstance(msg, dict) and "nudge" in msg:
                    lm_gen.inject_continuation(msg["nudge"])
                    pcm_in = torch.from_numpy(msg["pcm"]).to(device)
                else:
                    pcm_in = torch.from_numpy(msg).to(device)

                codes = mimi.encode(pcm_in)
                _ = other_mimi.encode(pcm_in)

                result_pcm = None
                result_text = None
                result_injecting = False

                for c in range(codes.shape[-1]):
                    tokens = lm_gen.step(codes[:, :, c : c + 1])
                    if tokens is not None:
                        pcm_out = mimi.decode(tokens[:, 1:9])
                        _ = other_mimi.decode(tokens[:, 1:9])
                        result_pcm = pcm_out.detach().cpu().numpy()  # [1, 1, frame_size]
                        result_text = tokens[0, 0, 0].item()
                        result_injecting = lm_gen._injecting_context

                out_q.put((result_pcm, result_text, result_injecting))

    except Exception as e:
        log("error", f"[Bot {name}] worker crashed: {e}")
        import traceback
        traceback.print_exc()
        out_q.put(("error", str(e)))


def _decode_text_token(
    token_id: int, tokenizer: sentencepiece.SentencePieceProcessor
) -> str:
    if token_id not in (0, 3):
        return tokenizer.id_to_piece(token_id).replace("\u2581", " ")
    return ["EPAD", "BOS", "EOS", "PAD"][token_id]


def main():
    parser = argparse.ArgumentParser(
        description="Bot-to-bot conversation (multi-process, CUDA graphs enabled)."
    )

    # Output
    parser.add_argument("--output-wav", required=True, help="Stereo WAV path (A=left, B=right)")
    parser.add_argument("--output-text-a", required=True, help="Bot A transcript JSON")
    parser.add_argument("--output-text-b", required=True, help="Bot B transcript JSON")
    parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds")

    # Prompts
    parser.add_argument("--text-prompt-a", type=str, default="", help="System prompt for Bot A")
    parser.add_argument("--text-prompt-b", type=str, default="", help="System prompt for Bot B")
    parser.add_argument("--voice-prompt-a", required=True, help="Voice prompt file for Bot A")
    parser.add_argument("--voice-prompt-b", required=True, help="Voice prompt file for Bot B")
    parser.add_argument("--greeting-a", type=str, default=None, help="Text for Bot A to speak first")

    # GPUs (physical GPU indices, not CUDA device strings)
    parser.add_argument("--gpu-a", type=int, default=0, help="Physical GPU index for Bot A")
    parser.add_argument("--gpu-b", type=int, default=1, help="Physical GPU index for Bot B")

    # Model weights
    parser.add_argument("--moshi-weight", type=str, default=None, help="Shared Moshi checkpoint")
    parser.add_argument("--moshi-weight-a", type=str, default=None, help="Override for Bot A")
    parser.add_argument("--moshi-weight-b", type=str, default=None, help="Override for Bot B")
    parser.add_argument("--mimi-weight", type=str, default=None, help="Shared Mimi checkpoint")
    parser.add_argument("--mimi-weight-a", type=str, default=None, help="Override for Bot A")
    parser.add_argument("--mimi-weight-b", type=str, default=None, help="Override for Bot B")

    # Shared assets
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path")
    parser.add_argument("--voice-prompt-dir", type=str, default=None)
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)

    # Sampling
    parser.add_argument("--temp-audio", type=float, default=0.8)
    parser.add_argument("--temp-text", type=float, default=0.7)
    parser.add_argument("--topk-audio", type=int, default=250)
    parser.add_argument("--topk-text", type=int, default=25)
    parser.add_argument("--rep-penalty", type=float, default=0.0,
                       help="Repetition penalty for text tokens (>1.0 to enable, e.g. 1.3)")
    parser.add_argument("--rep-penalty-window", type=int, default=30)
    parser.add_argument("--greedy", action="store_true")

    # Nudge (silence-breaking)
    parser.add_argument("--nudge-after", type=float, default=0,
                        help="Seconds of mutual silence before nudging Bot A (0=disabled)")
    parser.add_argument("--max-nudges", type=int, default=0,
                        help="Maximum nudge injections per dialogue (0=unlimited)")

    # Context injection (puppeteer)
    parser.add_argument("--context-injections", type=str, default=None,
                        help="JSON file with scheduled context injections: "
                             "[{\"frame\": int, \"text\": str}, ...]")

    # Screech guard
    parser.add_argument("--screech-guard", action="store_true",
                        help="Enable screech detection and auto-recovery via 'OK' injection")
    parser.add_argument("--screech-hf-threshold", type=float, default=0.40,
                        help="HF energy ratio threshold for screech detection (default: 0.40)")
    parser.add_argument("--screech-confirm-frames", type=int, default=3,
                        help="Consecutive PAD+HF frames needed to confirm screech (default: 3)")
    parser.add_argument("--screech-cooldown", type=float, default=0.5,
                        help="Seconds of cooldown between screech injections (default: 0.5)")

    # Misc
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--cpu-offload", action="store_true")

    args = parser.parse_args()

    # --- Resolve shared paths in main process ---
    from huggingface_hub import hf_hub_download

    hf_hub_download(args.hf_repo, "config.json")  # increment counter

    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        tokenizer_path = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)

    voice_prompt_dir = _get_voice_prompt_dir(args.voice_prompt_dir, args.hf_repo)
    if not os.path.exists(voice_prompt_dir):
        raise FileNotFoundError(f"voice_prompt_dir not found: {voice_prompt_dir}")

    voice_path_a = os.path.join(voice_prompt_dir, args.voice_prompt_a)
    voice_path_b = os.path.join(voice_prompt_dir, args.voice_prompt_b)
    for label, path in [("A", voice_path_a), ("B", voice_path_b)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Voice prompt for Bot {label} not found: {path}")

    # --- Build per-bot configs ---
    shared = {
        "hf_repo": args.hf_repo,
        "tokenizer_path": tokenizer_path,
        "temp_audio": args.temp_audio,
        "temp_text": args.temp_text,
        "topk_audio": args.topk_audio,
        "topk_text": args.topk_text,
        "rep_penalty": args.rep_penalty,
        "rep_penalty_window": args.rep_penalty_window,
        "greedy": bool(args.greedy),
        "cpu_offload": args.cpu_offload,
        "seed": args.seed,
        "screech_guard": args.screech_guard,
        "screech_hf_threshold": args.screech_hf_threshold,
        "screech_confirm_frames": args.screech_confirm_frames,
        "screech_cooldown": args.screech_cooldown,
    }
    config_a = {
        **shared,
        "name": "A",
        "moshi_weight": args.moshi_weight_a or args.moshi_weight,
        "mimi_weight": args.mimi_weight_a or args.mimi_weight,
        "text_prompt": args.text_prompt_a,
        "voice_prompt_path": voice_path_a,
        "greeting": args.greeting_a,
    }
    config_b = {
        **shared,
        "name": "B",
        "moshi_weight": args.moshi_weight_b or args.moshi_weight,
        "mimi_weight": args.mimi_weight_b or args.mimi_weight,
        "text_prompt": args.text_prompt_b,
        "voice_prompt_path": voice_path_b,
        "greeting": None,
    }

    # --- Spawn worker processes ---
    ctx = mp.get_context("spawn")
    in_a, out_a = ctx.Queue(), ctx.Queue()
    in_b, out_b = ctx.Queue(), ctx.Queue()

    _log("info", f"spawning Bot A on GPU {args.gpu_a}, Bot B on GPU {args.gpu_b}")
    proc_a = ctx.Process(target=_worker, args=(args.gpu_a, config_a, in_a, out_a))
    proc_b = ctx.Process(target=_worker, args=(args.gpu_b, config_b, in_b, out_b))
    proc_a.start()
    proc_b.start()

    # Wait for both to finish loading
    ready_a = out_a.get()
    ready_b = out_b.get()
    if ready_a[0] == "error":
        raise RuntimeError(f"Bot A failed to start: {ready_a[1]}")
    if ready_b[0] == "error":
        raise RuntimeError(f"Bot B failed to start: {ready_b[1]}")
    assert ready_a[0] == "ready" and ready_b[0] == "ready"

    _, frame_size, frame_rate, sample_rate = ready_a
    frame_size = int(frame_size)
    _log("info", f"both bots ready (frame_size={frame_size}, rate={frame_rate}Hz)")

    # Tokenizer for text decoding in main process
    tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

    # Screech guard is now handled inside LMGen.process_transformer_output()
    # (configured per-worker above). No external detector needed in frame loop.

    # Load scheduled context injections (for puppeteered eval)
    scheduled_injections: dict[int, str] = {}
    if args.context_injections:
        with open(args.context_injections) as f:
            inj_list = json.load(f)
        for inj in inj_list:
            scheduled_injections[int(inj["frame"])] = inj["text"]
        _log("info", f"loaded {len(scheduled_injections)} scheduled context injections")

    # --- Frame loop ---
    total_frames = int(args.duration * frame_rate)
    silence = np.zeros((1, 1, frame_size), dtype=np.float32)
    prev_pcm_a = silence.copy()
    prev_pcm_b = silence.copy()

    frames_a: List[np.ndarray] = []
    frames_b: List[np.ndarray] = []
    text_tokens_a: List[str] = []
    text_tokens_b: List[str] = []

    # Silence tracking & nudge state
    NUDGE_PHRASES = ["So,", "Right,", "Now,", "Anyway,"]
    nudge_threshold = int(args.nudge_after * frame_rate) if args.nudge_after > 0 else 0
    mutual_silence_count = 0
    nudge_count = 0
    prev_text_id_a: Optional[int] = 3  # PAD
    prev_text_id_b: Optional[int] = 3
    prev_injecting_a: bool = False  # track context injection state for nudge gating

    _log("info", f"starting conversation for {args.duration}s ({total_frames} frames)")
    if nudge_threshold > 0:
        _log("info", f"nudge enabled: after {args.nudge_after}s silence, max {args.max_nudges} nudges")
    t0 = time.time()

    for frame_idx in range(total_frames):
        # Track mutual silence from previous frame
        a_silent = prev_text_id_a in (0, 3) or prev_text_id_a is None
        b_silent = prev_text_id_b in (0, 3) or prev_text_id_b is None
        if a_silent and b_silent:
            mutual_silence_count += 1
        else:
            mutual_silence_count = 0

        # Inject scheduled context at the right frame (puppeteer eval)
        scheduled_text = scheduled_injections.get(frame_idx)
        if scheduled_text is not None:
            in_a.put({"context": scheduled_text, "pcm": prev_pcm_b})
            in_b.put(prev_pcm_a)
            _log("info", f"[CONTEXT @{frame_idx}] injecting into Bot A ({len(scheduled_text)} chars)")
        # Inject nudge if mutual silence exceeds threshold (skip during context injection)
        elif (
            nudge_threshold > 0
            and mutual_silence_count >= nudge_threshold
            and (args.max_nudges == 0 or nudge_count < args.max_nudges)
            and not prev_injecting_a
        ):
            phrase = NUDGE_PHRASES[nudge_count % len(NUDGE_PHRASES)]
            tokens = tokenizer.encode(phrase)
            # Alternate nudge target: even → Bot A, odd → Bot B
            if nudge_count % 2 == 0:
                in_a.put({"nudge": tokens, "pcm": prev_pcm_b})
                in_b.put(prev_pcm_a)
                _log("info", f"[NUDGE #{nudge_count + 1}] injecting '{phrase}' into Bot A")
            else:
                in_a.put(prev_pcm_b)
                in_b.put({"nudge": tokens, "pcm": prev_pcm_a})
                _log("info", f"[NUDGE #{nudge_count + 1}] injecting '{phrase}' into Bot B")
            nudge_count += 1
            mutual_silence_count = 0
        else:
            in_a.put(prev_pcm_b)
            in_b.put(prev_pcm_a)

        # Both workers process in parallel — block until both done
        result_a = out_a.get()
        result_b = out_b.get()

        pcm_a, text_id_a, injecting_a = result_a
        pcm_b, text_id_b, injecting_b = result_b

        if pcm_a is not None:
            prev_pcm_a = pcm_a
            frames_a.append(pcm_a[0, 0])
            if injecting_a:
                # Context injection — record as <CTX> placeholder, don't log as speech
                text_tokens_a.append("<CTX>")
            else:
                text_str = _decode_text_token(text_id_a, tokenizer)
                text_tokens_a.append(text_str)
                if text_id_a not in (0, 3):
                    _log("info", f"[A] {text_str}")

        if pcm_b is not None:
            prev_pcm_b = pcm_b
            frames_b.append(pcm_b[0, 0])
            if injecting_b:
                text_tokens_b.append("<CTX>")
            else:
                text_str = _decode_text_token(text_id_b, tokenizer)
                text_tokens_b.append(text_str)
                if text_id_b not in (0, 3):
                    _log("info", f"[B] {text_str}")

        prev_text_id_a = text_id_a if not injecting_a else 3  # treat as silent for nudge tracking
        prev_text_id_b = text_id_b if not injecting_b else 3
        prev_injecting_a = injecting_a

        if (frame_idx + 1) % 125 == 0:
            elapsed = time.time() - t0
            _log("info", f"frame {frame_idx + 1}/{total_frames} ({elapsed:.1f}s elapsed)")

    elapsed = time.time() - t0
    _log("info", f"conversation done in {elapsed:.1f}s (nudges={nudge_count})")

    # --- Stop workers ---
    in_a.put(None)
    in_b.put(None)
    proc_a.join(timeout=10)
    proc_b.join(timeout=10)

    # --- Write outputs ---
    if not frames_a or not frames_b:
        _log("error", "no audio frames generated")
        return

    pcm_a_full = np.concatenate(frames_a, axis=-1)
    pcm_b_full = np.concatenate(frames_b, axis=-1)

    max_len = max(pcm_a_full.shape[-1], pcm_b_full.shape[-1])
    if pcm_a_full.shape[-1] < max_len:
        pcm_a_full = np.pad(pcm_a_full, (0, max_len - pcm_a_full.shape[-1]))
    if pcm_b_full.shape[-1] < max_len:
        pcm_b_full = np.pad(pcm_b_full, (0, max_len - pcm_b_full.shape[-1]))

    stereo = np.stack([pcm_a_full, pcm_b_full], axis=0)
    sphn.write_wav(args.output_wav, stereo, sample_rate)
    _log("info", f"wrote stereo WAV to {args.output_wav}")

    with open(args.output_text_a, "w") as f:
        json.dump(text_tokens_a, f, ensure_ascii=False)
    with open(args.output_text_b, "w") as f:
        json.dump(text_tokens_b, f, ensure_ascii=False)

    readable_a = "".join(t for t in text_tokens_a if t not in ("EPAD", "BOS", "EOS", "PAD", "<CTX>"))
    readable_b = "".join(t for t in text_tokens_b if t not in ("EPAD", "BOS", "EOS", "PAD", "<CTX>"))
    _log("info", f"Bot A said: {readable_a.strip()}")
    _log("info", f"Bot B said: {readable_b.strip()}")


if __name__ == "__main__":
    main()
