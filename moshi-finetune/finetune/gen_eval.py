"""Generation-based evaluation for Moshi finetuning.

Starts a Moshi server with merged LoRA weights, connects Gemini Live API as
the client via the gemini_eval bridge, evaluates conversation quality (audio
profile metrics via WhisperX) and content quality (LLM transcript review),
then logs results to wandb.

Standalone CLI:
    python -m finetune.gen_eval \
        --checkpoint runs/.../checkpoints/checkpoint_000200/consolidated \
        --run-dir runs/.../ \
        --step 200 \
        --config configs/pharma_demo.yaml

Also callable inline from train.py: gen_eval.run(ckpt_dir, run_dir, step, args)
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def _load_env():
    """Load .env file from project root into os.environ."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
    except ImportError:
        # Fallback: parse KEY=VALUE lines manually
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

_load_env()

logger = logging.getLogger("gen_eval")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # voice-training/
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
EVAL_PROFILE_SCRIPT = PROJECT_ROOT / "pipeline" / "eval_conversation_profile.py"
MOSHI_SERVER_CWD = PROJECT_ROOT / "personaplex" / "moshi"

DEFAULT_HF_REPO = "nvidia/personaplex-7b-v1"


def _extract_transcript(log_text: str, marker: str) -> str:
    """Extract transcript from gemini_eval subprocess logs.

    Parses lines like "[Gemini heard]: some text" or "[Gemini said]: some text"
    from the subprocess stderr and joins them.
    """
    parts = []
    for line in log_text.splitlines():
        idx = line.find(marker)
        if idx >= 0:
            text = line[idx + len(marker):].strip()
            if text.startswith(":"):
                text = text[1:].strip()
            if text:
                parts.append(text)
    return " ".join(parts)


def _tail_text(path: Path, max_chars: int = 2000) -> str:
    """Return the tail of a text file for error logs."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def _load_eval_prompts(path: str) -> dict:
    """Load eval prompt configurations from JSON.

    Resolution order:
      1. Absolute path → use as-is
      2. Sibling of this file (finetune/eval_prompts.json)
      3. Relative to moshi-finetune/ dir

    Returns a dict with:
      - "review_prompt": template string for LLM review (with {scenario},
        {reference_block}, {text_a}, {text_b}, {injection_task} placeholders)
      - "prompts": list of eval prompt dicts

    Backward-compatible: if the JSON file is a plain list, wraps it with
    a default review prompt template.
    """
    prompts_path = Path(path)
    if prompts_path.is_absolute() and prompts_path.exists():
        pass
    elif (Path(__file__).parent / prompts_path.name).exists():
        prompts_path = Path(__file__).parent / prompts_path.name
    else:
        prompts_path = PROJECT_ROOT / "moshi-finetune" / path
    with open(prompts_path) as f:
        data = json.load(f)
    # Backward-compatible: plain list → wrap with default template
    if isinstance(data, list):
        return {"review_prompt": None, "prompts": data}
    return data


def merge_lora(ckpt_dir: Path, output_path: Path, hf_repo: str) -> Path:
    """Merge LoRA weights into base model, returning path to merged safetensors."""
    sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))
    from merge_lora import merge
    return merge(ckpt_dir=ckpt_dir, output_path=output_path, hf_repo=hf_repo)


def _start_moshi_server(
    merged_weight: Path,
    gpu_id: int,
    port: int,
    hf_repo: str,
    log_path: Path,
) -> subprocess.Popen:
    """Start a Moshi server subprocess and wait until it's ready."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Remove torchrun distributed env vars — they would confuse the server
    for key in list(env.keys()):
        if key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                    "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK",
                    "TORCHELASTIC_RUN_ID", "OMP_NUM_THREADS"):
            del env[key]

    cmd = [
        str(PYTHON), "-m", "moshi.server",
        "--moshi-weight", str(merged_weight),
        "--port", str(port),
        "--host", "localhost",
        "--hf-repo", hf_repo,
        "--device", "cuda:0",  # CUDA_VISIBLE_DEVICES remaps physical GPU
    ]
    logger.info(f"Starting Moshi server on GPU {gpu_id}, port {port}...")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        cwd=str(MOSHI_SERVER_CWD),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_file.close()

    # Wait for server to be ready (look for "Application started" or listen on port)
    import socket
    deadline = time.time() + 120  # 2 min timeout for model loading
    ready = False
    while time.time() < deadline:
        # Check if process died
        if proc.poll() is not None:
            logger.error(
                f"Moshi server exited with code {proc.returncode}; "
                f"tail of {log_path}:\n{_tail_text(log_path, 1000)}"
            )
            raise RuntimeError("Moshi server failed to start")

        # Try connecting to port
        try:
            with socket.create_connection(("localhost", port), timeout=1):
                ready = True
                break
        except (ConnectionRefusedError, OSError):
            time.sleep(2)

    if not ready:
        proc.kill()
        raise RuntimeError(
            f"Moshi server not ready after 120s on port {port}; "
            f"tail of {log_path}:\n{_tail_text(log_path, 1000)}"
        )

    logger.info(f"Moshi server ready on port {port}; logs at {log_path}")
    return proc


def run_dialogues(
    merged_weight: Path,
    eval_prompts: list[dict],
    output_dir: Path,
    gpu_a: int,
    port: int,
    duration: float,
    timeout: int,
    hf_repo: str,
    temp_audio: float = 0.55,
    temp_text: float = 0.7,
    topk_audio: int = 100,
    topk_text: int = 30,
    gemini_model: str = "models/gemini-3.1-flash-live-preview",
    gemini_voice: str = "Puck",
) -> list[dict]:
    """Start Moshi server, run Gemini↔Moshi dialogues, return results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    GEMINI_EVAL_SCRIPT = PROJECT_ROOT / "pipeline" / "gemini_eval.py"

    # Build a clean env for the gemini_eval subprocess (strip torchrun vars)
    clean_env = os.environ.copy()
    for key in list(clean_env.keys()):
        if key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
                    "MASTER_ADDR", "MASTER_PORT", "GROUP_RANK",
                    "TORCHELASTIC_RUN_ID", "OMP_NUM_THREADS",
                    "CUDA_VISIBLE_DEVICES"):
            del clean_env[key]

    # Start Moshi server once for all dialogues
    server_proc = None
    try:
        server_proc = _start_moshi_server(
            merged_weight,
            gpu_a,
            port,
            hf_repo,
            output_dir / "moshi_server.log",
        )
        moshi_url = f"ws://localhost:{port}/api/chat"

        for i, prompt in enumerate(eval_prompts):
            dial_id = prompt["id"]
            wav_path = output_dir / f"{dial_id}.wav"

            logger.info(
                f"[{i+1}/{len(eval_prompts)}] Running dialogue {dial_id} via Gemini bridge"
                + (f" +{len(prompt.get('context_injections', []))} injections"
                   if prompt.get("context_injections") else "")
            )
            t0 = time.time()

            try:
                # Write prompt config to temp JSON for the subprocess
                prompt_file = output_dir / f"{dial_id}_prompt.json"
                with open(prompt_file, "w") as f:
                    json.dump(prompt, f, ensure_ascii=False)

                # Run gemini_eval as subprocess using the .venv Python
                # (which has aiohttp, websockets, sphn, etc.)
                cmd = [
                    str(PYTHON), str(GEMINI_EVAL_SCRIPT),
                    "--moshi-url", moshi_url,
                    "--system-prompt", prompt["client_prompt"],
                    "--text-prompt", prompt.get("broker_prompt", ""),
                    "--greeting", prompt.get("greeting", ""),
                    "--voice-prompt", prompt.get("voice_broker", ""),
                    "--output", str(wav_path),
                    "--duration", str(duration),
                    "--gemini-model", gemini_model,
                    "--gemini-voice", gemini_voice,
                    "--audio-temperature", str(temp_audio),
                    "--text-temperature", str(temp_text),
                    "--audio-topk", str(topk_audio),
                    "--text-topk", str(topk_text),
                ]

                # Pass context injections via a temp file
                ctx_inj_path = None
                if prompt.get("context_injections"):
                    ctx_inj_path = output_dir / f"{dial_id}_injections.json"
                    with open(ctx_inj_path, "w") as f:
                        json.dump(prompt["context_injections"], f, ensure_ascii=False)
                    cmd.extend(["--context-injections", str(ctx_inj_path)])

                stderr_log = output_dir / f"{dial_id}_stderr.log"
                with open(stderr_log, "w") as stderr_f:
                    proc = subprocess.run(
                        cmd,
                        env=clean_env,
                        stdout=subprocess.PIPE,
                        stderr=stderr_f,
                        text=True,
                        timeout=timeout,
                    )
                elapsed = time.time() - t0

                # Parse transcripts from subprocess stderr log
                stderr_text = stderr_log.read_text()
                moshi_transcript = _extract_transcript(stderr_text, "[Gemini heard]")
                gemini_transcript = _extract_transcript(stderr_text, "[Gemini said]")

                if proc.returncode != 0:
                    error_tail = stderr_text[-1000:]
                    logger.error(f"Dialogue {dial_id} failed (rc={proc.returncode}):\n{error_tail}")
                    results.append({"id": dial_id, "status": "error", "error": error_tail})
                elif not wav_path.exists():
                    logger.error(f"Dialogue {dial_id}: no WAV produced")
                    results.append({"id": dial_id, "status": "error", "error": "no WAV file"})
                else:
                    logger.info(f"Dialogue {dial_id} completed in {elapsed:.0f}s")
                    results.append({
                        "id": dial_id,
                        "status": "ok",
                        "wav_path": str(wav_path),
                        "moshi_transcript": moshi_transcript,
                        "gemini_transcript": gemini_transcript,
                        "elapsed_s": elapsed,
                        "system_prompt": prompt.get("broker_prompt", ""),
                    })
                    if prompt.get("context_injections"):
                        results[-1]["context_injections"] = prompt["context_injections"]

            except subprocess.TimeoutExpired:
                logger.error(f"Dialogue {dial_id} timed out after {timeout}s")
                results.append({"id": dial_id, "status": "timeout"})
            except Exception as e:
                logger.error(f"Dialogue {dial_id} exception: {e}")
                results.append({"id": dial_id, "status": "error", "error": str(e)})

    finally:
        if server_proc is not None:
            logger.info("Shutting down Moshi server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait()
            logger.info("Moshi server stopped")

    return results


def evaluate_profiles(wav_paths: list[str], gpu_id: int) -> dict:
    """Run eval_conversation_profile.py as subprocess for GPU isolation."""
    if not wav_paths:
        return {}
    if not EVAL_PROFILE_SCRIPT.exists():
        logger.warning(f"Profile eval script not found: {EVAL_PROFILE_SCRIPT}")
        return {}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy WAVs into temp dir
        for wp in wav_paths:
            shutil.copy2(wp, tmpdir)

        out_json = Path(tmpdir) / "profile_results.json"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # The eval_conversation_profile.py hardcodes CUDA_VISIBLE_DEVICES="0" at
        # module level (line 31), which would override our subprocess env. Create a
        # patched copy that respects the env-provided GPU assignment instead.
        script_text = EVAL_PROFILE_SCRIPT.read_text()
        script_text = script_text.replace(
            'os.environ["CUDA_VISIBLE_DEVICES"] = "0"',
            '# CUDA_VISIBLE_DEVICES set via subprocess env',
        )
        patched_script = Path(tmpdir) / "eval_conversation_profile.py"
        patched_script.write_text(script_text)

        cmd = [
            str(PYTHON),
            str(patched_script),
            tmpdir,
            "--out", str(out_json),
        ]

        logger.info(f"Running conversation profile eval on GPU {gpu_id} ({len(wav_paths)} files)")
        try:
            proc = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if proc.returncode != 0:
                logger.error(f"Profile eval failed:\n{proc.stderr[-500:]}")
                return {}
            if not out_json.exists():
                logger.error("Profile eval produced no output JSON")
                return {}

            with open(out_json) as f:
                summaries = json.load(f)

            if summaries and isinstance(summaries, list):
                return summaries[0]
            return {}

        except subprocess.TimeoutExpired:
            logger.error("Profile eval timed out")
            return {}
        except Exception as e:
            logger.error(f"Profile eval exception: {e}")
            return {}


def _failed_review() -> dict:
    """Return a review dict with NaN scores for when review fails."""
    return {
        "coherence": float("nan"),
        "naturalness": float("nan"),
        "effectiveness": float("nan"),
        "grounding": float("nan"),
        "grounding_pct": float("nan"),
        "n_claims": 0,
        "n_correct": 0,
        "n_garbled": 0,
        "n_fabricated": 0,
        "injection_usage_pct": float("nan"),
        "injection_accuracy_pct": float("nan"),
        "broker_claims": [],
        "injection_results": [],
        "notes": "review failed",
    }


def _compute_grounding_metrics(parsed: dict, n_expected_injections: int) -> dict:
    """Compute derived grounding metrics from structured LLM output."""
    claims = parsed.get("broker_claims", [])
    injection_results = parsed.get("injection_results", [])

    n_claims = len(claims)
    n_correct = sum(1 for c in claims if c.get("verdict") == "correct")
    n_garbled = sum(1 for c in claims if c.get("verdict") == "garbled")
    n_fabricated = sum(1 for c in claims if c.get("verdict") == "fabricated")

    # Grounding pct: fraction of claims that are correct (0-100)
    grounding_pct = (n_correct / n_claims * 100) if n_claims > 0 else float("nan")

    # Backward-compat 1-5 scale: linear map from fraction correct
    grounding = (1 + 4 * n_correct / n_claims) if n_claims > 0 else float("nan")

    # Injection metrics
    n_used = sum(1 for r in injection_results if r.get("used"))
    n_accurate = sum(1 for r in injection_results if r.get("used") and r.get("accurate"))
    injection_usage_pct = (n_used / n_expected_injections * 100) if n_expected_injections > 0 else float("nan")
    injection_accuracy_pct = (n_accurate / n_used * 100) if n_used > 0 else float("nan")

    return {
        "notes": parsed.get("notes", ""),
        "coherence": parsed.get("coherence", float("nan")),
        "naturalness": parsed.get("naturalness", float("nan")),
        "effectiveness": parsed.get("effectiveness", float("nan")),
        "grounding": grounding,
        "grounding_pct": grounding_pct,
        "n_claims": n_claims,
        "n_correct": n_correct,
        "n_garbled": n_garbled,
        "n_fabricated": n_fabricated,
        "injection_usage_pct": injection_usage_pct,
        "injection_accuracy_pct": injection_accuracy_pct,
        "broker_claims": claims,
        "injection_results": injection_results,
    }


def _review_single(dialogue: dict, prompt: dict, model: str, review_prompt_template: str | None = None) -> dict:
    """Send transcript to Claude API for structured review with fact-checking.

    Uses Gemini transcriptions (moshi_transcript / gemini_transcript) rather
    than Moshi's internal token stream.
    """
    import anthropic

    text_a = dialogue.get("moshi_transcript", "")
    text_b = dialogue.get("gemini_transcript", "")

    # Build reference material with numbered injections
    broker_prompt_full = prompt.get('broker_prompt', '')
    context_injections = prompt.get('context_injections', [])
    reference_block = f"BROKER SYSTEM PROMPT (full):\n{broker_prompt_full}\n"
    if context_injections:
        reference_block += "\nINJECTED CONTEXT (provided to broker mid-conversation):\n"
        for i, c in enumerate(context_injections, 1):
            if isinstance(c, dict):
                reference_block += f"  INJECTION {i}: {c['text']}\n"

    injection_task = ""
    if context_injections:
        injection_task = f"""
TASK 3 — Injection usage:
For each of the {len(context_injections)} numbered injections above, determine:
- Was the key information from this injection used by the broker?
- If used, were the specific details (dollar amounts, names, percentages) accurate?
Include one entry in "injection_results" per injection."""

    if review_prompt_template:
        review_prompt = review_prompt_template.format(
            scenario=prompt.get('scenario', 'Unknown'),
            reference_block=reference_block,
            text_a=text_a,
            text_b=text_b,
            injection_task=injection_task,
        )
    else:
        # Default review prompt with transcription leniency note
        review_prompt = f"""You are evaluating a simulated voice conversation between an AI agent and a client.

CRITICAL — TRANSCRIPTION LIMITATIONS: These transcripts come from automatic speech
recognition and are UNRELIABLE for specific words. Person names, drug names,
company names, dollar amounts, percentages, and technical terms are frequently
mangled beyond recognition (e.g. "Wiggly Care" = "Wegovy Care", "Lex a pro" =
"Lexapro", "Bobby Dalton" might appear as "Bobby Dolton" or be missing entirely).
Do NOT penalize the agent for how the transcript renders specific words. Instead,
evaluate based on the GENERAL STRUCTURE and INTENT of the conversation:
- Did the agent follow the right conversational flow?
- Did the agent attempt to address the right topics from their system prompt?
- Did the agent appear to reference injected information at roughly the right time?
- Did the agent fabricate entire topics or directions not in the prompt?
When classifying claims, only mark something as "garbled" or "fabricated" if you
are confident the agent actually said something wrong — not because the transcript
garbled it. When in doubt, classify as "correct".

SCENARIO: {prompt.get('scenario', 'Unknown')}

{reference_block}
AGENT SAID:
{text_a}

CLIENT SAID:
{text_b}

TASK 1 — Quality ratings (1-5 each):
1. COHERENCE: Does the conversation flow logically? Are there non-sequiturs, abrupt topic changes, or excessive repetition?
2. NATURALNESS: Does it sound like a real voice conversation? Appropriate turn-taking, natural pacing?
3. EFFECTIVENESS: Does the agent follow the call protocol and work toward its goal? Is it professional and on-topic?

TASK 2 — Factual claims:
Extract the agent's major factual claims (topics discussed, information referenced,
actions proposed). Focus on STRUCTURAL accuracy — did the agent discuss the right
topics and reference the right information sources? Ignore exact wording.
- "correct": the agent clearly discussed or referenced information from the system prompt or injections
- "garbled": the agent attempted to reference provided information but conveyed the WRONG meaning
  (NOT just a transcription spelling error — the underlying intent must be wrong)
- "fabricated": the agent introduced an entire topic, claim, or recommendation with no basis in the provided context
{injection_task}
Respond ONLY with this JSON (no other text):
{{"notes": "<1-2 sentence summary>", "coherence": <int 1-5>, "naturalness": <int 1-5>, "effectiveness": <int 1-5>, "broker_claims": [{{"claim": "<topic/information the agent discussed>", "verdict": "<correct|garbled|fabricated>", "reference": "<which source, or none>"}}], "injection_results": [{{"index": <1-based int>, "used": <bool>, "accurate": <bool>, "detail": "<brief note>"}}]}}"""

    client = anthropic.Anthropic()
    n_injections = len(context_injections)
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": review_prompt}],
            )
            text = response.content[0].text.strip()
            # Extract JSON from response
            if "{" in text:
                json_str = text[text.index("{"):text.rindex("}") + 1]
                parsed = json.loads(json_str)
                return _compute_grounding_metrics(parsed, n_injections)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response for {dialogue['id']}, attempt {attempt+1}")
        except Exception as e:
            logger.warning(f"LLM review error for {dialogue['id']}, attempt {attempt+1}: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)

    return _failed_review()


def review_transcripts(
    dialogue_results: list[dict],
    eval_prompts: list[dict],
    model: str,
    review_prompt_template: str | None = None,
) -> list[dict]:
    """Review transcripts via Claude API concurrently."""
    ok_results = [d for d in dialogue_results if d["status"] == "ok"]
    if not ok_results:
        return []

    prompts_by_id = {p["id"]: p for p in eval_prompts}
    reviews = []

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = []
        for d in ok_results:
            prompt = prompts_by_id.get(d["id"], {})
            futures.append(pool.submit(_review_single, d, prompt, model, review_prompt_template))

        for d, future in zip(ok_results, futures):
            try:
                review = future.result(timeout=120)
                review["id"] = d["id"]
                reviews.append(review)
            except Exception as e:
                logger.warning(f"Review failed for {d['id']}: {e}")
                failed = _failed_review()
                failed["id"] = d["id"]
                failed["notes"] = str(e)
                reviews.append(failed)

    return reviews


def _find_wandb_run_id(run_dir: Path) -> str | None:
    """Extract wandb run ID from run directory."""
    wandb_dir = run_dir / "wandb"
    if not wandb_dir.exists():
        return None
    run_dirs = sorted(
        d for d in wandb_dir.iterdir()
        if d.is_dir() and d.name.startswith("run-")
    )
    if not run_dirs:
        return None
    parts = run_dirs[-1].name.rsplit("-", 1)
    if len(parts) == 2:
        return parts[1]
    return None


def log_results(
    dialogue_results: list[dict],
    profile: dict,
    reviews: list[dict],
    step: int,
    run_dir: Path,
    wandb_project: str | None,
    wav_paths: list[str],
) -> dict:
    """Log metrics to wandb and save results.json. Returns the metrics dict."""
    import math

    n_ok = sum(1 for d in dialogue_results if d["status"] == "ok")

    # Aggregate LLM review scores
    def safe_mean(vals):
        finite = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return sum(finite) / len(finite) if finite else float("nan")

    coherence = safe_mean([r["coherence"] for r in reviews])
    naturalness = safe_mean([r["naturalness"] for r in reviews])
    effectiveness = safe_mean([r["effectiveness"] for r in reviews])
    grounding = safe_mean([r.get("grounding", float("nan")) for r in reviews])

    # Structured grounding detail metrics
    grounding_pct = safe_mean([r.get("grounding_pct", float("nan")) for r in reviews])
    total_claims = sum(r.get("n_claims", 0) for r in reviews)
    total_correct = sum(r.get("n_correct", 0) for r in reviews)
    total_garbled = sum(r.get("n_garbled", 0) for r in reviews)
    total_fabricated = sum(r.get("n_fabricated", 0) for r in reviews)
    injection_usage = safe_mean([r.get("injection_usage_pct", float("nan")) for r in reviews])
    injection_accuracy = safe_mean([r.get("injection_accuracy_pct", float("nan")) for r in reviews])

    # Build metrics dict
    metrics = {
        "gen_eval/n_dialogues_ok": n_ok,
        "gen_eval/coherence_mean": coherence,
        "gen_eval/naturalness_mean": naturalness,
        "gen_eval/effectiveness_mean": effectiveness,
        "gen_eval/grounding_mean": grounding,
        "gen_eval/grounding_correct_pct": grounding_pct,
        "gen_eval/grounding_n_claims": total_claims,
        "gen_eval/grounding_n_correct": total_correct,
        "gen_eval/grounding_n_garbled": total_garbled,
        "gen_eval/grounding_n_fabricated": total_fabricated,
        "gen_eval/injection_usage_pct": injection_usage,
        "gen_eval/injection_accuracy_pct": injection_accuracy,
    }

    # Extract profile metrics
    if profile:
        tg = profile.get("turn_gap", {})
        sr = profile.get("speech_rate", {})
        bc = profile.get("backchannels", {})
        td = profile.get("turn_duration", {})
        ld = profile.get("loudness", {})

        profile_metrics = {
            "gen_eval/turn_gap_median_ms": tg.get("median_ms", float("nan")),
            "gen_eval/speech_rate_wps": sr.get("mean_wps", float("nan")),
            "gen_eval/backchannel_rate_per_hour": bc.get("mean_rate_per_hour", float("nan")),
            "gen_eval/turn_duration_median_s": td.get("median_s", float("nan")),
            "gen_eval/loudness_interturn_sd": ld.get("inter_turn_sd_dB", float("nan")),
            "gen_eval/pct_overlap": tg.get("pct_overlap", float("nan")),
        }
        metrics.update(profile_metrics)

    # Filter out NaN for wandb (it doesn't handle NaN scalars well)
    wandb_metrics = {k: v for k, v in metrics.items() if not (isinstance(v, float) and math.isnan(v))}

    # Log to wandb
    if wandb_project:
        try:
            import wandb

            run_id = _find_wandb_run_id(run_dir)
            if run_id:
                wandb.init(
                    resume="allow",
                    id=run_id,
                    project=wandb_project,
                    dir=run_dir,
                )

                wandb.log(wandb_metrics, step=step)

                # Log audio artifacts
                for wp in wav_paths:
                    name = Path(wp).stem
                    try:
                        if wp.endswith(".mp3"):
                            wandb.log({f"gen_eval/audio/{name}": wandb.Audio(wp)}, step=step)
                        else:
                            wandb.log({f"gen_eval/audio/{name}": wandb.Audio(wp, sample_rate=24000)}, step=step)
                    except Exception as e:
                        logger.warning(f"Failed to log audio {name}: {e}")

                # Log per-dialogue table
                if reviews:
                    columns = [
                        "id", "coherence", "naturalness", "effectiveness",
                        "grounding", "grounding_pct",
                        "n_claims", "n_correct", "n_garbled", "n_fabricated",
                        "inj_usage_pct", "inj_accuracy_pct", "notes",
                    ]
                    table = wandb.Table(columns=columns)
                    for r in reviews:
                        table.add_data(
                            r.get("id", ""),
                            r.get("coherence", ""),
                            r.get("naturalness", ""),
                            r.get("effectiveness", ""),
                            r.get("grounding", ""),
                            r.get("grounding_pct", ""),
                            r.get("n_claims", 0),
                            r.get("n_correct", 0),
                            r.get("n_garbled", 0),
                            r.get("n_fabricated", 0),
                            r.get("injection_usage_pct", ""),
                            r.get("injection_accuracy_pct", ""),
                            r.get("notes", ""),
                        )
                    wandb.log({f"gen_eval/review_table_step_{step}": table}, step=step)

                # Don't call wandb.finish() — the training loop owns the run lifecycle.
            else:
                logger.warning("No wandb run ID found, skipping wandb logging")
        except Exception as e:
            logger.error(f"wandb logging failed: {e}")

    return metrics


def run(
    ckpt_dir: str | Path,
    run_dir: str | Path,
    step: int,
    args,
) -> dict | None:
    """Main gen_eval entry point. Called from train.py or standalone.

    Returns metrics dict or None on fatal error.
    """
    ckpt_dir = Path(ckpt_dir)
    run_dir = Path(run_dir)
    gen_args = args.gen_eval

    # Output directory for this step
    step_dir = run_dir / "gen_eval" / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    hf_repo = args.moshi_paths.hf_repo_id or DEFAULT_HF_REPO

    # 1. Merge LoRA → temp merged safetensors
    merged_path = step_dir / "merged_model.safetensors"
    logger.info(f"Merging LoRA from {ckpt_dir} → {merged_path}")
    try:
        merge_lora(ckpt_dir, merged_path, hf_repo)
    except Exception as e:
        logger.error(f"LoRA merge failed: {e} — aborting gen_eval for step {step}")
        return None

    # 2. Load eval prompts
    prompts_path = gen_args.eval_prompts_path
    if not prompts_path:
        prompts_path = str(Path(__file__).parent / "eval_prompts.json")
    eval_data = _load_eval_prompts(prompts_path)
    eval_prompts = eval_data["prompts"]
    review_prompt_template = eval_data.get("review_prompt")
    logger.info(f"Loaded {len(eval_prompts)} eval prompts from {prompts_path}")

    # 3. Run Gemini↔Moshi dialogues
    dialogue_results = run_dialogues(
        merged_weight=merged_path,
        eval_prompts=eval_prompts,
        output_dir=step_dir / "dialogues",
        gpu_a=gen_args.gpu_a,
        port=gen_args.moshi_port,
        duration=gen_args.dialogue_duration,
        timeout=gen_args.dialogue_timeout,
        hf_repo=hf_repo,
        temp_audio=gen_args.temp_audio,
        temp_text=gen_args.temp_text,
        topk_audio=gen_args.topk_audio,
        topk_text=gen_args.topk_text,
        gemini_model=gen_args.gemini_model,
        gemini_voice=gen_args.gemini_voice,
    )

    n_ok = sum(1 for d in dialogue_results if d["status"] == "ok")
    logger.info(f"{n_ok}/{len(eval_prompts)} dialogues succeeded")

    # Clean up merged weights (large file, ~17GB)
    if merged_path.exists():
        logger.info(f"Removing merged weights: {merged_path}")
        merged_path.unlink()

    if n_ok == 0:
        logger.error("All dialogues failed — skipping profile + LLM eval")
        metrics = log_results(dialogue_results, {}, [], step, run_dir,
                              args.wandb.project, [])
        _save_results(step_dir, step, metrics, dialogue_results, [], {})
        return metrics

    # 4. Evaluate conversation profiles (WhisperX)
    wav_paths = [d["wav_path"] for d in dialogue_results if d["status"] == "ok"]
    profile = {}
    try:
        profile = evaluate_profiles(wav_paths, gen_args.gpu_a)
    except Exception as e:
        logger.error(f"Profile eval failed: {e}")

    # 4b. Convert WAVs to MP3 to save disk space
    mp3_paths = []
    for wp in wav_paths:
        mp3_path = Path(wp).with_suffix(".mp3")
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", wp, "-q:a", "2", str(mp3_path)],
                capture_output=True, timeout=60,
            )
            if mp3_path.exists():
                Path(wp).unlink()
                mp3_paths.append(str(mp3_path))
            else:
                mp3_paths.append(wp)
        except Exception as e:
            logger.warning(f"MP3 conversion failed for {wp}: {e}")
            mp3_paths.append(wp)
    wav_paths = mp3_paths

    # 5. LLM transcript review
    reviews = []
    if gen_args.llm_review:
        try:
            reviews = review_transcripts(dialogue_results, eval_prompts, gen_args.llm_model, review_prompt_template)
        except Exception as e:
            logger.error(f"LLM review failed: {e}")

    # 6. Log results
    metrics = log_results(
        dialogue_results, profile, reviews, step, run_dir,
        args.wandb.project, wav_paths,
    )
    _save_results(step_dir, step, metrics, dialogue_results, reviews, profile)

    return metrics


def _save_results(step_dir: Path, step: int, metrics: dict, dialogues: list, reviews: list, profile: dict):
    """Save results.json to disk."""
    results_path = step_dir / "results.json"
    results_data = {
        "step": step,
        "metrics": {k: v for k, v in metrics.items()},
        "dialogues": dialogues,
        "reviews": reviews,
        "profile_summary": profile,
    }
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")


def main():
    """CLI entry point for standalone gen_eval."""
    import argparse

    from finetune.args import TrainArgs

    parser = argparse.ArgumentParser(description="Generation eval for Moshi finetuning")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint consolidated dir")
    parser.add_argument("--run-dir", required=True, help="Training run directory")
    parser.add_argument("--step", type=int, required=True, help="Training step")
    parser.add_argument("--config", required=True, help="Training config YAML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    train_args = TrainArgs.load(args.config, drop_extra_fields=False)
    # Override gen_eval to enabled for standalone run
    train_args.gen_eval.enable = True

    result = run(
        ckpt_dir=Path(args.checkpoint),
        run_dir=Path(args.run_dir),
        step=args.step,
        args=train_args,
    )

    if result:
        print(f"\nGen eval complete. Metrics:")
        for k, v in sorted(result.items()):
            print(f"  {k}: {v}")
    else:
        print("\nGen eval failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
