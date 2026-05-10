#!/usr/bin/env python3
"""Retry failed records from a generate_dialogues_sync.py output JSONL.

Reads the JSONL, retries any record where dialogue is null (error),
and writes a new JSONL with failures replaced by fresh attempts.

Usage:
    python retry_failures.py outbound_insurance.jsonl
    python retry_failures.py outbound_insurance.jsonl -w 8
    python retry_failures.py outbound_insurance.jsonl -o patched.jsonl  # write to separate file
"""

import argparse
import json
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import anthropic

# Reuse generation logic from the main script
sys.path.insert(0, ".")
from generate_dialogues_sync import (
    SYSTEM_PROMPT, build_user_prompt, parse_response, DEFAULT_MODEL,
)

# Strip <thinking>...</thinking> blocks that some models emit as literal text
_THINKING_RE = re.compile(r'<thinking>.*?</thinking>\s*', re.DOTALL)

RETRY_MAX_TOKENS = 8192  # sufficient for brief+transcript without thinking bloat


def generate_single_fixed(
    client: anthropic.Anthropic,
    seed: dict,
    dialogue_id: str,
    model: str,
    max_retries: int = 3,
    base_delay: float = 5.0,
) -> dict:
    """Like generate_single but strips <thinking> text and uses tighter max_tokens."""
    user_prompt = build_user_prompt(seed)
    # Prepend instruction to suppress thinking in text output
    patched_system = (
        "IMPORTANT: Do NOT wrap your response in <thinking> tags. "
        "Output the <brief> and <transcript> directly.\n\n"
        + SYSTEM_PROMPT
    )
    now = datetime.now(timezone.utc).isoformat()

    for attempt in range(max_retries):
        try:
            log.info("[%s] calling API (attempt %d)...", dialogue_id, attempt + 1)
            response = client.messages.create(
                model=model,
                max_tokens=RETRY_MAX_TOKENS,
                system=patched_system,
                messages=[{"role": "user", "content": user_prompt}],
            )

            text_parts = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
            full_text = "\n".join(text_parts)

            # Strip any literal <thinking> blocks
            full_text = _THINKING_RE.sub("", full_text)

            brief, transcript, injections = parse_response(full_text)

            if not brief or not transcript:
                log.warning("[%s] Parse failed — missing %s",
                            dialogue_id, "brief" if not brief else "transcript")
                return {
                    "id": dialogue_id,
                    "seed": seed,
                    "user_prompt": user_prompt,
                    "text_prompt": None,
                    "dialogue": None,
                    "context_injections": [],
                    "raw_response": full_text[:3000],
                    "model": model,
                    "error_type": "parse_error",
                    "error": f"Missing {'brief' if not brief else 'transcript'}",
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                    "generated_at": now,
                }

            return {
                "id": dialogue_id,
                "seed": seed,
                "user_prompt": user_prompt,
                "text_prompt": brief,
                "dialogue": transcript,
                "context_injections": injections,
                "model": model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                "generated_at": now,
            }

        except anthropic.RateLimitError:
            delay = base_delay * (2 ** attempt)
            log.warning("[%s] Rate limited (attempt %d/%d). Waiting %.0fs...",
                        dialogue_id, attempt + 1, max_retries, delay)
            time.sleep(delay)

        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                delay = base_delay * (2 ** attempt)
                log.warning("[%s] Server error %d (attempt %d/%d). Waiting %.0fs...",
                            dialogue_id, e.status_code, attempt + 1, max_retries, delay)
                time.sleep(delay)
            else:
                log.error("[%s] API error %d: %s", dialogue_id, e.status_code, e.message)
                return {
                    "id": dialogue_id, "seed": seed, "user_prompt": user_prompt,
                    "text_prompt": None, "dialogue": None, "model": model,
                    "error_type": f"api_error_{e.status_code}",
                    "error": str(e.message), "generated_at": now,
                }

        except anthropic.APIConnectionError:
            delay = base_delay * (2 ** attempt)
            log.warning("[%s] Connection error (attempt %d/%d). Waiting %.0fs...",
                        dialogue_id, attempt + 1, max_retries, delay)
            time.sleep(delay)

    log.error("[%s] Failed after %d attempts.", dialogue_id, max_retries)
    return {
        "id": dialogue_id, "seed": seed, "user_prompt": user_prompt,
        "text_prompt": None, "dialogue": None, "model": model,
        "error_type": "max_retries_exhausted",
        "error": f"Failed after {max_retries} attempts", "generated_at": now,
    }

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Retry failed dialogue generations.")
    parser.add_argument("input", help="Input JSONL file with mixed success/error records")
    parser.add_argument("-o", "--output", help="Output JSONL (default: overwrite input)")
    parser.add_argument("-w", "--workers", type=int, default=8, help="Parallel workers (default: 8)")
    parser.add_argument("--model", default=None, help="Override model (default: use original)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per record")
    args = parser.parse_args()

    output_path = args.output or args.input

    # Read all records
    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Split into success and failure
    succeeded = {r["id"]: r for r in records if r.get("dialogue") is not None}
    failed = [r for r in records if r.get("dialogue") is None]

    if not failed:
        log.info("No failures found — nothing to retry.")
        return

    log.info("Found %d succeeded, %d failed. Retrying failures with %d workers...",
             len(succeeded), len(failed), args.workers)

    client = anthropic.Anthropic()
    retry_succeeded = 0
    retry_failed = 0
    results_lock = threading.Lock()

    def _retry(record):
        model = args.model or record.get("model", DEFAULT_MODEL)
        return generate_single_fixed(
            client, record["seed"], record["id"], model,
            max_retries=args.max_retries,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_retry, r): r["id"] for r in failed}

        for future in as_completed(futures):
            dial_id = futures[future]
            result = future.result()

            if result.get("dialogue") is not None:
                with results_lock:
                    retry_succeeded += 1
                    succeeded[dial_id] = result
                tokens = result["usage"]
                log.info("%s — OK (%d in / %d out tokens)", dial_id,
                         tokens["input_tokens"], tokens["output_tokens"])
            else:
                with results_lock:
                    retry_failed += 1
                    succeeded[dial_id] = result  # keep updated error record
                log.error("%s — STILL FAILED: %s", dial_id, result.get("error", "unknown"))

    log.info("Retry complete — recovered: %d, still failed: %d", retry_succeeded, retry_failed)

    # Reconstruct in original order (by dial-NNNNN id)
    all_ids = [r["id"] for r in records]
    final_records = [succeeded[did] for did in all_ids]

    with open(output_path, "w", encoding="utf-8") as f:
        for r in final_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_ok = sum(1 for r in final_records if r.get("dialogue") is not None)
    total_err = sum(1 for r in final_records if r.get("dialogue") is None)
    log.info("Written to %s — succeeded: %d, errored: %d (of %d total)",
             output_path, total_ok, total_err, len(final_records))


if __name__ == "__main__":
    main()
