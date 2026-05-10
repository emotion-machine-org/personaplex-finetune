#!/usr/bin/env python3
"""Generate PersonaPlex-style system prompts for broker training.

Reads dialogues from dialogues_1.jsonl, uses Claude to generate rich
broker system prompts (context, task, constraints), and adds text_prompt
to the corresponding stereo_wav/dial-XXXXX.json files.

Supports parallel processing and resuming (skips files that already have text_prompt).
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

GENERATION_PROMPT = """Generate a system prompt (UNDER 128 WORDS) for a voice AI playing an insurance broker in a phone call.

## Metadata
{seed_json}

## Dialogue
{dialogue}

## Instructions

Write a CONCISE pre-call brief — 128 words max. Dense paragraph, no markdown headers, no bullet points.

CRITICAL: Only include information the broker would know BEFORE the call starts — client name, company, account history, their policies on file, the broker's own expertise. Extract specific details like company names, policy types, dollar amounts, and account facts from the dialogue, but ONLY things the broker would already have in their system or know from prior interactions.

DO NOT include anything that only becomes apparent DURING the call — the client's emotional state, their specific questions, confusions, or how the conversation unfolds. This is a pre-call brief, not a post-call summary.

Example:

"You are Mark Antonelli at Coastal Pacific Insurance, speaking with Fatima Al-Rashid, facilities manager at Hartwell Industrial (new client, <1yr). Hartwell has cargo operations through 3 ports including a new Tacoma facility and leases 2 barges. Their account includes a marine package: hull & machinery, protection & indemnity, and cargo warehouse-to-warehouse coverage. The dec pages were last issued 6 months ago at onboarding. Your specialty is environmental liability but you handle marine accounts. Warm, relationship-focused communication style."

Output ONLY the system prompt text. No quotes, no code blocks, no preamble. MUST be under 128 words."""


def load_dialogues(path: Path) -> dict[str, dict]:
    """Load dialogues keyed by ID."""
    dialogues = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                dialogues[d["id"]] = d
    return dialogues


def process_one(json_path: Path, dial_id: str, data: dict, dialogue: dict) -> tuple[bool, str]:
    """Generate prompt and write to JSON. Returns (success, message)."""
    try:
        client = anthropic.Anthropic()
        seed_json = json.dumps(dialogue["seed"], indent=2)
        text = GENERATION_PROMPT.format(
            seed_json=seed_json,
            dialogue=dialogue["dialogue"],
        )
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": text}],
        )
        text_prompt = response.content[0].text.strip()
        data["text_prompt"] = text_prompt
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        return True, f"{dial_id}: OK ({len(text_prompt)} chars)"
    except Exception as e:
        return False, f"{dial_id}: FAILED: {e}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dialogues", default="dialogues_1.jsonl", help="Path to dialogues JSONL"
    )
    parser.add_argument(
        "--stereo-dir", default="data/stereo_wav", help="Directory with dial-XXXXX.json files"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print first prompt without calling API"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing text_prompt fields"
    )
    parser.add_argument(
        "--workers", type=int, default=10, help="Number of parallel workers (default: 10)"
    )
    args = parser.parse_args()

    dialogues = load_dialogues(Path(args.dialogues))
    stereo_dir = Path(args.stereo_dir)
    json_files = sorted(stereo_dir.glob("dial-*.json"))

    if not json_files:
        print(f"No dial-*.json files found in {stereo_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(json_files)} JSON files, {len(dialogues)} dialogues")

    # Match JSON files to dialogues, skip already-done (resume)
    to_process = []
    skipped = 0
    for json_path in json_files:
        dial_id = json_path.stem
        if dial_id not in dialogues:
            print(f"  Skipping {dial_id}: no matching dialogue")
            continue

        with open(json_path) as f:
            data = json.load(f)

        if "text_prompt" in data and not args.overwrite:
            skipped += 1
            continue

        to_process.append((json_path, dial_id, data))

    if skipped:
        print(f"Resuming: skipped {skipped} already done")
    print(f"{len(to_process)} files to process")

    if not to_process:
        print("Nothing to do.")
        return

    if args.dry_run:
        jp, did, _ = to_process[0]
        seed_json = json.dumps(dialogues[did]["seed"], indent=2)
        print("\n=== DRY RUN: Prompt for", did, "===\n")
        print(
            GENERATION_PROMPT.format(
                seed_json=seed_json,
                dialogue=dialogues[did]["dialogue"][:2000] + "\n[...truncated for dry run]",
            )
        )
        return

    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one, jp, did, data, dialogues[did]): did
            for jp, did, data in to_process
        }
        for i, future in enumerate(as_completed(futures), 1):
            ok, msg = future.result()
            if ok:
                success += 1
            else:
                failed += 1
            print(f"[{i}/{len(to_process)}] {msg}")

    print(f"\nDone: {success} succeeded, {failed} failed")


if __name__ == "__main__":
    main()
