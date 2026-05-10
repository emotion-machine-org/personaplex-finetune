"""Parse dialogues from JSONL into VibeVoice script format."""
import json
import re
import sys
from pathlib import Path


def parse_dialogue_to_vibevoice(dialogue_text: str) -> str | None:
    """Convert a dialogue string to VibeVoice 'Speaker N:' format.

    BROKER -> Speaker 1, CLIENT -> Speaker 2.
    """
    lines = dialogue_text.strip().split('\n')
    output_lines = []

    # Pattern: BROKER (Name): text  or  CLIENT (Name): text
    broker_client_pattern = re.compile(r'^(BROKER|CLIENT)\s*\([^)]*\):\s*(.*)$')
    # Pattern: USER: text  or  COMPANION_NAME: text (any non-USER speaker)
    user_companion_pattern = re.compile(r'^([A-Z][A-Z ]+):\s*(.*)$')

    # First pass: detect format and find companion name
    companion_name = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = user_companion_pattern.match(line)
        if match and match.group(1) != "USER":
            companion_name = match.group(1)
            break

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip markers
        if line.startswith('[') and line.endswith(']'):
            continue

        match = broker_client_pattern.match(line)
        if match:
            role = match.group(1)
            text = match.group(2).strip()
            speaker_num = 1 if role == "BROKER" else 2
        elif companion_name:
            match = user_companion_pattern.match(line)
            if match:
                role = match.group(1)
                text = match.group(2).strip()
                speaker_num = 2 if role == "USER" else 1
            else:
                continue
        else:
            continue

        if not match:
            continue
        # Normalize unicode
        text = text.replace('\u2014', '--')  # em-dash
        text = text.replace('\u2013', '-')   # en-dash
        text = text.replace('\u2018', "'").replace('\u2019', "'")  # smart single quotes
        text = text.replace('\u201c', '"').replace('\u201d', '"')  # smart double quotes
        text = text.replace('\u2026', '...')  # ellipsis
        output_lines.append(f"Speaker {speaker_num}: {text}")

    if not output_lines:
        return None
    return '\n'.join(output_lines)


def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dialogues_1.jsonl")
    scripts_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/scripts")
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    scripts_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(input_path) as f:
        for line in f:
            if count >= n:
                break
            entry = json.loads(line)
            dialogue = entry.get("dialogue")
            if not dialogue:
                continue

            dial_id = entry.get("id", f"dial-{count:05d}")
            script = parse_dialogue_to_vibevoice(dialogue)
            if not script:
                continue

            # Count words - skip if too long
            word_count = len(script.split())
            if word_count > 3000:
                print(f"Skipping {dial_id}: {word_count} words (too long)")
                continue

            out_path = scripts_dir / f"{dial_id}.txt"
            out_path.write_text(script)
            print(f"Wrote {out_path} ({word_count} words)")
            count += 1

    print(f"\nParsed {count} dialogues to {scripts_dir}")


if __name__ == "__main__":
    main()
