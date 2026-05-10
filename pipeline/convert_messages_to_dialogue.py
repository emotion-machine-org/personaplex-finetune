"""Convert messages-format JSONL to dialogue-format JSONL for the pipeline.

Input format:  {id, system_prompt, assistant_name, messages: [{role, content}, ...]}
Output format: {id, text_prompt, dialogue, context_injections: []}
"""
import json
import sys
from pathlib import Path


def convert_record(record: dict) -> dict:
    assistant_name = record.get("assistant_name", "ASSISTANT").upper()
    messages = record["messages"]

    # Build dialogue text: ASSISTANT_NAME: text / USER: text
    lines = []
    for msg in messages:
        if msg["role"] == "assistant":
            lines.append(f"{assistant_name}: {msg['content']}")
        elif msg["role"] == "user":
            lines.append(f"USER: {msg['content']}")

    return {
        "id": record["id"],
        "text_prompt": record.get("system_prompt", ""),
        "dialogue": "\n".join(lines),
        "context_injections": [],
    }


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.jsonl> <output.jsonl>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            record = json.loads(line)
            converted = convert_record(record)
            fout.write(json.dumps(converted) + "\n")
            count += 1

    print(f"Converted {count} records: {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
