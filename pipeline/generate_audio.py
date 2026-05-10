"""Generate audio from dialogues using VibeVoice 7B with multi-GPU support."""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

from tqdm import tqdm


def parse_dialogue(dialogue_text):
    """Convert dialogue to VibeVoice 'Speaker N:' format.

    Supports both BROKER/CLIENT format and USER/COMPANION format.
    """
    lines = dialogue_text.strip().split('\n')
    output_lines = []
    broker_client_pattern = re.compile(r'^(BROKER|CLIENT)\s*\([^)]*\):\s*(.*)$')
    user_companion_pattern = re.compile(r'^([A-Z][A-Z ]+):\s*(.*)$')

    # Detect companion name (first non-USER uppercase speaker)
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
        if not line or (line.startswith('[') and line.endswith(']')):
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
        text = text.replace('\u2014', '--').replace('\u2013', '-')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2026', '...')
        output_lines.append(f"Speaker {speaker_num}: {text}")
    return '\n'.join(output_lines) if output_lines else None


def load_dialogues(jsonl_path, max_words=3000):
    """Load and parse dialogues from JSONL."""
    dialogues = []
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            dialogue = entry.get("dialogue")
            if not dialogue:
                continue
            script = parse_dialogue(dialogue)
            if not script:
                continue
            word_count = len(script.split())
            if word_count > max_words:
                print(f"Skipping {entry.get('id', '?')}: {word_count} words (>{max_words})")
                continue
            dialogues.append({
                "id": entry.get("id", f"dial-{len(dialogues):05d}"),
                "script": script,
                "word_count": word_count,
            })
    return dialogues


def assign_voices(dialogues, voices_dir, seed=42):
    """Assign 2 random distinct voices per dialogue, deterministic per ID."""
    voice_files = sorted(str(p) for p in Path(voices_dir).glob("*.wav"))
    if len(voice_files) < 2:
        raise ValueError(f"Need >=2 voices in {voices_dir}, found {len(voice_files)}")
    assignments = {}
    for d in dialogues:
        rng = random.Random(f"{seed}_{d['id']}")
        v1, v2 = rng.sample(voice_files, 2)
        assignments[d["id"]] = [v1, v2]
    return assignments


def make_batches(dialogues, batch_size):
    """Group dialogues into batches sorted by word count for efficient padding."""
    sorted_dialogues = sorted(dialogues, key=lambda d: d["word_count"])
    return [sorted_dialogues[i:i + batch_size] for i in range(0, len(sorted_dialogues), batch_size)]


def generate_shard(args, dialogues, voice_assignments):
    """Load model on one GPU and generate audio for a shard of dialogues."""
    import torch
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'VibeVoice'))
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_tag = f"[GPU {os.environ.get('CUDA_VISIBLE_DEVICES', args.device)}]"

    # Resume: skip already-generated files
    if args.resume:
        before = len(dialogues)
        dialogues = [
            d for d in dialogues
            if not (output_dir / f"{d['id']}.wav").exists()
            or (output_dir / f"{d['id']}.wav").stat().st_size <= 1000
        ]
        print(f"{gpu_tag} {before - len(dialogues)} already done, {len(dialogues)} remaining")

    if not dialogues:
        print(f"{gpu_tag} Nothing to generate.")
        return

    # Load model
    print(f"{gpu_tag} Loading model from {args.model_path}...")
    processor = VibeVoiceProcessor.from_pretrained(args.model_path)
    try:
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            device_map=args.device, attn_implementation="flash_attention_2",
        )
        print(f"{gpu_tag} Using flash_attention_2")
    except Exception as e:
        print(f"{gpu_tag} flash_attn failed ({e}), falling back to sdpa")
        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16,
            device_map=args.device, attn_implementation="sdpa",
        )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)

    batch_size = args.batch_size
    batches = make_batches(dialogues, batch_size)
    print(f"{gpu_tag} Model loaded, generating {len(dialogues)} dialogues in {len(batches)} batches (bs={batch_size})\n")

    done_count = 0
    for bi, batch in enumerate(batches):
        ids = [d["id"] for d in batch]
        wc_range = f"{batch[0]['word_count']}-{batch[-1]['word_count']}"
        print(f"{gpu_tag} Batch {bi+1}/{len(batches)}: {len(batch)} dialogues ({wc_range} words) {ids}")

        try:
            texts = [d["script"] for d in batch]
            voices_list = [voice_assignments[d["id"]] for d in batch]
            inputs = processor(
                text=texts,
                voice_samples=voices_list,
                padding=True, return_tensors="pt", return_attention_mask=True,
            )
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(args.device)

            start = time.time()
            outputs = model.generate(
                **inputs, max_new_tokens=None, cfg_scale=args.cfg_scale,
                tokenizer=processor.tokenizer,
                generation_config={'do_sample': False},
                verbose=False, is_prefill=True,
            )
            elapsed = time.time() - start

            for j, d in enumerate(batch):
                out_path = output_dir / f"{d['id']}.wav"
                audio = outputs.speech_outputs[j]
                if audio is not None:
                    processor.save_audio(audio, output_path=str(out_path))
                    n_samples = audio.shape[-1]
                    duration = n_samples / 24000
                    print(f"{gpu_tag}   {d['id']}: {duration:.1f}s audio")
                else:
                    print(f"{gpu_tag}   {d['id']}: no audio output")

            total_audio = sum(
                o.shape[-1] / 24000 for o in outputs.speech_outputs if o is not None
            )
            rtf = elapsed / total_audio if total_audio > 0 else float('inf')
            done_count += len(batch)
            print(f"{gpu_tag}   Batch done: {total_audio:.1f}s audio in {elapsed:.1f}s (RTF={rtf:.2f}x) [{done_count}/{len(dialogues)}]")

        except Exception as e:
            if "OutOfMemoryError" in type(e).__name__ or "32BitIndexMath" in str(e):
                reason = "OOM" if "OutOfMemoryError" in type(e).__name__ else "tensor too large"
                print(f"{gpu_tag}   {reason} on batch (bs={len(batch)}), falling back to bs=1...")
                torch.cuda.empty_cache()
                for d in batch:
                    try:
                        inputs = processor(
                            text=[d["script"]],
                            voice_samples=[voice_assignments[d["id"]]],
                            padding=True, return_tensors="pt", return_attention_mask=True,
                        )
                        for k, v in inputs.items():
                            if torch.is_tensor(v):
                                inputs[k] = v.to(args.device)
                        start = time.time()
                        outputs = model.generate(
                            **inputs, max_new_tokens=None, cfg_scale=args.cfg_scale,
                            tokenizer=processor.tokenizer,
                            generation_config={'do_sample': False},
                            verbose=False, is_prefill=True,
                        )
                        elapsed = time.time() - start
                        out_path = output_dir / f"{d['id']}.wav"
                        if outputs.speech_outputs[0] is not None:
                            processor.save_audio(outputs.speech_outputs[0], output_path=str(out_path))
                            n_samples = outputs.speech_outputs[0].shape[-1]
                            duration = n_samples / 24000
                            print(f"{gpu_tag}   {d['id']}: {duration:.1f}s audio in {elapsed:.1f}s (fallback bs=1)")
                        done_count += 1
                    except Exception as inner_e:
                        print(f"{gpu_tag}   Error on {d['id']} (fallback): {inner_e}")
                        torch.cuda.empty_cache()
                        done_count += 1
            else:
                print(f"{gpu_tag}   Error on batch: {e}")
                traceback.print_exc()
                done_count += len(batch)


def main():
    parser = argparse.ArgumentParser(description="Generate audio with VibeVoice 7B")
    parser.add_argument("--model-path", default="vibevoice/VibeVoice-7B")
    parser.add_argument("--dialogues", default="dialogues_1.jsonl", help="Input JSONL file")
    parser.add_argument("--output-dir", default="dialogues_1_data/mono_wav")
    parser.add_argument("--scripts-dir", default="dialogues_1_data/scripts",
                        help="Save parsed scripts here (for downstream pipeline steps)")
    parser.add_argument("--voices-dir", default="dialogues_1_data/speaker_samples")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpus", default=None,
                        help="Comma-separated GPU IDs for multi-GPU, e.g. '0,1,2,3'")
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=1.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-words", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of dialogues to generate in parallel")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # --- Multi-GPU launcher mode ---
    if args.gpus:
        gpu_ids = [int(g) for g in args.gpus.split(",")]

        # Pre-parse dialogues, save scripts and voice assignments before spawning
        dialogues = load_dialogues(args.dialogues, args.max_words)
        print(f"Loaded {len(dialogues)} dialogues from {args.dialogues}")

        voice_assignments = assign_voices(dialogues, args.voices_dir, args.seed)

        # Save scripts for downstream steps (create_stereo.py)
        scripts_dir = Path(args.scripts_dir)
        scripts_dir.mkdir(parents=True, exist_ok=True)
        for d in dialogues:
            (scripts_dir / f"{d['id']}.txt").write_text(d["script"])
        print(f"Saved {len(dialogues)} scripts to {scripts_dir}")

        # Save voice assignments
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "voice_assignments.json", "w") as f:
            json.dump(voice_assignments, f, indent=2)
        print(f"Saved voice assignments to {output_dir / 'voice_assignments.json'}")

        # Launch one subprocess per GPU
        procs = []
        for i, gpu_id in enumerate(gpu_ids):
            cmd = [
                sys.executable, os.path.abspath(__file__),
                "--model-path", args.model_path,
                "--dialogues", args.dialogues,
                "--output-dir", args.output_dir,
                "--scripts-dir", args.scripts_dir,
                "--voices-dir", args.voices_dir,
                "--device", "cuda:0",
                "--shard", str(i),
                "--num-shards", str(len(gpu_ids)),
                "--cfg-scale", str(args.cfg_scale),
                "--seed", str(args.seed),
                "--max-words", str(args.max_words),
                "--batch-size", str(args.batch_size),
            ]
            if args.resume:
                cmd.append("--resume")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["TQDM_DISABLE"] = "1"
            log_path = output_dir / f"shard_{i}_gpu{gpu_id}.log"
            log_file = open(log_path, "w")
            print(f"Launching shard {i}/{len(gpu_ids)} on GPU {gpu_id} (log: {log_path})")
            procs.append((subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file), log_file))

        # Poll output dir for completed .wav files and show a single progress bar
        expected = {d["id"] for d in dialogues}
        done_before = {p.stem for p in Path(args.output_dir).glob("*.wav")
                       if p.stem in expected and p.stat().st_size > 1000}
        total = len(expected)
        pbar = tqdm(total=total, initial=len(done_before), desc="Generating audio",
                    unit="dial", dynamic_ncols=True)
        while any(p.poll() is None for p, _ in procs):
            done_now = {p.stem for p in Path(args.output_dir).glob("*.wav")
                        if p.stem in expected and p.stat().st_size > 1000}
            new = len(done_now) - pbar.n
            if new > 0:
                pbar.update(new)
            time.sleep(2)
        # Final sweep after all processes exit
        done_now = {p.stem for p in Path(args.output_dir).glob("*.wav")
                    if p.stem in expected and p.stat().st_size > 1000}
        new = len(done_now) - pbar.n
        if new > 0:
            pbar.update(new)
        pbar.close()

        for _, log_file in procs:
            log_file.close()

        exit_codes = [p.returncode for p, _ in procs]
        failed = [(i, c) for i, c in enumerate(exit_codes) if c != 0]
        if failed:
            print(f"Warning: shards failed: {failed}")
            for i, code in failed:
                log_path = output_dir / f"shard_{i}_gpu{gpu_ids[i]}.log"
                print(f"  See log: {log_path}")
            sys.exit(1)
        else:
            print(f"All {len(gpu_ids)} shards complete.")
        return

    # --- Single-GPU worker mode ---
    dialogues = load_dialogues(args.dialogues, args.max_words)
    voice_assignments = assign_voices(dialogues, args.voices_dir, args.seed)

    # Shard filtering
    if args.shard is not None and args.num_shards is not None:
        dialogues = [d for i, d in enumerate(dialogues) if i % args.num_shards == args.shard]
        print(f"Shard {args.shard}/{args.num_shards}: {len(dialogues)} dialogues")

    # Save scripts (idempotent, each shard writes its own subset)
    scripts_dir = Path(args.scripts_dir)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    for d in dialogues:
        (scripts_dir / f"{d['id']}.txt").write_text(d["script"])

    generate_shard(args, dialogues, voice_assignments)


if __name__ == "__main__":
    main()
