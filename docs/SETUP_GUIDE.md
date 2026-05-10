# PersonaPlex Finetuning VM Setup Guide
## Zero to Hero on a Fresh GCP VM

**Reference VM:** GCP preemptible, single H100 80GB, Debian 12 (Bookworm).
The same recipe works on bare metal or any cloud with NVIDIA GPUs; the
disk-mount and firewall steps are GCP-specific.

---

## Phase 1: OS & System Setup

### 1.1 Fix Debian Version (if Bullseye/EOL)

```bash
cat /etc/os-release  # Check current version

# If Bullseye (11), upgrade to Bookworm (12):
sudo sed -i 's/bullseye/bookworm/g' /etc/apt/sources.list /etc/apt/sources.list.d/*.list
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y -o Dpkg::Options::="--force-confold"
sudo DEBIAN_FRONTEND=noninteractive apt-get full-upgrade -y -o Dpkg::Options::="--force-confold"
sudo reboot
```

### 1.2 Install Base Packages

```bash
sudo apt-get update
sudo apt-get install -y -qq ufw unattended-upgrades fail2ban mdadm ffmpeg gh
```

### 1.3 Security Hardening

```bash
sudo ufw default deny incoming && sudo ufw default allow outgoing && sudo ufw allow ssh && sudo ufw --force enable
# fail2ban and unattended-upgrades are active by default after install
# Verify SSH is pubkey-only (no password auth, no root login)
```

---

## Phase 2: Disks & Storage

### 2.1 Persistent SSD (`/mnt/data`) -- format only on first use

```bash
# Identify the persistent disk (usually /dev/nvme0n2, ~500GB)
lsblk

# Format (ONLY if new/empty disk -- this destroys data!)
sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/nvme0n2

# Mount
sudo mkdir -p /mnt/data
sudo mount -o discard,defaults /dev/nvme0n2 /mnt/data
sudo chmod 775 /mnt/data

# Add to fstab for persistence across reboots
DISK_UUID=$(sudo blkid -s UUID -o value /dev/nvme0n2)
echo "UUID=$DISK_UUID /mnt/data ext4 discard,defaults,nofail 0 2" | sudo tee -a /etc/fstab
```

### 2.2 Local SSD RAID0 (`/mnt/scratch`) -- recreated every boot

Local SSDs are wiped on preemption. A systemd service recreates them on boot.

```bash
# Create the setup script
sudo tee /usr/local/sbin/setup-local-ssds.sh << 'SCRIPT'
#!/bin/bash
set -e
RAID_DEV=/dev/md0
MOUNT_POINT=/mnt/scratch
DEVICES=(/dev/nvme1n1 /dev/nvme2n1)
mdadm --stop "$RAID_DEV" 2>/dev/null || true
mdadm --create "$RAID_DEV" --level=0 --raid-devices=${#DEVICES[@]} "${DEVICES[@]}" --force --run
mkfs.ext4 -m 0 -F -E lazy_itable_init=0,lazy_journal_init=0,discard "$RAID_DEV"
mkdir -p "$MOUNT_POINT"
mount -o discard,defaults "$RAID_DEV" "$MOUNT_POINT"
chmod 777 "$MOUNT_POINT"
echo "Local SSDs mounted at $MOUNT_POINT ($(lsblk -dn -o SIZE "$RAID_DEV") RAID0)"
SCRIPT
sudo chmod +x /usr/local/sbin/setup-local-ssds.sh

# Create systemd service
sudo tee /etc/systemd/system/setup-local-ssds.service << 'UNIT'
[Unit]
Description=Assemble and mount local SSD RAID0
After=local-fs.target
Before=docker.service

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/setup-local-ssds.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable setup-local-ssds.service
sudo systemctl start setup-local-ssds.service
```

### 2.3 Directory Structure on Persistent Disk

```bash
mkdir -p /mnt/data/{cache/huggingface,runs,models,personaplex_data}
```

### 2.4 Move HuggingFace Cache to Persistent Disk

```bash
# If HF cache exists on root disk, move it:
rsync -a ~/.cache/huggingface/ /mnt/data/cache/huggingface/
rm -rf ~/.cache/huggingface
ln -s /mnt/data/cache/huggingface ~/.cache/huggingface
```

---

## Phase 3: NVIDIA Drivers & CUDA

### 3.1 Check Current State

```bash
nvidia-smi  # Check driver version & CUDA compatibility
```

### 3.2 Install/Upgrade CUDA Toolkit 12.8 + Drivers

```bash
# Add NVIDIA repo for Debian 12
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb -O /var/tmp/cuda-keyring.deb
sudo dpkg -i /var/tmp/cuda-keyring.deb
sudo apt-get update

# Install CUDA toolkit and compatible drivers
sudo apt-get install -y cuda-toolkit-12-8
sudo apt-get install -y cuda-drivers-570

sudo reboot
```

After reboot, verify:
```bash
nvidia-smi              # Should show driver 570+ and CUDA 12.8+
ls -la /usr/local/cuda  # Should point to cuda-12.8
```

---

## Phase 4: Claude Code (Optional but Recommended)

```bash
curl -fsSL https://claude.ai/install.sh | bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
```

---

## Phase 5: Conda & Python

### 5.1 Suppress Conda Base Auto-activation

If conda is pre-installed on the VM image:
```bash
conda config --set auto_activate_base false
```

Python 3.10 from conda is used as the base interpreter.

---

## Phase 6: Clone Repo & Set Up Python Environments

### 6.1 Clone the Repo

```bash
gh auth login
mkdir -p ~/projects && cd ~/projects
git clone <your-fork-of-voice-training>.git
cd voice-training
```

### 6.2 Main Venv (TTS pipeline, audio processing)

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip

# Install main requirements
.venv/bin/pip install -r requirements.txt
```

### 6.3 Install flash-attn (Prebuilt Wheel -- Do NOT Build from Source)

Building flash-attn from source takes forever. Use the prebuilt wheel:

```bash
# Download the correct wheel for CUDA 12 + torch 2.8+ + Python 3.10
gh release download v2.8.3 --repo Dao-AILab/flash-attention \
  --pattern 'flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl' \
  --dir /var/tmp

.venv/bin/pip install /var/tmp/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl --no-deps
```

> **Note:** If torch version changes, find the matching wheel at
> https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3

### 6.4 Fix Triton Version (Critical!)

triton 3.6.0 breaks `torch.compile(backend='inductor')` by removing `triton_key`. Downgrade:

```bash
.venv/bin/pip install triton==3.4.0
```

### 6.5 Fix torchcodec Version (if needed)

torchcodec 0.11.0 requires `libnvrtc.so.13` which isn't available with CUDA 12.8. Downgrade:

```bash
.venv/bin/pip install torchcodec==0.10.0
```

### 6.6 Moshi-Finetune Venv

```bash
cd moshi-finetune
python3 -m venv .venv
.venv/bin/pip install -e .
# Ensure correct triton for torch 2.6:
.venv/bin/pip install triton==3.2.0
cd ..
```

---

## Phase 7: Model Downloads

### 7.1 PersonaPlex Base Model

```bash
# This downloads ~29GB to the HF cache (now on /mnt/data via symlink)
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/personaplex-7b-v1')"
```

### 7.2 VibeVoice TTS Model

The VibeVoice-7B model should be at `vibevoice/VibeVoice-7B` inside the repo.
Check if it's tracked in git or needs separate download.

```bash
ls -la vibevoice/VibeVoice-7B/  # Verify model weights exist (~17GB)
```

---

## Phase 8: Patch PersonaPlex Compatibility

PersonaPlex's `config.json` is sparse (just `{"model_type": "personaplex", "version": "7b-v1"}`) and causes multiple issues. The upstream `moshi` package (from pip) is used for LoRA support, but needs patching.

### 8.1 Patch moshi loaders for dep_q

In the moshi-finetune venv's `moshi/models/loaders.py`, apply these fixes:

1. **Hardcode `dep_q = 16`** in `get_moshi_lm()`:
   ```python
   lm_kwargs["dep_q"] = 16
   ```

2. **Handle sparse lm_kwargs** -- when config has no `dim` key, ignore the sparse config and use `_lm_kwargs` defaults

3. **Strip non-model keys** like `version` before passing to model constructor

4. **Fix `get_mimi`** to use `num_codebooks = 8` (not 16) when `dep_q` is missing from `lm_config`

> These patches may already be committed to the voice-training repo. Check `git log` for relevant commits.

---

## Phase 9: Data Preparation Pipeline

The full pipeline for preparing training data:

### 9.1 Prepare Speaker Samples

Place voice reference WAV files in the speaker_samples directory:
```bash
mkdir -p /mnt/data/personaplex_data/<dataset>/speaker_samples
# Copy speaker sample WAVs here
```

### 9.2 Parse Dialogues

```bash
cd ~/projects/voice-training
.venv/bin/python pipeline/parse_dialogues.py \
  /mnt/data/personaplex_data/<dataset>/<dataset>.jsonl \
  /mnt/data/personaplex_data/<dataset>/scripts \
  10000
```

### 9.3 Generate Mono Audio (TTS)

```bash
.venv/bin/python pipeline/generate_audio.py \
  --model-path vibevoice/VibeVoice-7B \
  --dialogues /mnt/data/personaplex_data/<dataset>/<dataset>.jsonl \
  --output-dir /mnt/data/personaplex_data/<dataset>/mono_wav \
  --scripts-dir /mnt/data/personaplex_data/<dataset>/scripts \
  --voices-dir /mnt/data/personaplex_data/<dataset>/speaker_samples \
  --gpus 0 --resume --batch-size 16
```

> **Batch size notes:** 16 is the sweet spot on H100. 64 causes int32 overflow in conv1d. 32 works but 16 is safer. OOM fallback retries individual dialogues automatically.

### 9.4 Create Stereo WAV + Alignment

```bash
.venv/bin/python pipeline/create_stereo.py \
  --mono-dir /mnt/data/personaplex_data/<dataset>/mono_wav \
  --scripts-dir /mnt/data/personaplex_data/<dataset>/scripts \
  --output-dir /mnt/data/personaplex_data/<dataset>/stereo_wav \
  --dialogues /mnt/data/personaplex_data/<dataset>/<dataset>.jsonl \
  --whisper-model large-v3 \
  --gpus 0 --resume --batch-size 8
```

### 9.5 Compute Injection Offsets

```bash
.venv/bin/python pipeline/compute_injection_offsets.py \
  --stereo-dir /mnt/data/personaplex_data/<dataset>/stereo_wav
```

### 9.6 Create Train/Eval Manifest

```bash
.venv/bin/python pipeline/create_manifest.py \
  --stereo-dir /mnt/data/personaplex_data/<dataset>/stereo_wav \
  --train-output /mnt/data/personaplex_data/<dataset>/dataset/train.jsonl \
  --eval-output /mnt/data/personaplex_data/<dataset>/dataset/eval.jsonl \
  --eval-fraction 0.05
```

---

## Phase 10: Training

### 10.1 Training Config

Use `configs/pharma_demo.yaml` as the template, or `configs/pharma_demo.yaml`
for the patient-support demo. Replace every `<EDIT_ME>` path with your absolute
repo or data path before launching. Key parameters:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `hf_repo_id` | `nvidia/personaplex-7b-v1` | Base model |
| `lora.rank` | 64 | LoRA rank |
| `lora.scaling` | 2.0 | |
| `duration_sec` | 80.0 | Training window length |
| `text_padding_weight` | 0.5 | Loss weight on silence (lower = less silence in output) |
| `first_codebook_weight_multiplier` | 10.0 | Boost text coherence |
| `batch_size` | 32 | |
| `max_steps` | 128 | |
| `lr` | 2e-5 | |
| `eval_freq` / `ckpt_freq` | 32 | |
| `run_dir` | `/mnt/data/runs/pharma_demo-v0-X` | Change per run |
| `wandb.project` | `personaplex-insurance` | Set `wandb.entity` to your own WandB entity |

### 10.2 Launch Training

```bash
cd ~/projects/voice-training/moshi-finetune

CUDA_VISIBLE_DEVICES=0 .venv/bin/torchrun --nproc-per-node 1 \
  -m train ../configs/pharma_demo.yaml
```

Alternative without torchrun:
```bash
CUDA_VISIBLE_DEVICES=0 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 \
  MASTER_ADDR=localhost MASTER_PORT=29500 \
  .venv/bin/python -m train ../configs/pharma_demo.yaml
```

### 10.3 Gen Eval

Gen eval runs automatically during training if `gen_eval.enable: true`. To run manually:

```bash
cd ~/projects/voice-training/moshi-finetune

~/projects/voice-training/.venv/bin/python -m finetune.gen_eval \
  --checkpoint /mnt/data/runs/pharma_demo-v0-X/checkpoints/checkpoint_000032/consolidated \
  --run-dir /mnt/data/runs/pharma_demo-v0-X \
  --step 32 \
  --config ~/projects/voice-training/configs/pharma_demo.yaml
```

> Gen eval starts a Moshi server with the merged LoRA checkpoint and connects Gemini Live as the user simulator. Set `GEMINI_API_KEY` before running it.

---

## Phase 11: Merge LoRA & Serve

### 11.1 Merge LoRA Weights

```bash
cd ~/projects/voice-training
.venv/bin/python pipeline/merge_lora.py \
  --checkpoint /mnt/data/runs/pharma_demo-v0-X/checkpoints/checkpoint_000096/consolidated \
  --output /mnt/data/runs/pharma_demo-v0-X/merged/model.safetensors
```

### 11.2 Build Client UI

```bash
sudo apt install nodejs npm   # if not installed
cd ~/projects/voice-training/personaplex/client
npm install && npm run build
cd ../..
```

### 11.3 Install Moshi Inference Package

The inference server uses the moshi package from `personaplex/moshi/`. Install it in the main venv:

```bash
.venv/bin/pip install sentencepiece==0.2.0 "sphn>=0.1.4,<0.2"
.venv/bin/pip install ./personaplex/moshi/
```

> **Note:** This pins torch <2.5 which conflicts with whisperx/pyannote. For a machine that does both training and inference, consider a separate venv for inference.

### 11.4 Serve with Moshi Server

```bash
cd ~/projects/voice-training

# Base model (no LoRA):
.venv/bin/python -m moshi.server --static personaplex/client/dist

# With merged LoRA:
.venv/bin/python -m moshi.server \
  --moshi-weight /mnt/data/runs/pharma_demo-v0-X/merged/model.safetensors \
  --static personaplex/client/dist
```

Access via `http://localhost:8998`. If using VS Code Remote / Cursor, port 8998 is auto-forwarded — just open `http://localhost:8998` in your local browser. No SSL needed (browsers allow mic on localhost).

For external IP access, SSL is required (browsers block mic on non-HTTPS non-localhost):
```bash
SSL_DIR=$(mktemp -d)
openssl req -x509 -newkey rsa:2048 -keyout $SSL_DIR/key.pem -out $SSL_DIR/cert.pem -days 365 -nodes -subj '/CN=localhost'
.venv/bin/python -m moshi.server \
  --moshi-weight /mnt/data/runs/pharma_demo-v0-X/merged/model.safetensors \
  --static personaplex/client/dist --ssl $SSL_DIR
```
Also needs GCloud firewall rule: `gcloud compute firewall-rules create allow-moshi-8998 --allow tcp:8998 --direction INGRESS --source-ranges 0.0.0.0/0` (run from local machine, not VM — VM may lack API scopes).

---

## Known Pitfalls & Gotchas

| Issue | Symptom | Fix |
|-------|---------|-----|
| Wrong `apex` package | `pip install apex` installs Pyramid web framework | Don't install it. FusedRMSNorm warning is cosmetic. |
| triton 3.6.0 | `AttributeError: triton_key` during torch.compile | `pip install triton==3.4.0` |
| torchcodec 0.11.0 | `libnvrtc.so.13 not found` | `pip install torchcodec==0.10.0` |
| flash-attn build | Takes 1+ hours from source | Use prebuilt wheel from GH releases |
| PersonaPlex sparse config | `KeyError: dep_q`, dimension mismatches | Patch `moshi/models/loaders.py` (dep_q=16, num_codebooks=8) |
| Root disk full | HF cache fills 128GB boot disk | Symlink `~/.cache/huggingface` to `/mnt/data/cache/huggingface` |
| Batch size 64 | int32 overflow in conv1d (32BitIndexMath) | Use batch_size 16 |
| `text_prompt` wrong | Model learns Claude's meta-instructions instead of persona | Ensure `text_prompt` is the actual system prompt for the model |
| 65% silence in output | Model produces mostly silence | Lower `text_padding_weight` (0.5 -> 0.05), increase `duration_sec`, lower sampling temps |
| Disk full during pip install | `OSError: [Errno 28] No space left on device` | `.venv/bin/pip cache purge` frees GBs |
| HTTPS on plain HTTP server | `ERR_SSL_PROTOCOL_ERROR`, `BadStatusLine: \x16\x03\x01` in logs | Use `http://` not `https://`, or add `--ssl` flag |
| Mic blocked in browser | UI loads but no audio captured | Must use `localhost` or HTTPS — browsers block mic on plain HTTP non-localhost |
| GCloud firewall rule from VM | `insufficient authentication scopes` | Run `gcloud compute firewall-rules create` from local machine instead |

---

## Quick Reference: Full Pipeline Commands (Copy-Paste)

```bash
# === ONE-TIME SETUP ===
# (Phases 1-8 above)

# === PER-DATASET PIPELINE ===
DATASET=/mnt/data/personaplex_data/YOUR_DATASET
DIALOGUES=$DATASET/your_dialogues.jsonl
cd ~/projects/voice-training

# 1. Parse dialogues
.venv/bin/python pipeline/parse_dialogues.py $DIALOGUES $DATASET/scripts 10000

# 2. Generate mono audio
.venv/bin/python pipeline/generate_audio.py \
  --model-path vibevoice/VibeVoice-7B \
  --dialogues $DIALOGUES --output-dir $DATASET/mono_wav \
  --scripts-dir $DATASET/scripts --voices-dir $DATASET/speaker_samples \
  --gpus 0 --resume --batch-size 16

# 3. Create stereo
.venv/bin/python pipeline/create_stereo.py \
  --mono-dir $DATASET/mono_wav --scripts-dir $DATASET/scripts \
  --output-dir $DATASET/stereo_wav --dialogues $DIALOGUES \
  --whisper-model large-v3 --gpus 0 --resume --batch-size 8

# 4. Compute injection offsets
.venv/bin/python pipeline/compute_injection_offsets.py --stereo-dir $DATASET/stereo_wav

# 5. Create manifest
.venv/bin/python pipeline/create_manifest.py \
  --stereo-dir $DATASET/stereo_wav \
  --train-output $DATASET/dataset/train.jsonl \
  --eval-output $DATASET/dataset/eval.jsonl \
  --eval-fraction 0.05

# 6. Train
cd moshi-finetune
CUDA_VISIBLE_DEVICES=0 .venv/bin/torchrun --nproc-per-node 1 -m train ../configs/YOUR_CONFIG.yaml
```

---

## Environment variables (every shell)

The training and eval code reads several env vars. Keep them in
`/etc/profile.d/voice-training.sh` so they're available system-wide, then
ensure non-login shells pick them up by adding this to `~/.bashrc`:

```bash
for f in /etc/profile.d/*.sh; do
    [ -r "$f" ] && . "$f"
done
```

| Variable | Purpose |
|----------|---------|
| `NCCL_NET=Socket` | Forces NCCL to use TCP sockets (single-GPU GCP instances have no InfiniBand) |
| `ANTHROPIC_API_KEY` | Claude API key for dialogue generation and LLM transcript review |
| `GEMINI_API_KEY` | Gemini key for `pipeline/gemini_eval.py` and checkpoint gen eval |
| `HF_TOKEN` | HuggingFace token if you need to gate-pull a private model |

A starter `.env.example` lives at the repo root.

## Single-GPU gen eval

Gen eval runs Gemini Live ↔ Moshi dialogues + LLM review at checkpoint time.
On a single GPU (e.g., one H100 80GB), set `gpu_a` to that device in the
training config:

```yaml
gen_eval:
  enable: true
  freq: 128
  gpu_a: 0
  moshi_port: 8998
```

The training loop offloads the model, optimizer, and Mimi to CPU before gen
eval, freeing ~80 GB on an H100. The merged Moshi server runs on the freed
GPU while Gemini Live provides the simulated client. After gen eval completes,
training state is restored and training
resumes.

What gen eval does:

1. Merges LoRA into base model (CPU, saves ~17 GB safetensors to disk).
2. Starts a Moshi server with the merged weights and runs the configured
   Gemini Live ↔ Moshi eval dialogues.
3. Runs conversation profile eval via WhisperX (turn gaps, speech rate,
   backchannels).
4. Runs LLM transcript review (coherence, naturalness, effectiveness,
   grounding).
5. Logs metrics to wandb + saves `results.json`.

The `anthropic` package needs to be installed in the moshi-finetune venv:

```bash
moshi-finetune/.venv/bin/pip install anthropic
```
