# Quick Launch Reference

Replace `<REPO>` with the absolute path to your `voice-training` checkout
and `<HF_CACHE>` with `~/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/<rev>`.

## A/B Test Server (dual model)
```bash
NO_CUDA_GRAPH=1 CUDA_VISIBLE_DEVICES=0,1 .venv/bin/python -m moshi.server \
  --host 0.0.0.0 --port 8998 --device cuda:0 \
  --moshi-weight <REPO>/runs/<your_run>/checkpoints/<ckpt>/model.safetensors \
  --ab-moshi-weight <HF_CACHE>/model.safetensors \
  --ab-device cuda:1 --model-label finetuned --ab-label base \
  --static ../../personaplex/client/dist \
  --voice-prompt-dir <HF_CACHE>/voices \
  --ssl <ssl_dir>
```

## Single Model Server (simpler, for debugging)
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m moshi.server \
  --host 0.0.0.0 --port 8998 --device cuda \
  --moshi-weight <REPO>/runs/<your_run>/checkpoints/<ckpt>/model.safetensors \
  --static ../../personaplex/client/dist \
  --voice-prompt-dir <HF_CACHE>/voices \
  --ssl <ssl_dir>
```

## Build Client
```bash
cd <REPO>/personaplex/client
npm run build
```

## SSH Port Forward
```bash
ssh -L 8998:localhost:8998 user@host
# Then open https://localhost:8998
```

## Key env vars
- `NO_CUDA_GRAPH=1` — required for multi-GPU, prevents broken CUDA graph capture on second GPU
- `CUDA_VISIBLE_DEVICES=0,1` — which GPUs to use
