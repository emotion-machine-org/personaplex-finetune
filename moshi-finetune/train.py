import dataclasses
import gc
import logging
import os
import pprint
import shutil
from contextlib import ExitStack
from pathlib import Path

import fire
import torch
import torch.cuda
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from torch.optim import AdamW, lr_scheduler

# from torch.profiler import ProfilerActivity, profile

from finetune.args import TrainArgs
from finetune.checkpointing import Checkpointer
from finetune.data.data_loader import build_data_loader
from finetune.data.interleaver import InterleavedTokenizer, Interleaver
from finetune.distributed import (
    BACKEND,
    avg_aggregate,
    get_rank,
    get_world_size,
    is_torchrun,
    set_device,
)
from finetune.eval import evaluate
from finetune.loss import compute_loss_with_mask
from finetune.mixed_precision import (
    downcast_mixed_precision,
    prepare_mixed_precision,
    upcast_mixed_precision,
)
from finetune.monitoring.metrics_logger import (
    MetricsLogger,
    eval_log_msg,
    get_eval_logs,
    get_train_logs,
    train_log_msg,
)
from finetune.monitoring.utils import set_logger
from finetune.utils import TrainState, logged_closing, set_random_seed
from finetune.wrapped_model import get_fsdp_model
from moshi.models import loaders

logger = logging.getLogger("train")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def _fix_optim_state_dtype(optimizer, model):
    """Ensure optimizer state tensors are fp32 on the correct device.

    optimizer.load_state_dict may cast state tensors to match the param's
    current dtype (bf16), but the mixed-precision scheme requires fp32 state.
    """
    for p in model.parameters():
        if p in optimizer.state:
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in optimizer.state[p]:
                    t = optimizer.state[p][key]
                    if t.dtype != torch.float32 or t.device != p.device:
                        optimizer.state[p][key] = t.to(
                            device=p.device, dtype=torch.float32
                        )


def _nuke_cuda_tensors():
    """Replace all CUDA tensor data with empty CPU tensors to free GPU memory.

    FSDP holds internal references to flat-parameter storage that survive
    del/gc.collect/empty_cache. Walking gc.get_objects() while the model is
    still referenced is the only reliable way to reach every CUDA tensor
    in the process.

    IMPORTANT: must be called BEFORE del model — otherwise gc can't find
    FSDP's internal flat_params.
    """
    gc.collect()
    nuked = 0
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            try:
                obj.data = torch.empty(0, device="cpu")
                nuked += 1
            except Exception:
                pass
    gc.collect()
    torch.cuda.empty_cache()
    return nuked


def _run_gen_eval_with_teardown(model, optimizer, mimi, ckpt_dir, run_dir, step, args,
                                checkpoint_info, param_dtype, optim_dtype, scheduler, state):
    """Nuke all CUDA tensors, run gen_eval, then rebuild model from checkpoint.

    After a real FSDP forward/backward, ~7-8 GB of GPU memory per rank is held
    by FSDP's internal flat-parameter storage. Normal del/gc/empty_cache can't
    free it. We walk gc.get_objects() to replace all CUDA tensor data with empty
    CPU tensors, then rebuild everything from the just-saved checkpoint.

    Returns (model, optimizer, mimi) — callers must update their references.
    """
    import datetime
    import time

    rank = get_rank()

    # 1. Nuke all CUDA tensors (while model is still referenced so gc can find them)
    n = _nuke_cuda_tensors()
    logger.info(f"[Rank {rank}] Nuked {n} CUDA tensors")
    free, total = torch.cuda.mem_get_info()
    logger.info(f"[Rank {rank}] GPU after nuke: {free/1e9:.1f}/{total/1e9:.1f} GB free")

    # 2. Destroy PG — the nuke also corrupts NCCL's internal CUDA tensors,
    #    so we must tear down the communicator before it notices.
    #    File-based signaling replaces barriers for the gen_eval window.
    dist.barrier()
    dist.destroy_process_group()

    # 3. Run gen_eval on rank 0; other ranks poll for completion file
    signal_file = Path(run_dir) / f".gen_eval_done_{step}"
    signal_file.unlink(missing_ok=True)

    if rank == 0:
        try:
            from finetune.gen_eval import run as run_gen_eval
            run_gen_eval(ckpt_dir, run_dir, step, args)
        except Exception as e:
            logger.error(f"Gen eval failed at step {step}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            signal_file.touch()
    else:
        while not signal_file.exists():
            time.sleep(2)

    # 4. Re-init PG with fresh NCCL state
    nccl_timeout = datetime.timedelta(hours=1)
    dist.init_process_group(backend=BACKEND, timeout=nccl_timeout)
    dist.barrier()
    signal_file.unlink(missing_ok=True)

    # 5. Rebuild model, optimizer, mimi from checkpoint
    logger.info(f"[Rank {rank}] Rebuilding model from checkpoint...")
    lora_path = str(Path(ckpt_dir) / "lora.safetensors")
    if not Path(lora_path).exists():
        lora_path = None

    mimi = checkpoint_info.get_mimi(device="cuda")
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    model = get_fsdp_model(args, checkpoint_info, resume_lora_path=lora_path)

    optimizer = AdamW(
        model.parameters(),
        lr=args.optim.lr,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=args.optim.weight_decay,
    )

    # Restore optimizer state from checkpoint
    saved = Checkpointer.load_train_state(Path(ckpt_dir))
    if saved is not None and "optimizer" in saved:
        if get_world_size() > 1:
            sharded = FullyShardedDataParallel.shard_full_optim_state_dict(
                saved["optimizer"], model
            )
            optimizer.load_state_dict(sharded)
        else:
            optimizer.load_state_dict(saved["optimizer"])
        _fix_optim_state_dtype(optimizer, model)

    # Restore scheduler: create a new one positioned at current step
    new_scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.optim.lr,
        total_steps=args.max_steps,
        pct_start=args.optim.pct_start,
        last_epoch=state.step - 1 if state.step > 0 else -1,
    )
    # Sync LRs
    for i, lr_val in enumerate(new_scheduler.get_last_lr()):
        optimizer.param_groups[i]['lr'] = lr_val

    prepare_mixed_precision(
        model.parameters(), param_dtype=param_dtype, optim_dtype=optim_dtype
    )

    return model, optimizer, mimi, new_scheduler


def train(config: str, resume_from: str | None = None):
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    if resume_from is not None:
        args.resume_from = resume_from
    set_logger(logging.INFO)

    with ExitStack() as exit_stack:
        _train(args, exit_stack)
    logger.info("Closed everything!")


def _train(args: TrainArgs, exit_stack: ExitStack):
    # 1. Initial setup and checks
    set_random_seed(args.seed)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Init NCCL
    if "LOCAL_RANK" in os.environ:
        set_device()
        logger.info("Going to init comms...")

        # Gen eval can take 15+ min (merge + dialogues + profile + LLM review);
        # non-rank-0 processes wait at barriers, so set a generous timeout.
        import datetime
        nccl_timeout = datetime.timedelta(hours=1) if args.gen_eval.enable else datetime.timedelta(minutes=10)
        dist.init_process_group(backend=BACKEND, timeout=nccl_timeout)
    else:
        logger.error(
            "PyTorch environment is not correctly initialized. This message should only be displayed when testing."
        )

    # 2. Resolve resume checkpoint
    resume_step = 0
    resume_lora_path = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if (resume_path / "checkpoints").exists():
            # It's a run directory — find latest checkpoint
            ckpts = sorted(
                d for d in (resume_path / "checkpoints").iterdir() if d.is_dir()
            )
            if not ckpts:
                raise RuntimeError(f"No checkpoints found in {resume_path / 'checkpoints'}")
            latest_ckpt = ckpts[-1]
        else:
            latest_ckpt = resume_path

        ckpt_name = latest_ckpt.name
        if ckpt_name.startswith("checkpoint_"):
            resume_step = int(ckpt_name.split("_")[1])

        consolidated = latest_ckpt / "consolidated"
        if not consolidated.exists():
            consolidated = latest_ckpt

        lora_file = consolidated / "lora.safetensors"
        if lora_file.exists():
            resume_lora_path = str(lora_file)
        else:
            raise RuntimeError(f"No lora.safetensors found at {lora_file}")

        main_logger_info(f"Resuming from step {resume_step}, loading LoRA from {resume_lora_path}")

    # 2b. Init run dir
    main_logger_info(f"Run dir: {args.run_dir}")
    run_dir = Path(args.run_dir)

    if is_torchrun() and not args.resume_from:
        if run_dir.exists() and not args.overwrite_run_dir:
            raise RuntimeError(
                f"Run dir {run_dir} already exists. Make sure to either rename `run_dir` or remove {run_dir}."
            )
        elif run_dir.exists():
            main_logger_info(f"Removing run dir {run_dir}...")
            shutil.rmtree(run_dir)

    if args.full_finetuning:
        assert not args.lora.enable, "LoRA should not be enabled for full finetuning."
    else:
        assert args.lora.enable, "LoRA should be enabled for partial finetuning"

    dist.barrier()
    run_dir.mkdir(exist_ok=True, parents=True)

    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        args.save(args_path)

    main_logger_info(f"TrainArgs: {pprint.pformat(dataclasses.asdict(args))}")

    # 3. Get loggers
    resume_run_dir = Path(args.resume_from) if args.resume_from else None
    metrics_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="train",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
        resume_run_dir=resume_run_dir,
    )
    exit_stack.enter_context(logged_closing(metrics_logger, "metrics_logger"))

    eval_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="eval",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
        resume_run_dir=resume_run_dir,
    )
    exit_stack.enter_context(logged_closing(eval_logger, "eval_logger"))

    # 4.1 Load function calling audio encoder and tokenizer
    main_logger_info("Loading Mimi and Moshi...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo=args.moshi_paths.hf_repo_id,
        moshi_weights=args.moshi_paths.moshi_path,
        mimi_weights=args.moshi_paths.mimi_path,
        tokenizer=args.moshi_paths.tokenizer_path,
        config_path=args.moshi_paths.config_path,
    )

    # Use the fully-merged lm_config from the checkpoint info (defaults +
    # repo overrides + model-type specific settings like dep_q=16 for PersonaPlex).
    # If the checkpoint config is sparse (e.g. PersonaPlex has no dep_q/dim), fall back to defaults.
    if checkpoint_info.lm_config and "dep_q" in checkpoint_info.lm_config:
        lm_config = dict(checkpoint_info.lm_config)
    else:
        lm_config = dict(loaders._lm_kwargs)
        lm_config["dep_q"] = 16  # PersonaPlex: all 16 codebooks in depformer
    lm_config["lora"] = args.lora.enable
    lm_config["lora_rank"] = args.lora.rank
    lm_config["lora_scaling"] = args.lora.scaling

    mimi = checkpoint_info.get_mimi(device="cuda")
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    # 4.2 Load and shard model, prepare interleaver for audio/text tokens.
    model = get_fsdp_model(args, checkpoint_info, resume_lora_path=resume_lora_path)

    spm = checkpoint_info.get_text_tokenizer()

    interleaver = Interleaver(
        spm,
        mimi.frame_rate,
        model.text_padding_token_id,
        model.end_of_text_padding_id,
        model.zero_token_id,
        keep_main_only=True,
    )
    interleaved_tokenizer = InterleavedTokenizer(
        mimi,
        interleaver,
        duration_sec=args.duration_sec,
        system_prompt_enabled=args.system_prompt.enable,
        audio_silence_frames=args.system_prompt.audio_silence_frames,
        prompt_budget_frames=args.system_prompt.prompt_budget_frames if args.system_prompt.enable else 0,
    )
    logger.info(
        f"Tokenizer: duration_sec={args.duration_sec}, "
        f"chunk_step_sec={interleaved_tokenizer.chunk_step_sec:.1f}, "
        f"prompt_budget_frames={interleaved_tokenizer.prompt_budget_frames}"
    )

    # 5. Load data loaders
    data_loader = build_data_loader(
        instruct_tokenizer=interleaved_tokenizer,
        args=args.data,
        batch_size=args.batch_size,
        seed=args.seed,
        rank=get_rank(),  # DDP rank
        world_size=get_world_size(),  # DDP world_size
        is_eval=False,
    )

    def make_eval_data_loader():
        return build_data_loader(
            instruct_tokenizer=interleaved_tokenizer,
            args=args.data,
            batch_size=args.batch_size,
            seed=None,
            rank=get_rank(),  # DDP rank
            world_size=get_world_size(),  # DDP world_size
            is_eval=True,
        )

    # 6. Load model
    # Define mixed precision
    param_dtype = getattr(torch, args.param_dtype)
    optim_dtype = torch.float32

    assert args.lora is not None, "`args.lora` should be set to a valid value."

    # 7. Load optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.optim.lr,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=args.optim.weight_decay,
    )

    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.optim.lr,
        total_steps=args.max_steps,
        pct_start=args.optim.pct_start,
    )

    state = TrainState(args.max_steps)

    # 7b. Restore optimizer, scheduler, and training state from checkpoint
    if resume_step > 0:
        resume_ckpt_dir = Path(resume_lora_path).parent
        saved = Checkpointer.load_train_state(resume_ckpt_dir)

        if args.force_warm_restart:
            main_logger_info("force_warm_restart=True — ignoring saved optimizer/scheduler")
            saved = None

        if saved is not None:
            main_logger_info("Restoring optimizer, scheduler, and training state from checkpoint")
            # Restore optimizer
            if "optimizer" in saved:
                if get_world_size() > 1:
                    sharded = FullyShardedDataParallel.shard_full_optim_state_dict(
                        saved["optimizer"], model
                    )
                    optimizer.load_state_dict(sharded)
                else:
                    optimizer.load_state_dict(saved["optimizer"])
                _fix_optim_state_dtype(optimizer, model)

            # Restore scheduler
            if "scheduler" in saved:
                scheduler.load_state_dict(saved["scheduler"])
                # Sync optimizer param group LRs with restored scheduler
                for i, lr in enumerate(scheduler.get_last_lr()):
                    optimizer.param_groups[i]['lr'] = lr

            # Restore training state
            ts = saved.get("train_state", {})
            state.step = ts.get("step", resume_step)
            state.elapsed_time = ts.get("elapsed_time", 0.0)
            state.n_seen_tokens = ts.get("n_seen_tokens", 0)
        else:
            main_logger_info(
                "No train_state.pt found — warm restart "
                "(optimizer state lost, scheduler approximated)"
            )
            state.step = resume_step
            # Re-create scheduler positioned at resume step
            scheduler = lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.optim.lr,
                total_steps=args.max_steps,
                pct_start=args.optim.pct_start,
                last_epoch=resume_step - 1,
            )

    # 8. Initialize checkpointer
    if args.do_ckpt:
        checkpointer = Checkpointer(
            model=model,
            state=state,
            config=lm_config,
            run_dir=run_dir,
            optimizer=optimizer,
            scheduler=scheduler,
            num_ckpt_keep=args.num_ckpt_keep,
            full_finetuning=args.full_finetuning,
        )
    # 9. Prepare mixed precision
    prepare_mixed_precision(
        model.parameters(), param_dtype=param_dtype, optim_dtype=optim_dtype
    )

    # 11. Step-0 baseline eval (before any training)
    if state.step == 0 and not args.skip_zero_eval:
        # Validation loss
        if args.do_eval:
            main_logger_info("Step-0 baseline validation eval...")
            evaluate(model, make_eval_data_loader(), state, args)
            eval_logs = get_eval_logs(
                0, 0.0,
                state.this_eval_perplexity,
                state.this_eval_loss,
                text_eval_loss=state.this_text_loss,
                audio_eval_loss=state.this_audio_loss,
                real_token_eval_loss=state.this_real_token_eval_loss,
                pad_token_eval_loss=state.this_pad_token_eval_loss,
                eval_real_token_pct=state.this_eval_real_token_pct,
                eval_pred_pad_pct=state.this_eval_pred_pad_pct,
            )
            main_logger_info(eval_log_msg(eval_logs))
            eval_logger.log(eval_logs, step=0)

        # Gen eval (base model performance)
        if args.gen_eval.enable and args.do_ckpt:
            main_logger_info("Step-0 baseline gen eval...")
            checkpointer.save_checkpoint(
                save_only_lora=not args.full_finetuning and args.save_adapters,
                dtype=param_dtype,
            )
            ckpt_dir = run_dir / "checkpoints" / "checkpoint_000000" / "consolidated"

            model, optimizer, mimi, scheduler = _run_gen_eval_with_teardown(
                model, optimizer, mimi, ckpt_dir, run_dir, 0, args,
                checkpoint_info, param_dtype, optim_dtype, scheduler, state,
            )
            # Update all references to rebuilt objects
            interleaved_tokenizer.mimi = mimi
            if args.do_ckpt:
                checkpointer = Checkpointer(
                    model=model, state=state, config=lm_config,
                    run_dir=run_dir, optimizer=optimizer, scheduler=scheduler,
                    num_ckpt_keep=args.num_ckpt_keep, full_finetuning=args.full_finetuning,
                )

    # 12. NEFTune: patch forward_embeddings to add noise during training
    if args.neftune_alpha > 0:
        _lm = model.module if hasattr(model, "module") else model
        _orig_forward_embeddings = _lm.forward_embeddings

        def _neftune_forward_embeddings(input_: torch.Tensor):
            if _lm.training:
                # NEFTune: uniform noise scaled by alpha / sqrt(seq_len * embed_dim)
                dims = input_.shape[-1]
                seq_len = input_.shape[-2]
                mag = args.neftune_alpha / (dims * seq_len) ** 0.5
                input_ = input_ + torch.zeros_like(input_).uniform_(-mag, mag)
            return _orig_forward_embeddings(input_)

        _lm.forward_embeddings = _neftune_forward_embeddings
        main_logger_info(f"NEFTune enabled: alpha={args.neftune_alpha}")

    # 13. train!
    model.train()
    torch.cuda.empty_cache()

    while state.step < args.max_steps:
        state.start_step()
        is_last_step = state.step == args.max_steps

        optimizer.zero_grad()

        loss = torch.tensor([0.0], device="cuda")
        lora_l2_accum = 0.0
        text_loss_accum = 0.0
        audio_loss_accum = 0.0
        real_token_loss_accum = 0.0
        pad_token_loss_accum = 0.0
        real_token_count_accum = 0
        pad_token_count_accum = 0
        lora_l2_accum = 0.0
        n_batch_tokens: int = 0
        n_real_tokens: int = 0
        per_cb_accum: dict[str, float] = {}
        n_ctx_injected_frames: int = 0
        n_ctx_total_frames: int = 0
        n_ctx_samples: int = 0
        # Injection pipeline stats (aggregated across microbatches)
        inj_total: int = 0
        inj_in_window: int = 0
        inj_placed: int = 0
        inj_truncated: int = 0
        inj_drop_anchor: int = 0
        inj_drop_space: int = 0
        inj_drop_final: int = 0
        inj_tokens_placed: int = 0
        inj_tokens_requested: int = 0

        for i in range(args.num_microbatches):
            batch = next(data_loader)
            codes = batch.codes

            # Track context injection stats
            if batch.context_masks is not None:
                n_ctx_injected_frames += batch.context_masks.sum().item()
                n_ctx_total_frames += batch.context_masks.numel()
                n_ctx_samples += batch.context_masks.any(dim=-1).sum().item()
            else:
                n_ctx_total_frames += codes.shape[-1] * codes.shape[0]
            if batch.injection_stats is not None:
                s = batch.injection_stats
                inj_total += s.total
                inj_in_window += s.in_window
                inj_placed += s.placed
                inj_truncated += s.truncated
                inj_drop_anchor += s.drop_anchor_overlap
                inj_drop_space += s.drop_no_space
                inj_drop_final += s.drop_final_overlap
                inj_tokens_placed += s.tokens_placed
                inj_tokens_requested += s.tokens_requested

            condition_tensors = None
            if batch.condition_attributes is not None:
                condition_tensors = model.condition_provider.prepare(
                    batch.condition_attributes
                )

            # forward / backward
            output = model(codes=codes, condition_tensors=condition_tensors)
            _text_padding_ids = {
                model.text_padding_token_id,
                model.end_of_text_padding_id,
            }
            text_result = compute_loss_with_mask(
                output.text_logits,
                codes[:, : model.audio_offset],
                output.text_mask,
                mode="text",
                text_padding_weight=args.text_padding_weight,
                text_padding_ids=_text_padding_ids,
                prompt_lengths=batch.prompt_lengths,
                context_masks=batch.context_masks,
                return_text_diagnostics=True,
            )
            text_loss, text_diag = text_result
            audio_loss, per_cb = compute_loss_with_mask(
                output.logits,
                codes[:, model.audio_offset : model.audio_offset + model.dep_q],
                output.mask,
                mode="audio",
                first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
                prompt_lengths=batch.prompt_lengths,
                return_per_codebook=True,
            )
            for k, v in per_cb.items():
                key = f"audio_cb{k}_loss"
                per_cb_accum[key] = per_cb_accum.get(key, 0.0) + v

            if args.audio_loss_weight > 0:
                mb_loss = text_loss + args.audio_loss_weight * audio_loss
            else:
                mb_loss = text_loss

            # L2 regularization on lora_B weights only (pulls toward base model).
            # lora_A is Kaiming-initialized (non-zero); lora_B is zero-initialized.
            # Only lora_B growth represents drift from base model behavior.
            if args.lora_l2_weight > 0:
                l2_reg = sum(
                    p.pow(2).sum()
                    for n, p in model.named_parameters()
                    if p.requires_grad and "lora_B" in n
                )
                mb_loss = mb_loss + args.lora_l2_weight * l2_reg
                lora_l2_accum += (args.lora_l2_weight * l2_reg).detach().item()

            mb_loss.backward()

            loss += mb_loss.detach()
            text_loss_accum += text_loss.detach().item()
            audio_loss_accum += audio_loss.detach().item()
            real_token_loss_accum += text_diag["real_token_loss"]
            pad_token_loss_accum += text_diag["pad_token_loss"]
            real_token_count_accum += text_diag["real_token_count"]
            pad_token_count_accum += text_diag["pad_token_count"]

            # PAD prediction rate: what % of the model's greedy predictions are padding?
            with torch.no_grad():
                text_preds = output.text_logits.detach().argmax(dim=-1)  # [B, 1, T]
                n_pred_total = text_preds.numel()
                n_pred_pad = sum(
                    (text_preds == pid).sum().item() for pid in _text_padding_ids
                )

            n_batch_tokens += output.text_mask.numel() + output.mask.numel()
            n_real_tokens += (
                torch.sum(output.text_mask).item() + torch.sum(output.mask).item()
            )

            if i < args.num_microbatches - 1:
                # synchronize CUDA to re-run backward
                assert args.num_microbatches > 1  # should not happen
                torch.cuda.synchronize()

        if args.num_microbatches > 1:
            loss /= args.num_microbatches
            for p in model.parameters():
                if p.requires_grad:
                    assert p.grad is not None
                    p.grad.div_(args.num_microbatches)

        # Average per-codebook losses across microbatches
        for k in per_cb_accum:
            per_cb_accum[k] /= args.num_microbatches

        # upcast params for optimizer update
        upcast_mixed_precision(model.parameters(), optim_dtype=optim_dtype)

        # clip grad norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        # optimizer step
        optimizer.step()

        # downcast params for forward & backward
        downcast_mixed_precision(model.parameters(), param_dtype=param_dtype)

        last_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Host sync
        loss_item = loss.item()
        avg_loss = avg_aggregate(loss_item)

        if args.do_eval and (
            (args.eval_freq > 0 and state.step % args.eval_freq == 0) or is_last_step
        ):
            # write perplexity to state
            evaluate(model, make_eval_data_loader(), state, args)

            eval_logs = get_eval_logs(
                state.step,
                avg_loss,
                state.this_eval_perplexity,
                state.this_eval_loss,
                text_eval_loss=state.this_text_loss,
                audio_eval_loss=state.this_audio_loss,
                real_token_eval_loss=state.this_real_token_eval_loss,
                pad_token_eval_loss=state.this_pad_token_eval_loss,
                eval_real_token_pct=state.this_eval_real_token_pct,
                eval_pred_pad_pct=state.this_eval_pred_pad_pct,
            )

            main_logger_info(eval_log_msg(eval_logs))
            eval_logger.log(eval_logs, step=state.step)

        # Timing
        state.end_step(n_batch_tokens)

        if state.step % args.log_freq == 0:
            train_logs = get_train_logs(
                state,
                avg_loss,
                n_real_tokens,
                last_lr,
                torch.cuda.max_memory_allocated(),
                torch.cuda.memory_allocated(),
                args,
            )
            # Text/audio loss breakdown
            train_logs["text_loss"] = text_loss_accum / args.num_microbatches
            train_logs["audio_loss"] = audio_loss_accum / args.num_microbatches
            if lora_l2_accum > 0:
                train_logs["lora_l2_loss"] = lora_l2_accum / args.num_microbatches

            # Decomposed text loss diagnostics: real tokens vs padding tokens
            train_logs["real_token_loss"] = real_token_loss_accum / args.num_microbatches
            train_logs["pad_token_loss"] = pad_token_loss_accum / args.num_microbatches
            total_text_tokens = real_token_count_accum + pad_token_count_accum
            if total_text_tokens > 0:
                train_logs["real_token_pct"] = 100.0 * real_token_count_accum / total_text_tokens
            if n_pred_total > 0:
                train_logs["pred_pad_pct"] = 100.0 * n_pred_pad / n_pred_total
            if lora_l2_accum > 0:
                train_logs["lora_l2_loss"] = lora_l2_accum / args.num_microbatches

            # Context injection stats
            if n_ctx_total_frames > 0:
                train_logs["ctx_injected_pct"] = 100.0 * n_ctx_injected_frames / n_ctx_total_frames
                train_logs["ctx_samples"] = n_ctx_samples
            if inj_total > 0:
                train_logs["inj_total"] = inj_total
                train_logs["inj_in_window"] = inj_in_window
                train_logs["inj_placed"] = inj_placed
                train_logs["inj_truncated"] = inj_truncated
                train_logs["inj_drop_anchor"] = inj_drop_anchor
                train_logs["inj_drop_space"] = inj_drop_space
                train_logs["inj_drop_final"] = inj_drop_final
                if inj_in_window > 0:
                    train_logs["inj_place_rate"] = 100.0 * inj_placed / inj_in_window
                if inj_tokens_requested > 0:
                    train_logs["inj_token_yield"] = 100.0 * inj_tokens_placed / inj_tokens_requested

            # Add per-codebook audio losses for diagnostics
            train_logs.update(per_cb_accum)
            main_logger_info(train_log_msg(state, logs=train_logs, loss=avg_loss))
            metrics_logger.log(train_logs, step=state.step)

        if args.do_ckpt and (
            (args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0) or is_last_step
        ):
            checkpointer.save_checkpoint(
                save_only_lora=not args.full_finetuning and args.save_adapters,
                dtype=param_dtype,
            )

            # --- Gen eval: offload training state, run eval, restore ---
            should_run_gen_eval = args.gen_eval.enable and (
                (args.gen_eval.freq > 0 and state.step % args.gen_eval.freq == 0)
                or is_last_step
            )

            if should_run_gen_eval:
                ckpt_dir = (
                    run_dir / "checkpoints" / f"checkpoint_{state.step:06d}" / "consolidated"
                )

                main_logger_info(
                    f"Gen eval at step {state.step}: nuking CUDA state for gen eval..."
                )
                model, optimizer, mimi, scheduler = _run_gen_eval_with_teardown(
                    model, optimizer, mimi, ckpt_dir, run_dir, state.step, args,
                    checkpoint_info, param_dtype, optim_dtype, scheduler, state,
                )
                # Update all references to rebuilt objects
                interleaved_tokenizer.mimi = mimi
                checkpointer = Checkpointer(
                    model=model, state=state, config=lm_config,
                    run_dir=run_dir, optimizer=optimizer, scheduler=scheduler,
                    num_ckpt_keep=args.num_ckpt_keep, full_finetuning=args.full_finetuning,
                )
                model.train()

    main_logger_info("done!")


if __name__ == "__main__":
    """See README.md for usage."""
    fire.Fire(train)
