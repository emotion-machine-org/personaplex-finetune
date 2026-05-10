import logging
from typing import Iterator

import torch
import torch.cuda
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from finetune.args import TrainArgs

from .data.data_loader import Batch
from .distributed import get_rank, get_world_size
from .loss import compute_loss_with_mask
from .utils import TrainState

logger = logging.getLogger("eval")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def evaluate(
    model: FullyShardedDataParallel,
    eval_data_loader: Iterator[Batch],
    state: TrainState,
    args: TrainArgs,
):
    num_samples = torch.tensor([0], device="cuda", dtype=torch.long)

    text_loss = torch.tensor(0.0).cuda()
    audio_loss = torch.tensor(0.0).cuda()
    real_token_loss_sum = 0.0
    pad_token_loss_sum = 0.0
    real_token_count_sum = 0
    pad_token_count_sum = 0
    pred_pad_total = 0
    pred_pad_count = 0

    _text_padding_ids = {
        model.text_padding_token_id,
        model.end_of_text_padding_id,
    }

    model.eval()
    for batch in eval_data_loader:
        num_samples += 1
        if num_samples > 40 // get_world_size():
            break
        with torch.no_grad():
            codes = batch.codes
            condition_tensors = None
            if batch.condition_attributes is not None:
                condition_tensors = model.condition_provider.prepare(
                    batch.condition_attributes
                )

            output = model(codes=codes, condition_tensors=condition_tensors)
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
            batch_text_loss, text_diag = text_result
            text_loss += batch_text_loss
            real_token_loss_sum += text_diag["real_token_loss"]
            pad_token_loss_sum += text_diag["pad_token_loss"]
            real_token_count_sum += text_diag["real_token_count"]
            pad_token_count_sum += text_diag["pad_token_count"]

            # PAD prediction rate
            text_preds = output.text_logits.argmax(dim=-1)
            pred_pad_total += text_preds.numel()
            pred_pad_count += sum(
                (text_preds == pid).sum().item() for pid in _text_padding_ids
            )

            audio_loss += compute_loss_with_mask(
                output.logits,
                codes[:, model.audio_offset : model.audio_offset + model.dep_q],
                output.mask,
                mode="audio",
                first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
                prompt_lengths=batch.prompt_lengths,
            )
    eval_loss = text_loss + audio_loss
    all_num_samples = [torch.zeros_like(num_samples) for _ in range(get_world_size())]

    torch.distributed.all_gather(all_num_samples, num_samples)

    total_num_samples = int(torch.tensor(all_num_samples).sum().item())
    # sum loss
    main_logger_info("Eval finished!")

    dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(text_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(audio_loss, op=dist.ReduceOp.SUM)
    text_loss /= total_num_samples
    audio_loss /= total_num_samples
    eval_loss /= total_num_samples

    n_eval = max(1, int(num_samples.item()))
    state.this_eval_loss = eval_loss.item()
    state.this_eval_perplexity = (2**eval_loss).item()
    state.this_audio_loss = audio_loss.item()
    state.this_text_loss = text_loss.item()
    state.this_real_token_eval_loss = real_token_loss_sum / n_eval
    state.this_pad_token_eval_loss = pad_token_loss_sum / n_eval
    total_eval_text_tokens = real_token_count_sum + pad_token_count_sum
    state.this_eval_real_token_pct = (
        100.0 * real_token_count_sum / total_eval_text_tokens if total_eval_text_tokens > 0 else 0.0
    )
    state.this_eval_pred_pad_pct = (
        100.0 * pred_pad_count / pred_pad_total if pred_pad_total > 0 else 0.0
    )

    # train mode!
    model.train()
