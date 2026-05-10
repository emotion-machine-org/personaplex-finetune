import torch
from torch.nn import functional as F


def compute_loss_with_mask(
    logits: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    mode: str,
    first_codebook_weight_multiplier: float = 1.0,
    text_padding_weight: float = 1.0,
    text_padding_ids: set[int] | None = None,
    prompt_lengths: list[int] | None = None,
    context_masks: torch.Tensor | None = None,
    return_per_codebook: bool = False,
    n_codebooks_per_stream: int = 8,
    return_text_diagnostics: bool = False,
):
    target = torch.where(target_mask, target, torch.zeros_like(target))

    weights = target_mask.float()
    if mode == "audio":
        # Upweight the first (semantic) codebook of each audio stream.
        # With dep_q=16 (PersonaPlex), streams start at indices 0 and 8.
        K = weights.shape[1]
        for stream_start in range(0, K, n_codebooks_per_stream):
            weights[:, stream_start] *= first_codebook_weight_multiplier
    elif mode == "text":
        assert text_padding_ids is not None
        # Build a mask identifying padding positions BEFORE applying weight
        is_padding = torch.zeros_like(weights, dtype=torch.bool)
        for id in text_padding_ids:
            is_padding |= (target == id)
        for id in text_padding_ids:
            weights[target == id] *= text_padding_weight

    # Mask out system prompt frames (PersonaPlex: no loss on prompt region)
    if prompt_lengths is not None:
        for b, pl in enumerate(prompt_lengths):
            if pl > 0:
                weights[b, :, :pl] = 0.0

    # Mask out context injection frames for text loss only.
    # Audio loss stays active — model learns silence/filler during injections.
    if context_masks is not None and mode == "text":
        for b in range(context_masks.shape[0]):
            weights[b, :, context_masks[b]] = 0.0

    logits = logits.view(-1, logits.size(-1)).float()
    target = target.view(-1)
    weights = weights.view(-1)
    ce = F.cross_entropy(logits, target, reduction="none")
    mb_loss = torch.where(weights > 0.0, ce * weights, torch.zeros_like(ce))
    mb_loss = torch.sum(mb_loss) / torch.sum(weights).clamp(min=1)

    # Text diagnostics: decomposed loss on real vs padding tokens
    if return_text_diagnostics and mode == "text":
        is_padding_flat = is_padding.view(-1)
        valid_flat = (target_mask.float().view(-1) > 0)
        # Apply prompt/context masking to valid positions
        if prompt_lengths is not None:
            prompt_mask = target_mask.float().clone()
            for b, pl in enumerate(prompt_lengths):
                if pl > 0:
                    prompt_mask[b, :, :pl] = 0.0
            if context_masks is not None:
                for b in range(context_masks.shape[0]):
                    prompt_mask[b, :, context_masks[b]] = 0.0
            valid_flat = (prompt_mask.view(-1) > 0)
        elif context_masks is not None:
            ctx_mask = target_mask.float().clone()
            for b in range(context_masks.shape[0]):
                ctx_mask[b, :, context_masks[b]] = 0.0
            valid_flat = (ctx_mask.view(-1) > 0)

        real_mask = valid_flat & ~is_padding_flat
        pad_mask = valid_flat & is_padding_flat

        real_denom = real_mask.float().sum().clamp(min=1)
        pad_denom = pad_mask.float().sum().clamp(min=1)

        real_token_loss = (ce * real_mask.float()).sum() / real_denom
        pad_token_loss = (ce * pad_mask.float()).sum() / pad_denom

        diag = {
            "real_token_loss": real_token_loss.item(),
            "pad_token_loss": pad_token_loss.item(),
            "real_token_count": int(real_mask.float().sum().item()),
            "pad_token_count": int(pad_mask.float().sum().item()),
        }
        return mb_loss, diag

    if return_per_codebook and mode == "audio":
        B, K, T = target_mask.shape
        ce_3d = ce.detach().view(B, K, T)
        # Use unweighted mask (no first_codebook_multiplier) so per-cb losses
        # are directly comparable across codebooks.
        mask_3d = target_mask.float()
        if prompt_lengths is not None:
            for b, pl in enumerate(prompt_lengths):
                if pl > 0:
                    mask_3d[b, :, :pl] = 0.0
        per_cb = {}
        for k in range(K):
            denom = mask_3d[:, k].sum()
            if denom > 0:
                # Use torch.where to avoid NaN * 0 = NaN from delayed positions
                # whose logits are filled with NaN by _undelay_sequence.
                masked_ce = torch.where(mask_3d[:, k] > 0, ce_3d[:, k], torch.zeros_like(ce_3d[:, k]))
                per_cb[k] = masked_ce.sum().item() / denom.item()
            else:
                per_cb[k] = 0.0
        return mb_loss, per_cb

    return mb_loss
