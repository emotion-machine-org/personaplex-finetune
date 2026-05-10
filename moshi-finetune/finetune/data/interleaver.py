import json
import logging
import math
import os
from collections import deque
from dataclasses import dataclass, field
from functools import reduce

logger = logging.getLogger("interleaver")

import numpy as np
import sentencepiece
import sphn
import torch
from moshi.conditioners import ConditionAttributes

Alignment = tuple[str, tuple[float, float], str]
TokenizedAlignment = tuple[list[int], tuple[float, float], str]

# Pre-computed Mimi tokens for silence (agent not speaking) and 440 Hz sine wave (user placeholder).
# These match the PersonaPlex constants used during hybrid system-prompt injection.
SILENCE_TOKENS = np.array([948, 243, 1178, 546, 1736, 1030, 1978, 2008], dtype=np.int64)
SINE_TOKENS = np.array([430, 1268, 381, 1611, 1095, 1495, 56, 472], dtype=np.int64)


@dataclass
class InjectionStats:
    """Per-sample context injection diagnostics."""
    total: int = 0          # injections in the JSON
    in_window: int = 0      # fell within the chunk window
    placed: int = 0         # successfully spliced
    truncated: int = 0      # placed but truncated to fit
    drop_no_offset: int = 0       # missing frame_offset
    drop_out_of_window: int = 0   # frame_offset outside chunk
    drop_anchor_overlap: int = 0  # anchor + neighbors all have broker speech
    drop_no_space: int = 0        # available run ≤ 0
    drop_final_overlap: int = 0   # final placement check found broker tokens
    tokens_placed: int = 0        # total tokens actually spliced
    tokens_requested: int = 0     # total tokens before truncation


@dataclass
class Sample:
    codes: torch.Tensor
    condition_attributes: ConditionAttributes | None = None
    prompt_length: int = 0  # number of time-frames that are system prompt (for loss masking)
    context_mask: torch.Tensor | None = None  # [T] boolean, True = injection frame
    injection_stats: InjectionStats | None = None


@dataclass
class Batch:
    codes: torch.Tensor
    condition_attributes: list[ConditionAttributes] | None = None
    prompt_lengths: list[int] | None = None
    context_masks: torch.Tensor | None = None  # [B, T] boolean
    injection_stats: InjectionStats | None = None  # aggregated across batch

    @classmethod
    def collate(cls, batch: list[Sample]) -> "Batch":
        codes = torch.cat([b.codes for b in batch])
        condition_attributes = None
        if batch[0].condition_attributes is not None:
            condition_attributes = [b.condition_attributes for b in batch]
        prompt_lengths = None
        if any(b.prompt_length > 0 for b in batch):
            prompt_lengths = [b.prompt_length for b in batch]
        context_masks = None
        if any(b.context_mask is not None for b in batch):
            T = codes.shape[-1]
            masks = []
            for b in batch:
                if b.context_mask is not None:
                    m = b.context_mask[:T]
                    if m.shape[0] < T:
                        m = torch.nn.functional.pad(m, (0, T - m.shape[0]), value=False)
                    masks.append(m)
                else:
                    masks.append(torch.zeros(T, dtype=torch.bool, device=codes.device))
            context_masks = torch.stack(masks)
        # Aggregate injection stats
        injection_stats = None
        sample_stats = [b.injection_stats for b in batch if b.injection_stats is not None]
        if sample_stats:
            injection_stats = InjectionStats()
            for s in sample_stats:
                injection_stats.total += s.total
                injection_stats.in_window += s.in_window
                injection_stats.placed += s.placed
                injection_stats.truncated += s.truncated
                injection_stats.drop_no_offset += s.drop_no_offset
                injection_stats.drop_out_of_window += s.drop_out_of_window
                injection_stats.drop_anchor_overlap += s.drop_anchor_overlap
                injection_stats.drop_no_space += s.drop_no_space
                injection_stats.drop_final_overlap += s.drop_final_overlap
                injection_stats.tokens_placed += s.tokens_placed
                injection_stats.tokens_requested += s.tokens_requested
        return Batch(codes, condition_attributes, prompt_lengths, context_masks, injection_stats)


def tokenize(
    tokenizer: sentencepiece.SentencePieceProcessor,
    text: str,
    bos: bool = True,
    alpha: float | None = None,
):
    """Tokenize the given string, accounting for new lines, potentially adding a BOS token."""
    nl_piece = tokenizer.encode("\n")[-1]
    if alpha is not None:
        tokens = tokenizer.encode(
            text.split("\n"), enable_sampling=True, alpha=alpha, nbest_size=-1
        )
    else:
        tokens = tokenizer.encode(text.split("\n"))
    tokens = reduce(lambda a, b: [*a, nl_piece, *b], tokens)
    if bos:
        tokens = [tokenizer.bos_id(), *tokens]
    return tokens


def wrap_with_system_tags(text: str) -> str:
    """Wrap text in <system> tags as PersonaPlex expects."""
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


class Interleaver:
    """Interleaver with basic featuress
    Args:
        tokenizer: text tokenizer used by the model.
        audio_frame_rate (float): frame rate of the audio tokenizer.
        text_padding (int): special token used for text padding.
        end_of_text_padding (int): special token used to indicate end of text padding.
        zero_padding (int): special token id indicating that a 0 should be used instead
            of an actual embedding.
        in_word_padding (int | None): padding used within a word segment. Will default to `text_padding`.
        keep_main_only (bool): if True, will only keep the alignments with the main speaker.
        keep_and_shift (bool): if True, will not drop any alignment, except for those with negative duration.
        use_bos_eos: (bool): if True, inserts BOS, EOS for change of turns.
        audio_delay (float): delay between the text and audio.
            A positive value means the text will be ahead of the audio.
        proba (float): probability of keeping the text.
        device: device location for the output tensors.
    """

    def __init__(
        self,
        tokenizer: sentencepiece.SentencePieceProcessor,
        audio_frame_rate: float,
        text_padding: int,
        end_of_text_padding: int,
        zero_padding: int,
        in_word_padding: int | None = None,
        keep_main_only: bool = False,
        main_speaker_label: str = "SPEAKER_BROKER",
        use_bos_eos: bool = False,
        keep_and_shift: bool = False,
        audio_delay: float = 0.0,
        proba: float = 1.0,
        device: str | torch.device = "cuda",
    ):
        self.tokenizer = tokenizer
        self.audio_frame_rate = audio_frame_rate
        self.text_padding = text_padding
        self.end_of_text_padding = end_of_text_padding
        self.zero_padding = zero_padding
        self.in_word_padding = (
            self.text_padding if in_word_padding is None else in_word_padding
        )
        self.keep_main_only = keep_main_only
        self.main_speaker_label = main_speaker_label
        self.use_bos_eos = use_bos_eos
        self.keep_and_shift = keep_and_shift
        self.audio_delay = audio_delay
        self.proba = proba
        self.device = device

    @property
    def special_tokens(self) -> set[int]:
        """Return the set of special tokens used by this interleaver."""
        return {
            self.text_padding,
            self.end_of_text_padding,
            self.tokenizer.bos_id(),
            self.tokenizer.eos_id(),
            self.zero_padding,
            self.in_word_padding,
        }

    def _tokenize(self, alignments: list[Alignment]) -> list[TokenizedAlignment]:
        # Tokenizes each word individually into a list of ints.
        out = []
        for word, ts, speaker in alignments:
            toks = tokenize(self.tokenizer, word.strip(), bos=False)
            out.append((toks, ts, speaker))
        return out

    def _keep_main_only(
        self, alignments: list[TokenizedAlignment], main_speaker: str
    ) -> list[TokenizedAlignment]:
        return [a for a in alignments if a[2] == main_speaker]

    def _keep_those_with_duration(
        self, alignments: list[TokenizedAlignment]
    ) -> list[TokenizedAlignment]:
        # Removes all words with negative or 0 durations.
        return [a for a in alignments if a[1][0] < a[1][1]]

    def _add_delay(
        self, alignments: list[TokenizedAlignment]
    ) -> list[TokenizedAlignment]:
        # Delay the audio with respect to the text, e.g. positive values mean the audio is late on the text.
        return [
            (a[0], (a[1][0] - self.audio_delay, a[1][1] - self.audio_delay), a[2])
            for a in alignments
            if a[1][1] > self.audio_delay
        ]

    def _insert_bos_eos(
        self, alignments: list[TokenizedAlignment], main_speaker: str
    ) -> list[TokenizedAlignment]:
        # EOS and BOS is different from what it was in the old Interleaver, it is now symmetrical:
        # if the main speaker talks after another speaker (or is the first to talk), BOS is prepended to the first word.
        # Similary, if any other speaker speaks either first, or after the main speaker, a EOS is prepended.
        # This is in contrast with the legacy Interleaver, where the EOS would be inserted immediately
        # at the end of the turn of the main speaker.
        out: list[TokenizedAlignment] = []
        last_speaker = None
        for toks, ts, speaker in alignments:
            toks = list(toks)
            if speaker == last_speaker:
                pass
            elif speaker == main_speaker:
                toks.insert(0, self.tokenizer.bos_id())
            elif last_speaker == main_speaker:
                assert out
                toks.insert(0, self.tokenizer.eos_id())
            last_speaker = speaker
            out.append((toks, ts, speaker))
        return out

    def build_token_stream(
        self,
        alignments: list[TokenizedAlignment] | None,
        segment_duration: float,
    ) -> torch.Tensor:
        """Builds the token stream from the tokenized alignments."""
        T = math.ceil(segment_duration * self.audio_frame_rate)
        if alignments is None:
            text_tokens = [self.zero_padding] * T
        else:
            text_tokens = [self.text_padding] * T
            i = 0
            to_append_stack: deque = deque()
            last_word_end = -1
            for t in range(T):
                while (
                    i < len(alignments)
                    and alignments[i][1][0] * self.audio_frame_rate < t + 1
                ):
                    tokenized = alignments[i][0]
                    last_word_end = int(alignments[i][1][1] * self.audio_frame_rate)
                    if self.keep_and_shift:
                        to_append_stack.extend(tokenized)
                    else:
                        to_append_stack = deque(tokenized)
                    i += 1
                if to_append_stack:
                    if t > 0 and text_tokens[t - 1] in [
                        self.text_padding,
                        self.in_word_padding,
                    ]:
                        text_tokens[t - 1] = self.end_of_text_padding
                    next_token = to_append_stack.popleft()
                    text_tokens[t] = next_token
                elif t <= last_word_end:
                    text_tokens[t] = self.in_word_padding
        if self.audio_delay < 0:
            prefix_length = int(self.audio_frame_rate * -self.audio_delay)
            text_tokens[:prefix_length] = [self.zero_padding] * prefix_length
        return torch.tensor(text_tokens, device=self.device).view(1, 1, -1)

    def prepare_item(
        self,
        alignments: list[Alignment] | None,
        segment_duration: float,
        main_speaker: str | None = None,
    ) -> torch.Tensor:
        """Responsible with processing the alignments and calling `build_token_stream`."""
        if alignments is None:
            tokenized = None
        else:
            tokenized = self._tokenize(sorted(alignments, key=lambda x: x[1][0]))
            if self.keep_main_only:
                main_speaker = main_speaker or self.main_speaker_label
                tokenized = self._keep_main_only(tokenized, main_speaker)
            elif self.use_bos_eos:
                main_speaker = main_speaker or self.main_speaker_label
                tokenized = self._insert_bos_eos(tokenized, main_speaker)
            tokenized = self._keep_those_with_duration(tokenized)
            if self.audio_delay != 0:
                tokenized = self._add_delay(tokenized)
        return self.build_token_stream(tokenized, segment_duration)


def dicho(alignment, val, i=0, j=None):
    if j is None:
        j = len(alignment)
    if i == j:
        return i
    k = (i + j) // 2
    if alignment[k][1][0] < val:
        return dicho(alignment, val, k + 1, j)
    else:
        return dicho(alignment, val, i, k)


class InterleavedTokenizer:
    def __init__(
        self,
        mimi,
        interleaver,
        duration_sec: float,
        system_prompt_enabled: bool = False,
        audio_silence_frames: int = 6,
        prompt_budget_frames: int = 0,
    ):
        self.mimi = mimi
        self.interleaver = interleaver
        self.duration_sec = duration_sec
        self.num_audio_frames = math.ceil(duration_sec * mimi.frame_rate)
        self.system_prompt_enabled = system_prompt_enabled
        self.audio_silence_frames = audio_silence_frames
        # When system_prompt_enabled, the prompt steals frames from each chunk's
        # conversation window.  If chunks step by the full duration_sec, injections
        # near each chunk boundary fall into the "prompt gap" and are dropped.
        # prompt_budget_frames should be >= the longest prompt in the dataset;
        # the chunking step is reduced by this amount so that conversation windows
        # tile without gaps.
        self.prompt_budget_frames = prompt_budget_frames
        self.chunk_step_sec = (self.num_audio_frames - prompt_budget_frames) / mimi.frame_rate
        # Cache for encoded voice prompts: path -> [1, K, T]
        self._voice_prompt_cache: dict[str, torch.Tensor] = {}

    def _encode_voice_prompt(self, voice_prompt_path: str, base_path: str) -> torch.Tensor:
        """Load and encode a voice prompt WAV through Mimi. Returns [1, K, T_frames]."""
        if not os.path.isabs(voice_prompt_path):
            voice_prompt_path = os.path.join(os.path.dirname(base_path), voice_prompt_path)

        if voice_prompt_path in self._voice_prompt_cache:
            return self._voice_prompt_cache[voice_prompt_path]

        sample_pcm, sample_sr = sphn.read(voice_prompt_path)
        if sample_sr != self.mimi.sample_rate:
            sample_pcm = sphn.resample(
                sample_pcm, src_sample_rate=sample_sr, dst_sample_rate=self.mimi.sample_rate
            )
        # Ensure mono
        if sample_pcm.ndim == 2 and sample_pcm.shape[0] > 1:
            sample_pcm = sample_pcm[:1]
        elif sample_pcm.ndim == 1:
            sample_pcm = sample_pcm[None, :]

        voice_tensor = torch.tensor(sample_pcm, dtype=torch.float32, device="cuda").unsqueeze(0)  # [1, 1, T]
        voice_codes = self.mimi.encode(voice_tensor)  # [1, K, T_frames]

        self._voice_prompt_cache[voice_prompt_path] = voice_codes
        return voice_codes

    def _build_system_prompt_prefix(
        self, text_prompt: str, voice_prompt_path: str | None, base_path: str
    ) -> tuple[torch.Tensor, int]:
        """Build a PersonaPlex-style system prompt prefix.

        Layout: [voice frames] [silence] [text prompt frames] [silence]

        Returns:
            prompt_codes: [1, 1+2K, T_prompt] tensor of token codes
            prompt_length: number of time frames in the prompt
        """
        K = 8  # audio codebooks per channel
        device = "cuda"
        sil = self.audio_silence_frames

        silence = torch.tensor(SILENCE_TOKENS, device=device, dtype=torch.long)  # [K]
        sine = torch.tensor(SINE_TOKENS, device=device, dtype=torch.long)  # [K]

        # Voice prompt
        voice_codes = None
        voice_frames = 0
        if voice_prompt_path:
            voice_codes = self._encode_voice_prompt(voice_prompt_path, base_path)  # [1, K, T_voice]
            voice_frames = voice_codes.shape[-1]

        # Text prompt
        wrapped = wrap_with_system_tags(text_prompt)
        prompt_tokens = tokenize(self.interleaver.tokenizer, wrapped, bos=False)
        text_frames = len(prompt_tokens)

        total = voice_frames + sil + text_frames + sil

        # Text channel: default to text_padding
        text_ch = torch.full((1, 1, total), self.interleaver.text_padding, device=device, dtype=torch.long)

        # Agent audio: default to SILENCE_TOKENS
        agent_ch = silence.view(1, K, 1).repeat(1, 1, total)

        # User audio: always SINE_TOKENS
        user_ch = sine.view(1, K, 1).repeat(1, 1, total)

        # Fill in voice prompt agent audio
        if voice_codes is not None:
            agent_ch[:, :, :voice_frames] = voice_codes

        # Fill in text prompt tokens (after voice + first silence)
        text_start = voice_frames + sil
        text_ch[0, 0, text_start:text_start + text_frames] = torch.tensor(
            prompt_tokens, device=device, dtype=torch.long
        )

        # Combine: [text, agent_audio, user_audio]
        prompt_codes = torch.cat([text_ch, agent_ch, user_ch], dim=1)  # [1, 1+2K, T_prompt]
        return prompt_codes, total

    def __call__(self, wav: np.ndarray, start_sec: float, path: str) -> Sample:
        with torch.no_grad():
            info_file = os.path.splitext(path)[0] + ".json"
            with open(info_file) as f:
                data = json.load(f)

            # Build system prompt prefix if enabled
            prompt_length = 0
            prompt_codes = None
            if self.system_prompt_enabled:
                text_prompt = data.get("text_prompt")
                if text_prompt is not None:
                    voice_prompt = data.get("voice_prompt")
                    prompt_codes, prompt_length = self._build_system_prompt_prefix(
                        text_prompt, voice_prompt, path
                    )
                    # Clamp prompt to not exceed total frames
                    if prompt_length >= self.num_audio_frames:
                        prompt_codes = prompt_codes[..., : self.num_audio_frames - 1]
                        prompt_length = self.num_audio_frames - 1

            conv_frames = self.num_audio_frames - prompt_length

            # Encode conversation audio
            audio_tensor = torch.Tensor(wav).cuda()
            audio_tokens = self.mimi.encode(audio_tensor[:, None])
            audio_tokens = audio_tokens[..., :conv_frames]
            this_num_audio_frames = audio_tokens.shape[-1]
            audio_tokens = torch.nn.functional.pad(
                audio_tokens,
                (0, conv_frames - this_num_audio_frames),
                value=self.interleaver.zero_padding,
            )
            audio_tokens = audio_tokens.view(1, -1, conv_frames)

            # Build text tokens for conversation
            alignments = data["alignments"]
            start_alignment = dicho(alignments, start_sec)
            end_alignment = dicho(alignments, start_sec + self.chunk_step_sec)
            alignments = [
                (a[0], (a[1][0] - start_sec, a[1][1] - start_sec), a[2])
                for a in alignments[start_alignment:end_alignment]
            ]

            text_tokens = self.interleaver.prepare_item(
                alignments, this_num_audio_frames
            )
            text_tokens = torch.nn.functional.pad(
                text_tokens,
                (0, conv_frames - text_tokens.shape[-1]),
                value=self.interleaver.zero_padding,
            )

            conv_codes = torch.cat([text_tokens, audio_tokens], dim=1)

            # ── Context injection splicing (v2: insert-silence) ──
            #
            # V1 (commented below) tried to fit injection tokens into EXISTING
            # silent gaps in the text channel. This caused ~40% of in-window
            # injections to be truncated (inj_token_yield ~56%) because the
            # broker's speech often resumed before all ~40 tokens could fit.
            #
            # V2 (active) INSERTS N silence frames at the anchor point, shifting
            # all subsequent content right. This guarantees full injection
            # placement (100% yield) and teaches the model the correct behavior:
            # pause and absorb context before speaking. Trade-off: each injection
            # pushes ~3.2s of conversation off the tail of the chunk (~4% of 80s).
            #
            # To revert to v1: uncomment the "V1 SPLICE" block below, comment
            # out the "V2 INSERT-SILENCE" block.
            context_injections = data.get("context_injections")
            context_mask = None
            inj_stats = None
            if context_injections:
                inj_stats = InjectionStats(total=len(context_injections))
                frame_rate = self.mimi.frame_rate
                chunk_start_frame = int(start_sec * frame_rate)

                # Build silence frame for insertion: text=PAD, agent audio=SILENCE, user audio=SILENCE
                silence_text = self.interleaver.text_padding
                silence_agent = torch.tensor(SILENCE_TOKENS, device="cuda", dtype=torch.long)  # [8]
                silence_user = torch.tensor(SILENCE_TOKENS, device="cuda", dtype=torch.long)   # [8]

                # Collect (local_offset, tokens) pairs, sorted by offset so
                # cumulative shift is tracked correctly
                placements = []
                for inj in context_injections:
                    fo = inj.get("frame_offset")
                    if fo is None:
                        inj_stats.drop_no_offset += 1
                        continue
                    local_offset = fo - chunk_start_frame
                    if local_offset < 0 or local_offset >= conv_frames:
                        inj_stats.drop_out_of_window += 1
                        continue

                    inj_stats.in_window += 1
                    wrapped = f"<context> {inj['text'].strip()} </context>"
                    tokens = tokenize(self.interleaver.tokenizer, wrapped, bos=False)
                    inj_stats.tokens_requested += len(tokens)
                    placements.append((local_offset, tokens))

                placements.sort(key=lambda x: x[0])

                # Insert silence frames and place injection tokens
                # Process in order so cumulative_shift tracks correctly
                cumulative_shift = 0
                context_mask_list = []  # list of (start, end) ranges in the expanded tensor

                for local_offset, tokens in placements:
                    N = len(tokens)
                    insert_at = local_offset + cumulative_shift

                    # Build insertion block: [1, 17, N]
                    insert_block = torch.zeros(1, conv_codes.shape[1], N, device="cuda", dtype=torch.long)
                    # Text channel (0): injection tokens
                    insert_block[0, 0, :] = torch.tensor(tokens, device="cuda", dtype=torch.long)
                    # Agent audio channels (1-8): silence (broker is quiet during injection)
                    for k in range(8):
                        insert_block[0, 1 + k, :] = silence_agent[k]
                    # User audio channels (9-16): keep original conversation audio.
                    # ~70% of injections are proactive (during client speech), so the
                    # model must learn to read context while the user is talking.
                    # Copy user audio from the frames being displaced by this insertion.
                    avail = conv_codes.shape[-1] - insert_at
                    copy_n = min(N, avail)
                    if copy_n > 0:
                        insert_block[0, 9:17, :copy_n] = conv_codes[0, 9:17, insert_at:insert_at + copy_n]
                    # If injection is longer than remaining frames, pad tail with silence
                    if copy_n < N:
                        for k in range(8):
                            insert_block[0, 9 + k, copy_n:] = silence_user[k]

                    # Insert into conv_codes
                    left = conv_codes[:, :, :insert_at]
                    right = conv_codes[:, :, insert_at:]
                    conv_codes = torch.cat([left, insert_block, right], dim=-1)

                    context_mask_list.append((insert_at, insert_at + N))
                    cumulative_shift += N
                    inj_stats.placed += 1
                    inj_stats.tokens_placed += N

                # Trim back to conv_frames (lose content from the tail)
                conv_codes = conv_codes[:, :, :conv_frames]

                # Build context mask
                context_mask = torch.zeros(conv_frames, dtype=torch.bool, device="cuda")
                for start, end in context_mask_list:
                    # Clamp to conv_frames in case insertion pushed past the boundary
                    clamped_end = min(end, conv_frames)
                    if start < conv_frames:
                        context_mask[start:clamped_end] = True

                if not context_mask.any():
                    context_mask = None
                # ── END V2 INSERT-SILENCE ──

                # ── V1 SPLICE (deprecated — fit into existing gaps, ~40% truncation) ──
                # This version searched for contiguous silent runs around each
                # anchor and placed as many tokens as would fit, truncating the
                # rest. It never modified the audio channels or sequence length.
                #
                # inj_stats = InjectionStats(total=len(context_injections))
                # frame_rate = self.mimi.frame_rate
                # chunk_start_frame = int(start_sec * frame_rate)
                # context_mask = torch.zeros(conv_frames, dtype=torch.bool, device="cuda")
                # available_tokens = {
                #     self.interleaver.text_padding,
                #     self.interleaver.zero_padding,
                #     self.interleaver.end_of_text_padding,
                # }
                # text_channel = conv_codes[0, 0]
                #
                # for inj in context_injections:
                #     fo = inj.get("frame_offset")
                #     if fo is None:
                #         inj_stats.drop_no_offset += 1
                #         continue
                #     local_offset = fo - chunk_start_frame
                #     if local_offset < 0 or local_offset >= conv_frames:
                #         inj_stats.drop_out_of_window += 1
                #         continue
                #
                #     inj_stats.in_window += 1
                #     wrapped = f"<context> {inj['text'].strip()} </context>"
                #     tokens = tokenize(self.interleaver.tokenizer, wrapped, bos=False)
                #     full_token_len = len(tokens)
                #     inj_stats.tokens_requested += full_token_len
                #
                #     if local_offset + len(tokens) > conv_frames:
                #         tokens = tokens[:conv_frames - local_offset]
                #     if not tokens:
                #         inj_stats.drop_no_space += 1
                #         continue
                #
                #     anchor = local_offset
                #     if text_channel[anchor].item() not in available_tokens:
                #         if anchor > 0 and text_channel[anchor - 1].item() in available_tokens:
                #             anchor -= 1
                #         elif anchor + 1 < conv_frames and text_channel[anchor + 1].item() in available_tokens:
                #             anchor += 1
                #         else:
                #             inj_stats.drop_anchor_overlap += 1
                #             continue
                #
                #     run_start = anchor
                #     while run_start > 0 and text_channel[run_start - 1].item() in available_tokens:
                #         run_start -= 1
                #     run_end = anchor + 1
                #     while run_end < conv_frames and text_channel[run_end].item() in available_tokens:
                #         run_end += 1
                #
                #     available_frames = run_end - run_start
                #     if available_frames <= 0:
                #         inj_stats.drop_no_space += 1
                #         continue
                #
                #     was_truncated = False
                #     if len(tokens) <= available_frames:
                #         place_start = anchor
                #         if place_start + len(tokens) > run_end:
                #             place_start = run_end - len(tokens)
                #     else:
                #         tokens = tokens[:available_frames]
                #         place_start = run_start
                #         was_truncated = True
                #
                #     place_end = place_start + len(tokens)
                #     existing = text_channel[place_start:place_end]
                #     if any(t.item() not in available_tokens for t in existing):
                #         inj_stats.drop_final_overlap += 1
                #         continue
                #
                #     conv_codes[0, 0, place_start:place_end] = torch.tensor(
                #         tokens, device="cuda", dtype=torch.long
                #     )
                #     context_mask[place_start:place_end] = True
                #     inj_stats.placed += 1
                #     inj_stats.tokens_placed += len(tokens)
                #     if was_truncated:
                #         inj_stats.truncated += 1
                #
                # if not context_mask.any():
                #     context_mask = None
                # ── END V1 SPLICE ──

            # Prepend system prompt if present
            if prompt_codes is not None:
                codes = torch.cat([prompt_codes, conv_codes], dim=-1)
            else:
                codes = conv_codes

            # Build full context mask (prepend zeros for prompt prefix)
            full_context_mask = None
            if context_mask is not None:
                if prompt_length > 0:
                    full_context_mask = torch.cat([
                        torch.zeros(prompt_length, dtype=torch.bool, device="cuda"),
                        context_mask,
                    ])
                else:
                    full_context_mask = context_mask
                # Pad to match codes length
                T = codes.shape[-1]
                if full_context_mask.shape[0] < T:
                    full_context_mask = torch.nn.functional.pad(
                        full_context_mask, (0, T - full_context_mask.shape[0]), value=False
                    )

            return Sample(codes, data.get("text_conditions", None), prompt_length=prompt_length, context_mask=full_context_mask, injection_stats=inj_stats)
