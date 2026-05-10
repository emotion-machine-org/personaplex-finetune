import logging
import os
from dataclasses import dataclass, field

from simple_parsing.helpers import Serializable

from .data.args import DataArgs


@dataclass
class SystemPromptArgs(Serializable):
    """PersonaPlex-style hybrid system prompt support.

    When enabled, each training sample's JSON can include:
      - "text_prompt": role-conditioning text (wrapped in <system> tags)
      - "voice_prompt": path to a voice sample WAV for voice cloning

    The system prompt is prepended to each training sequence as:
      voice frames → silence → text prompt frames → silence
    Loss is masked over the prompt region.
    """

    enable: bool = False
    audio_silence_frames: int = 6  # ~0.48s of silence at 12.5 Hz
    # Budget (in frames at 12.5 Hz) reserved for the system prompt when
    # computing chunk step size.  Should be >= the longest prompt in the
    # dataset.  When > 0, the audio chunking step is reduced by this many
    # frames so that no context injections fall into the "prompt gap"
    # between chunks.  Use compute_prompt_budget.py to find the right value.
    # 0 = legacy behaviour (step = duration_sec, some injections lost).
    prompt_budget_frames: int = 0


@dataclass
class GenEvalArgs(Serializable):
    """Generation-based evaluation: Gemini↔Moshi dialogues at checkpoint time.

    Starts a Moshi server with merged LoRA weights, connects Gemini Live API
    as the client, runs dialogues, and evaluates via LLM review.
    """

    enable: bool = False
    freq: int = 0  # steps between gen_evals (0 = only at end)
    gpu_a: int = 3  # physical GPU for Moshi server
    dialogue_duration: float = 60.0  # seconds per dialogue
    seed: int = 42
    eval_prompts_path: str = ""
    llm_review: bool = True
    llm_model: str = "claude-sonnet-4-20250514"
    dialogue_timeout: int = 300
    temp_audio: float = 0.55  # audio sampling temperature
    temp_text: float = 0.7  # text sampling temperature
    topk_audio: int = 100  # audio top-k candidates
    topk_text: int = 30  # text top-k candidates
    # Gemini Live API settings
    gemini_model: str = "models/gemini-3.1-flash-live-preview"
    gemini_voice: str = "Puck"
    moshi_port: int = 8998  # port for the Moshi server subprocess
    # Deprecated (kept for config backward compatibility)
    gpu_b: int = 4
    nudge_after: float = 2.0
    max_nudges: int = 0
    rep_penalty: float = 0.0
    rep_penalty_window: int = 30


@dataclass
class PuppeteerArgs(Serializable):
    """Configuration for puppeteer context injection during training data preparation."""

    enable: bool = False
    max_tokens_per_injection: int = 50  # hard cap on injection length
    context_tag: str = "context"  # wrapping tag: <context>...</context>


@dataclass
class LoraArgs(Serializable):
    enable: bool = False
    rank: int = 64
    scaling: float = 2.0
    ft_embed: bool = False
    skip_depformer: bool = True

    def __post_init__(self) -> None:
        if self.enable:
            assert self.rank > 0
            assert self.scaling > 0.0


@dataclass
class OptimArgs(Serializable):
    lr: float = 1e-4
    weight_decay: float = 0.1
    pct_start: float = 0.05


@dataclass
class WandbArgs(Serializable):
    project: str | None = None  # Fill this argument to use wandb.
    offline: bool = False
    key: str | None = None
    run_name: str | None = None

    def __post_init__(self) -> None:
        if self.project is not None:
            try:
                import wandb  # noqa: F401
            except ImportError:
                raise ImportError(
                    "`wandb` not installed. Either make sure `wandb` is installed or set `wandb:project` to None."
                )

            if len(self.project) == 0:
                raise ValueError("`wandb.project` must not be an empty string.")


@dataclass
class ModelPaths(Serializable):
    hf_repo_id: str | None = "kyutai/moshiko-pytorch-bf16"
    mimi_path: str | None = None
    moshi_path: str | None = None
    tokenizer_path: str | None = None
    config_path: str | None = None

    def __post_init__(self) -> None:
        if self.hf_repo_id is not None and self.config_path is None:
            print(
                "Warning: `hf_repo_id` is set but `config_path` is None. "
                "This will load default models."
            )


@dataclass
class TrainArgs(Serializable):
    data: DataArgs

    run_dir: str  # Path to the directory where everything will be saved. It needs to be empty.
    # Name of the wandb run, if None it will be set to the name of the run_dir.
    moshi_paths: ModelPaths = field(default_factory=ModelPaths)
    first_codebook_weight_multiplier: float = 1.0
    text_padding_weight: float = 0.5
    # Weight for audio loss in the total training loss: total = text + weight * audio.
    # 1.0 = default (equal weighting).  0.0 = text-only training (no audio
    # gradients reach the backbone LoRA).  Intermediate values (e.g. 0.1)
    # keep a small audio regularisation signal.
    audio_loss_weight: float = 1.0
    # L2 regularization on LoRA parameters to prevent drift from base model.
    # LoRA weights are initialized to zero, so L2 pulls them back toward the
    # base model.  0.0 = disabled.  Try 1e-4 to start.
    lora_l2_weight: float = 0.0
    # NEFTune: add uniform noise to embeddings during training to regularize
    # and prevent overfitting.  0.0 = disabled.  Try 5.0-15.0 to start.
    # Noise magnitude is scaled by alpha / sqrt(seq_len * embed_dim).
    neftune_alpha: float = 0.0

    optim: OptimArgs = field(default_factory=OptimArgs)
    seed: int = 0
    # Number of steps to accumulate gradients before doing an optimizer step.
    num_microbatches: int = 1

    duration_sec: float = 10
    batch_size: int = 1
    max_norm: float = 1.0  # Gradient clipping.
    max_steps: int = 100  # Number of training steps.
    log_freq: int = 1  # Number of steps between each logging.

    # Number of steps between each checkpoint saving. If inferior to 1, only the last checkpoint will be saved.
    ckpt_freq: int = 0
    save_adapters: bool = True
    # If False, no checkpoints will be saved. This is useful for development.
    do_ckpt: bool = True
    num_ckpt_keep: int | None = 3
    eval_freq: int = 0
    do_eval: bool = False
    skip_zero_eval: bool = False  # skip the step-0 baseline eval and gen_eval

    # Efficiency
    # Determines whether gradient checkpointing should be utilized or not
    # during the training process. Gradient checkpointing can be beneficial in
    # reducing memory usage at the cost of slightly longer training times.
    gradient_checkpointing: bool = True

    world_size: int | None = field(init=False, default=None)

    # logging
    wandb: WandbArgs = field(default_factory=WandbArgs)

    # LoRA
    lora: LoraArgs | None = field(default_factory=LoraArgs)
    full_finetuning: bool = False

    # PersonaPlex-style system prompt
    system_prompt: SystemPromptArgs = field(default_factory=SystemPromptArgs)

    # Generation eval (bot-to-bot dialogues at checkpoint time)
    gen_eval: GenEvalArgs = field(default_factory=GenEvalArgs)

    # Puppeteer context injection
    puppeteer: PuppeteerArgs = field(default_factory=PuppeteerArgs)

    param_dtype: str = "bfloat16"

    overwrite_run_dir: bool = False

    # Path to a run directory or checkpoint directory to resume from.
    # Loads LoRA weights and continues training from the checkpoint step.
    # Optimizer state is NOT restored (warm restart).
    resume_from: str | None = None
    # When True, skip restoring optimizer/scheduler from train_state.pt on
    # resume.  Use this when changing max_steps or lr for a resumed run so
    # the scheduler is re-created for the new config.
    force_warm_restart: bool = False

    def __post_init__(self) -> None:
        assert getattr(self, "world_size", None) is None
        self.world_size = int(os.environ.get("WORLD_SIZE", -1))

        if self.wandb.offline:
            command = f"cd {self.run_dir}; wandb sync --sync-all"
            logging.info(f"to sync wandb offline, run: {command}")

        assert self.num_microbatches >= 1

        assert self.num_ckpt_keep is None or self.num_ckpt_keep >= 1

        if not self.save_adapters:
            logging.warning(
                "You have disabled `save_adapters` and are thus merging the "
                "trained LoRA checkpoint into the base model upon checkpointing. "
                "This might lead to OOM errors - make sure you have enough CPU "
                "and GPU memory."
            )
