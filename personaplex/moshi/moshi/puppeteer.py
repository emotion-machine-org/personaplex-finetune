"""Async puppeteer process — monitors conversation and injects grounding context.

Runs alongside the voice model, receiving transcript fragments each frame and
proactively injecting context (tool-call results, policy data, etc.) during
client speech windows so the broker has the information loaded before responding.

I/O Contract
============

Input (from orchestrator, each frame):
    {
        "frame_idx": int,           # current frame number
        "text_a": str | None,       # Bot A (broker) text token this frame (decoded string or None if PAD)
        "text_b": str | None,       # Bot B (client) text token this frame
        "injecting": bool,          # True if context injection is currently in progress
    }

Output (to orchestrator, when ready):
    {
        "type": "context",
        "text": str,                # directive coaching text, under 50 tokens
                                    # e.g. "Quote the Berkshire at $13,100. Emphasize savings."
    }

    The orchestrator calls lm_gen.inject_context(text, tokenizer) upon receipt.
    Output is async — the puppeteer does NOT block the 12.5 Hz frame loop.

Integration Points
==================

bot_to_bot.py:
    - Orchestrator sends transcript fragments to puppeteer each frame via queue
    - Orchestrator checks puppeteer output queue (non-blocking) and forwards
      context messages to the worker via {"context": text, "pcm": pcm}

server.py:
    - External puppeteer connects via websocket and sends JSON:
      {"type": "context", "text": "..."}
    - Server calls lm_gen.inject_context() directly

Tool Definitions (for Claude API calls)
========================================

The puppeteer calls Claude with tools like:
    - lookup_policy(policy_number) -> policy details
    - search_client(name) -> client history, prior claims
    - get_quote(coverage_type, limits) -> current pricing
    - check_underwriter_holds() -> pending approvals/deadlines

These are stub definitions — actual tool implementations connect to the
broker's CRM/AMS/rating systems.
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ConversationState:
    """Accumulated state from transcript fragments."""
    frame_idx: int = 0
    broker_tokens: list[str] = field(default_factory=list)
    client_tokens: list[str] = field(default_factory=list)
    # Track which KNOWN/UNKNOWN items have been discussed
    discussed_topics: set[str] = field(default_factory=set)
    # Track pending injections to avoid duplicates
    pending_injection: bool = False


class Puppeteer:
    """Async puppeteer that monitors conversation and produces context injections.

    Usage:
        puppeteer = Puppeteer(system_prompt=text_prompt, tools=tool_defs)
        asyncio.create_task(puppeteer.run())

        # Each frame from orchestrator:
        puppeteer.feed(frame_idx, text_a, text_b, injecting)

        # Check for output (non-blocking):
        ctx = puppeteer.get_context()
        if ctx:
            lm_gen.inject_context(ctx["text"], tokenizer)
    """

    def __init__(
        self,
        system_prompt: str,
        tools: Optional[list[dict]] = None,
        model: str = "claude-sonnet-4-6",
        max_pending_tokens: int = 50,
    ):
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.model = model
        self.max_pending_tokens = max_pending_tokens

        self.state = ConversationState()
        self._input_queue: asyncio.Queue = asyncio.Queue()
        self._output_queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    def feed(self, frame_idx: int, text_a: Optional[str], text_b: Optional[str], injecting: bool):
        """Feed a transcript fragment from the orchestrator (called each frame)."""
        self._input_queue.put_nowait({
            "frame_idx": frame_idx,
            "text_a": text_a,
            "text_b": text_b,
            "injecting": injecting,
        })

    def get_context(self) -> Optional[dict]:
        """Non-blocking check for context injection output."""
        try:
            return self._output_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def run(self):
        """Main puppeteer loop.

        Pseudocode:
            while running:
                1. Drain input queue, update conversation state
                2. Accumulate broker/client tokens into readable transcript
                3. When client starts a new utterance (broker went silent):
                   a. Check if client's topic matches an UNKNOWN item
                   b. If yes, fire tool calls to retrieve information
                   c. Compose directive coaching text from tool results
                   d. Push to output queue for injection during client speech
                4. When broker hedges ("let me check", "one moment"):
                   a. Reactive fallback — fire tool calls
                   b. Inject during the resulting pause
                5. Rate-limit: max 1 injection in flight at a time
        """
        self._running = True
        logger.info("Puppeteer started")

        while self._running:
            # Drain all available input
            fragments = []
            try:
                while True:
                    fragments.append(self._input_queue.get_nowait())
            except asyncio.QueueEmpty:
                pass

            if not fragments:
                await asyncio.sleep(0.01)  # ~100Hz poll, well above 12.5Hz frame rate
                continue

            for frag in fragments:
                self._update_state(frag)

            # --- Decision logic (pseudocode) ---
            #
            # transcript = self._build_transcript()
            # if self._should_inject_proactively():
            #     # Client is speaking about an UNKNOWN topic
            #     tool_results = await self._call_tools(topic)
            #     coaching = self._compose_coaching(tool_results)
            #     self._output_queue.put_nowait({"type": "context", "text": coaching})
            #
            # elif self._broker_is_hedging():
            #     # Reactive fallback
            #     tool_results = await self._call_tools(topic)
            #     coaching = self._compose_coaching(tool_results)
            #     self._output_queue.put_nowait({"type": "context", "text": coaching})

    def stop(self):
        self._running = False

    def _update_state(self, fragment: dict):
        """Update conversation state from a single frame's transcript fragment."""
        self.state.frame_idx = fragment["frame_idx"]
        if fragment["text_a"] is not None:
            self.state.broker_tokens.append(fragment["text_a"])
        if fragment["text_b"] is not None:
            self.state.client_tokens.append(fragment["text_b"])
        self.state.pending_injection = fragment["injecting"]

    def _build_transcript(self) -> str:
        """Build readable transcript from accumulated tokens."""
        # TODO: Segment broker/client tokens into turns, join into readable text
        raise NotImplementedError

    def _should_inject_proactively(self) -> bool:
        """Check if client is asking about an UNKNOWN topic that needs tool calls."""
        # TODO: Compare recent client speech against UNKNOWN items from system_prompt
        # Return True if we should proactively retrieve information
        raise NotImplementedError

    def _broker_is_hedging(self) -> bool:
        """Check if broker just produced a hedge phrase ("let me check", etc.)."""
        # TODO: Pattern-match recent broker tokens against hedge phrases
        raise NotImplementedError

    async def _call_tools(self, topic: str) -> dict:
        """Call Claude API with tools to retrieve information for the given topic.

        Pseudocode:
            client = anthropic.AsyncAnthropic()
            response = await client.messages.create(
                model=self.model,
                system=f"You are a real-time assistant for an insurance broker. "
                       f"The broker's briefing: {self.system_prompt}\n"
                       f"Retrieve the specific information needed to answer the client's question.",
                messages=[{"role": "user", "content": f"Client is asking about: {topic}\n"
                          f"Conversation so far: {self._build_transcript()}"}],
                tools=self.tools,
            )
            # Process tool_use blocks, execute tools, return results
        """
        raise NotImplementedError

    def _compose_coaching(self, tool_results: dict) -> str:
        """Compose directive coaching text from tool results.

        Must be under 50 tokens. Lead with action, follow with key facts.
        Example: "Quote the Berkshire at $13,100. Emphasize savings vs current 12.3% increase."

        Pseudocode:
            client = anthropic.AsyncAnthropic()
            response = await client.messages.create(
                model=self.model,
                system="Compose a brief directive coaching instruction for the broker. "
                       "Under 30 words. Lead with what to DO, then the key fact.",
                messages=[{"role": "user", "content": f"Tool results: {tool_results}"}],
            )
            return response.content[0].text
        """
        raise NotImplementedError
