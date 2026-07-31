"""
Acheron Thinking (冥河思考) — Biomimetic deliberation for the Ouro architecture.

Replaces linear Chain-of-Thought prompting with genuine state-attractor dynamics:
- State evolves through self-referential loops (no output tokens generated)
- STM (c_state) carries the "train of thought" through forget/input gating
- Temporal queue (c_state_queue) enables reflection over thought history
- Mem matrices (long-term memory) stay FROZEN during thinking
- Convergence detection via energy monitoring: E(t) = ||c_t - c_{t-1}||^2

Thinking Modes:
  "focused"   — convergent, tight energy threshold, fast settling
  "divergent" — exploratory, noise-injected, broader search

This is inference-only; no training required.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from naxi.v_0d1.gridman.config import Config, RUNNING_CONFIG
from naxi.v_0d1.gridman.core import Gridman


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class AcheronConfig:
    """Thinking hyperparameters."""

    max_steps: int = 50
    energy_threshold: float = 1e-3
    noise_std: float = 0.02
    noise_interval: int = 3
    min_steps: int = 3
    generation_temperature: float = 0.7
    verbose: bool = True
    show_thought_preview: bool = False  # off by default, toggle with /preview
    thought_preview_tokens: int = 24  # ~8 Chinese chars (3 bytes each)


# ── Trajectory records ──────────────────────────────────────────────────────

@dataclass
class ThoughtStep:
    step: int
    energy: float
    state_norm: float
    delta_norm: float
    converged: bool = False


@dataclass
class ThinkingResult:
    output_ids: torch.Tensor
    output_text: str
    trajectory: list[ThoughtStep]
    total_steps: int
    converged: bool
    elapsed_ms: float
    mode: str


# ── Acheron Thinking Engine ──────────────────────────────────────────────────

class AcheronThinking:
    """
    Biomimetic thinking engine for Ouro/Gridman models.

    STM state evolves through self-referential loops. Mem matrices (long-term
    memory) are PRESERVED — only working memory changes during thinking.

    Usage:
        model = Gridman(config).to(device)
        load_checkpoint(model, is_sft=True)
        thinker = AcheronThinking(model, config)
        result = thinker.think("明月几时有", mode="focused")
    """

    def __init__(self, model: Gridman, config: Config = RUNNING_CONFIG,
                 acheron_config: Optional[AcheronConfig] = None):
        self.model = model
        self.config = config
        self.ac = acheron_config or AcheronConfig()
        self.tokenizer = config.tokenizer
        self.device = config.device
        self.embed_dim = config.embed_dim

    # ── Public API ───────────────────────────────────────────────────────

    @torch.no_grad()
    def think(self, query: str, mode: str = "focused",
              max_steps: Optional[int] = None,
              energy_threshold: Optional[float] = None) -> ThinkingResult:
        """Think about a query and generate a response."""
        t0 = time.time()
        max_steps = max_steps or self.ac.max_steps
        threshold = energy_threshold or self._mode_threshold(mode)

        # Phase 0: Encode query → seed STM state
        input_ids = self._build_sft_input(query)
        self._forward_query(input_ids)

        # Header
        if self.ac.verbose:
            print(f"\n{'='*60}")
            print(f"  Acheron Thinking — mode: {mode}")
            print(f"  Query: {query[:80]}{'...' if len(query) > 80 else ''}")
            print(f"  Max steps: {max_steps} | Threshold: {threshold:.1e}")
            print(f"{'='*60}\n")
            if self.ac.show_thought_preview:
                print(f"  {'Step':>4s}  {'Energy':>10s}  {'|State|':>10s}  {'Preview':<40s}  Status")
                print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*40}  {'─'*20}")
            else:
                print(f"  {'Step':>4s}  {'Energy':>10s}  {'|State|':>10s}  Status")
                print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*20}")

        # Phase 1: Deliberative loop — only STM evolves
        trajectory: list[ThoughtStep] = []
        prev_state: Optional[torch.Tensor] = None
        converged = False

        for step in range(max_steps):
            self._think_step(input_ids)

            # Reflective mode: after each think step, reflect on recent thoughts
            if mode == "reflective" and step > 0:
                self._reflect_step(input_ids)

            current_state = self._get_current_state()

            if prev_state is not None and step >= self.ac.min_steps:
                delta_norm = torch.norm(current_state - prev_state).item()
                energy = (delta_norm ** 2) / self.embed_dim
                converged = energy < threshold
            else:
                delta_norm = float('inf')
                energy = float('inf')

            state_norm = torch.norm(current_state).item() / math.sqrt(self.embed_dim)

            trajectory.append(ThoughtStep(step=step, energy=energy,
                                          state_norm=state_norm,
                                          delta_norm=delta_norm,
                                          converged=converged))

            if self.ac.verbose:
                status = "⚡ CONVERGED" if converged else ("·" if step < self.ac.min_steps else "→")
                if self.ac.show_thought_preview:
                    preview = self._peek_thought(input_ids, self.ac.thought_preview_tokens)
                    preview_clean = preview.replace('\n', ' ')[:40]
                    print(f"  {step:4d}  {energy:10.6f}  {state_norm:10.4f}  {preview_clean:<40s}  {status}")
                else:
                    print(f"  {step:4d}  {energy:10.6f}  {state_norm:10.4f}  {status}")

            if converged:
                break
            if mode == "divergent" and step > 0 and step % self.ac.noise_interval == 0:
                self._inject_noise()
            prev_state = current_state

        # Phase 2: Generate from converged state (collect all chunks)
        output_text = ""
        for chunk in self._generate_stream(input_ids):
            output_text += chunk
        output_ids = torch.tensor([self.tokenizer.encode(output_text)], device=self.device, dtype=torch.long)
        elapsed_ms = (time.time() - t0) * 1000

        result = ThinkingResult(output_ids=output_ids, output_text=output_text,
                                trajectory=trajectory, total_steps=len(trajectory),
                                converged=converged, elapsed_ms=elapsed_ms, mode=mode)

        if self.ac.verbose:
            print(f"\n  ── Result ──")
            print(f"  Steps: {result.total_steps} | Converged: {converged}")
            print(f"  Time: {elapsed_ms:.0f}ms")
            print(f"  Output: {output_text[:120]}{'...' if len(output_text) > 120 else ''}")
            print()

        return result

    def reset_state(self):
        """Soft reset: clear STM working memory, keep trained mem matrices."""
        ouro = self.model.core_ouro
        ouro.stm.mem_clear()
        ouro.c_state_queue.zero_()
        for block in ouro.ouro_blocks:
            for layer in block.ouro_layers:
                if layer.need_stm:
                    layer.ouro_stm.mem_clear()

    @torch.no_grad()
    def think_stream(self, query: str, mode: str = "focused",
                     max_steps: Optional[int] = None,
                     energy_threshold: Optional[float] = None):
        """
        Think and stream output chunks as they are generated.
        Yields (phase, data):
          ("trajectory", ThoughtStep) — during thinking
          ("text", str)              — during generation
          ("done", ThinkingResult)   — final result
        """
        t0 = time.time()
        max_steps = max_steps or self.ac.max_steps
        threshold = energy_threshold or self._mode_threshold(mode)

        input_ids = self._build_sft_input(query)
        self._forward_query(input_ids)

        trajectory: list[ThoughtStep] = []
        prev_state = None
        converged = False

        for step in range(max_steps):
            self._think_step(input_ids)
            if mode == "reflective" and step > 0:
                self._reflect_step(input_ids)

            current_state = self._get_current_state()
            if prev_state is not None and step >= self.ac.min_steps:
                delta_norm = torch.norm(current_state - prev_state).item()
                energy = (delta_norm ** 2) / self.embed_dim
                converged = energy < threshold
            else:
                energy = float('inf')

            ts = ThoughtStep(step=step, energy=energy,
                             state_norm=torch.norm(current_state).item() / math.sqrt(self.embed_dim),
                             delta_norm=float('inf') if energy == float('inf') else math.sqrt(energy * self.embed_dim),
                             converged=converged)
            trajectory.append(ts)
            yield ("trajectory", ts)

            if converged:
                break
            if mode == "divergent" and step > 0 and step % self.ac.noise_interval == 0:
                self._inject_noise()
            prev_state = current_state

        output_text = ""
        for chunk in self._generate_stream(input_ids):
            output_text += chunk
            yield ("text", chunk)

        elapsed_ms = (time.time() - t0) * 1000
        result = ThinkingResult(
            output_ids=torch.tensor([self.tokenizer.encode(output_text)], device=self.device),
            output_text=output_text, trajectory=trajectory,
            total_steps=len(trajectory), converged=converged,
            elapsed_ms=elapsed_ms, mode=mode)
        yield ("done", result)

    def show_trajectory(self, result: ThinkingResult):
        """Print thought trajectory."""
        traj = result.trajectory
        if not traj:
            print("(empty trajectory)")
            return
        print(f"\n  Thought Trajectory ({result.mode}, {len(traj)} steps):")
        print(f"  {'Step':>4s}  {'Energy':>10s}  {'|State|':>10s}")
        print(f"  {'─'*4}  {'─'*10}  {'─'*10}")
        converged_at = next((t.step for t in traj if t.converged), None)
        for t in traj:
            marker = ""
            if converged_at is not None and t.step == converged_at:
                marker = " ← converged"
            elif t.step < self.ac.min_steps:
                marker = " (warmup)"
            e = f"{t.energy:.6f}" if t.energy != float('inf') else "∞"
            print(f"  {t.step:4d}  {e:>10s}  {t.state_norm:10.4f}{marker}")
        finite = [t.energy for t in traj if t.energy != float('inf')]
        if finite:
            print(f"\n  Energy range: {min(finite):.6f} → {max(finite):.6f}")
            print(f"  Final energy: {finite[-1]:.6f}")

    # ── Internal: Encoding & Forward ─────────────────────────────────────

    def _build_sft_input(self, text: str) -> torch.Tensor:
        """SFT format: [EOS][USER]...query...[EOS][ASSISTANT]"""
        t = self.tokenizer
        ids = [t.eos_token_id, t.user_token_id] + t.encode(text) + \
              [t.eos_token_id, t.assistant_token_id]
        return torch.tensor([ids], device=self.device, dtype=torch.long)

    def _forward_query(self, input_ids: torch.Tensor) -> None:
        """Encode query through Ouro. Full mem_sync — query ingest
        naturally updates knowledge with seq_len damping."""
        self.model.eval()
        with torch.amp.autocast(self.config.device_type, dtype=torch.bfloat16):
            _ = self.model(input_ids, lock_mem=False)
        self.model.core_ouro.mem_sync()

    def _think_step(self, input_ids: torch.Tensor) -> None:
        """
        One thinking step: re-read the full query with the evolving state.

        Feeding the complete sequence (seq_len > 1) applies the mem damping
        factor 1/seq_len — knowledge (mem matrices) participates gently
        through associative retrieval instead of being frozen or corrupted.
        """
        self.model.eval()
        with torch.amp.autocast(self.config.device_type, dtype=torch.bfloat16):
            _ = self.model(input_ids, lock_mem=False)
        self.model.core_ouro.mem_sync()

    def _get_current_state(self) -> torch.Tensor:
        return self.model.core_ouro.stm.active_c(batch_size=1).detach().clone()

    def _reflect_step(self, input_ids: torch.Tensor) -> None:
        """
        Reflective step: a second re-read of the query.

        The temporal queue now holds the prior think-step's state, so this
        second pass naturally reflects on it via OuroTemporalAttention —
        no state vectors are pushed through the input processors (that
        was the instability source). Deliberation without distribution shift.
        """
        self.model.eval()
        with torch.amp.autocast(self.config.device_type, dtype=torch.bfloat16):
            _ = self.model(input_ids, lock_mem=False)
        self.model.core_ouro.mem_sync()
    def _peek_thought(self, input_ids: torch.Tensor, n_tokens: int = 10) -> str:
        """
        Mini autoregressive generation from current state (lock_mem=True).
        Shows what the model would say if it stopped thinking now.
        """
        self.model.eval()
        gen = input_ids.clone()
        for _ in range(n_tokens):
            with torch.amp.autocast(self.config.device_type, dtype=torch.bfloat16):
                logits = self.model(gen, lock_mem=True)
            probs = F.softmax(logits[:, -1, :256] / 0.7, dim=-1)
            nt = torch.multinomial(probs, num_samples=1).item()
            if nt == self.tokenizer.eos_token_id:
                break
            gen = torch.cat([gen, torch.tensor([[nt]], device=self.device)], dim=1)
        new_ids = gen[0, input_ids.shape[1]:].tolist()
        return self.tokenizer.decode(new_ids)

    def _inject_noise(self) -> None:
        """Inject controlled noise into STM for divergent mode."""
        stm = self.model.core_ouro.stm
        if stm._runtime_c_state is not None:
            current_norm = torch.norm(stm._runtime_c_state).item()
            scale = self.ac.noise_std * max(current_norm / math.sqrt(self.embed_dim), 0.01)
            stm._runtime_c_state = stm._runtime_c_state + \
                torch.randn_like(stm._runtime_c_state) * scale

    # ── Internal: UTF-8 helpers ──────────────────────────────────────────

    @staticmethod
    def _split_valid_utf8(patch: list[int]) -> tuple[list[int], list[int]]:
        """Split patch at last valid UTF-8 boundary. Returns (valid, leftover)."""
        n = len(patch)
        for i in range(1, min(5, n + 1)):
            token = patch[-i]
            if token > 255 or token <= 127:
                return patch, []
            if 192 <= token <= 247:
                expected = 2 if 192 <= token <= 223 else (3 if 224 <= token <= 239 else 4)
                return (patch, []) if i == expected else (patch[:-i], patch[-i:])
        return patch, []

    # ── Internal: Generation ─────────────────────────────────────────────

    def _generate_stream(self, input_ids: torch.Tensor):
        """
        Streaming generator — yields text chunks as UTF-8 characters complete.
        Matches the ChatSession's incremental decoding pattern.
        """
        self.model.eval()
        patch_size = self.config.patch_size
        current_patch: list[int] = list(input_ids[0].tolist())
        all_generated: list[int] = []
        cached_logits = None
        display_offset = 0

        while True:
            if len(current_patch) > 0:
                p = torch.tensor([current_patch], device=self.device, dtype=torch.long)
                lock = len(current_patch) < patch_size
                with torch.amp.autocast(self.config.device_type, dtype=torch.bfloat16):
                    logits = self.model(p, lock)
                cur_logits = logits[:, -1, :]
                if not lock:
                    self.model.core_ouro.mem_sync()
                    cached_logits = cur_logits
                    current_patch = []
            elif cached_logits is not None:
                cur_logits = cached_logits
                cached_logits = None

            probs = F.softmax(cur_logits / self.ac.generation_temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            if next_token == self.tokenizer.eos_token_id:
                # Flush remaining
                valid, _ = self._split_valid_utf8(all_generated)
                if len(valid) > display_offset:
                    yield self.tokenizer.decode(valid[display_offset:])
                break
            all_generated.append(next_token)
            current_patch.append(next_token)

            # Yield new completable characters
            valid, _ = self._split_valid_utf8(all_generated)
            if len(valid) > display_offset:
                chunk = self.tokenizer.decode(valid[display_offset:])
                display_offset = len(valid)
                if chunk:
                    yield chunk

    def _mode_threshold(self, mode: str) -> float:
        return {"focused": 1e-3, "divergent": 5e-5, "reflective": 1e-3}.get(
            mode, self.ac.energy_threshold)




def acheron_chat(model: Gridman, config: Config = RUNNING_CONFIG,
                 acheron_config: Optional[AcheronConfig] = None) -> None:
    """Interactive chat loop with Acheron Thinking."""
    thinker = AcheronThinking(model, config, acheron_config)
    last_result: Optional[ThinkingResult] = None

    print("\n" + "=" * 60)
    print("  Acheron Thinking Chat")
    print("  /focus | /diverge | /reflect  — 思考模式")
    print("  /trajectory | /preview | /clear | /quit")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.\n")
            break
        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("/quit", "/exit"):
            print("Bye.\n")
            break
        if cmd == "/clear":
            thinker.reset_state()
            last_result = None
            print("  [State cleared]\n")
            continue
        if cmd == "/trajectory":
            if last_result:
                thinker.show_trajectory(last_result)
            else:
                print("  [No trajectory yet]\n")
            continue
        if cmd == "/preview":
            thinker.ac.show_thought_preview = not thinker.ac.show_thought_preview
            print(f"  [Thought preview: {'ON' if thinker.ac.show_thought_preview else 'OFF'}]\n")
            continue

        mode, query = "focused", user_input
        for prefix, m in [("/focus ", "focused"), ("/diverge ", "divergent"),
                          ("/reflect ", "reflective"), ("/think ", "focused")]:
            if user_input.startswith(prefix):
                mode, query = m, user_input[len(prefix):].strip()
                break

        if not query:
            print("  [Empty query]\n")
            continue

        # Print header
        if thinker.ac.verbose:
            t = thinker._mode_threshold(mode)
            ms = max_steps = thinker.ac.max_steps
            print(f"\n{'='*60}")
            print(f"  Acheron Thinking — mode: {mode}")
            print(f"  Query: {query[:80]}{'...' if len(query) > 80 else ''}")
            print(f"  Max steps: {ms} | Threshold: {t:.1e}")
            print(f"{'='*60}\n")
            print(f"  {'Step':>4s}  {'Energy':>10s}  {'|State|':>10s}  Status")
            print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*20}")

        # Stream thinking + generation
        last_result = None
        for phase, data in thinker.think_stream(query, mode=mode):
            if phase == "trajectory":
                ts = data
                if thinker.ac.verbose:
                    status = "⚡ CONVERGED" if ts.converged else ("·" if ts.step < thinker.ac.min_steps else "→")
                    print(f"  {ts.step:4d}  {ts.energy:10.6f}  {ts.state_norm:10.4f}  {status}")
            elif phase == "text":
                print(data, end="", flush=True)
            elif phase == "done":
                last_result = data
        print("\n")
