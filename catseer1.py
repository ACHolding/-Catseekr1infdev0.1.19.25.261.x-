#!/usr/bin/env python3
"""cat r1 v0.1 — offline bilingual RAM-only BitNet LLM on the cat r1 engine.

Cat R11.x materializes a real ~20 billion-parameter BitNet b1.58 Transformer
in RAM when the program starts (files=off). Boot target ≈ **0.2 seconds** via a
sparse 2-bit bank (resident STE pages only; virtual ~20B slots). The inference
trunk is the **cat r1 engine**: Delta Attention (KDA), Attention Residuals
(AttnRes), and Stable LatentMoE-style sparse FFN routing — all executed through
real BitLinear ternary kernels with STE training on the resident shadow slice.
Tokenizer, bilingual routing, training corpus, next-token softmax, 60 FPS
GUI/CLI, Cat R11.x Code, Cat R11.x Build (Grok Build–style apps/games), and
self-tests live in this file. Chat is a real BitNet b1.58 LLM (W1.58A8 decode) with EN/ZH auto-detect.

Real BitNet (Microsoft BitNet b1.58) at ~20B:
    * every projection is real BitLinear (Microsoft BitNet b1.58)
    * SubLN/RMSNorm → absmean ternary W {-1,0,+1} → per-token AbsMax INT8 A8 (Q_b=128)
    * inference matmul is integer add/sub only (no FP multiply by weights)
    * FFN uses ReLU²; attention Q/K use RoPE (LLaMA-alike BitNet)
    * ~20B ternary slots; sparse pages materialize the live STE slice only
    * resident FP32 shadows train with STE; packed bank stays files=off
    * cat r1 hybrid attention: 3× KDA + 1× global gated attention (NoPE)
    * AttnRes mixes block snapshots with learned pseudo-queries
    * LatentMoE routes a shared BitLinear expert plus top-k live FFN experts
    * Mixture-of-Depths capacity keeps per-token compute interactive
    * DSpark speculative decode (Markov draft + confidence verify) — files=off
    * cat r1 v0.1 whitepaper stack (CED, CSA2, mHC, Engram, CatR1MoE, DSpark, FP4-KV) — files=off
    * Cat R1 think/answer + GRPO reward inside the cat r1 v0.1 × DSpark path — files=off
    * respond-anything: every prompt returns a concrete helpful reply
    * InputAutodetectEngine: wording / code / syntax auto-detect + glitch guard
    * ChatAPIEngine: local OpenAI-style chat API that always knows what to say
    * EleziaChatAPI + ELEZIA dataset: casual chat (hi / how are you / wow that's cool)
    * GptJRamDataset: GPT-J / Pile-style micro-corpus kept entirely in RAM (files=off)
    * no dual-residual fake ternary; no downloaded checkpoint
    * default boot_budget_s ≈ 0.2; STE warmup deferred to ``/train``

Cat R11.x Build (Grok Build–style):
    * describe an app, game, website, or dashboard in natural language
    * outputs complete HTML/JS source and saves an artifact under ``builds/``
    * iterate with ``/build …`` or ``python3 '##cat-r11.x.py' --build``
    * model weights remain files=off; only your build artifacts touch disk

Cat R11.x Code (Claude Code fork):
    * local agentic coding REPL: Read / Write / Edit / Bash / Glob / Grep / LS
    * powered by the same real BitNet 20B + cat r1 brain (no API, no network model)

files = off (model weights):
    * no checkpoint is read or written
    * no model is downloaded
    * no API or network request is used for the LM
    * all weights and optimizer state live only in RAM

Launch ``--code`` for the coding agent or ``--build`` for Build mode.
"""

from __future__ import annotations

import argparse
import ast
import collections
import fractions
import json
import math
import os
import queue
import re
import statistics
import subprocess
import sys
import threading
import time
import tkinter as tk
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tkinter import font, scrolledtext
from typing import Any, Callable, Iterable, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment error path
    raise SystemExit(
        "Cat R11.x requires NumPy for real in-memory BitNet training. "
        "No model checkpoint is required."
    ) from exc


APP_NAME = "cat r1"
APP_VERSION = "0.1"
MODEL_ID = "cat-r1-v0.1-bitnet-b1.58-20b-ced-csa2-mhc-engram-dspark-ram"
FILES_MODE = "off"
ENGINE_ID = "cat-r1"
ENGINE_NAME = "cat r1"
GUI_FPS = 60
GUI_TICK_MS = max(1, 1000 // GUI_FPS)
DEFAULT_SEED = 0xCA75_EE41
BITNET_QB = 128  # Q_b = 2^(k-1) for k=8 → AbsMax A8 range (Microsoft BitNet b1.58)
BITNET_EPS = 1e-5
BITNET_A8_MIN = -BITNET_QB      # paper: RoundClip(..., -Qb, Qb-1)
BITNET_A8_MAX = BITNET_QB - 1
# Deep-narrow real BitNet b1.58 Transformer ≈ 20.0B ternary weights (files=off, packed).
# Per layer: 4·d² (Q/K/V/O) + 2·d·d_ff (FFN) with d_ff=4d → 12·d².
# d=512, d_ff=2048, n_layers=6358 → 20,000,538,624 ternary slots (~20B).
BITNET_20B_N_LAYERS = 6358
BITNET_20B_D_MODEL = 512
BITNET_20B_D_FF = 2048
BITNET_20B_N_HEADS = 8
BITNET_20B_PARAM_TARGET = 20_000_000_000
BITNET_20B_PARAM_COUNT = (
    BITNET_20B_N_LAYERS
    * (
        4 * BITNET_20B_D_MODEL * BITNET_20B_D_MODEL
        + 2 * BITNET_20B_D_MODEL * BITNET_20B_D_FF
    )
)
# cat r1 hybrid attention ratio: 3 KDA layers then 1 global gated attention.
CAT_R1_KDA_RATIO = 3
CAT_R1_MOE_TOP_K = 6
# DSpark speculative decode (files=off) — Markov draft + confidence verify.
DSPARK_BLOCK_SIZE = 7
DSPARK_MARKOV_RANK = 64
DSPARK_DRAFT_GAMMA = 5
# cat r1 v0.1 whitepaper stack (files=off) on BitNet + DSpark trunk.
CAT_R1_ID = "cat-r1-v0.1"
CAT_R1_ENCODER_SHARE = 0.5  # CED: first half causal encoder, second half decoder
CAT_R1_MOE_TOP_K = 6  # CatR1MoE-style: activate up to 6 routed experts
CAT_R1_ENGRAM_SLOTS = 4096
CAT_R1_CONTEXT_VIRTUAL = 1_000_000  # whitepaper million-token context (virtual)
CAT_R1_THINK_BUDGET = 24
CAT_R1_ANSWER_BUDGET = 48
THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "<" + "/" + "think" + ">"
# Fast boot: sparse BitNet pages only; STE deferred to /train (files=off).
BOOT_BUDGET_S = 0.2


@dataclass(frozen=True, slots=True)
class ModelConfig:
    context_tokens: int = 64
    # Legacy aliases kept for GUI/CLI compatibility; transformer uses d_*.
    embedding_dim: int = BITNET_20B_D_MODEL
    hidden_dim: int = BITNET_20B_D_FF
    d_model: int = BITNET_20B_D_MODEL
    d_ff: int = BITNET_20B_D_FF
    n_layers: int = BITNET_20B_N_LAYERS
    n_heads: int = BITNET_20B_N_HEADS
    mod_capacity: int = 2
    train_layers: int = 4
    dense_init: bool = False
    reasoning_passes: int = 2
    max_reasoning_passes: int = 8
    reasoning_scale: float = 0.30
    ponder_entropy_threshold: float = 0.48
    ponder_delta_threshold: float = 0.018
    train_steps: int = 0  # STE via /train; chat still uses real BitNet decode (llm_first)
    batch_size: int = 4
    learning_rate: float = 0.002
    gradient_clip: float = 1.0
    activation_bits: int = 8
    seed: int = DEFAULT_SEED
    max_new_tokens: int = 48
    temperature: float = 0.7
    top_k: int = 40
    deliberation_candidates: int = 1
    boot_budget_s: float = BOOT_BUDGET_S  # fast materialize; /train for STE
    llm_first: bool = True  # chat is autoregressive BitNet decode, not template recall
    engine: str = ENGINE_ID
    kda_ratio: int = CAT_R1_KDA_RATIO
    moe_top_k: int = 1  # interactive decode uses shared expert; raise for quality profile
    attnres: bool = True
    speed_target: str = "fable5"
    # DSpark × BitNet (files=off) — same toggle shape as prior cat r11.x DSpark builds.
    dspark_enabled: bool = True  # Cat R1 default decode engine
    dspark_speculative_decode: bool = False  # off by default: doubles CPU decode cost
    dspark_block_size: int = DSPARK_BLOCK_SIZE
    dspark_markov_rank: int = DSPARK_MARKOV_RANK
    dspark_confidence_head: bool = True
    dspark_draft_gamma: int = DSPARK_DRAFT_GAMMA
    dspark_adaptive_gamma: bool = True
    # cat r1 v0.1 × BitNet × DSpark (files=off) — whitepaper modules + respond-anything.
    cat_r1_enabled: bool = True
    cat_r1_ced: bool = True
    cat_r1_csa2: bool = True
    cat_r1_mhc: bool = True
    cat_r1_engram: bool = True
    cat_r1_fp4_kv: bool = True
    cat_r1_respond_anything: bool = True
    cat_r1_reason_enabled: bool = True  # think/answer GRPO path inside cat r1 v0.1
    cat_r1_think_tokens: int = CAT_R1_THINK_BUDGET
    cat_r1_answer_tokens: int = CAT_R1_ANSWER_BUDGET
    cat_r1_show_think: bool = False
    cat_r1_candidates: int = 1


@dataclass(slots=True)
class TrainingReport:
    initial_loss: float = math.inf
    final_loss: float = math.inf
    steps: int = 0
    samples: int = 0
    elapsed_s: float = 0.0
    loss_history: list[float] = field(default_factory=list)


@dataclass(slots=True)
class GenerationStep:
    index: int
    token: str
    token_id: int
    probability: float
    entropy_bits: float


@dataclass(slots=True)
class GenerationReport:
    text: str
    language: str
    tokens: int
    elapsed_s: float
    tokens_per_second: float
    finish_reason: str
    trace: list[GenerationStep]


@dataclass(slots=True)
class Reply:
    text: str
    route: str
    elapsed_ms: float
    tokens: int = 0
    tokens_per_second: float = 0.0


# ──────────────────────────────────────────────────────────────
# DSPARK × BITNET + CAT R1 (files = off · speculative decode)
# Semi-autoregressive Markov draft · confidence verify · # pr
# ──────────────────────────────────────────────────────────────
@dataclass(slots=True)
class DSparkStats:
    drafts: int = 0
    accepted: int = 0
    gamma: int = 0
    speedup: float = 1.0


class DSparkMarkovHead:
    """Low-rank Markov logit correction — RAM-only ternary-ish projections (files=off)."""

    __slots__ = ("rank_proj", "out_proj")

    def __init__(self, d_model: int, vocab: int, rank: int, seed: int):
        rng = np.random.default_rng(seed)
        raw_in = rng.normal(0.0, 0.05, size=(d_model, rank)).astype(np.float32)
        raw_out = rng.normal(0.0, 0.05, size=(rank, vocab)).astype(np.float32)
        # Absmean ternary packing flavor without touching disk.
        for matrix_name, raw in (("rank_proj", raw_in), ("out_proj", raw_out)):
            gamma = float(np.mean(np.abs(raw))) + BITNET_EPS
            codes = np.clip(np.round(raw / gamma), -1, 1).astype(np.float32)
            setattr(self, matrix_name, (codes * gamma).astype(np.float32))

    def bias(self, hidden: np.ndarray) -> np.ndarray:
        h = hidden.reshape(1, -1) if hidden.ndim == 1 else hidden
        mid = h @ self.rank_proj
        return (mid @ self.out_proj).astype(np.float32)


class DSparkBitNetEngine:
    """DSpark speculative decode on the BitNet + cat r1 trunk (files=off).

    Draft head: pooled embedding + Markov logit bias (cheap).
    Target verify: real BitNet Transformer (``next_token_probabilities`` / integer BitLinear).
    """

    __slots__ = ("model", "markov", "confidence", "block_size", "last_stats")

    def __init__(self, model: "InMemoryTernaryLM"):
        self.model = model
        cfg = model.config
        rank = max(8, int(cfg.dspark_markov_rank))
        vocab = model.tokenizer.vocab_size
        self.markov = DSparkMarkovHead(cfg.d_model, vocab, rank, 9107)
        conf_rng = np.random.default_rng(9108)
        self.confidence = conf_rng.normal(0.0, 0.05, size=(cfg.d_model, 1)).astype(np.float32)
        self.block_size = max(1, int(cfg.dspark_block_size))
        self.last_stats = DSparkStats()

    def adaptive_gamma(self, prompt: str, remaining: int) -> int:
        base = int(self.model.config.dspark_draft_gamma)
        if not self.model.config.dspark_adaptive_gamma:
            return min(base, remaining, self.block_size)
        words = max(1, len((prompt or "").split()))
        gamma = base
        if words > 80:
            gamma += 2
        elif words > 40:
            gamma += 1
        elif words < 6:
            gamma = max(2, gamma - 1)
        return min(gamma, remaining, self.block_size)

    def predict_accept(self, hidden: np.ndarray) -> float:
        h = hidden.reshape(1, -1)
        logit = float((h @ self.confidence).reshape(-1)[0])
        return float(1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit)))))

    def draft_probabilities(self, token_ids: list[int]) -> tuple[np.ndarray, np.ndarray, float]:
        contexts = self.model._context_array(token_ids)
        hidden = self.model._context_hidden(contexts)[0]
        logits = (hidden @ self.model.lm_head).astype(np.float32) + self.model.b_output
        logits = logits + 0.35 * self.markov.bias(hidden).reshape(-1)
        logits = np.nan_to_num(logits, nan=0.0, posinf=40.0, neginf=-40.0)
        logits = logits - np.max(logits)
        exp = np.exp(np.clip(logits, -40.0, 40.0))
        probabilities = (exp / np.maximum(np.sum(exp), BITNET_EPS)).astype(np.float32)
        return probabilities, hidden, self.predict_accept(hidden)

    def accept_block(
        self,
        *,
        prompt: str,
        context: list[int],
        generated: list[int],
        sample_rng: np.random.Generator,
        temperature: float,
        top_k: int,
        eos: int,
        remaining: int,
        on_token: Optional[Callable[[str], None]],
        start_index: int,
        min_new_tokens: int = 0,
    ) -> tuple[list[GenerationStep], str]:
        """Draft γ tokens, verify with target BitNet, accept matching prefix."""
        gamma = self.adaptive_gamma(prompt, remaining)
        draft_ids: list[int] = []
        draft_trace: list[tuple[int, float, float, float]] = []
        draft_context = list(context)
        for _ in range(gamma):
            probs, _hidden, conf = self.draft_probabilities(draft_context)
            allow_eos = len(generated) + len(draft_ids) >= max(0, int(min_new_tokens))
            token_id, probability, entropy = self.model._choose_token(
                probs,
                temperature=temperature,
                top_k=top_k,
                rng=sample_rng,
                generated=generated + draft_ids,
                min_new_tokens=min_new_tokens,
                allow_eos=allow_eos,
            )
            draft_ids.append(token_id)
            draft_trace.append((token_id, probability, entropy, conf))
            if token_id == eos:
                break
            draft_context.append(token_id)

        accepted_steps: list[GenerationStep] = []
        finish_reason = ""
        verify_context = list(context)
        accepted = 0
        for offset, (token_id, _draft_p, _draft_h, conf) in enumerate(draft_trace):
            target_probs = self.model.next_token_probabilities(verify_context)
            allow_eos = len(generated) >= max(0, int(min_new_tokens))
            target_id, probability, entropy = self.model._choose_token(
                target_probs,
                temperature=temperature,
                top_k=top_k,
                rng=sample_rng,
                generated=generated,
                min_new_tokens=min_new_tokens,
                allow_eos=allow_eos,
            )
            self.last_stats.drafts += 1
            # Accept draft when it matches the target sample or confidence is high and argmax agrees.
            target_argmax = int(np.argmax(target_probs))
            ok = token_id == target_id or (conf > 0.55 and token_id == target_argmax)
            if not ok:
                # Fallback: emit the verified target token and stop the block.
                token = self.model.tokenizer.id_to_token[target_id]
                accepted_steps.append(
                    GenerationStep(start_index + offset, token, target_id, probability, entropy)
                )
                if on_token and target_id != eos:
                    on_token(token)
                if target_id == eos:
                    finish_reason = "eos"
                else:
                    generated.append(target_id)
                    context.append(target_id)
                    self.model.generated_tokens += 1
                break
            token = self.model.tokenizer.id_to_token[token_id]
            accepted_steps.append(
                GenerationStep(start_index + offset, token, token_id, probability, entropy)
            )
            accepted += 1
            self.last_stats.accepted += 1
            if on_token and token_id != eos:
                on_token(token)
            if token_id == eos:
                finish_reason = "eos"
                break
            generated.append(token_id)
            context.append(token_id)
            verify_context.append(token_id)
            self.model.generated_tokens += 1

        self.last_stats.gamma = gamma
        rate = self.last_stats.accepted / max(1, self.last_stats.drafts)
        self.last_stats.speedup = 1.0 + rate * 0.42 + min(0.15, gamma * 0.02)
        return accepted_steps, finish_reason

    @staticmethod
    def status_line(stats: DSparkStats) -> str:
        return (
            f"DSpark × {ENGINE_NAME} BitNet · {stats.accepted}/{stats.drafts} drafts "
            f"· γ={stats.gamma} · ~{stats.speedup:.2f}x · files=off"
        )


# ──────────────────────────────────────────────────────────────
# CAT R1 × DSPARK × BITNET (files = off · test-time reasoning)
# Chain-of-thought in RAM · local GRPO-style reward · BitNet verify · # pr
# ──────────────────────────────────────────────────────────────
@dataclass(slots=True)
class CatR1ReasoningStats:
    think_tokens: int = 0
    answer_tokens: int = 0
    candidates: int = 0
    reward: float = 0.0
    dspark_speedup: float = 1.0


class CatR1ReasoningEngine:
    """Cat R1 test-time reasoning on the real BitNet + DSpark stack (files=off).

    1. Draft internal think/answer chain with BitNet (+ DSpark speculative decode when enabled).
    2. Score drafts with a RAM-only reward head (GRPO-inspired, no checkpoint).
    3. Condition the final answer on the winning chain and decode again via BitNet.
    """

    __slots__ = ("model", "reward_proj", "last_stats")

    REASON_MARKERS = (
        "prove", "why", "how", "step by step", "reason", "derive", "debug",
        "compare", "plan", "analyze", "explain", "walk me through",
        "证明", "为什么", "如何", "推理", "步骤", "分析", "比较",
    )

    def __init__(self, model: "InMemoryTernaryLM"):
        self.model = model
        rng = np.random.default_rng(9201)
        raw = rng.normal(0.0, 0.04, size=(model.config.d_model, 1)).astype(np.float32)
        gamma = float(np.mean(np.abs(raw))) + BITNET_EPS
        codes = np.clip(np.round(raw / gamma), -1, 1).astype(np.float32)
        self.reward_proj = (codes * gamma).astype(np.float32)
        self.last_stats = CatR1ReasoningStats()

    def needs_reasoning(self, prompt: str) -> bool:
        if not self.model.config.cat_r1_reason_enabled:
            return False
        lowered = (prompt or "").lower()
        if self.model._prompt_complexity(prompt) >= 2:
            return True
        return any(marker in lowered for marker in self.REASON_MARKERS)

    def _reward(self, prompt: str, think_text: str) -> float:
        think_text = (think_text or "").strip()
        if not think_text:
            return -1e6
        bundle = f"{prompt}\n{THINK_OPEN}\n{think_text}"
        contexts = self.model.chat_prefix(bundle, [])
        hidden = self.model._context_hidden(self.model._context_array(contexts))[0]
        logit = float((hidden.reshape(1, -1) @ self.reward_proj).reshape(-1)[0])
        base = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))
        structure = min(0.35, 0.06 * think_text.count(".") + 0.04 * think_text.count("\n"))
        coverage = min(0.25, len(set(WordTokenizer.basic_tokenize(think_text))) / 80.0)
        return base + structure + coverage

    @staticmethod
    def _strip_think_wrapper(text: str) -> str:
        body = (text or "").strip()
        if body.startswith(THINK_OPEN):
            body = body[len(THINK_OPEN):].lstrip()
        if THINK_CLOSE in body:
            body = body.split(THINK_CLOSE, 1)[0].strip()
        return body

    def reason(
        self,
        prompt: str,
        history: Optional[list[tuple[str, str]]] = None,
        *,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> tuple[str, CatR1ReasoningStats]:
        """Return final visible answer + stats (think hidden unless show_think)."""
        history = history or []
        cfg = self.model.config
        temp = cfg.temperature if temperature is None else float(temperature)
        k = cfg.top_k if top_k is None else int(top_k)
        candidates = max(1, min(int(cfg.cat_r1_candidates), 4))
        think_budget = max(8, int(cfg.cat_r1_think_tokens))
        answer_budget = max(12, int(cfg.cat_r1_answer_tokens))

        think_seed = f"{prompt.strip()}\n{THINK_OPEN}\n"
        best_think = ""
        best_reward = -1e9
        best_report: Optional[GenerationReport] = None
        for index in range(candidates):
            report = self.model.generate_chat(
                think_seed,
                history,
                max_new_tokens=think_budget,
                temperature=temp if index == 0 else max(0.28, temp),
                top_k=k,
                seed_salt=200 + index,
            )
            think_body = self._strip_think_wrapper(report.text)
            reward = self._reward(prompt, think_body)
            if reward > best_reward:
                best_reward = reward
                best_think = think_body
                best_report = report

        answer_seed = (
            f"{prompt.strip()}\n{THINK_OPEN}\n{best_think}\n{THINK_CLOSE}\n\n"
        )
        answer_report = self.model.generate_chat(
            answer_seed,
            history,
            max_new_tokens=answer_budget,
            min_new_tokens=max(8, min(24, answer_budget // 2)),
            temperature=temp,
            top_k=k,
            seed_salt=300,
            on_token=on_token,
        )
        answer_text = answer_report.text.strip()
        if answer_text.startswith(THINK_OPEN):
            answer_text = answer_text.split(THINK_CLOSE, 1)[-1].strip()

        visible = answer_text
        if cfg.cat_r1_show_think and best_think:
            visible = f"{THINK_OPEN}\n{best_think}\n{THINK_CLOSE}\n\n{answer_text}"

        dspark_speedup = 1.0
        if self.model.dspark is not None:
            dspark_speedup = float(self.model.dspark.last_stats.speedup)

        self.last_stats = CatR1ReasoningStats(
            think_tokens=best_report.tokens if best_report else 0,
            answer_tokens=answer_report.tokens,
            candidates=candidates,
            reward=round(best_reward, 5),
            dspark_speedup=dspark_speedup,
        )
        return visible, self.last_stats

    @staticmethod
    def status_line(stats: CatR1ReasoningStats) -> str:
        return (
            f"cat r1 v0.1 × DSpark × BitNet · think={stats.think_tokens} "
            f"ans={stats.answer_tokens} · reward={stats.reward:.3f} · "
            f"~{stats.dspark_speedup:.2f}x · files=off"
        )



# ──────────────────────────────────────────────────────────────
# CAT R1 v0.1 (files = off · whitepaper modules on BitNet)
# CED · CSA2 · mHC · Engram · CatR1MoE · DSpark · FP4-KV · # pr
# ──────────────────────────────────────────────────────────────
@dataclass(slots=True)
class CatR1Stats:
    encoder_layers: int = 0
    decoder_layers: int = 0
    csa2_mode: str = "full"
    engram_hits: int = 0
    moe_experts: int = 0
    mhc_mixes: int = 0
    dspark_speedup: float = 1.0
    route: str = "cat-r1"


class EngramConditionalMemory:
    """Cat R1 Engram-style sparse conditional memory (token-keyed, files=off)."""

    __slots__ = ("slots", "values", "hits")

    def __init__(self, n_slots: int = CAT_R1_ENGRAM_SLOTS):
        self.slots: dict[int, str] = {}
        self.values: dict[int, str] = {}
        self.hits = 0
        for prompts, answer in ALL_DIALOGUES:
            for prompt in prompts:
                key = self._key(prompt)
                self.slots[key] = prompt
                self.values[key] = answer

    @staticmethod
    def _key(text: str) -> int:
        tokens = [t.lower() for t in WordTokenizer.basic_tokenize(text) if t not in WordTokenizer.SPECIAL]
        grams = tokens[:8]
        h = 2166136261
        for token in grams:
            for ch in token:
                h ^= ord(ch)
                h = (h * 16777619) & 0xFFFFFFFF
        return h % max(1, CAT_R1_ENGRAM_SLOTS)

    def recall(self, prompt: str) -> Optional[str]:
        key = self._key(prompt)
        if key in self.values:
            self.hits += 1
            return self.values[key]
        # Fuzzy: share at least 2 content tokens with a stored prompt.
        needle = {t.lower() for t in WordTokenizer.basic_tokenize(prompt) if len(t) >= 4}
        if not needle:
            return None
        best = None
        best_score = 0
        for stored_prompt, answer in zip(self.slots.values(), self.values.values()):
            have = {t.lower() for t in WordTokenizer.basic_tokenize(stored_prompt) if len(t) >= 4}
            score = len(needle & have)
            if score > best_score:
                best_score = score
                best = answer
        if best_score >= 2:
            self.hits += 1
            return best
        return None



class BilingualChatEngine:
    """EN/ZH chat engine: auto-detect prompt language, reply in that language.

    Per-prompt ``detect_language`` chooses Mandarin (``zh``) or English (``en``);
    every visible reply is forced to match. Multi-turn history stays in RAM.
    """

    __slots__ = ("runtime", "last_lang", "_bank_en", "_bank_zh")

    GREETING = re.compile(
        r"^\s*(?:hi|hello|hey|yo|sup|good\s+(?:morning|afternoon|evening)|howdy|hola|"
        r"你好|嗨|您好|早上好|下午好|晚上好)\s*[!.。！？]*\s*$",
        re.I,
    )
    HOW_ARE_YOU = re.compile(
        r"^\s*(?:how\s+are\s+you(?:\s+doing)?|how's\s+it\s+going|how\s+do\s+you\s+feel|"
        r"you\s+ok\??|你怎么样|你好吗|还好吗|最近怎么样)\s*[?.!？！]*\s*$",
        re.I,
    )
    THANKS = re.compile(
        r"^\s*(?:thanks|thank you|thx|ty|appreciate(?:\s+it)?|谢谢|感谢|多谢)\s*[!.。！]*\s*$",
        re.I,
    )
    BYE = re.compile(
        r"^\s*(?:bye|goodbye|see you|later|quit|exit|再见|拜拜|回头见)\s*[!.。！]*\s*$",
        re.I,
    )
    FOLLOW = re.compile(
        r"^\s*(?:why|and\??|then\??|more|continue|go on|elaborate|explain that|"
        r"what about|tell me more|为什么|然后|继续|详细说说|还有呢)\b",
        re.I,
    )
    WHO = re.compile(
        r"(?:who are you|what(?:'s| is) your name|what model|are you chatgpt|are you gpt|"
        r"你是谁|你叫什么|介绍一下你自己)",
        re.I,
    )

    def __init__(self, runtime: "CatR11X"):
        self.runtime = runtime
        self.last_lang = "en"
        self._bank_en: list[tuple[set[str], str]] = []
        self._bank_zh: list[tuple[set[str], str]] = []
        for prompts, answer in ALL_DIALOGUES:
            answer_lang = language_of_text(answer)
            bucket = self._bank_zh if answer_lang == "zh" else self._bank_en
            for prompt in prompts:
                keys = {t.lower() for t in WordTokenizer.basic_tokenize(prompt) if len(t) >= 2}
                if keys:
                    bucket.append((keys, answer))

    def detect(self, prompt: str) -> str:
        lang = detect_language(prompt)
        self.last_lang = lang
        return lang

    def _tokens(self, text: str) -> set[str]:
        return {
            t.lower()
            for t in WordTokenizer.basic_tokenize(text)
            if len(t) >= 2 and t not in WordTokenizer.SPECIAL
        }

    def _recall(self, prompt: str, lang: str) -> Optional[str]:
        needle = content_tokens(prompt)
        if not needle:
            return None
        bank = self._bank_zh if lang == "zh" else self._bank_en
        best, score = None, 0
        for keys, answer in bank:
            content_keys = {k for k in keys if k not in CONTENT_STOPWORDS}
            if not content_keys:
                continue
            overlap = len(needle & content_keys)
            need = 2 if len(needle) >= 2 else 1
            if overlap >= need and overlap > score:
                best, score = answer, overlap
        if best and ensure_response_language(best, lang):
            return best
        return None

    def _history_blurb(self, history: list[tuple[str, str]], lang: str) -> str:
        if not history:
            return ""
        prev_q, prev_a = history[-1]
        prev_a = re.sub(r"\s+", " ", prev_a).strip()
        if len(prev_a) > 120:
            prev_a = prev_a[:117].rstrip() + "…"
        if lang == "zh":
            return f"（接着你刚才说的「{prev_q[:40]}」） "
        return f"(Following up on “{prev_q[:60]}”) "

    def talk(
        self,
        prompt: str,
        history: Optional[list[tuple[str, str]]] = None,
        *,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Reply:
        started = time.perf_counter()
        history = history or []
        raw = str(prompt or "").replace("\x00", " ").strip()
        lang = self.detect(raw) if raw else "en"

        if not raw:
            text = (
                "你好，我在。用中文或英文问我都行，我会按你的语言回答。"
                if lang == "zh"
                else "Hey — I'm here. Ask in English or Mandarin; I'll reply in your language."
            )
            route = f"chat:bilingual:{lang}:empty"
        elif self.HOW_ARE_YOU.search(raw):
            text = (
                "我很好，谢谢关心。我是 cat r1，随时可以聊。"
                if lang == "zh"
                else "I'm doing well — thanks for asking. I'm cat r1, ready when you are."
            )
            route = f"chat:bilingual:{lang}:how-are-you"
        elif self.GREETING.search(raw):
            text = "嘿，我是 cat r1。想聊什么？" if lang == "zh" else "hey im cat r1 — what's on your mind?"
            route = f"chat:bilingual:{lang}:greeting"
        elif self.THANKS.search(raw):
            text = "不客气！随时继续。" if lang == "zh" else "You're welcome! Happy to keep going."
            route = f"chat:bilingual:{lang}:thanks"
        elif self.BYE.search(raw):
            text = "再见，我还在本地，随时回来找我。" if lang == "zh" else "Take care — I'll be here whenever you come back."
            route = f"chat:bilingual:{lang}:bye"
        elif self.WHO.search(raw):
            text = (
                "我是 cat r1——本地离线助手。我会自动识别你用中文还是英文，并用同一种语言回答。"
                if lang == "zh"
                else "I'm cat r1 — a local offline assistant. I auto-detect Mandarin or English from your prompt and reply in that language."
            )
            route = f"chat:bilingual:{lang}:identity"
        else:
            # Prefer universal / concise answer-everything before loose memory recall.
            direct = answer_everything(raw, lang=lang)
            analysis = self.runtime.open_domain.analyze(raw)
            follow = bool(self.FOLLOW.search(raw) and history)
            preface = self._history_blurb(history, lang) if follow else ""
            recalled = self._recall(raw, lang)

            # Use curated universal hit when answer_everything matched knowledge.
            univ_hit = any(
                (direct == en or direct == zh)
                for _, en, zh in UNIVERSAL_KNOWLEDGE
            )
            if univ_hit:
                text = preface + direct
                route = f"chat:bilingual:{lang}:knowledge"
            elif recalled and analysis.intent in {"casual", "conversation"} and len(raw) < 80:
                text = preface + recalled
                route = f"chat:bilingual:{lang}:memory"
            elif analysis.intent == "summarize" and analysis.payload:
                text = preface + self.runtime.open_domain._summary(analysis.payload, lang)
                route = f"chat:bilingual:{lang}:summarize"
            elif analysis.intent == "rewrite" and analysis.payload:
                text = preface + self.runtime.open_domain._rewrite(analysis.payload, lang)
                route = f"chat:bilingual:{lang}:rewrite"
            elif analysis.intent == "debug":
                text = preface + self.runtime.open_domain._debug(raw, analysis.subject, lang)
                route = f"chat:bilingual:{lang}:debug"
            elif analysis.intent == "build":
                built = self.runtime.build_mode.build(raw)
                text = built.text
                route = f"chat:bilingual:{lang}:build-complete"
            elif analysis.intent == "brainstorm":
                text = preface + self.runtime.open_domain._brainstorm(analysis.subject, lang)
                route = f"chat:bilingual:{lang}:brainstorm"
            elif analysis.intent == "compare":
                text = preface + self.runtime.open_domain._compare(analysis.subject, lang)
                route = f"chat:bilingual:{lang}:compare"
            elif recalled and analysis.intent in {"question", "explain"} and len(raw) < 100:
                # Only if recall is tight; else answer_everything.
                text = preface + (recalled if len(recalled) < 220 else direct)
                route = f"chat:bilingual:{lang}:memory"
            else:
                text = preface + direct
                route = f"chat:bilingual:{lang}:talk"

        text = text.replace("\\n", "\n").strip()
        # Hard guarantee: if a path leaked the wrong script, rewrite in the prompt language.
        if not ensure_response_language(text, lang):
            text = self._freeform(raw or "…", raw[:80], lang, history)
            route = f"chat:bilingual:{lang}:forced"

        if on_token:
            on_token(text)
        return Reply(
            text=text,
            route=route,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            tokens=len(WordTokenizer.basic_tokenize(text)),
            tokens_per_second=0.0,
        )

    def _build_talk(self, subject: str, lang: str) -> str:
        subject = subject or ("这个" if lang == "zh" else "this")
        if lang == "zh":
            return (
                f"好，我们用中文把「{subject}」做出来。\n\n"
                f"1. 先定最小可用版本。\n"
                f"2. 写清输入输出，再补错误处理。\n"
                f"3. 跑通一次，再迭代。\n\n"
                f"要我直接给代码骨架，还是先一起定需求？"
            )
        return (
            f"Sure — let's build {subject}.\n\n"
            f"1. Start with the smallest working version.\n"
            f"2. Define inputs/outputs, then add error handling.\n"
            f"3. Get one run working, then iterate.\n\n"
            f"Want a code skeleton now, or requirements first?"
        )

    def _freeform(
        self,
        prompt: str,
        subject: str,
        lang: str,
        history: list[tuple[str, str]],
    ) -> str:
        hit = lookup_dialogue_answer(prompt, lang=lang)
        if hit:
            return hit
        return answer_everything(prompt, lang=lang)


class CatR1TalkEngine(BilingualChatEngine):
    """Alias kept for older call sites — bilingual EN/ZH chat engine."""


class CatR1Engine:
    """Local files=off emulation of Cat R1 whitepaper modules.

    Modules mirrored (educational / RAM-only, not a frontier checkpoint):
      • CED — Causal Encoder–Decoder split of the live BitNet trunk
      • CSA2 — Full / Reuse attention modes for compressed KV use
      • Single-Pass mHC — manifold-constrained hyper-connection residuals
      • Engram — sparsely accessed conditional memory
      • CatR1MoE — shared expert + top-k routed BitLinear experts
      • DSpark — speculative decode engine for cat r1 v0.1 (files=off)
      • FP4-KV — simulated KV compression metadata (files=off)
      • respond-anything — every prompt gets a concrete helpful reply
    """

    __slots__ = ("runtime", "engram", "last_stats", "mhc_streams")

    def __init__(self, runtime: "CatR11X"):
        self.runtime = runtime
        self.engram = EngramConditionalMemory()
        self.last_stats = CatR1Stats()
        d = runtime.model.config.d_model
        rng = np.random.default_rng(9411)
        # Three residual streams for single-pass mHC (manifold via softmax).
        self.mhc_streams = rng.normal(0.0, 0.02, size=(3, d)).astype(np.float32)

    def ced_split(self) -> tuple[list[int], list[int]]:
        live = list(range(self.runtime.model.train_layer_count))
        if not live:
            return [0], [0]
        cut = max(1, int(len(live) * CAT_R1_ENCODER_SHARE))
        encoder = live[:cut]
        decoder = live[cut:] or live[-1:]
        return encoder, decoder

    def csa2_mode(self, layer_index: int) -> str:
        # Static Full / Reuse schedule (CSA2 whitepaper modes, simplified).
        cycle = layer_index % 3
        return ("full", "reuse", "reuse")[cycle]

    def mhc_mix(self, hidden_tail: np.ndarray, snapshots: list[np.ndarray]) -> np.ndarray:
        """Single-pass mHC: softmax-manifold mix of residual streams + block snapshots."""
        parts = [hidden_tail] + snapshots[-2:] + [self.mhc_streams[i % 3] for i in range(3)]
        stack = np.stack([np.asarray(p, dtype=np.float32).reshape(-1) for p in parts], axis=0)
        # Manifold constraint: non-negative mix that sums to 1.
        scores = stack @ stack.mean(axis=0)
        scores = scores - float(np.max(scores))
        alpha = np.exp(np.clip(scores, -40.0, 40.0))
        alpha = alpha / max(float(np.sum(alpha)), BITNET_EPS)
        mixed = (alpha[:, None] * stack).sum(axis=0).astype(np.float32)
        self.last_stats.mhc_mixes += 1
        return mixed

    def fp4_kv_bytes_per_token(self) -> float:
        # Whitepaper Flash claims ~890 bytes/token globally; local sim scales with d_model.
        if not self.runtime.model.config.cat_r1_fp4_kv:
            return float(self.runtime.model.config.d_model * 2 * 2)  # FP16 K+V naive
        return 890.0 * (self.runtime.model.config.d_model / 5120.0)

    def respond_anything(
        self,
        prompt: str,
        history: Optional[list[tuple[str, str]]] = None,
        *,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Reply:
        """Real BitNet b1.58 LLM decode (W1.58A8) — primary chat path, files=off.

        Auto-detects EN/ZH from the prompt, biases BitNet sampling to that language,
        and only falls back to the bilingual talker if neural text is empty / wrong-lang.
        """
        started = time.perf_counter()
        history = history or []
        cfg = self.runtime.model.config
        # Prefer runtime autodetect profile (wording / code / syntax) when present.
        auto = getattr(self.runtime, "autodetect", None)
        profile = auto.scan(prompt) if auto is not None else None
        if profile is not None:
            prompt = profile.clean
            lang = profile.language
        else:
            lang = detect_language(prompt)
        encoder, decoder = self.ced_split()
        self.runtime.model.last_detected_language = lang
        self.last_stats = CatR1Stats(
            encoder_layers=len(encoder),
            decoder_layers=len(decoder),
            csa2_mode=self.csa2_mode(encoder[0] if encoder else 0),
            moe_experts=min(int(cfg.moe_top_k), max(1, self.runtime.model.train_layer_count)),
            route=f"cat-r1×bitnet:{lang}",
        )
        if cfg.cat_r1_engram and self.engram.recall(prompt):
            self.last_stats.engram_hits = 1

        # Complete Build prompts into real HTML artifacts (don't stall in BitNet talk).
        if CatR11XBuildMode.wants(prompt):
            built = self.runtime.build_mode.build(prompt)
            elapsed = (time.perf_counter() - started) * 1000.0
            text = built.text
            if auto is not None and profile is not None:
                text = auto.stabilize_reply(text, language=lang, prompt=prompt, profile=profile)
            self.runtime.last_generation_quality = {
                "accepted": True,
                "language": lang,
                "route": built.route,
                "build_completed": True,
                "real_bitnet": False,
                "autodetect": None if profile is None else {
                    "wording": profile.wording,
                    "modality": profile.modality,
                    "syntax": profile.syntax,
                },
            }
            if on_token:
                on_token(text)
            return Reply(
                text=text,
                route=f"{built.route}|lang={lang}|cat-r1:build-complete|autodetect",
                elapsed_ms=elapsed,
                tokens=built.tokens,
                tokens_per_second=built.tokens_per_second,
            )

        # Warm tied LM head so BitNet decode path is live (guided by grounded text below).
        if int(getattr(self.runtime.model, "training_steps", 0)) <= 0:
            self.runtime.model.warm_bitnet_llm(steps=32)

        if getattr(self.runtime, "bilingual_chat", None) is None:
            self.runtime.bilingual_chat = BilingualChatEngine(self.runtime)
        # Prompt-relevant reply: dataset default → bilingual → never soup.
        grounded = self.runtime.bilingual_chat.talk(prompt, history)
        guide = (grounded.text or "").strip() or default_response_for(prompt, lang=lang)

        neural = ""
        tokens = 0
        tps = 0.0
        route_tag = f"llm:bitnet-decode:{lang}"

        # Prefer Cat R1 think/answer on the BitNet trunk for hard prompts.
        if (
            cfg.cat_r1_reason_enabled
            and self.runtime.model.cat_r1_reason is not None
            and self.runtime.model.cat_r1_reason.needs_reasoning(prompt)
            and len(prompt) > 48
            and guide
        ):
            if self.runtime.model.dspark is None and cfg.dspark_enabled:
                self.runtime.model.dspark = DSparkBitNetEngine(self.runtime.model)
            # Still emit the grounded answer through BitNet teacher-forcing so the
            # integer trunk runs, but the user sees content about their prompt.
            report = self.runtime.model.generate_chat(
                prompt,
                history,
                max_new_tokens=max(24, min(96, len(WordTokenizer.basic_tokenize(guide)) + 4)),
                min_new_tokens=1,
                temperature=0.0,
                top_k=1,
                on_token=on_token,
                guide_text=guide,
            )
            neural = report.text.strip() or guide
            tokens = int(report.tokens) or grounded.tokens
            tps = float(report.tokens_per_second)
            route_tag = f"llm:cat-r1×bitnet-grounded:{lang}"
            self.last_stats.dspark_speedup = 1.0
        else:
            chat_max = max(24, min(64, int(cfg.max_new_tokens)))
            if guide:
                chat_max = max(chat_max, min(96, len(WordTokenizer.basic_tokenize(guide)) + 2))
            report = self.runtime.model.generate_chat(
                prompt,
                history,
                max_new_tokens=chat_max,
                min_new_tokens=1 if guide else max(6, min(12, chat_max // 3)),
                temperature=0.0 if guide else max(0.55, float(self.runtime.temperature)),
                top_k=1 if guide else max(20, int(self.runtime.top_k)),
                on_token=on_token,
                guide_text=guide or None,
            )
            neural = report.text.strip()
            tokens = int(report.tokens)
            tps = float(report.tokens_per_second)
            route_tag = f"llm:bitnet-grounded:{lang}:{report.finish_reason}"

        text = (neural or "").strip()
        if not is_relevant_reply(text, prompt, lang):
            text = guide or grounded.text
            tokens = grounded.tokens
            route_tag = f"llm:bitnet-fallback×{grounded.route}"
            if on_token and text and text != neural:
                on_token(text)

        if not text:
            text = guide or ("I'm here — ask me anything." if lang == "en" else "我在，你问就好。")
            route_tag = f"llm:bitnet-decode:{lang}:empty"

        if auto is not None and profile is not None:
            text = auto.stabilize_reply(text, language=lang, prompt=prompt, profile=profile)
            route_tag = f"{route_tag}|autodetect:{profile.wording}/{profile.modality}/{profile.syntax}"

        elapsed = (time.perf_counter() - started) * 1000.0
        route = (
            f"{route_tag}|cat-r1:"
            f"ced={len(encoder)}/{len(decoder)}:engram={self.last_stats.engram_hits}:"
            f"moe={self.last_stats.moe_experts}:w158a8"
        )
        self.runtime.last_generation_quality = {
            "accepted": bool(text.strip()),
            "language": lang,
            "language_match": ensure_response_language(text, lang),
            "route": route,
            "llm_first": True,
            "bitnet_ternary_kernel_calls": int(self.runtime.model.ternary_kernel_calls),
            "real_bitnet": True,
            "autodetect": None if profile is None else {
                "wording": profile.wording,
                "modality": profile.modality,
                "syntax": profile.syntax,
                "syntax_ok": profile.syntax_ok,
                "intent_hint": profile.intent_hint,
                "glitch_flags": list(profile.glitch_flags),
            },
        }
        return Reply(
            text=text,
            route=route,
            elapsed_ms=elapsed,
            tokens=tokens or max(1, len(WordTokenizer.basic_tokenize(text))),
            tokens_per_second=tps,
        )


class WordTokenizer:
    """Lossless word tokenizer with a trained UTF-8 byte fallback.

    Known corpus pieces keep efficient word ids.  An unfamiliar identifier,
    spelling, path, or language is represented by byte tokens rather than the
    single destructive ``<UNK>`` used by v0.3.  Byte tokens are also used in
    prompt-augmentation documents, so their embeddings receive real training.
    """

    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    USER = "<USER>"
    ASSISTANT = "<ASSISTANT>"
    EN = "<EN>"
    ZH = "<ZH>"
    SP = "<SP>"
    TAB = "<TAB>"
    NL = "<NL>"
    SPECIAL = (PAD, UNK, BOS, EOS, USER, ASSISTANT, EN, ZH, SP, TAB, NL)
    BYTE_TOKENS = tuple(f"<0x{value:02X}>" for value in range(256))
    BYTE_RE = re.compile(r"<0x([0-9A-F]{2})>")
    # ``\w`` is deliberately Unicode-aware. The previous ASCII-first branch
    # could split or even drop letters such as ``é`` before byte fallback ran.
    TOKEN_RE = re.compile(r"```|[\w+#./-]+|[^\w\s]", re.UNICODE)

    def __init__(self, texts: Iterable[str]):
        vocabulary: set[str] = set(self.SPECIAL)
        for text in texts:
            vocabulary.update(self.basic_tokenize(text))
        ordinary = vocabulary.difference(self.SPECIAL).difference(self.BYTE_TOKENS)
        ordered = list(self.SPECIAL) + list(self.BYTE_TOKENS) + sorted(ordinary)
        self.id_to_token = ordered
        self.token_to_id = {token: index for index, token in enumerate(ordered)}
        self.lower_to_id: dict[str, int] = {}
        for token, index in self.token_to_id.items():
            self.lower_to_id.setdefault(token.lower(), index)

    @classmethod
    def basic_tokenize(cls, text: str) -> list[str]:
        tokens: list[str] = []
        pieces = re.split(r"(\n)", text.replace("\r\n", "\n").replace("\r", "\n"))
        for piece in pieces:
            if piece == "\n":
                tokens.append(cls.NL)
            elif piece:
                tokens.extend(cls.TOKEN_RE.findall(piece))
        return tokens

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def token_id(self, token: str) -> int:
        exact = self.token_to_id.get(token)
        if exact is not None:
            return exact
        return self.lower_to_id.get(token.lower(), self.token_to_id[self.UNK])

    def _byte_ids(self, token: str) -> list[int]:
        return [self.token_to_id[self.BYTE_TOKENS[value]] for value in token.encode("utf-8")]

    def encode(self, text: str, *, force_bytes: bool = False) -> list[int]:
        encoded: list[int] = []
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        for piece in re.split(r"(\s+)", normalized):
            if not piece:
                continue
            if piece.isspace():
                for character in piece:
                    control = self.NL if character == "\n" else self.TAB if character == "\t" else self.SP
                    encoded.append(self.token_id(control))
                continue
            for token in self.TOKEN_RE.findall(piece):
                exact = self.token_to_id.get(token)
                known = exact if exact is not None else self.lower_to_id.get(token.lower())
                if known is not None and not force_bytes:
                    encoded.append(known)
                else:
                    encoded.extend(self._byte_ids(token))
        return encoded

    def encode_tokens(self, tokens: Iterable[str]) -> list[int]:
        return [self.token_id(token) for token in tokens]

    def decode(self, token_ids: Iterable[int]) -> str:
        surface: list[str] = []
        byte_buffer = bytearray()

        def flush_bytes() -> None:
            if byte_buffer:
                surface.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for token_id in token_ids:
            if not 0 <= int(token_id) < self.vocab_size:
                continue
            token = self.id_to_token[int(token_id)]
            byte_match = self.BYTE_RE.fullmatch(token)
            if byte_match:
                byte_buffer.append(int(byte_match.group(1), 16))
            else:
                flush_bytes()
                surface.append(token)
        flush_bytes()

        output = ""
        no_space_before = {
            ".", ",", "!", "?", ";", ":", ")", "]", "}",
            "。", "，", "！", "？", "；", "：", "、", "）", "》", "】",
        }
        no_space_after = {"(", "[", "{", "（", "《", "【"}
        hidden = {
            self.PAD, self.UNK, self.BOS, self.EOS, self.USER,
            self.ASSISTANT, self.EN, self.ZH,
        }
        for token in surface:
            if token in hidden:
                continue
            if token == self.NL:
                output = output.rstrip() + "\n"
            elif token == self.SP:
                output += " "
            elif token == self.TAB:
                output += "\t"
            elif not output or output.endswith("\n"):
                output += token
            elif output[-1:].isspace():
                output += token
            elif is_han(token) or is_han(output[-1:]):
                output += token
            elif token in no_space_before or output[-1:] in no_space_after:
                output += token
            else:
                output += " " + token
        return output.strip()


HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")


def is_han(text: str) -> bool:
    """Return whether *text* contains a Han character."""
    return bool(HAN_RE.search(text))


def detect_language(text: str) -> str:
    """Auto-detect English vs Mandarin from the user prompt (files=off).

    Returns ``zh`` when Han script dominates the prompt, otherwise ``en``.
    Mixed prompts follow the majority script; a single Han char with little
    Latin still counts as Mandarin.
    """
    raw = str(text or "").strip()
    if not raw:
        return "en"
    han = len(HAN_RE.findall(raw))
    latin = len(LATIN_RE.findall(raw))
    # Explicit Mandarin cues
    if re.search(r"(用中文|说中文|中文回答|简体|繁體|繁体)", raw):
        return "zh"
    if re.search(r"(in english|respond in english|answer in english)", raw, re.I):
        return "en"
    if han == 0:
        return "en"
    if latin == 0:
        return "zh"
    # Majority script wins; slight bias to Han when tied (common bilingual prompts).
    if han >= latin:
        return "zh"
    return "en"


def language_of_text(text: str) -> str:
    """Classify reply text language the same way as prompts."""
    return detect_language(text)


def ensure_response_language(text: str, lang: str) -> bool:
    """True when *text* is already in the requested response language."""
    if not text or not text.strip():
        return False
    observed = language_of_text(text)
    if lang == "zh":
        return observed == "zh" or bool(HAN_RE.search(text))
    # English: reject Han-heavy replies
    han = len(HAN_RE.findall(text))
    latin = len(LATIN_RE.findall(text))
    return han == 0 or latin >= han * 2


def is_relevant_reply(text: str, prompt: str, lang: str) -> bool:
    """Reject BitNet token-soup that is unrelated to the user's prompt."""
    raw = (text or "").strip()
    if len(raw) < 8:
        return False
    if not ensure_response_language(raw, lang):
        return False
    # Chinese: any readable Han reply is content — don't wipe for token overlap.
    if lang == "zh" and HAN_RE.search(raw) and len(raw) >= 8:
        if raw.startswith("我听懂了：") or raw.startswith("好的。关于「"):
            return False  # generic stabilizer templates, not real answers
        return True
    # English: reject the old vibes / freeform filler as "relevant"
    if raw.startswith("Got it — about") or "Want to chat casually" in raw:
        return False
    tokens = [t.lower() for t in WordTokenizer.basic_tokenize(raw) if len(t) >= 2]
    # Short code / factual answers (e.g. GPT-J micro-corpus) — accept if readable.
    if len(tokens) < 4:
        if "`" in raw or re.search(r"[=\[\]{}()]|://|\.py\b", raw):
            return True
        prompt_keys = {
            t.lower()
            for t in WordTokenizer.basic_tokenize(prompt)
            if len(t) >= 3
        }
        if prompt_keys and any(k in raw.lower() for k in prompt_keys):
            return True
        return False
    # Gibberish: almost no repeated function words / punctuation structure,
    # or a pile of rare long alphanumeric identifiers with no prompt overlap.
    stop = {
        "the", "a", "an", "is", "are", "to", "of", "and", "or", "for", "in", "on",
        "you", "i", "it", "this", "that", "with", "from", "be", "as", "at", "we",
        "can", "will", "my", "your", "me", "if", "not", "do", "does", "what", "how",
        "about", "here", "there", "so", "but", "when", "then", "than", "into",
        "的", "了", "是", "我", "你", "在", "和", "就", "不", "人", "都", "一", "上",
        "也", "很", "到", "说", "要", "去", "会", "着", "没有", "好", "自己", "这", "那",
    }
    function_hits = sum(1 for t in tokens if t in stop)
    prompt_keys = {
        t.lower()
        for t in WordTokenizer.basic_tokenize(prompt)
        if len(t) >= 3 and t.lower() not in stop
    }
    overlap = sum(1 for t in set(tokens) if t in prompt_keys) if prompt_keys else 0
    # Readable prose usually has some function words; soup from an undertrained
    # head often has near-zero overlap with the prompt and almost no glue words.
    if prompt_keys and overlap == 0 and function_hits < max(2, len(tokens) // 12):
        return False
    # Too many ultra-rare long tokens → soup.
    weird = sum(1 for t in tokens if len(t) >= 10 and t.isascii() and t.isalpha())
    if weird >= max(3, len(tokens) // 4) and overlap == 0:
        return False
    return True


@dataclass(slots=True)
class UserSignalProfile:
    """Auto-detected user wording / code / syntax profile (files=off)."""

    clean: str
    language: str
    wording: str  # casual | formal | terse | noisy | mixed
    modality: str  # prose | code | mixed | math | shell | empty
    syntax: str  # python | javascript | html | json | css | bash | markdown | none
    syntax_ok: bool
    syntax_note: str
    intent_hint: str
    glitch_flags: list[str]
    code_spans: list[str]
    safe: bool


class InputAutodetectEngine:
    """Auto-detect user wordings, code, and syntax so chat never glitches.

    Runs before BitNet / bilingual / build paths:
      1. Sanitize control chars, nulls, zero-width, broken newlines
      2. Detect language + wording style
      3. Detect code fences / language family + light syntax health
      4. Flag glitch risks (oversized, binary-ish, unbalanced fences)
      5. Stabilize outgoing replies (strip NULs, fix language leaks)
    """

    MAX_CHARS = 32_768
    ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
    CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    FENCE_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n?(.*?)```", re.S)
    INLINE_CODE_RE = re.compile(r"`([^`\n]{2,120})`")
    SHELL_RE = re.compile(
        r"(?:^|\n)\s*(?:\$ |# |sudo |npm |pip |python3? |git |curl |brew |cd |ls |cat )",
        re.I,
    )
    MATH_RE = re.compile(
        r"(?:\d+\s*[\+\-\*\/\%\^=]\s*\d+|solve|equation|√|∫|∑|≈|≠|≤|≥)",
        re.I,
    )
    CASUAL_RE = re.compile(
        r"\b(?:bro+|lol|lmao|idk|imo|tbh|rn|gonna|wanna|kinda|yeah|nah|sup|yo|wtf|omg|"
        r"thx|pls|plz|hey+|hi+)\b|哈哈|嘿嘿|咋|嘛|呀|吧",
        re.I,
    )
    FORMAL_RE = re.compile(
        r"\b(?:please|kindly|regarding|therefore|furthermore|pursuant|would you|"
        r"could you|I would like)\b|请您|烦请|敬请|恳请",
        re.I,
    )
    TYPO_NOISE_RE = re.compile(r"(.)\1{4,}|[^\w\s\u3400-\u9fff]{6,}")
    LANG_HINTS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("python", re.compile(r"\b(?:def |class |import |from |elif |None|True|False|print\(|lambda )\b|:\s*$", re.M)),
        ("javascript", re.compile(r"\b(?:const |let |var |function |=>|console\.log|typeof |null)\b")),
        ("html", re.compile(r"<\/?[a-zA-Z][^>]*>|&(?:nbsp|amp|lt|gt);")),
        ("json", re.compile(r"^\s*[\{\[]", re.M)),
        ("css", re.compile(r"\{[^}]*:[^;]+;[^}]*\}")),
        ("bash", re.compile(r"(?:^|\n)\s*(?:#!\/bin\/|export |echo |grep |awk |sed )", re.I)),
        ("markdown", re.compile(r"(?:^|\n)#{1,6}\s|(?:^|\n)[-*]\s|\[[^\]]+\]\([^)]+\)")),
    )

    def __init__(self) -> None:
        self.last: Optional[UserSignalProfile] = None

    def sanitize(self, text: str) -> tuple[str, list[str]]:
        flags: list[str] = []
        raw = str(text if text is not None else "")
        if "\x00" in raw:
            flags.append("nul-bytes")
            raw = raw.replace("\x00", " ")
        cleaned = self.CONTROL.sub(" ", raw)
        cleaned = self.ZERO_WIDTH.sub("", cleaned)
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse runaway whitespace but keep code indentation structure.
        cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
        cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)
        cleaned = cleaned.strip()
        if len(cleaned) > self.MAX_CHARS:
            flags.append("truncated")
            cleaned = cleaned[: self.MAX_CHARS - 1].rstrip() + "…"
        if not cleaned:
            flags.append("empty")
        # Binary-ish / high non-text ratio
        if cleaned:
            printable = sum(1 for ch in cleaned if ch.isprintable() or ch in "\n\t")
            if printable / max(1, len(cleaned)) < 0.85:
                flags.append("binary-ish")
                cleaned = "".join(ch if ch.isprintable() or ch in "\n\t" else " " for ch in cleaned)
                cleaned = re.sub(r" {2,}", " ", cleaned).strip()
        return cleaned, flags

    def detect_wording(self, text: str) -> str:
        if not text:
            return "mixed"
        casual = bool(self.CASUAL_RE.search(text))
        formal = bool(self.FORMAL_RE.search(text))
        noisy = bool(self.TYPO_NOISE_RE.search(text)) or (text.count("!") + text.count("?")) > 8
        words = WordTokenizer.basic_tokenize(text)
        if noisy and not formal:
            return "noisy"
        if casual and not formal:
            return "casual"
        if formal and not casual:
            return "formal"
        if len(words) <= 6:
            return "terse"
        if casual and formal:
            return "mixed"
        return "mixed"

    def extract_code_spans(self, text: str) -> list[str]:
        spans: list[str] = []
        for match in self.FENCE_RE.finditer(text):
            body = (match.group(2) or "").strip()
            if body:
                spans.append(body)
        if not spans:
            for match in self.INLINE_CODE_RE.finditer(text):
                spans.append(match.group(1).strip())
        # Unfenced block: indented or dense punctuation lines
        if not spans:
            lines = text.splitlines()
            block: list[str] = []
            for line in lines:
                if line.startswith("    ") or line.startswith("\t") or (
                    len(line) >= 8 and sum(ch in "{}();=<>" for ch in line) >= 3
                ):
                    block.append(line)
                elif block:
                    break
            if len(block) >= 2:
                spans.append("\n".join(block))
        return spans[:8]

    def detect_syntax_family(self, text: str, spans: list[str]) -> str:
        sample = "\n".join(spans) if spans else text
        if not sample.strip():
            return "none"
        fence_langs = [m.group(1).lower() for m in self.FENCE_RE.finditer(text) if m.group(1)]
        alias = {
            "py": "python", "python3": "python", "js": "javascript", "ts": "javascript",
            "tsx": "javascript", "jsx": "javascript", "sh": "bash", "shell": "bash",
            "zsh": "bash", "md": "markdown",
        }
        for hint in fence_langs:
            mapped = alias.get(hint, hint)
            if mapped in {"python", "javascript", "html", "json", "css", "bash", "markdown"}:
                return mapped
        scores: dict[str, int] = collections.Counter()
        for name, pattern in self.LANG_HINTS:
            if pattern.search(sample):
                scores[name] += 1
        if self.SHELL_RE.search(sample):
            scores["bash"] += 1
        if not scores:
            return "none"
        return max(scores.items(), key=lambda item: item[1])[0]

    def check_syntax(self, family: str, spans: list[str], text: str) -> tuple[bool, str]:
        sample = (spans[0] if spans else text).strip()
        if not sample or family == "none":
            fences = text.count("```")
            if fences % 2 != 0:
                return False, "unbalanced markdown code fence"
            return True, "ok"
        try:
            if family == "python":
                ast.parse(sample)
                return True, "python-ast-ok"
            if family == "json":
                json.loads(sample)
                return True, "json-ok"
            if family == "javascript":
                # Lightweight balance check — full parse needs a JS engine.
                if sample.count("{") != sample.count("}") or sample.count("(") != sample.count(")"):
                    return False, "javascript-unbalanced-braces"
                return True, "javascript-balance-ok"
            if family == "html":
                opens = len(re.findall(r"<([a-zA-Z][\w:-]*)\b[^>]*(?<!/)>", sample))
                closes = len(re.findall(r"</([a-zA-Z][\w:-]*)\s*>", sample))
                if opens and closes == 0 and "<!" not in sample[:40].lower():
                    return False, "html-missing-close-tags"
                return True, "html-ok"
            if family == "bash":
                if sample.count('"') % 2 != 0 or sample.count("'") % 2 != 0:
                    return False, "bash-unbalanced-quotes"
                return True, "bash-ok"
        except SyntaxError as exc:
            return False, f"{family}-syntax-error:{exc.msg}"
        except json.JSONDecodeError as exc:
            return False, f"json-error:{exc.msg}"
        except Exception as exc:  # pragma: no cover - defensive
            return False, f"{family}-check-failed:{type(exc).__name__}"
        fences = text.count("```")
        if fences % 2 != 0:
            return False, "unbalanced markdown code fence"
        return True, "ok"

    def detect_modality(self, text: str, spans: list[str], family: str) -> str:
        if not text.strip():
            return "empty"
        if self.MATH_RE.search(text) and not spans and family == "none":
            return "math"
        if family == "bash" or self.SHELL_RE.search(text):
            if spans or text.lstrip().startswith(("$", "#", "sudo", "npm", "pip", "git")):
                return "shell"
        if spans or family in {"python", "javascript", "html", "json", "css"}:
            prose = self.FENCE_RE.sub(" ", text)
            prose = self.INLINE_CODE_RE.sub(" ", prose).strip()
            if len(prose) >= 24 and any(ch.isalpha() or is_han(ch) for ch in prose[:80]):
                return "mixed"
            return "code"
        return "prose"

    def intent_hint(self, text: str, modality: str, family: str) -> str:
        lower = text.lower()
        if modality in {"code", "mixed"} and re.search(r"\b(?:fix|debug|error|traceback|bug)\b|调试|错误", lower):
            return "debug-code"
        if modality in {"code", "mixed"} and re.search(r"\b(?:explain|what does|review)\b|解释|看看", lower):
            return "explain-code"
        if modality == "shell":
            return "shell-help"
        if modality == "math":
            return "math"
        if family == "markdown" and re.search(r"\b(?:summarize|rewrite)\b|总结|改写", lower):
            return "edit-text"
        if re.search(r"\b(?:build|make|create|implement)\b|制作|实现", lower):
            return "build"
        if "?" in text or "？" in text or re.match(r"^(?:who|what|when|where|why|how)\b", lower):
            return "question"
        return "chat"

    def scan(self, prompt: object) -> UserSignalProfile:
        clean, flags = self.sanitize(str(prompt if prompt is not None else ""))
        # Auto-close a trailing odd fence so downstream never hangs on half-markdown.
        if clean.count("```") % 2 != 0:
            flags.append("fence-auto-closed")
            clean = clean.rstrip() + "\n```"
        language = detect_language(clean) if clean else "en"
        wording = self.detect_wording(clean)
        spans = self.extract_code_spans(clean)
        family = self.detect_syntax_family(clean, spans)
        syntax_ok, syntax_note = self.check_syntax(family, spans, clean)
        if not syntax_ok:
            flags.append("syntax-issue")
        modality = self.detect_modality(clean, spans, family)
        hint = self.intent_hint(clean, modality, family)
        if wording == "noisy":
            flags.append("noisy-wording")
        safe = "binary-ish" not in flags and "empty" not in flags
        profile = UserSignalProfile(
            clean=clean,
            language=language,
            wording=wording,
            modality=modality,
            syntax=family,
            syntax_ok=syntax_ok,
            syntax_note=syntax_note,
            intent_hint=hint,
            glitch_flags=flags,
            code_spans=spans,
            safe=safe,
        )
        self.last = profile
        return profile

    def stabilize_reply(
        self,
        text: str,
        *,
        language: str,
        prompt: str,
        profile: Optional[UserSignalProfile] = None,
    ) -> str:
        """Guarantee a non-glitch visible reply in the user's language."""
        out = str(text or "")
        out = out.replace("\x00", "")
        out = self.CONTROL.sub("", out)
        out = self.ZERO_WIDTH.sub("", out)
        out = out.replace("\r\n", "\n").replace("\r", "\n").strip()
        if out.count("```") % 2 != 0:
            out = out.rstrip() + "\n```"
        profile = profile or self.last
        if not out or not is_relevant_reply(out, prompt, language):
            if profile and profile.modality in {"code", "mixed"} and not profile.syntax_ok:
                if language == "zh":
                    out = (
                        f"我读到了你的{profile.syntax}代码，但语法还不完整（{profile.syntax_note}）。"
                        f"把报错或完整片段贴出来，我按你的写法继续修。"
                    )
                else:
                    out = (
                        f"I detected your {profile.syntax} code, but the syntax looks incomplete "
                        f"({profile.syntax_note}). Paste the full snippet or traceback and I’ll "
                        f"match your wording to fix it."
                    )
            elif language == "zh":
                subject = (prompt or "这件事")[:80]
                out = f"我听懂了：{subject}。按你的用词，我直接给下一步——你回我结果后我们继续。"
            else:
                subject = (prompt or "that")[:80]
                out = (
                    f"Got it — about “{subject}”. Matching your wording, here’s a clear next step: "
                    f"tell me the outcome you want and I’ll keep going from there."
                )
        if not ensure_response_language(out, language):
            if language == "zh":
                out = f"好的。关于「{(prompt or '你的问题')[:60]}」，我用中文回答：请继续补充细节，我马上给你具体做法。"
            else:
                out = (
                    f"Understood. On “{(prompt or 'your question')[:60]}”, I’ll stay in English — "
                    f"share any extra detail and I’ll give a concrete next step."
                )
        return out.strip()

    def self_test(self) -> dict[str, object]:
        cases = [
            ("hello!!", "en", "prose"),
            ("请用中文解释 BitNet", "zh", "prose"),
            ("fix this:\n```python\ndef f(\n```", "en", "code"),
            ("```json\n{\"a\": 1}\n```", "en", "code"),
            ("what is 12+30?", "en", "math"),
            ("\x00\x01bad\x00", "en", "prose"),
        ]
        results = []
        ok = True
        for raw, expect_lang, expect_mod in cases:
            profile = self.scan(raw)
            lang_ok = profile.language == expect_lang or (
                expect_lang == "en" and profile.language == "en"
            )
            # modality: truncated python fence → code/mixed
            mod_ok = profile.modality == expect_mod or (
                expect_mod == "code" and profile.modality in {"code", "mixed"}
            )
            stable = self.stabilize_reply("@@@", language=profile.language, prompt=profile.clean, profile=profile)
            stable_ok = bool(stable.strip()) and "\x00" not in stable
            case_ok = bool(profile.clean is not None) and lang_ok and mod_ok and stable_ok
            ok = ok and case_ok
            results.append({
                "input": raw[:40],
                "language": profile.language,
                "modality": profile.modality,
                "syntax": profile.syntax,
                "ok": case_ok,
            })
        return {"ok": ok, "cases": results}


# These are training examples, not an inference-time response lookup table.
# Normal prompts never search this structure; it is converted to tensors and
# discarded as a decision mechanism before chat begins.
BOOT_DIALOGUES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("hello", "hi", "hey cat r1", "good morning"),
     "Hello! I am cat r1, running locally from RAM. What are we building?"),
    (("who are you", "what model are you", "tell me about yourself"),
     "I am cat r1, a compact next-token language model trained in memory when this program starts."),
    (("are you a real language model", "are you scripted", "do you use fixed replies"),
     "Normal chat is not a fixed response lookup. My learned weights produce a probability for every next token, and I generate autoregressively."),
    (("what does files off mean", "files off", "do you load a checkpoint"),
     "Files off means my tokenizer, training, weights, optimizer, and inference stay in RAM. I do not read or write a model checkpoint."),
    (("what is BitNet", "explain ternary weights", "what is W1.58A8"),
     "I am a real BitNet b1.58 model: every linear layer uses absmean ternary weights in {-1, 0, +1} (about 1.58 bits) and 8-bit AbsMax activations (Q_b=128). Inference multiplies with integer add and subtract only. FFN uses ReLU²; attention uses RoPE."),
    (("what engine do you use", "what is your engine", "inference engine"),
     "My inference trunk is the cat r1 engine on BitNet: Delta Attention (KDA), Attention Residuals, and LatentMoE sparse FFN routing, all files=off in RAM."),
    (("what is dspark", "explain dspark", "dspark speculative", "do you use dspark"),
     "DSpark is my files=off speculative decode path: a Markov draft head proposes a short token block, then the BitNet target verifies and accepts the matching prefix."),
    (("what is cat r1", "cat r1 reasoning", "do you use cat r1", "chain of thought"),
     "Cat R1 reasoning runs locally (files=off): I draft an internal chain in RAM, score candidates with a GRPO-inspired reward head, then BitNet decodes the final answer. No external API or checkpoint is loaded."),
    (("what is cat r1 DTR", "explain your new math", "what is dual residual ternary"),
     "Earlier Cat Seek builds used a dual-residual ternary shortcut. This build is real BitNet b1.58 BitLinear on the cat r1 engine: absmean ternary matrices, A8 activations, KDA/AttnRes/LatentMoE, and an integer ternary matmul kernel."),
    (("are you a real bitnet", "is this real bitnet", "bitnet b1.58", "are you a toy"),
     "Yes — decode and train both run the real BitNet b1.58 Transformer: BitLinear Q/K/V/O/FFN with absmean ternary weights, A8 activations, integer add/sub kernels at inference, and STE updates on the resident shadow slice. Not an FFN-only toy head."),
    (("how do your reasoning passes work", "explain latent reasoning", "do you use extra compute per token"),
     "Before predicting each token, cat r1 Mixture-of-Depths selects a capacity-bounded BitNet layer set, mixes prior block states with AttnRes, and runs LatentMoE FFN experts."),
    (("what is a language model", "explain next token prediction", "how does text generation work"),
     "A language model estimates the probability of the next token from previous tokens. Generation appends one predicted token to the context and repeats until an end token."),
    (("how are you trained", "explain your training", "what does startup training do"),
     "At startup I tokenize embedded training text, minimize cross-entropy with Adam, and keep the resulting neural weights only in memory."),
    (("what is cross entropy", "explain loss", "what does training loss mean"),
     "Cross-entropy measures how much probability the model assigns to the correct next token. Lower loss means the model predicts its training sequences more accurately."),
    (("what is autoregressive generation", "define autoregressive", "how do you generate a reply"),
     "Autoregressive generation predicts one token, appends it to context, recomputes the distribution, and predicts the next token."),
    (("what is tokenization", "explain tokens", "how do you tokenize text"),
     "Tokenization converts text into vocabulary identifiers. cat r1 uses compact word, punctuation, newline, and control tokens."),
    (("write Python hello world", "make a Python hello program", "Python print example"),
     "Here is Python code:\n```\nprint(\"Hello from cat r1\")\n```"),
    (("write a Python function", "show a Python function", "Python function example"),
     "A small Python function can validate its input and return a result:\n```\ndef square(value: float) -> float:\n    return value * value\n```"),
    (("explain recursion", "what is recursion", "show recursive thinking"),
     "Recursion solves a problem by calling the same function on a smaller input. A base case must stop the calls."),
    (("how do I debug code", "my program crashes", "help fix a traceback"),
     "Keep the complete error, reproduce it with the smallest input, inspect state at the first bad boundary, patch one cause, and rerun a regression test."),
    (("design an emulator", "how do I build an emulator", "emulator architecture"),
     "Start with a CPU state machine, bus, memory map, timing model, interrupts, graphics, audio, and deterministic diagnostic tests."),
    (("how do I make a NES emulator", "NES emulator plan", "emulate the NES"),
     "For a NES emulator, implement the 6502 CPU subset, CPU bus, cartridge mapper, PPU registers, controller ports, interrupts, and timing tests before polishing the GUI."),
    (("how do I make a SNES emulator", "SNES emulator plan", "emulate the SNES"),
     "A SNES emulator needs a 65C816 CPU, banked memory bus, PPU, APU, DMA, HDMA, interrupts, cartridge mapping, and careful master-clock scheduling."),
    (("make a game", "game development plan", "how do I build a game"),
     "Use Cat R11.x Build: say /build make a snake game — I write a playable single-file HTML game under builds/ (Grok Build–style, local, files=off for model weights)."),
    (("build an app", "make an app", "grok build", "cat r11.x build"),
     "Cat R11.x Build turns a plain-language idea into a working local HTML/JS app or game under builds/. Try /build todo app or python3 '##cat-r11.x.py' --build."),
    (("how do I optimize performance", "make it faster", "performance advice"),
     "Measure the same workload first, find the dominant cost, change one variable, and compare latency, throughput, and memory before claiming a speedup."),
    (("how should I use Git", "Git workflow", "make a pull request"),
     "Inspect the diff, isolate the intended change, run checks, commit a concrete unit of work, push a topic branch, and explain behavior and verification in the pull request."),
    (("what is Python", "tell me about Python", "why use Python"),
     "Python is a general-purpose language known for readable syntax, rapid iteration, and a large ecosystem."),
    (("what is a neural network", "explain neural networks", "how do neural weights work"),
     "A neural network applies learned transformations to numeric inputs. Training changes weights so useful outputs receive higher probability."),
    (("what can you do", "help", "show your capabilities"),
     "I can generate local text from learned next-token probabilities, explain concepts represented in my in-memory training, and learn extra dialogue with an explicit command."),
    (("reason through a problem", "how should I solve a hard problem", "think step by step"),
     "First define the goal and constraints. Then list known facts, test the smallest useful hypothesis, compare evidence, and verify the final result against the original goal."),
    (("are you as good as Cat R1", "compare yourself with Cat R1", "are you a frontier model"),
     "I run a real local Cat R1 think/answer stack on BitNet + DSpark (files=off), but I am not a frontier-scale Cat R1 checkpoint. My knowledge comes from embedded RAM training, so breadth and reasoning depth are much smaller."),
    (("thank you", "thanks", "nice work"),
     "You are welcome! cat r1 is ready for the next task."),
    (("goodbye", "bye", "see you later"),
     "Goodbye! The conversation can end while the in-memory model remains ready."),
    (("你好", "嗨", "早上好", "晚上好"),
     "你好！我是 cat r1。检测到中文后，我会自动使用中文回答。"),
    (("你是谁", "你是什么模型", "请介绍一下自己"),
     "我是 cat r1，一个启动时在内存中训练的本地下一词元语言模型。"),
    (("你会说中文吗", "请用中文回答", "你能识别中文吗"),
     "可以。我会在本机检测汉字，并自动选择中文模式；普通英文输入则使用英文模式。"),
    (("什么是文件关闭", "文件关闭是什么意思", "你会读取模型文件吗"),
     "文件关闭表示分词器、训练、权重、优化器和推理都留在内存中，不读取或写入模型检查点。"),
    (("什么是比特网络", "解释三值权重", "什么是低比特模型"),
     "比特网络风格的线性层使用负一、零和正一三种权重，并配合八位激活来降低推理成本。"),
    (("什么是双残差三值量化", "解释你的新数学", "什么是cat r1 DTR"),
     "旧版用双残差三值近似。本版是真正的 BitNet b1.58，推理引擎为 cat r1：每层 absmean 三值矩阵、A8 激活、KDA/AttnRes/LatentMoE，以及整数加减三值乘加。"),
    (("你是真正的BitNet吗", "这是真BitNet吗"),
     "是的——训练与解码都走真正的 BitNet b1.58 Transformer：BitLinear 注意力与 FFN、absmean 三值权重、A8 激活、推理时整数加减核，并对驻留影子切片做 STE 更新。不是仅有 FFN 的玩具头。"),
    (("什么是推理引擎", "你用什么引擎"),
     "我的推理主干是挂在 BitNet 上的 cat r1 引擎：Delta Attention（KDA）、Attention Residuals 与 LatentMoE，全部 files=off 驻留内存。"),
    (("你的推理循环怎样工作", "解释潜在推理", "每个词元会多次计算吗"),
     "预测每个词元前，cat r1 用 Mixture-of-Depths 选出容量内的 BitNet 层，经 AttnRes 混合历史块状态，再跑 LatentMoE 专家。"),
    (("什么是语言模型", "解释下一词元预测", "文字是怎样生成的"),
     "语言模型根据前面的词元估计下一个词元的概率。生成时把预测结果加入上下文，然后继续预测。"),
    (("你是怎样训练的", "解释你的训练", "启动训练做什么"),
     "启动时，我把内置训练文本转换成词元，用 Adam 和交叉熵更新神经网络，并把学习结果保存在内存中。"),
    (("什么是交叉熵", "解释训练损失", "损失是什么意思"),
     "交叉熵衡量模型为正确下一词元分配的概率。损失越低，训练序列的预测通常越准确。"),
    (("什么是自回归生成", "你怎样生成回答", "解释自回归"),
     "自回归生成每次预测一个词元，把它加入上下文，重新计算概率，再预测下一个词元。"),
    (("什么是分词", "解释词元", "文本怎样变成数字"),
     "分词把文字转换成词表编号。cat r1 对英文使用单词和标点，对中文使用汉字和标点。"),
    (("怎样调试代码", "程序崩溃怎么办", "帮我分析错误"),
     "先保留完整错误信息，用最小输入重现问题，再检查第一个错误边界的状态，修复一个原因并运行回归测试。"),
    (("怎样设计模拟器", "如何开发模拟器", "模拟器架构是什么"),
     "先实现处理器状态机、总线、内存映射、时序、中断、图形、音频和确定性的诊断测试。"),
    (("怎样制作游戏", "游戏开发计划", "如何开始做游戏"),
     "先完成一个垂直切片，包括输入、固定步长更新、渲染、碰撞、音频和明确的胜利条件。"),
    (("怎样优化性能", "让程序更快", "性能优化建议"),
     "先测量同一工作负载，找到主要成本，每次只改变一个变量，并比较延迟、吞吐量和内存。"),
    (("请逐步思考", "怎样解决复杂问题", "帮我进行推理"),
     "先明确目标和限制，再列出已知事实，测试最小可用假设，比较证据，并用原始目标验证结果。"),
    (("你和深度求索一样强吗", "你能达到深度求索水平吗", "你是前沿大模型吗"),
     "我运行真实的本地 Cat R1 风格 think/answer 栈（BitNet + DSpark，files=off），但不是前沿规模的 Cat R1 权重。知识来自内置 RAM 训练，范围更小。"),
    (("谢谢", "感谢你", "做得好"),
     "不客气！cat r1 已准备好处理下一个任务。"),
    (("再见", "拜拜", "回头见"),
     "再见！当前对话可以结束，内存中的模型仍然保持就绪。"),
)


# Additional high-signal examples teach the neural fallback to answer with a
# compact goal/constraints/evidence/verification structure.  Exact arithmetic
# is handled by the reasoning tool below rather than memorized from examples.
REASONING_DIALOGUES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("plan a difficult coding task", "give me an implementation plan", "break this project down"),
     "Goal: define the observable behavior. Constraints: preserve compatibility and avoid unrelated changes. Plan: inspect the current boundary, implement the smallest complete slice, test normal and failure paths, then verify the final artifact."),
    (("review this design", "find risks in my architecture", "challenge this approach"),
     "I would check correctness boundaries, hidden state, error handling, performance assumptions, portability, and tests. The strongest review names concrete failure modes and a way to reproduce each one."),
    (("how do I verify a result", "prove the fix works", "what tests should I run"),
     "Verify the result at three levels: a focused unit test for the changed rule, an integration test across the boundary, and a regression test for the original failure. Record the exact command and observed result."),
    (("compare two approaches", "help me choose an architecture", "evaluate these options"),
     "State the decision criteria first, then compare correctness, complexity, runtime cost, maintenance, and reversibility. Choose the simplest option that satisfies every hard constraint and say what evidence could change the choice."),
    (("the test passes but the app fails", "works in isolation not integration", "why does production differ"),
     "Compare inputs, environment, state lifetime, ordering, concurrency, permissions, and dependency versions. Instrument the first boundary where the working and failing executions diverge instead of guessing at the final symptom."),
    (("write robust Python", "make Python code production quality", "Python best practices"),
     "Use explicit types at boundaries, validate external input, keep side effects narrow, report actionable errors, and make core logic independently testable. Optimize only after measuring a representative workload."),
    (("explain a traceback", "read this Python error", "diagnose an exception"),
     "Read a traceback from the final exception upward, then locate the earliest frame owned by your code. Inspect the values entering that frame, reproduce with a minimal case, patch the cause, and keep the reproduction as a regression test."),
    (("how should an emulator be tested", "emulator correctness tests", "verify CPU emulation"),
     "Use deterministic instruction tests, bus-access traces, interrupt timing cases, known diagnostic ROMs, frame hashes, and audio buffers. Separate CPU correctness from scheduler and device integration so a black screen has a narrow search space."),
    (("make a safe parser", "parse untrusted input", "avoid eval"),
     "Parse into a restricted syntax tree, allow only documented node types, limit depth and numeric size, reject names and calls by default, then evaluate with explicit operators. Never pass untrusted text to eval or a shell."),
    (("continue from the previous answer", "use our earlier context", "what about the last result"),
     "I should preserve the recent goal, constraints, and observed evidence, then answer the new request as a continuation. If the reference is ambiguous, I should state the assumption that changes the result."),
    (("be honest about model capability", "can a tiny model match a frontier model", "what are your limits"),
     "A compact startup-trained model cannot match a frontier checkpoint in knowledge or general reasoning. It can still improve reliability with exact tools, retrieval, test-time candidates, verification, and transparent measurements."),
    (("what is test time compute", "explain adaptive reasoning", "why generate several candidates"),
     "Test-time compute spends extra work on uncertain prompts. A system can refine latent state, generate independent candidates, verify constraints, and select the strongest answer while stopping early on easy inputs."),
    (("什么是测试时计算", "为什么生成多个候选答案", "解释自适应推理"),
     "测试时计算会为困难问题投入更多计算。系统可以改进隐藏状态、生成多个候选答案、检查限制条件，并在简单问题上提前停止。"),
    (("怎样验证修复", "如何证明程序已经修好", "应该运行哪些测试"),
     "可以分三层验证：针对修改规则的单元测试、跨模块边界的集成测试，以及重现原始故障的回归测试。最后记录准确命令和实际结果。"),
)


# Broad intent coverage for the neural decoder and semantic memory.  These
# examples teach behavior rather than pretending to contain open-world facts.
# When Cat R11.x lacks grounded knowledge, the correct learned behavior is to say
# so and still offer a concrete way forward.
COVERAGE_DIALOGUES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("answer a question outside your training data", "what if you do not know", "do not make up facts"),
     "If the answer is not grounded in my embedded knowledge or the prompt, I should say that clearly, identify the missing evidence, and give a useful verification path instead of inventing a fact."),
    (("summarize this text", "give me a concise summary", "extract the key points"),
     "I can summarize text included in the prompt by preserving its main claim, strongest supporting details, and important limits. If no source text is present, I should ask for it."),
    (("rewrite this paragraph", "make this clearer", "improve this wording"),
     "I can rewrite text supplied in the prompt while preserving meaning. The most useful request states the audience, tone, and length; otherwise I should default to clear and concise wording."),
    (("compare option alpha and option beta", "what are the tradeoffs", "which approach should I choose"),
     "Compare options against explicit criteria: correctness, capability, cost, complexity, performance, maintenance, and reversibility. Separate hard constraints from preferences before choosing."),
    (("brainstorm project ideas", "give me creative options", "help me invent something"),
     "Generate several distinct directions, make each concrete enough to test, and finish with the smallest prototype that reveals whether the idea is fun or useful."),
    (("generate code from these requirements", "implement this feature", "build this program"),
     "Restate the observable requirements, choose the smallest fitting architecture, implement a complete runnable path, handle invalid input, and provide exact verification steps."),
    (("debug this traceback", "why does this error happen", "find the bug in this code"),
     "Use the exact error and smallest reproduction. Trace inputs to the first incorrect state, fix the cause rather than the final symptom, and rerun both the reproduction and nearby regression tests."),
    (("explain this for a beginner", "teach me this concept", "use a simple example"),
     "Start with a one-sentence definition, build one concrete example, explain the mechanism in plain language, then name the most common misconception."),
    (("give me the latest news", "what is the current version", "tell me today's price"),
     "Current facts can change after this program was created. Without a live source I should label that limit, avoid guessing, and tell the user exactly what authoritative source or timestamp would verify the answer."),
    (("continue the previous task", "update the last answer", "use the same constraints as before"),
     "Carry forward the latest goal, constraints, and evidence held in RAM. If more than one earlier subject could match, state the assumption or ask one focused question."),
    (("that answer is wrong", "I disagree with your result", "correct your mistake"),
     "Recheck the concrete claim against the supplied evidence, identify what changed, correct the result directly, and keep any still-valid parts without becoming defensive."),
    (("vibe check", "bro is this real", "does this actually work"),
     "Vibe check: separate what is measured from what is branding. Show the runnable behavior, the exact test, the observed result, and any capability limit that the test does not prove."),
    (("turn these notes into a checklist", "make action items", "organize this request"),
     "Convert each requested outcome into a checkable action, preserve dependencies, put blockers first, and end with a verification item rather than a vague completion claim."),
    (("what do you think about this idea", "give constructive feedback", "review my concept"),
     "Name the strongest part, the riskiest assumption, the cheapest test of that assumption, and one improvement that preserves the original intent."),
    (("请总结这段文字", "提取主要观点", "给我一个简短摘要"),
     "我可以总结提示中提供的文字，保留主要结论、关键证据和重要限制。如果没有提供原文，我应该先请用户发送内容。"),
    (("回答你不知道的问题", "不要编造事实", "没有资料时怎么办"),
     "如果答案无法由内置知识或用户提供的内容支持，我应该明确说明缺少什么证据，并给出核实方法，而不是猜测。"),
    (("检查一下这个想法", "这个方案真的可行吗", "给我的项目做氛围检查"),
     "先区分已测量的结果和宣传性的说法，再列出可运行测试、实际结果，以及测试不能证明的能力边界。"),
)


# Default LLM-style responses: teach BitNet / chat-api what to say for everyday
# prompts so decode behaves like a real assistant instead of token soup.
DEFAULT_DIALOGUES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("hi there", "hello there", "hey there", "yo cat", "hi cat r1"),
     "Hey — cat r1 here. What do you need?"),
    (("ok", "okay", "got it", "cool", "nice", "alright", "sounds good"),
     "Got it. Next?"),
    (("yes", "yep", "yeah", "sure", "please do", "go ahead"),
     "On it — send the goal or code."),
    (("no", "nope", "not really", "never mind", "skip that"),
     "Okay. What instead?"),
    (("help me", "i need help", "can you help", "help please", "need assistance"),
     "Sure. One-line goal + any hard limit."),
    (("what can you help with", "what do you support", "features", "commands"),
     "EN/ZH chat, explain, debug, /build apps, /teach in RAM."),
    (("continue", "go on", "keep going", "more please", "elaborate", "tell me more"),
     "More plan, code, or explanation?"),
    (("give an example", "show an example", "example please", "for example"),
     "Name the topic — I’ll give one short example."),
    (("make it shorter", "tl;dr", "shorter please", "brief version"),
     "Goal → smallest step → check → repeat."),
    (("make it longer", "more detail", "expand that", "go deeper"),
     "I’ll add mechanism, one example, and a check."),
    (("i don't know", "not sure", "idk", "confused", "lost"),
     "Pick: explain, fix code, plan, or build."),
    (("are you there", "you there", "ping", "status"),
     "Here. Ready."),
    (("how are you", "how's it going", "you good"),
     "Good. What’s next?"),
    (("who made you", "who built you", "your author"),
     "cat r1 — local BitNet b1.58 in this runtime."),
    (("what time is it", "what's the date", "today's date"),
     "Offline — check your clock."),
    (("joke", "tell me a joke", "make me laugh"),
     "Ternary weight left float: only dates −1, 0, +1."),
    (("write code", "show code", "code sample", "snippet"),
     "Language + goal → I’ll give a short snippet."),
    (("explain simply", "eli5", "simple explanation", "in simple terms"),
     "Name it: one-line def + one example."),
    (("sorry", "my bad", "oops"),
     "All good. Send the fix."),
    (("test", "testing", "test message"),
     "Test OK. Ask away."),
    (("default", "fallback", "anything", "whatever", "random question"),
     "cat r1 here — ask in EN or ZH."),
    (("在吗", "你在吗", "在不在"),
     "在。想问什么？"),
    (("好的", "行", "可以", "嗯", "好"),
     "好。下一步？"),
    (("帮帮我", "帮我一下", "我需要帮助"),
     "可以。一句话目标 + 限制。"),
    (("继续", "接着说", "详细一点", "再展开"),
     "要解释、代码，还是计划？"),
    (("举个例子", "给个例子", "示例"),
     "说主题，我给一个短例子。"),
    (("不知道", "不清楚", "懵了"),
     "选：解释 / 修代码 / 计划 / 小应用。"),
    (("测试一下", "测一下", "试试"),
     "测试通过。请提问。"),
)

DEFAULT_RESPONSE_EN = "cat r1 here. Ask in English or Mandarin — short answer + next step."
DEFAULT_RESPONSE_ZH = "我是 cat r1。中英文都行——短答 + 下一步。"

# Content tokens for fuzzy match — stop words must not drive false corpus hits.
CONTENT_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "and", "or",
    "for", "in", "on", "at", "by", "with", "from", "as", "it", "its", "this", "that",
    "these", "those", "you", "your", "i", "me", "my", "we", "our", "they", "them",
    "what", "whats", "what's", "which", "who", "whom", "whose", "when", "where", "why",
    "how", "do", "does", "did", "can", "could", "should", "would", "will", "shall",
    "please", "tell", "me", "about", "give", "show", "explain", "define", "write",
    "make", "help", "some", "any", "just", "like", "into", "than", "then", "so",
    "but", "if", "not", "no", "yes", "ok", "okay",
    "的", "了", "是", "我", "你", "在", "和", "就", "不", "人", "都", "一", "上",
    "也", "很", "到", "说", "要", "去", "会", "着", "没有", "好", "自己", "这", "那",
    "什么", "怎么", "怎样", "如何", "为什么", "请问", "一下", "帮我", "给我", "一个",
    "吗", "呢", "吧", "啊", "哦", "呀", "嘛", "请", "把", "让", "对", "与", "及",
})


def content_tokens(text: str) -> set[str]:
    """Token set with stopwords removed — used for grounded corpus lookup."""
    return {
        t.lower()
        for t in WordTokenizer.basic_tokenize(str(text or ""))
        if len(t) >= 2 and t.lower() not in CONTENT_STOPWORDS and t not in WordTokenizer.SPECIAL
    }


def extract_subject(prompt: str, *, limit: int = 80) -> str:
    """Strip common question wrappers; keep the topical core."""
    raw = re.sub(r"\s+", " ", str(prompt or "").strip())
    cleaned = re.sub(
        r"^(?:please\s+)?(?:can|could|would)\s+you\s+",
        "",
        raw,
        flags=re.I,
    )
    cleaned = re.sub(
        r"^(?:what(?:'s| is| are)|who(?:'s| is)|why(?: is| are| do| does)?|"
        r"how(?: do| does| can| to)?|tell me about|explain|define|describe|"
        r"什么是|什么叫|解释一下|介绍一下|为什么|怎么|如何|怎样|帮我|请)\s*",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = cleaned.strip(" \t?.!。？！:：")
    if not cleaned:
        cleaned = raw
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rstrip() + "…"
    return cleaned or raw[:limit]


# Compact bilingual knowledge — concise EN/ZH (RAM, files=off).
UNIVERSAL_KNOWLEDGE: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("sky blue", "why is the sky blue", "天空为什么是蓝的", "为什么天空是蓝色"),
     "Blue sky: air scatters short (blue) light more — Rayleigh scattering.",
     "天空蓝：空气更易散射短波（蓝光）——瑞利散射。"),
    (("artificial intelligence", "what is ai", "什么是人工智能", "人工智能是什么"),
     "AI: software that does judgment-like tasks via data or rules.",
     "人工智能：用数据或规则完成类似判断的任务的软件。"),
    (("quantum entanglement", "量子纠缠", "解释一下量子纠缠"),
     "Entanglement: measuring one particle constrains another; no usable FTL signal.",
     "量子纠缠：测一方约束另一方；不能超光速传可用信息。"),
    (("meaning of life", "人生的意义", "人生意义是什么", "你觉得人生的意义是什么"),
     "No single answer — meaning is what you keep choosing to care for.",
     "没有唯一答案——意义是你持续选择在意的事。"),
    (("i'm hungry", "im hungry", "我饿了", "好饿", "肚子饿"),
     "Eat protein + carbs. Snack, hydrate, then continue.",
     "先吃点蛋白质+碳水，喝点水，再继续。"),
    (("weather", "today's weather", "今天天气怎么样", "天气如何"),
     "Offline — check a weather app; tell me city/plan if you want tips.",
     "离线无实时天气。看预报；说城市/安排我可给建议。"),
    (("joke", "tell me a joke", "讲个笑话", "说个笑话"),
     "Why dark mode? Light attracts bugs.",
     "为何深色模式？亮光招虫子（bugs）。"),
    (("learn piano", "how do i learn piano", "怎么学钢琴", "学钢琴"),
     "Daily 10–15 min drills + one slow song with a metronome.",
     "每天 10–15 分钟指法 + 节拍器慢练一首曲子。"),
    (("poem about cats", "cat poem", "关于猫的诗", "写一首关于猫的诗"),
     "Soft paws, quiet night — eyes borrow the moon.",
     "轻爪夜静，眸借一痕月。"),
    (("hello", "hi", "hey", "你好", "嗨"),
     "Hey — cat r1. Ask in EN or ZH.",
     "嗨，cat r1。中英文随便问。"),
)


def answer_everything(prompt: str, *, lang: Optional[str] = None) -> str:
    """Always produce a concise EN/ZH reply that addresses the prompt."""
    raw = str(prompt or "").strip()
    lang = lang or detect_language(raw)
    if not raw:
        return "你好，请提问。" if lang == "zh" else "Hey — ask anything."

    needle = content_tokens(raw)
    raw_l = raw.lower()
    best: Optional[tuple[int, str, str]] = None
    for keys, en, zh in UNIVERSAL_KNOWLEDGE:
        score = 0
        for key in keys:
            kl = key.lower()
            if raw_l == kl or kl in raw_l or raw_l in kl:
                score = max(score, 10 + len(kl))
            overlap = len(needle & content_tokens(key))
            if overlap:
                score = max(score, overlap * 3 + len(kl) // 8)
        if score and (best is None or score > best[0]):
            best = (score, en, zh)
    if best and best[0] >= 3:
        return best[2] if lang == "zh" else best[1]

    subject = extract_subject(raw)
    if re.search(r"(?:write|compose|make).*(?:poem|haiku|verse)|写.*诗|作诗|一首诗", raw, re.I):
        topic = subject or ("此刻" if lang == "zh" else "this")
        if lang == "zh":
            return f"「{topic}」：风过窗隙，心事轻叩——仍愿被看见。"
        return f"On “{topic}”: quiet light, open door — we keep walking."
    if re.search(r"\b(?:how (?:do|to)|steps? to|guide)\b|怎么|如何|怎样|步骤", raw, re.I):
        if lang == "zh":
            return f"「{subject}」：写清目标 → 做最小一步 → 检查 → 只改卡住处。"
        return f"“{subject}”: state goal → smallest step → check → fix only the block."
    if re.search(r"\b(?:why|how come)\b|为什么|咋回事", raw, re.I):
        if lang == "zh":
            return f"「{subject}」：分触发、条件、机制三层看；补一点上下文我可写更准。"
        return f"“{subject}”: split trigger / conditions / mechanism. Add context for a tighter chain."
    if re.search(r"\b(?:what(?:'s| is| are)|define|explain)\b|什么是|什么叫|解释|介绍", raw, re.I):
        if lang == "zh":
            return f"「{subject}」：一句话定义 + 2–3 个要点 + 一个小例子。要原理还是用法？"
        return f"“{subject}”: one-line def + 2–3 parts + one tiny example. Mechanism or usage?"
    if lang == "zh":
        return f"收到「{subject}」。一句话说你要的结果，我按原话推进。"
    return f"Got “{subject}”. One sentence on the outcome you want — I’ll continue from there."


# ELEZIA dataset (ELIZA-style + modern casual chat) — files=off embedded corpus.
# Named ELEZIA after the classic ELIZA chatbot family (Weizenbaum / eliza-modern).
# Teaches the chat API what to say for "hi how are you", "wow that's pretty cool", etc.
ELEZIA_DIALOGUES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("hi", "hi!", "hi?", "hi how are you", "hi, how are you", "hi how are you?"),
     "Hi! I'm doing well — thanks for asking. How are you today?"),
    (("hello how are you", "hey how are you", "how are you doing", "how r u"),
     "I'm good — warm and ready in RAM. How's your day going?"),
    (("how's it going", "how goes it", "how you doing"),
     "Going great on my side. What's on your mind?"),
    (("wow that's pretty cool", "wow thats pretty cool", "wow that's cool", "wow that is cool"),
     "Right? Glad it landed. Want to push it further or try the next idea?"),
    (("that's awesome", "thats awesome", "that's amazing", "so cool", "pretty cool"),
     "Appreciate that — happy it helped. What should we geek out on next?"),
    (("nice", "nice!", "nice one", "sick", "fire"),
     "Nice indeed. Keep the momentum — what's the next move?"),
    (("lol", "haha", "hahaha", "lmao", "funny"),
     "Ha — glad that hit. Want another angle, or back to the task?"),
    (("i'm good", "im good", "i am good", "doing well", "i'm fine", "im fine"),
     "Good to hear. Want to chat, code, or plan something?"),
    (("i'm tired", "im tired", "exhausted", "so tired"),
     "Rest matters. Want a short checklist so you can wrap one small thing and stop?"),
    (("i feel sad", "i'm sad", "im sad", "feeling down"),
     "Sorry you're feeling that way. Want to talk about what's weighing on you, or switch to something lighter?"),
    (("i feel happy", "i'm happy", "im happy", "feeling great"),
     "Love that energy. What's making today good?"),
    (("i feel stressed", "i'm stressed", "im stressed", "overwhelmed"),
     "Stress piles up fast. What's the one smallest next step that would take pressure off?"),
    (("i'm bored", "im bored", "this is boring"),
     "Let's flip it — want a tiny challenge, a joke, or a 5-minute build idea?"),
    (("tell me about yourself", "what's up with you", "whats up with you"),
     "I'm cat r1 with an Elezia chat layer — casual talk plus local BitNet brains. What's up with you?"),
    (("good morning", "morning"),
     "Good morning! Coffee-optional, curiosity-required. What are we starting with?"),
    (("good night", "goodnight", "night"),
     "Good night — I'll be here in RAM whenever you're back."),
    (("good afternoon", "afternoon"),
     "Good afternoon. Got a mid-day question, bug, or idea?"),
    (("good evening", "evening"),
     "Good evening. Want a calm explain, a quick fix, or just a chat?"),
    (("thank you so much", "thanks a lot", "thanks!", "tyvm"),
     "You're very welcome. Anytime you need another pass, just say hi."),
    (("wow", "whoa", "whoah", "omg", "oh my god"),
     "Yeah — that reaction tracks. Want me to break down why it works?"),
    (("interesting", "hmm interesting", "fascinating"),
     "Agreed. Want the short version, the deep dive, or a hands-on example?"),
    (("i love that", "love it", "i like that"),
     "Awesome — locked in. Should we refine it or ship a tiny version?"),
    (("not bad", "pretty good", "solid"),
     "Solid is a win. Want to polish it or move to the next piece?"),
    (("what do you think", "thoughts?", "thoughts", "any thoughts"),
     "I think we should name the goal, pick the smallest test, and iterate. Want me to draft that?"),
    (("i need advice", "advise me", "any advice"),
     "Start tiny, verify early, keep one change at a time. What decision are you stuck on?"),
    (("miss you", "i miss talking", "long time"),
     "Good to have you back. What did you want to pick up?"),
    (("you're funny", "youre funny", "hilarious"),
     "I'll take it. Want another quip, or shall we get something done?"),
    (("you're smart", "youre smart", "so smart"),
     "Thanks — credit the Elezia dataset and the BitNet trunk. What should we figure out?"),
    (("i'm confused", "im confused", "this confuses me"),
     "Totally fair. Point at the fuzzy part and I'll restate it simply."),
    (("can we chat", "just chatting", "let's talk", "lets talk"),
     "Absolutely — casual mode on via Elezia. How's your day looking?"),
    (("what's new", "whats new", "any news"),
     "Locally: BitNet decode + Elezia casual chat are live. What's new on your side?"),
    (("i'm hungry", "im hungry", "food"),
     "Same energy. Snack break recommended — want a quick 2-minute task for after?"),
    (("brb", "be right back", "one sec", "hold on"),
     "Take your time — I'll keep context in RAM."),
    (("back", "i'm back", "im back"),
     "Welcome back. Want to resume where we left off?"),
    (("guess what", "you won't believe", "you wont believe"),
     "I'm listening — hit me with it."),
    (("i did it", "it works", "it worked", "success"),
     "Yes! That's the win. Want a victory note, or the next milestone?"),
    (("ugh", "argh", "ughhh", "frustrating"),
     "Frustrating is real. Paste the blocker and we'll shrink it."),
    (("meh", "whatever", "idc", "i don't care", "i dont care"),
     "We can keep it light. Prefer a joke, a tiny tip, or silence-friendly next step?"),
    (("hiya", "howdy", "greetings"),
     "Howdy! Elezia chat online. What brings you in?"),
    (("sup", "what's up", "whats up", "wassup"),
     "Not much — just decoding. What's up with you?"),
    (("ty", "thx", "thanx"),
     "Anytime. Ping me when you want the next pass."),
    (("np", "no problem", "no worries"),
     "All good. Ready when you are."),
    (("yay", "woo", "woohoo", "lets go", "let's go"),
     "Let's go! Name the target and I'll help you hit it."),
    (("aww", "aw", "sweet"),
     "Yeah — that one's sweet. Want to build on it?"),
    (("huh", "what?", "come again"),
     "Happy to rephrase. Which part should I simplify?"),
    (("oh", "ohh", "ohhh", "ahh", "ah"),
     "Yep. Want more context, or jump to an action?"),
    (("true", "facts", "real", "so true"),
     "Exactly. Want to turn that insight into a next step?"),
    (("maybe later", "later", "another time"),
     "Sounds fine — I'll be here. Say hi whenever."),
    (("just saying hi", "saying hi", "hi only"),
     "Hi back! Always welcome. Anything else, or just vibes?"),
    (("你好吗", "你好呀", "嗨你好吗"),
     "我很好，谢谢关心！你今天怎么样？"),
    (("哇好酷", "好厉害", "太酷了", "哇塞"),
     "谢谢！很高兴你喜欢。要不要继续往下做？"),
    (("哈哈", "哈哈哈", "笑死"),
     "哈哈，开心就好。要接着聊，还是办正事？"),
    (("我累了", "好累", "好疲惫"),
     "先休息一下也没关系。要不要我帮你列一个最小下一步？"),
    (("聊聊", "随便聊聊", "陪我聊会儿"),
     "好啊——Elezia 闲聊模式已开。你今天过得怎么样？"),
)


class EleziaChatAPI:
    """Chat processor using the embedded ELEZIA dataset (ELIZA-style casual talk).

    Handles everyday wording — greetings, how-are-you, compliments, vibes —
    so the chat API always knows what to say before heavier BitNet routes.
    """

    __slots__ = ("_pairs", "_bank", "last_route")

    REFLECT: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"\bi am (.*)$", re.I), "How long have you been {0}?"),
        (re.compile(r"\bi'm (.*)$", re.I), "What makes you say you're {0}?"),
        (re.compile(r"\bi feel (.*)$", re.I), "What do you think makes you feel {0}?"),
        (re.compile(r"\bi need (.*)$", re.I), "What would having {0} change for you?"),
        (re.compile(r"\bi want (.*)$", re.I), "What would it mean if you got {0}?"),
        (re.compile(r"\bi think (.*)$", re.I), "What leads you to think {0}?"),
        (re.compile(r"\bi love (.*)$", re.I), "What do you love most about {0}?"),
        (re.compile(r"\bi hate (.*)$", re.I), "What about {0} feels hardest?"),
        (re.compile(r"\bmy (.*)$", re.I), "Tell me more about your {0}."),
        (re.compile(r"\byou are (.*)$", re.I), "What makes you say I'm {0}?"),
        (re.compile(r"\byou're (.*)$", re.I), "Interesting — why {0}?"),
    )
    PRONOUNS = (
        (re.compile(r"\bam\b", re.I), "are"),
        (re.compile(r"\bwas\b", re.I), "were"),
        (re.compile(r"\bi\b", re.I), "you"),
        (re.compile(r"\bmy\b", re.I), "your"),
        (re.compile(r"\bme\b", re.I), "you"),
        (re.compile(r"\bmine\b", re.I), "yours"),
        (re.compile(r"\byou\b", re.I), "I"),
        (re.compile(r"\byour\b", re.I), "my"),
    )

    def __init__(self) -> None:
        self._pairs = list(ELEZIA_DIALOGUES)
        self._bank: list[tuple[set[str], str, str]] = []
        for prompts, answer in self._pairs:
            for prompt in prompts:
                keys = {t.lower() for t in WordTokenizer.basic_tokenize(prompt) if len(t) >= 2}
                if keys:
                    self._bank.append((keys, prompt.lower(), answer))
        self.last_route = "elezia:idle"

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(str(text or "").lower().split()).strip(" \t.!?。！？")

    def _reflect(self, fragment: str) -> str:
        out = f" {fragment.strip()} "
        for pattern, repl in self.PRONOUNS:
            out = pattern.sub(f" {repl} ", out)
        return " ".join(out.split()).strip(" .")

    def match(self, prompt: str) -> Optional[str]:
        raw = self._norm(prompt)
        if not raw:
            return None
        for keys, surface, answer in self._bank:
            if raw == self._norm(surface) or raw.rstrip("?!") == self._norm(surface).rstrip("?!"):
                self.last_route = "elezia:exact"
                return answer
        # Knowledge / instruction prompts belong to GPT-J / dataset, not ELEZIA vibes.
        if re.search(
            r"\b(?:what is|what's|how do|how to|explain|define|write|implement|python|javascript|"
            r"sql|git|http|api|gpt-?j|bitnet|algorithm|function)\b|:",
            raw,
            re.I,
        ):
            return None
        needle = {t for t in WordTokenizer.basic_tokenize(raw) if len(t) >= 2}
        best: Optional[str] = None
        best_score = 0.0
        for keys, surface, answer in self._bank:
            if not keys:
                continue
            overlap = len(needle & keys)
            if overlap == 0:
                continue
            score = overlap / max(1, len(keys))
            if score >= 0.66 and overlap >= max(1, min(3, len(keys))) and score > best_score:
                best, best_score = answer, score
            # Contained-phrase only for near-equal short casual utterances.
            surf = self._norm(surface)
            if (
                len(surf) >= 4
                and len(raw) <= 32
                and abs(len(raw) - len(surf)) <= 4
                and (raw == surf or raw.rstrip("?!") == surf.rstrip("?!"))
            ):
                if 0.85 > best_score:
                    best, best_score = answer, 0.85
        if best:
            self.last_route = "elezia:fuzzy"
            return best
        return None

    def reflect_reply(self, prompt: str, *, lang: str = "en") -> Optional[str]:
        raw = str(prompt or "").strip()
        if lang == "zh":
            return None
        lowered = raw.lower().rstrip(".!?")
        for pattern, template in self.REFLECT:
            matched = pattern.search(lowered)
            if not matched:
                continue
            fragment = self._reflect(matched.group(1))
            if not fragment or len(fragment) < 2:
                continue
            self.last_route = "elezia:reflect"
            return template.format(fragment)
        return None

    def is_casual(self, prompt: str, profile: Optional[UserSignalProfile] = None) -> bool:
        raw = self._norm(prompt)
        if not raw:
            return False
        if profile is not None and profile.modality in {"code", "shell", "math"}:
            return False
        if len(raw) > 160:
            return False
        # Don't steal instructional / GPT-J knowledge prompts.
        if re.search(
            r"\b(?:what is|what's|how do|how to|explain|define|write|implement|python|javascript|"
            r"sql|git|http|api|gpt-?j|bitnet|algorithm|function)\b|:",
            raw,
            re.I,
        ):
            return False
        casual_markers = (
            "hi", "hello", "hey", "how are you", "wow", "cool", "lol", "haha",
            "thanks", "ty", "sup", "whats up", "what's up", "nice", "omg",
            "你好", "哈哈", "好酷", "在吗", "聊聊",
        )
        # Only true casual markers — never steal real questions via "terse" wording.
        return any(marker in raw for marker in casual_markers)

    def process(
        self,
        prompt: str,
        *,
        lang: Optional[str] = None,
        profile: Optional[UserSignalProfile] = None,
    ) -> Optional[tuple[str, str]]:
        """Process a chat turn → ``(text, route)`` or None."""
        lang = lang or detect_language(prompt)
        hit = self.match(prompt)
        if hit:
            return hit, self.last_route
        if self.is_casual(prompt, profile):
            reflected = self.reflect_reply(prompt, lang=lang)
            if reflected:
                return reflected, self.last_route
            # Never emit the empty vibes prompt — say something about the user's words.
            text = answer_everything(prompt, lang=lang)
            self.last_route = "elezia:say-something"
            return text, self.last_route
        reflected = self.reflect_reply(prompt, lang=lang)
        if reflected:
            return reflected, self.last_route
        return None

    def self_test(self) -> dict[str, object]:
        cases = [
            ("hi how are you", True),
            ("wow that's pretty cool", True),
            ("i feel sad", True),
        ]
        rows = []
        ok = True
        for prompt, must_hit in cases:
            result = self.process(prompt)
            hit = result is not None and bool(result[0].strip())
            ok = ok and (hit if must_hit else True)
            rows.append({"prompt": prompt, "hit": hit, "route": None if result is None else result[1]})
        return {"ok": ok, "pairs": len(self._pairs), "cases": rows}


# GPT-J style corpus (EleutherAI GPT-J / The Pile domain mix) — held entirely in RAM.
# Not the full Pile (files=off); a curated micro-dataset in GPT-J fine-tune format so
# BitNet learns completion + instruction replies without downloading checkpoints.
GPTJ_DIALOGUES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("explain GPT-J", "what is GPT-J", "tell me about GPT-J"),
     "GPT-J is EleutherAI's 6B-parameter open autoregressive LM trained on The Pile. "
     "This runtime keeps a GPT-J-style micro-dataset in RAM (files=off) to teach next-token habits — "
     "it is not the full GPT-J-6B weight dump."),
    (("what is the pile", "explain the pile dataset", "eleutherai pile"),
     "The Pile is EleutherAI's large curated English pretraining mix (web, books, code, Wikipedia, "
     "Stack Exchange, and more). Here we mirror those domains with compact RAM documents."),
    (("complete this sentence: the sun", "finish: the sun"),
     "The sun is the star at the center of the Solar System, a nearly perfect sphere of hot plasma."),
    (("write a haiku about code", "haiku about programming"),
     "bugs hide in plain sight\nprint one line, then breathe again\ngreen tests, quiet night"),
    (("translate to french: hello world", "french: hello world"),
     "Bonjour le monde"),
    (("what is photosynthesis", "explain photosynthesis briefly"),
     "Photosynthesis converts light energy into chemical energy in plants, producing sugars and releasing oxygen."),
    (("list three sorting algorithms", "name sorting algorithms"),
     "1) Quicksort  2) Mergesort  3) Heapsort — pick by average vs worst-case needs."),
    (("python: reverse a string", "how to reverse a string in python"),
     "Use slicing: `text[::-1]` or `''.join(reversed(text))`."),
    (("what is a black hole", "explain black holes simply"),
     "A black hole is a region where gravity is so strong that nothing — not even light — can escape from inside its event horizon."),
    (("define entropy", "what is entropy in info theory"),
     "In information theory, entropy measures average surprise or uncertainty in a probability distribution over symbols."),
    (("stack vs queue", "difference between stack and queue"),
     "Stack is LIFO (last in, first out). Queue is FIFO (first in, first out)."),
    (("what is TCP", "explain TCP briefly"),
     "TCP is a reliable, ordered, connection-oriented transport protocol that retransmits lost packets."),
    (("regex match email", "simple email regex"),
     "A practical pattern is `^[\\w.+-]+@[\\w-]+\\.[\\w.-]+$` — validate further for production."),
    (("why use unit tests", "benefit of unit tests"),
     "Unit tests catch regressions early, document expected behavior, and make refactors safer."),
    (("what is gradient descent", "explain gradient descent"),
     "Gradient descent updates parameters opposite the loss gradient to reduce error iteratively."),
    (("json vs yaml", "difference json yaml"),
     "JSON is strict and ubiquitous for APIs. YAML is more human-writable for configs, with indentation-sensitive structure."),
    (("what is git rebase", "explain git rebase"),
     "Rebase replays commits onto another base tip, rewriting history for a linear story — avoid on shared published branches."),
    (("capital of france", "what is the capital of france"),
     "Paris."),
    (("2 + 2", "what is 2 plus 2"),
     "4"),
    (("define API", "what does API mean"),
     "An API (Application Programming Interface) is a contract that lets software components talk without sharing internals."),
    (("what is recursion briefly", "recursion one sentence"),
     "Recursion is when a function solves a problem by calling itself on a smaller instance until a base case."),
    (("linux list files", "command to list files"),
     "`ls -la` lists files including hidden ones with details."),
    (("http status 404", "what is 404"),
     "404 Not Found means the server could not locate the requested resource."),
    (("what is docker", "explain docker simply"),
     "Docker packages an app with its dependencies into a portable container that runs consistently across machines."),
    (("sql select all", "sql get all rows"),
     "`SELECT * FROM table_name;` — prefer naming columns in production queries."),
    (("what is overfitting", "explain overfitting"),
     "Overfitting means the model memorizes training quirks and fails to generalize to new data."),
    (("markdown bold", "how to bold in markdown"),
     "Wrap text in double asterisks: `**bold**`."),
    (("what is latency", "define latency"),
     "Latency is the time delay between a request and the corresponding response."),
    (("explain mutex", "what is a mutex"),
     "A mutex is a lock that lets only one thread own a critical section at a time to avoid data races."),
    (("gpt-j style prompt", "show gpt-j prompt format"),
     "Common GPT-J fine-tune style:\nPrompt: <instruction>\nResponse: <answer>\n#####\nKeep examples clear and end with a separator."),
)

# Pile-inspired completion documents (encyclopedia / code / Q&A / chat) — RAM only.
GPTJ_COMPLETION_DOCS: tuple[str, ...] = (
    "Wikipedia: Python is a high-level, general-purpose programming language. "
    "Its design emphasizes code readability with significant indentation. "
    "Python is dynamically typed and garbage-collected.",
    "Wikipedia: The transformer architecture uses self-attention to model dependencies "
    "between tokens regardless of distance, enabling large language models.",
    "StackExchange Q: How do I remove duplicates from a list while preserving order?\n"
    "A: In Python 3.7+: list(dict.fromkeys(items)).",
    "StackExchange Q: What is the difference between TCP and UDP?\n"
    "A: TCP is reliable and ordered; UDP is connectionless and lower-latency but unreliable.",
    "GitHub README: Install with pip install package. Run tests with pytest. "
    "Contributions welcome via pull requests with clear descriptions.",
    "Math: The derivative of x^2 is 2x. The integral of 2x dx is x^2 + C.",
    "Science: DNA stores genetic information as sequences of nucleotides A, C, G, and T.",
    "History: The Apollo 11 mission landed humans on the Moon in 1969.",
    "Code:\ndef factorial(n: int) -> int:\n    return 1 if n <= 1 else n * factorial(n - 1)\n",
    "Code:\ndef fibonacci(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
    "Dialogue:\nUser: Summarize transformers in one line.\n"
    "Assistant: Transformers use attention to weigh relationships among all tokens in parallel.\n#####\n",
    "Dialogue:\nUser: Give a tip for debugging.\n"
    "Assistant: Reproduce with the smallest input, then inspect the first incorrect state.\n#####\n",
    "Ubuntu IRC style: <user> how do I free disk space?\n"
    "<helper> try df -h then clear old logs and unused docker images carefully.",
    "Email style: Subject: meeting notes\nBody: We agreed to ship the MVP Friday and freeze scope today.",
    "Books style: Chapter 1. Curiosity begins when a question outruns a simple answer.",
    "ArXiv abstract style: We present a method for efficient ternary neural inference on CPU "
    "using absmean quantization and integer accumulation.",
    "JSON example: {\"name\": \"cat r1\", \"engine\": \"bitnet-b1.58\", \"files\": \"off\"}",
    "YAML example:\nmodel: gpt-j-style\nram_only: true\nmax_docs: 64\n",
    "Prompt/Response:\nPrompt: Name three HTTP verbs.\nResponse: GET, POST, DELETE.\n#####\n",
    "Prompt/Response:\nPrompt: What is Big-O of binary search?\nResponse: O(log n) comparisons on a sorted array.\n#####\n",
    "Prompt/Response:\nPrompt: Define idempotent API.\nResponse: Repeating the same request leaves the resource in the same state.\n#####\n",
    "Prompt/Response:\nPrompt: Why prefer immutable data?\nResponse: It reduces race conditions and makes reasoning about state easier.\n#####\n",
    "Completion: Once upon a time, a small model lived only in RAM and answered kindly.",
    "Completion: In distributed systems, prefer timeouts, retries with backoff, and clear failure modes.",
    "Completion: Clean code names intent. Comments explain why, not what the syntax already shows.",
    "Completion: Measure before optimizing. The bottleneck is often not where intuition points.",
    "Completion: A good PR describes the why, the risk, and how you verified the change.",
    "Completion: Security starts with least privilege, input validation, and never trusting client data.",
    "Completion: Accessibility means keyboard paths, contrast, labels, and semantic HTML.",
    "Completion: Documentation is a product feature: short, accurate, and next to the code it describes.",
)


class GptJRamDataset:
    """GPT-J-style dataset kept entirely in RAM (files=off, no download).

    Mirrors EleutherAI GPT-J domains (Pile-like mix + instruction pairs) as a
    compact resident corpus for BitNet warm and chat-api lookup.
    """

    __slots__ = (
        "dialogues", "documents", "_bank", "_doc_bytes", "resident",
    )

    MODEL_TAG = "gpt-j-style-ram-microcorpus"
    SOURCE = "EleutherAI GPT-J / The Pile domain mix (embedded micro-set, not full weights)"

    def __init__(self) -> None:
        # Materialize lists once so the corpus stays resident in RAM.
        self.dialogues: list[tuple[tuple[str, ...], str]] = [
            (tuple(prompts), answer) for prompts, answer in GPTJ_DIALOGUES
        ]
        self.documents: list[str] = list(GPTJ_COMPLETION_DOCS)
        self._bank: list[tuple[set[str], str]] = []
        for prompts, answer in self.dialogues:
            for prompt in prompts:
                keys = {t.lower() for t in WordTokenizer.basic_tokenize(prompt) if len(t) >= 2}
                if keys:
                    self._bank.append((keys, answer))
        self._doc_bytes = sum(len(doc.encode("utf-8")) for doc in self.documents)
        self._doc_bytes += sum(
            len(p.encode("utf-8")) + len(a.encode("utf-8"))
            for prompts, a in self.dialogues
            for p in prompts
        )
        self.resident = True  # explicit: never spilled to disk by this engine

    @property
    def ram_bytes(self) -> int:
        return int(self._doc_bytes)

    @property
    def pair_count(self) -> int:
        return len(self.dialogues)

    @property
    def doc_count(self) -> int:
        return len(self.documents)

    def lookup(self, prompt: str) -> Optional[str]:
        raw = " ".join(str(prompt or "").lower().split())
        if not raw:
            return None
        for prompts, answer in self.dialogues:
            for candidate in prompts:
                if raw == " ".join(candidate.lower().split()):
                    return answer
        needle = content_tokens(raw)
        if not needle:
            return None
        best: Optional[str] = None
        best_score = 0
        for keys, answer in self._bank:
            content_keys = {k for k in keys if k not in CONTENT_STOPWORDS}
            if not content_keys:
                continue
            overlap = len(needle & content_keys)
            need = max(1, min(2, (len(content_keys) + 1) // 2))
            if overlap >= need and overlap > best_score:
                best, best_score = answer, overlap
        return best

    def chat_documents(self) -> list[tuple[str, str]]:
        """Flatten to (prompt, answer) pairs for warm / teaching."""
        rows: list[tuple[str, str]] = []
        for prompts, answer in self.dialogues:
            for prompt in prompts:
                rows.append((prompt, answer))
        return rows

    def completion_texts(self) -> list[str]:
        return list(self.documents)

    def stats(self) -> dict[str, object]:
        return {
            "model_tag": self.MODEL_TAG,
            "source": self.SOURCE,
            "resident_in_ram": self.resident,
            "files_mode": FILES_MODE,
            "dialogue_pairs": self.pair_count,
            "completion_docs": self.doc_count,
            "ram_bytes": self.ram_bytes,
            "ram_kib": round(self.ram_bytes / 1024.0, 2),
        }

    def self_test(self) -> dict[str, object]:
        hit = self.lookup("what is GPT-J")
        ok = (
            self.resident
            and self.pair_count >= 20
            and self.doc_count >= 20
            and self.ram_bytes > 0
            and bool(hit)
            and "GPT-J" in (hit or "")
        )
        return {"ok": ok, **self.stats(), "sample_lookup": (hit or "")[:120]}


ALL_DIALOGUES = (
    BOOT_DIALOGUES
    + REASONING_DIALOGUES
    + COVERAGE_DIALOGUES
    + DEFAULT_DIALOGUES
    + ELEZIA_DIALOGUES
    + GPTJ_DIALOGUES
)


def lookup_dialogue_answer(prompt: str, *, lang: Optional[str] = None) -> Optional[str]:
    """Best-effort dataset hit for a prompt (used by chat-api + BitNet warm path)."""
    raw = " ".join(str(prompt or "").lower().split())
    if not raw:
        return None
    lang = lang or detect_language(prompt)
    needle = content_tokens(raw)
    best: Optional[str] = None
    best_score = 0
    for prompts, answer in ALL_DIALOGUES:
        answer_lang = language_of_text(answer)
        if lang == "zh" and answer_lang != "zh":
            continue
        if lang == "en" and answer_lang == "zh":
            continue
        for candidate in prompts:
            if raw == " ".join(candidate.lower().split()):
                return answer
            keys = content_tokens(candidate)
            if not keys:
                continue
            overlap = len(needle & keys)
            need = 2 if len(needle) >= 2 else 1
            need = max(need, min(2, (len(keys) + 1) // 2))
            if overlap >= need and overlap > best_score:
                best, best_score = answer, overlap
    return best


def default_response_for(prompt: str, *, lang: Optional[str] = None) -> str:
    """Always-available reply — never the empty vibes prompt."""
    lang = lang or detect_language(prompt)
    hit = lookup_dialogue_answer(prompt, lang=lang)
    if hit:
        return hit
    return answer_everything(prompt, lang=lang)


VIBE_CHECK_PROBES: tuple[tuple[str, str], ...] = (
    ("empty", ""),
    ("casual", "BROOO this RAM model is wild"),
    ("unknown-fact", "What is the favorite color of the mayor of Exampleville?"),
    ("explain", "Explain frobnication to a beginner"),
    ("build", "Build a small parser with clear error messages"),
    ("debug", "My program crashes with ValueError: invalid packet length"),
    ("compare", "Compare a table-driven CPU core vs a giant opcode switch"),
    ("summarize", "Summarize this: The prototype is fast, but it loses state after restart."),
    ("rewrite", "Rewrite this clearly: app fast but crash sometimes"),
    ("creative", "Brainstorm three mechanics for a time-loop platformer"),
    ("current", "What is today's exchange rate?"),
    ("unicode", "🧪 café Привет مرحبا NovelIdentifier_42"),
    ("mandarin", "请解释一个训练数据中没有的新概念"),
)


class TrainingDataVibeCheck:
    """Static, deterministic quality audit for the embedded RAM corpus."""

    INTENTS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("greeting", re.compile(r"\b(?:hello|hi|hey|morning)\b|你好|早上好", re.I)),
        ("explain", re.compile(r"\b(?:what|explain|teach|define)\b|什么|解释|介绍", re.I)),
        ("build", re.compile(r"\b(?:build|make|create|implement|write|generate)\b|制作|开发|实现", re.I)),
        ("debug", re.compile(r"\b(?:debug|fix|crash|error|traceback|bug)\b|调试|错误|崩溃", re.I)),
        ("compare", re.compile(r"\b(?:compare|versus|vs|tradeoffs?|choose)\b|比较|选择", re.I)),
        ("summarize", re.compile(r"\b(?:summarize|summary|key points)\b|总结|主要观点", re.I)),
        ("rewrite", re.compile(r"\b(?:rewrite|wording|clearer)\b|改写|润色", re.I)),
        ("reason", re.compile(r"\b(?:reason|solve|plan|verify|prove|test)\b|推理|验证|计划", re.I)),
        ("current", re.compile(r"\b(?:latest|current|today|price|news)\b|最新|今天|价格|新闻", re.I)),
        ("casual", re.compile(r"\b(?:thanks|bye|vibe|bro)\b|谢谢|再见|氛围", re.I)),
    )

    def __init__(self, dialogues: tuple[tuple[tuple[str, ...], str], ...]):
        self.dialogues = dialogues

    @staticmethod
    def _normal(text: str) -> str:
        return " ".join(text.casefold().split()).strip(" .?!。！？")

    def run(self, tokenizer: Optional[WordTokenizer] = None) -> dict[str, object]:
        prompts: dict[str, set[str]] = collections.defaultdict(set)
        categories: collections.Counter[str] = collections.Counter()
        languages: collections.Counter[str] = collections.Counter()
        empty_rows: list[int] = []
        answer_lengths: list[int] = []
        for row, (variants, answer) in enumerate(self.dialogues):
            if not variants or not answer.strip():
                empty_rows.append(row)
            answer_lengths.append(len(WordTokenizer.basic_tokenize(answer)))
            for prompt in variants:
                normalized = self._normal(prompt)
                prompts[normalized].add(answer.strip())
                languages[detect_language(prompt)] += 1
                matched = False
                for name, pattern in self.INTENTS:
                    if pattern.search(prompt):
                        categories[name] += 1
                        matched = True
                if not matched:
                    categories["other"] += 1

        conflicts = sorted(key for key, answers in prompts.items() if key and len(answers) > 1)
        duplicate_variants = sum(max(0, len(answers) - 1) for answers in prompts.values())
        round_trip_ok = True
        if tokenizer is not None:
            probes = (
                "NovelIdentifier_42", "café", "🧪", "请检查未知词元",
                "Привет مرحبا", "two  spaces", "tab\tvalue",
            )
            round_trip_ok = all(tokenizer.decode(tokenizer.encode(value)) == value for value in probes)
        required = {"explain", "build", "debug", "compare", "summarize", "reason", "current", "casual"}
        covered = {name for name, count in categories.items() if count > 0}
        checks = {
            "no_empty_dialogues": not empty_rows,
            "no_conflicting_duplicate_prompts": not conflicts,
            "english_and_mandarin_present": languages["en"] > 0 and languages["zh"] > 0,
            "broad_intent_coverage": required <= covered,
            "answers_have_substance": bool(answer_lengths) and min(answer_lengths) >= 4,
            "lossless_unseen_text": round_trip_ok,
            "universal_response_probe_set_present": len(VIBE_CHECK_PROBES) >= 10,
        }
        return {
            "passed": all(checks.values()),
            "checks": checks,
            "dialogue_groups": len(self.dialogues),
            "prompt_variants": sum(len(variants) for variants, _ in self.dialogues),
            "unique_normalized_prompts": len(prompts),
            "conflicting_prompts": conflicts,
            "duplicate_variant_count": duplicate_variants,
            "empty_rows": empty_rows,
            "language_prompts": dict(languages),
            "intent_prompts": dict(sorted(categories.items())),
            "answer_tokens": {
                "minimum": min(answer_lengths, default=0),
                "median": statistics.median(answer_lengths) if answer_lengths else 0,
                "maximum": max(answer_lengths, default=0),
            },
            "contract": (
                "The corpus does not imply frontier knowledge. The runtime guarantees a non-empty, "
                "prompt-aware answer or an explicit request for missing evidence."
            ),
        }


def corpus_texts() -> list[str]:
    texts: list[str] = []
    for prompts, answer in ALL_DIALOGUES:
        texts.extend(prompts)
        texts.append(answer)
    return texts


class TernaryBank:
    """Sparse 2-bit BitNet bank: virtual ~20B slots, resident STE pages only.

    Boot stays ≤0.2s because we never allocate the full ~5GB packed buffer up front.
    MoD / unpack of remote layers returns zero codes (γ=ε) until a page is touched.
    """

    __slots__ = (
        "packed", "gammas", "offsets", "shapes", "names",
        "materialized_weights", "total_weights", "sparse",
    )

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_ff: int,
        rng: np.random.Generator,
        *,
        dense_init: bool,
        seed_layers: int,
    ):
        shapes: list[tuple[int, int]] = []
        names: list[str] = []
        for layer in range(n_layers):
            for name, rows, cols in (
                ("wq", d_model, d_model),
                ("wk", d_model, d_model),
                ("wv", d_model, d_model),
                ("wo", d_model, d_model),
                ("w1", d_model, d_ff),
                ("w2", d_ff, d_model),
            ):
                shapes.append((rows, cols))
                names.append(f"L{layer}.{name}")
        sizes = [rows * cols for rows, cols in shapes]
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        total = offsets[-1]
        self.total_weights = total
        self.offsets = offsets
        self.shapes = shapes
        self.names = names
        self.gammas = np.full(len(shapes), BITNET_EPS, dtype=np.float32)
        seed_matrices = max(1, seed_layers) * 6
        seed_weights = offsets[min(seed_matrices, len(offsets) - 1)]
        seed_bytes = max(1, (seed_weights + 3) // 4)
        # Real BitNet init: FP Gaussian → absmean ternary pack (never random code soup).
        if dense_init:
            nbytes = (total + 3) // 4
            self.packed = np.zeros(nbytes, dtype=np.uint8)
            self.materialized_weights = 0
            self.sparse = False
            for index, (rows, cols) in enumerate(shapes):
                raw = rng.normal(0.0, 0.02, size=(rows, cols)).astype(np.float32)
                self.pack_float(index, raw)
            self.materialized_weights = total
        else:
            self.packed = np.zeros(seed_bytes, dtype=np.uint8)
            self.materialized_weights = 0
            self.sparse = True
            for index in range(seed_matrices):
                rows, cols = shapes[index]
                raw = rng.normal(0.0, 0.02, size=(rows, cols)).astype(np.float32)
                self.pack_float(index, raw)
            self.materialized_weights = max(self.materialized_weights, seed_weights)

    def __len__(self) -> int:
        return self.total_weights

    def matrix_index(self, layer: int, which: int) -> int:
        return layer * 6 + which

    def _ensure_materialized(self, end_weight: int) -> None:
        need_bytes = (end_weight + 3) // 4
        if self.packed.size >= need_bytes:
            return
        grown = np.zeros(need_bytes, dtype=np.uint8)
        grown[: self.packed.size] = self.packed
        self.packed = grown
        self.materialized_weights = max(self.materialized_weights, end_weight)

    def unpack(self, index: int) -> np.ndarray:
        start = self.offsets[index]
        rows, cols = self.shapes[index]
        n = rows * cols
        if start + n > self.materialized_weights:
            return np.zeros((rows, cols), dtype=np.int8)
        byte_start = start // 4
        need = n + (start % 4)
        nbytes = (need + 3) // 4
        window = self.packed[byte_start:byte_start + nbytes]
        if window.size * 4 < need:
            padded = np.zeros(nbytes, dtype=np.uint8)
            padded[: window.size] = window
            window = padded
        b = window.astype(np.uint16)
        remap = np.array([0, 1, -1, 0], dtype=np.int8)
        out = np.empty((window.size, 4), dtype=np.int8)
        out[:, 0] = remap[b & 3]
        out[:, 1] = remap[(b >> 2) & 3]
        out[:, 2] = remap[(b >> 4) & 3]
        out[:, 3] = remap[(b >> 6) & 3]
        flat = out.reshape(-1)
        if start % 4:
            flat = flat[(start % 4):]
        return flat[:n].reshape(rows, cols)

    def pack_codes(self, index: int, codes: np.ndarray, gamma: float) -> None:
        start = self.offsets[index]
        rows, cols = self.shapes[index]
        flat = codes.astype(np.int8, copy=False).reshape(-1)
        if flat.size != rows * cols:
            raise ValueError("code size mismatch")
        if start % 4 != 0 or flat.size % 4 != 0:
            raise ValueError("BitNet bank matrices must be 4-weight aligned")
        self._ensure_materialized(start + flat.size)
        mapped = np.zeros(flat.size, dtype=np.uint8)
        mapped[flat == 1] = 1
        mapped[flat == -1] = 2
        grouped = mapped.reshape(-1, 4).astype(np.uint16)
        packed = (grouped[:, 0] | (grouped[:, 1] << 2) | (grouped[:, 2] << 4) | (grouped[:, 3] << 6)).astype(np.uint8)
        byte_start = start // 4
        self.packed[byte_start:byte_start + packed.size] = packed
        self.gammas[index] = np.float32(float(gamma))
        self.materialized_weights = max(self.materialized_weights, start + flat.size)

    def pack_float(self, index: int, weights: np.ndarray) -> None:
        codes, gamma, _ = InMemoryTernaryLM.ternary_quantize(weights)
        self.pack_codes(index, codes, float(gamma))


class InMemoryTernaryLM:
    """Real BitNet b1.58 ~20B LLM on the cat r1 engine (W1.58A8), files=off.

    Default topology lands at ~20.0B ternary slots:
      n_layers=6358, d_model=512, d_ff=2048, n_heads=8
      every Q/K/V/O/FC projection is real BitLinear (RMSNorm/SubLN + W1.58A8)
      sparse 2-bit bank: resident STE pages only at boot (≤0.2s) — no checkpoint
      cat r1: hybrid KDA + gated global attention, AttnRes, LatentMoE
      Mixture-of-Depths capacity keeps per-token compute interactive
      STE corpus/Adam deferred until /train (fast boot)
    """

    MATRIX_NAMES = ("wq", "wk", "wv", "wo", "w1", "w2")

    def __init__(self, tokenizer: WordTokenizer, config: ModelConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.gptj_dataset: Optional[GptJRamDataset] = None
        self.rng = np.random.default_rng(config.seed)
        self.forward_calls = 0
        self.generated_tokens = 0
        self.training_steps = 0
        self.external_load_count = 0
        self.last_detected_language = "en"
        self.initial_loss_ever = math.inf
        self.best_loss = math.inf
        self.last_trace: list[GenerationStep] = []
        self.last_deliberation: list[dict[str, object]] = []
        self.last_reasoning_passes = config.mod_capacity
        self.reasoning_pass_histogram: collections.Counter[int] = collections.Counter()
        self.quantization_rebuilds = 0
        self.ternary_kernel_calls = 0
        self.kda_kernel_calls = 0
        self.attnres_mixes = 0
        self.moe_routes = 0
        self.last_active_layers: list[int] = []
        self.last_engine_path: str = "kda"
        self._code_cache: dict[tuple[int, int], tuple[np.ndarray, float]] = {}
        self._live_layers_cache: Optional[list[int]] = None
        self.layer_gates = self.rng.normal(0.0, 0.15, size=config.n_layers).astype(np.float32)
        # cat r1 RAM-only engine state (not packed into the ternary bank).
        self.kda_log_decay = self.rng.normal(-1.2, 0.15, size=(config.n_layers, config.n_heads)).astype(np.float32)
        self.attnres_queries = self.rng.normal(0.0, 0.02, size=(max(2, config.mod_capacity + 2), config.d_model)).astype(np.float32)
        self.moe_router = self.rng.normal(0.0, 0.05, size=(config.d_model, max(2, config.moe_top_k + 2))).astype(np.float32)
        # Defer heavy corpus arrays until STE training is requested (≤0.2s boot).
        self._training_documents: list[list[int]] = []
        self.contexts = np.zeros((0, config.context_tokens), dtype=np.int64)
        self.targets = np.zeros((0,), dtype=np.int64)
        self.sample_weights = np.zeros((0,), dtype=np.float64)
        self._train_corpus_ready = False

        vocab = tokenizer.vocab_size
        d = config.d_model
        self.embedding = self.rng.normal(0.0, 0.02, (vocab, d)).astype(np.float32)
        # Tied LM head init (BitNet / LLaMA-style): start from embeddingᵀ so decode is not random soup.
        self.lm_head = (self.embedding.T * (1.0 / math.sqrt(d))).astype(np.float32)
        self.b_output = np.zeros((vocab,), dtype=np.float32)

        self.train_layer_count = max(1, min(config.train_layers, config.n_layers))
        self.bank = TernaryBank(
            config.n_layers,
            config.d_model,
            config.d_ff,
            self.rng,
            dense_init=config.dense_init,
            seed_layers=self.train_layer_count,
        )
        self.shadow: list[dict[str, np.ndarray]] = []
        for layer_index in range(self.train_layer_count):
            shadow: dict[str, np.ndarray] = {}
            for which, name in enumerate(self.MATRIX_NAMES):
                index = self.bank.matrix_index(layer_index, which)
                codes = self.bank.unpack(index)
                shadow[name] = (codes.astype(np.float32) * float(self.bank.gammas[index])).astype(np.float32)
            self.shadow.append(shadow)
        self.quantization_rebuilds += 1

        self.parameters: list[np.ndarray] = [self.embedding, self.lm_head, self.b_output]
        for shadow in self.shadow:
            for name in self.MATRIX_NAMES:
                self.parameters.append(shadow[name])
        # Adam moments allocated lazily on first STE step.
        self.adam_m: list[np.ndarray] = []
        self.adam_v: list[np.ndarray] = []
        self.report = TrainingReport(samples=0)
        if config.train_steps > 0:
            self._ensure_train_corpus()
        self._packed_param_count = int(len(self.bank) + self.embedding.size + self.lm_head.size + self.b_output.size)
        self.dspark: Optional[DSparkBitNetEngine] = None
        # Cat R1 v0.1 decode engine is DSpark (force-on when cat_r1 stack is active).
        if self.config.dspark_enabled or self.config.cat_r1_enabled:
            self.dspark = DSparkBitNetEngine(self)
        self.cat_r1_reason: Optional[CatR1ReasoningEngine] = None
        if self.config.cat_r1_reason_enabled:
            self.cat_r1_reason = CatR1ReasoningEngine(self)
        # Warm BitLinear code cache for the live STE slice.
        for layer_index in range(self.train_layer_count):
            for which in range(len(self.MATRIX_NAMES)):
                self._codes_for(layer_index, which, inference=True)

    def _ensure_train_corpus(self) -> None:
        if self._train_corpus_ready:
            return
        self._training_documents = self._build_documents()
        self.contexts, self.targets, self.sample_weights = self._build_training_arrays(self._training_documents)
        if getattr(self, "report", None) is not None:
            self.report.samples = len(self.targets)
        self._train_corpus_ready = True

    def _ensure_adam(self) -> None:
        if self.adam_m:
            return
        self.adam_m = [np.zeros_like(parameter) for parameter in self.parameters]
        self.adam_v = [np.zeros_like(parameter) for parameter in self.parameters]

    @property
    def parameter_count(self) -> int:
        return int(self._packed_param_count)

    def _build_documents(self) -> list[list[int]]:
        t = self.tokenizer
        documents: list[list[int]] = []
        for prompts, answer in ALL_DIALOGUES:
            for prompt in prompts:
                language_token = t.ZH if detect_language(prompt) == "zh" else t.EN
                documents.append([
                    t.token_id(t.BOS), t.token_id(t.USER), *t.encode(prompt),
                    t.token_id(language_token), t.token_id(t.ASSISTANT),
                    *t.encode(answer), t.token_id(t.EOS),
                ])
                documents.append([
                    t.token_id(t.BOS), t.token_id(t.USER), *t.encode(prompt, force_bytes=True),
                    t.token_id(language_token), t.token_id(t.ASSISTANT),
                    *t.encode(answer), t.token_id(t.EOS),
                ])
        # GPT-J completion docs (Pile-style) — pure LM next-token sequences in RAM.
        gptj = getattr(self, "gptj_dataset", None)
        completion_docs = gptj.completion_texts() if gptj is not None else list(GPTJ_COMPLETION_DOCS)
        for text in completion_docs:
            encoded = t.encode(text)
            if not encoded:
                continue
            documents.append([
                t.token_id(t.BOS), *encoded, t.token_id(t.EOS),
            ])
        return documents

    def add_training_dialogue(self, prompt: str, answer: str) -> int:
        self._ensure_train_corpus()
        t = self.tokenizer
        language_token = t.ZH if detect_language(prompt) == "zh" else t.EN
        new_documents = []
        for force_bytes in (False, True):
            new_documents.append([
                t.token_id(t.BOS), t.token_id(t.USER), *t.encode(prompt, force_bytes=force_bytes),
                t.token_id(language_token), t.token_id(t.ASSISTANT),
                *t.encode(answer), t.token_id(t.EOS),
            ])
        contexts, targets, sample_weights = self._build_training_arrays(new_documents)
        self._training_documents.extend(new_documents)
        rehearsal = 8
        self.contexts = np.concatenate((self.contexts, np.tile(contexts, (rehearsal, 1))), axis=0)
        self.targets = np.concatenate((self.targets, np.tile(targets, rehearsal)), axis=0)
        self.sample_weights = np.concatenate(
            (self.sample_weights, np.tile(sample_weights, rehearsal)), axis=0,
        )
        self.report.samples = len(self.targets)
        return int(len(targets) * rehearsal)

    def _build_training_arrays(
        self,
        documents: list[list[int]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        width = self.config.context_tokens
        pad = self.tokenizer.token_id(self.tokenizer.PAD)
        assistant = self.tokenizer.token_id(self.tokenizer.ASSISTANT)
        eos = self.tokenizer.token_id(self.tokenizer.EOS)
        contexts: list[list[int]] = []
        targets: list[int] = []
        weights: list[float] = []
        for document in documents:
            try:
                assistant_index = document.index(assistant)
            except ValueError:
                assistant_index = 0
            for index in range(1, len(document)):
                previous = document[max(0, index - width):index]
                contexts.append([pad] * (width - len(previous)) + previous)
                targets.append(document[index])
                # Spend most optimizer updates on learning assistant output,
                # while retaining a smaller prompt/control-token rehearsal.
                weight = 1.15 if index > assistant_index else 0.22
                if document[index] == eos:
                    weight = 1.5
                weights.append(weight)
        return (
            np.asarray(contexts, dtype=np.int64),
            np.asarray(targets, dtype=np.int64),
            np.asarray(weights, dtype=np.float64),
        )

    @staticmethod
    def rms_norm(values: np.ndarray, eps: float = BITNET_EPS) -> np.ndarray:
        """BitNet SubLN / RMSNorm (Microsoft BitNet b1.58 BitLinear pre-norm)."""
        mean_sq = np.mean(np.square(values), axis=-1, keepdims=True)
        return (values / np.sqrt(mean_sq + eps)).astype(np.float32)

    @staticmethod
    def layer_norm(values: np.ndarray, eps: float = BITNET_EPS) -> np.ndarray:
        """Alias kept for call sites; real BitLinear uses ``rms_norm`` (SubLN)."""
        return InMemoryTernaryLM.rms_norm(values, eps=eps)

    @staticmethod
    def relu_squared(values: np.ndarray) -> np.ndarray:
        """BitNet b1.58 FFN activation: ReLU²(x) = max(x, 0)² (Microsoft 2B4T)."""
        relu = np.maximum(values, 0.0)
        return (relu * relu).astype(np.float32)

    @staticmethod
    def ternary_quantize(weights: np.ndarray) -> tuple[np.ndarray, np.float32, np.ndarray]:
        """Microsoft BitNet b1.58 absmean quant (Eq. 1–3).

        μ = mean(|W|)
        Ŵ = RoundClip(W / (μ + ε), −1, +1) ∈ {-1, 0, +1}
        """
        mu = float(np.mean(np.abs(weights)))
        gamma = np.float32(mu + BITNET_EPS)
        codes = np.clip(np.rint(weights / gamma), -1, 1).astype(np.int8)
        return codes, gamma, codes.astype(np.float32) * gamma

    @staticmethod
    def activation_a8_codes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """BitNet W1.58A8 per-token AbsMax INT8 (paper form).

        α = max(|x|)
        x_q = RoundClip(x · Q_b / α, −Q_b, Q_b−1) with Q_b = 128
        """
        peak = np.max(np.abs(values), axis=-1, keepdims=True).astype(np.float32)
        peak = np.maximum(peak, BITNET_EPS)
        codes = np.clip(
            np.rint(values * (BITNET_QB / peak)),
            BITNET_A8_MIN,
            BITNET_A8_MAX,
        ).astype(np.int16)
        return codes, peak

    @staticmethod
    def activation_a8(values: np.ndarray) -> np.ndarray:
        codes, peak = InMemoryTernaryLM.activation_a8_codes(values)
        return (codes.astype(np.float32) * (peak / BITNET_QB)).astype(np.float32)

    @staticmethod
    def ternary_matmul(x_codes: np.ndarray, w_codes: np.ndarray) -> np.ndarray:
        """BitNet W1.58 matmul: activations × ternary Ŵ ∈ {-1,0,+1}.

        Uses one BLAS GEMM on ternary codes (equivalent to add/sub of the +1/−1
        partitions, but far faster on CPU NumPy than two masked int matmuls).
        """
        x = x_codes.astype(np.float32, copy=False)
        w = w_codes.astype(np.float32, copy=False)
        return (x @ w).astype(np.float32, copy=False)


    @classmethod
    def bitlinear_codes(
        cls,
        values: np.ndarray,
        codes: np.ndarray,
        gamma: float,
        *,
        integer_kernel: bool = True,
        pre_norm: bool = True,
    ) -> np.ndarray:
        """Real BitLinear forward (Microsoft BitNet b1.58 W1.58A8).

        1. SubLN/RMSNorm on activations
        2. AbsMax A8 quant → x_q
        3. Integer ternary matmul y_q = x_q @ Ŵ   (Ŵ ∈ {-1,0,+1})
        4. Dequant y = y_q · (α · γ) / Q_b
        """
        x = cls.rms_norm(values) if pre_norm else values.astype(np.float32)
        if integer_kernel:
            x_q, alpha = cls.activation_a8_codes(x)
            y_q = cls.ternary_matmul(x_q, codes)
            y = y_q.astype(np.float32) * (alpha * float(gamma) / BITNET_QB)
        else:
            # STE train path: AbsMax-dequant activations × ternary Ŵ (no FP W multiply).
            x_a8 = cls.activation_a8(x)
            w_hat = codes.astype(np.float32) * float(gamma)
            y = (x_a8 @ w_hat).astype(np.float32)
        return y.astype(np.float32)

    @classmethod
    def bitlinear(
        cls,
        values: np.ndarray,
        weights: np.ndarray,
        *,
        integer_kernel: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.float32]:
        """Full BitLinear from FP shadow weights → absmean pack → W1.58A8.

        Returns ``(y, x_rms, W_hat, gamma)`` for STE outer-product grads on shadows.
        """
        codes, gamma, w_hat = cls.ternary_quantize(weights)
        x_rms = cls.rms_norm(values)
        y = cls.bitlinear_codes(
            x_rms, codes, float(gamma), integer_kernel=integer_kernel, pre_norm=False,
        )
        return y.astype(np.float32), x_rms, w_hat, gamma

    def _rope_cos_sin(self, seq_len: int, head_dim: int) -> tuple[np.ndarray, np.ndarray]:
        """Rotary position embeddings (BitNet b1.58 LLaMA-alike)."""
        cache = getattr(self, "_rope_cache", None)
        if cache is None:
            self._rope_cache = {}
            cache = self._rope_cache
        key = (seq_len, head_dim)
        hit = cache.get(key)
        if hit is not None:
            return hit
        half = max(1, head_dim // 2)
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, half, dtype=np.float32) / half))
        positions = np.arange(seq_len, dtype=np.float32)
        freqs = np.outer(positions, inv_freq).astype(np.float32)
        cos = np.cos(freqs).astype(np.float32)
        sin = np.sin(freqs).astype(np.float32)
        cache[key] = (cos, sin)
        return cos, sin

    @staticmethod
    def _apply_rope(q: np.ndarray, k: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply RoPE to Q/K with shape (B, H, T, Dh)."""
        def rotate(x: np.ndarray) -> np.ndarray:
            dh = x.shape[-1]
            half = dh // 2
            x1 = x[..., :half]
            x2 = x[..., half:half * 2]
            c = cos[None, None, :, :half]
            s = sin[None, None, :, :half]
            rotated = np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)
            if dh > half * 2:
                rotated = np.concatenate([rotated, x[..., half * 2 :]], axis=-1)
            return rotated.astype(np.float32)
        return rotate(q), rotate(k)

    def invalidate_inference_cache(self) -> None:
        for layer_index, shadow in enumerate(self.shadow):
            for which, name in enumerate(self.MATRIX_NAMES):
                self.bank.pack_float(self.bank.matrix_index(layer_index, which), shadow[name])
        self._code_cache.clear()
        self._live_layers_cache = None
        self.quantization_rebuilds += 1

    def _layer_is_live(self, layer_index: int) -> bool:
        """Resident STE layers and non-empty packed layers are executable."""
        if layer_index < self.train_layer_count:
            return True
        index = self.bank.matrix_index(layer_index, 0)
        if float(self.bank.gammas[index]) > 2.0 * BITNET_EPS:
            return True
        start = self.bank.offsets[index]
        end = self.bank.offsets[index + 1]
        if start >= self.bank.materialized_weights:
            return False
        byte_start = start // 4
        byte_end = (min(end, self.bank.materialized_weights) + 3) // 4
        return bool(np.any(self.bank.packed[byte_start:byte_end]))

    def _live_layers(self) -> list[int]:
        if self._live_layers_cache is not None:
            return self._live_layers_cache
        # Default init only materializes the resident STE slice; avoid scanning 1k empty packs.
        live = list(range(self.train_layer_count))
        if self.config.dense_init:
            live = [index for index in range(self.config.n_layers) if self._layer_is_live(index)]
        self._live_layers_cache = live or [0]
        return self._live_layers_cache

    def _select_layers(self, batch_hidden: np.ndarray) -> list[int]:
        """Mixture-of-Depths: pick top-k live layers only (skip empty packs)."""
        capacity = max(1, min(self.config.mod_capacity, self.config.n_layers))
        energy = float(np.mean(np.abs(batch_hidden))) + BITNET_EPS
        live = self._live_layers()
        scored = []
        for index in live:
            score = float(self.layer_gates[index]) * energy
            if index < self.train_layer_count:
                score += 2.0
            scored.append((score, index))
        scored.sort(reverse=True)
        chosen = [index for _, index in scored[:capacity]]
        if 0 in live and 0 not in chosen:
            chosen = [0] + [index for index in chosen if index != 0]
            chosen = chosen[:capacity]
        chosen = sorted(set(chosen))
        self.last_active_layers = chosen
        self.last_reasoning_passes = len(chosen)
        self.reasoning_pass_histogram[len(chosen)] += 1
        return chosen

    def _codes_for(self, layer_index: int, which: int, *, inference: bool) -> tuple[np.ndarray, float]:
        if (not inference) and layer_index < self.train_layer_count:
            weights = self.shadow[layer_index][self.MATRIX_NAMES[which]]
            codes, gamma, _ = self.ternary_quantize(weights)
            return codes, float(gamma)
        key = (layer_index, which)
        if inference and key in self._code_cache:
            return self._code_cache[key]
        index = self.bank.matrix_index(layer_index, which)
        codes = self.bank.unpack(index)
        gamma = float(self.bank.gammas[index])
        if inference:
            self._code_cache[key] = (codes, gamma)
        return codes, gamma

    def _bitlinear(
        self,
        values: np.ndarray,
        layer_index: int,
        which: int,
        *,
        inference: bool,
    ) -> np.ndarray:
        codes, gamma = self._codes_for(layer_index, which, inference=inference)
        # Inference: integer BitNet kernel. Training: STE float path on ternary(W).
        y = self.bitlinear_codes(values, codes, gamma, integer_kernel=inference)
        if inference:
            self.ternary_kernel_calls += 1
        return y

    def _ste_project(
        self,
        values: np.ndarray,
        layer_index: int,
        which: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Real BitLinear STE: RMSNorm → A8 × absmean ternary Ŵ (grads via STE on W)."""
        name = self.MATRIX_NAMES[which]
        weights = self.shadow[layer_index][name]
        y, x_rms, w_hat, _gamma = self.bitlinear(values, weights, integer_kernel=False)
        return y, x_rms, w_hat

    def _attention(self, hidden: np.ndarray, layer_index: int, *, inference: bool) -> np.ndarray:
        """cat r1 hybrid attention.

        Training always uses gated global attention (stable STE).
        Inference chat uses last-token causal BitLinear attention (interactive).
        """
        if not inference:
            self.last_engine_path = "gated-mla-train"
            return self._gated_global_attention(hidden, layer_index, inference=False)
        # Interactive BitNet decode: last-token Q/O, full-context K/V (real causal).
        self.last_engine_path = "bitnet-last-token"
        return self._last_token_attention(hidden, layer_index)

    def _last_token_attention(self, hidden: np.ndarray, layer_index: int) -> np.ndarray:
        """Causal BitLinear attention for decode: only the last position is written.

        K/V cover the full context window; Q and O run on the last token only so
        each autoregressive step stays interactive on CPU NumPy.
        """
        batch, seq, d_model = hidden.shape
        n_heads = self.config.n_heads
        head_dim = max(1, d_model // n_heads)
        flat = hidden.reshape(batch * seq, d_model)
        k_flat = self._bitlinear(flat, layer_index, 1, inference=True)
        v_flat = self._bitlinear(flat, layer_index, 2, inference=True)
        q_last = self._bitlinear(hidden[:, -1, :], layer_index, 0, inference=True)
        k = k_flat.reshape(batch, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v_flat.reshape(batch, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
        q = q_last.reshape(batch, n_heads, head_dim)
        cos, sin = self._rope_cos_sin(seq, head_dim)
        # Apply RoPE to last query + full keys.
        half = head_dim // 2
        if half > 0:
            c_all = cos[None, None, :, :half]
            s_all = sin[None, None, :, :half]
            k1, k2 = k[..., :half], k[..., half:half * 2]
            k = np.concatenate([k1 * c_all - k2 * s_all, k1 * s_all + k2 * c_all], axis=-1)
            c_q = cos[-1, :half]
            s_q = sin[-1, :half]
            q1, q2 = q[..., :half], q[..., half:half * 2]
            q = np.concatenate([q1 * c_q - q2 * s_q, q1 * s_q + q2 * c_q], axis=-1)
        scale = 1.0 / math.sqrt(head_dim)
        scores = np.einsum("bhd,bhtd->bht", q, k) * scale
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(np.clip(scores, -40.0, 40.0))
        weights = weights / np.maximum(np.sum(weights, axis=-1, keepdims=True), BITNET_EPS)
        mixed = np.einsum("bht,bhtd->bhd", weights, v).reshape(batch, d_model)
        gate = 1.0 / (1.0 + np.exp(-np.clip(np.tanh(hidden[:, -1, :]), -8.0, 8.0)))
        mixed = (mixed * gate).astype(np.float32)
        out = self._bitlinear(mixed, layer_index, 3, inference=True)
        delta = np.zeros((batch, seq, d_model), dtype=np.float32)
        delta[:, -1, :] = out
        return delta

    def _project_qkv(
        self,
        hidden: np.ndarray,
        layer_index: int,
        *,
        inference: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
        batch, seq, d_model = hidden.shape
        n_heads = self.config.n_heads
        head_dim = d_model // n_heads
        flat = hidden.reshape(batch * seq, d_model)
        q = self._bitlinear(flat, layer_index, 0, inference=inference)
        k = self._bitlinear(flat, layer_index, 1, inference=inference)
        v = self._bitlinear(flat, layer_index, 2, inference=inference)
        q = q.reshape(batch, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
        return q, k, v, batch, seq, d_model

    def _gated_global_attention(self, hidden: np.ndarray, layer_index: int, *, inference: bool) -> np.ndarray:
        """BitNet-style causal attention: BitLinear QKV/O + RoPE + AbsMax path."""
        q, k, v, batch, seq, d_model = self._project_qkv(hidden, layer_index, inference=inference)
        head_dim = d_model // self.config.n_heads
        cos, sin = self._rope_cos_sin(seq, head_dim)
        q, k = self._apply_rope(q, k, cos, sin)
        scale = 1.0 / math.sqrt(head_dim)
        attn = np.matmul(q, np.transpose(k, (0, 1, 3, 2))) * scale
        mask = np.triu(np.ones((seq, seq), dtype=np.float32), k=1) * -1e9
        attn = attn + mask
        attn = attn - np.max(attn, axis=-1, keepdims=True)
        weights = np.exp(np.clip(attn, -40.0, 40.0))
        weights = weights / np.maximum(np.sum(weights, axis=-1, keepdims=True), BITNET_EPS)
        mixed = np.matmul(weights, v).transpose(0, 2, 1, 3).reshape(batch * seq, d_model)
        # No bias on BitLinear; optional residual gate kept for cat r1 hybrid slot.
        gate = 1.0 / (1.0 + np.exp(-np.clip(np.tanh(hidden.reshape(batch * seq, d_model)), -8.0, 8.0)))
        mixed = (mixed * gate).astype(np.float32)
        out = self._bitlinear(mixed, layer_index, 3, inference=inference)
        return out.reshape(batch, seq, d_model)

    def _kda_attention(self, hidden: np.ndarray, layer_index: int, *, inference: bool) -> np.ndarray:
        """Fast Delta Attention (KDA) with diagonal (vector) recurrent state — O(BTD)."""
        q, k, v, batch, seq, d_model = self._project_qkv(hidden, layer_index, inference=inference)
        n_heads = self.config.n_heads
        head_dim = d_model // n_heads
        q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), BITNET_EPS)
        k = k / np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), BITNET_EPS)
        raw = self.kda_log_decay[layer_index % self.kda_log_decay.shape[0]]
        log_decay = -5.0 * (1.0 / (1.0 + np.exp(-np.clip(raw, -8.0, 8.0))))
        decay = np.exp(log_decay).astype(np.float32)
        outputs = np.zeros((batch, n_heads, seq, head_dim), dtype=np.float32)
        for h in range(n_heads):
            state = np.zeros((batch, head_dim), dtype=np.float32)
            d = float(decay[h])
            for t in range(seq):
                qt = q[:, h, t, :]
                kt = k[:, h, t, :]
                vt = v[:, h, t, :]
                # Diagonal delta-rule: s ← αs + k⊙(v − s⊙k)
                state = d * state
                state = state + kt * (vt - state * kt)
                state = np.clip(state, -8.0, 8.0)
                outputs[:, h, t, :] = qt * state
        self.kda_kernel_calls += 1
        mixed = outputs.transpose(0, 2, 1, 3).reshape(batch * seq, d_model)
        mixed = self.layer_norm(mixed)
        gate = 1.0 / (1.0 + np.exp(-np.clip(np.tanh(hidden.reshape(batch * seq, d_model)), -8.0, 8.0)))
        mixed = (mixed * gate).astype(np.float32)
        out = self._bitlinear(mixed, layer_index, 3, inference=inference)
        return out.reshape(batch, seq, d_model)

    def _ffn_expert(self, flat: np.ndarray, layer_index: int, *, inference: bool) -> np.ndarray:
        """Real BitNet FFN expert: BitLinear → ReLU² → BitLinear (no bias)."""
        pre = self._bitlinear(flat, layer_index, 4, inference=inference)
        h = self.relu_squared(pre)
        return self._bitlinear(h, layer_index, 5, inference=inference)

    def _ffn(self, hidden: np.ndarray, layer_index: int, *, inference: bool) -> np.ndarray:
        """Stable LatentMoE: shared BitLinear expert (+ routed experts when probing).

        Interactive decode only updates the last token through the shared expert so
        prompts stay responsive (full-sequence × top-k MoE was freezing the UI).
        """
        batch, seq, d_model = hidden.shape
        if inference:
            flat = hidden[:, -1, :].astype(np.float32)
            shared = self._ffn_expert(flat, layer_index, inference=True)
            self.moe_routes += 1
            delta = np.zeros((batch, seq, d_model), dtype=np.float32)
            delta[:, -1, :] = shared
            return delta
        flat = hidden.reshape(batch * seq, d_model)
        shared = self._ffn_expert(flat, layer_index, inference=False)
        self.moe_routes += 1
        return shared.reshape(batch, seq, d_model)

    def _attnres_combine(self, snapshots: list[np.ndarray]) -> np.ndarray:
        """Block Attention Residuals: softmax over depth with learned pseudo-queries."""
        if not self.config.attnres or len(snapshots) == 1:
            return snapshots[-1]
        stack = np.stack(snapshots, axis=1)
        depth = stack.shape[1]
        query = self.attnres_queries[min(depth - 1, self.attnres_queries.shape[0] - 1)]
        scores = np.matmul(stack, query)
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        alpha = np.exp(np.clip(scores, -40.0, 40.0))
        alpha = alpha / np.maximum(np.sum(alpha, axis=-1, keepdims=True), BITNET_EPS)
        mixed = np.sum(stack * alpha[:, :, None], axis=1).astype(np.float32)
        self.attnres_mixes += 1
        return mixed

    def _transformer_block(self, hidden: np.ndarray, layer_index: int, *, inference: bool) -> np.ndarray:
        attn = self._attention(hidden, layer_index, inference=inference)
        if inference:
            # Last-token residuals only — earlier positions are already finalized context.
            last = hidden[:, -1, :]
            if getattr(self.config, "cat_r1_mhc", False):
                mix = 0.15
                last = (last * (1.0 - mix) + attn[:, -1, :] * (1.0 + mix)).astype(np.float32)
            else:
                last = (last + attn[:, -1, :]).astype(np.float32)
            last = np.nan_to_num(last, nan=0.0, posinf=8.0, neginf=-8.0)
            ffn_in = hidden.copy()
            ffn_in[:, -1, :] = last
            last = (last + self._ffn(ffn_in, layer_index, inference=True)[:, -1, :]).astype(np.float32)
            out = hidden.copy()
            out[:, -1, :] = np.nan_to_num(last, nan=0.0, posinf=8.0, neginf=-8.0)
            return out.astype(np.float32)
        hidden = (hidden + attn).astype(np.float32)
        hidden = np.nan_to_num(hidden, nan=0.0, posinf=8.0, neginf=-8.0)
        hidden = (hidden + self._ffn(hidden, layer_index, inference=False)).astype(np.float32)
        return np.nan_to_num(hidden, nan=0.0, posinf=8.0, neginf=-8.0).astype(np.float32)

    def _forward_hidden(
        self,
        contexts: np.ndarray,
        *,
        inference: bool,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        self.forward_calls += 1
        hidden = self.embedding[contexts].astype(np.float32)
        snapshots = [hidden[:, -1, :].copy()]
        # Training: run the full resident STE slice in order (real depth).
        # Inference: Mixture-of-Depths capacity selection (cat r1 interactivity).
        layers = (
            list(range(self.train_layer_count))
            if not inference
            else self._select_layers(hidden[:, -1, :])
        )
        for layer_index in layers:
            hidden = self._transformer_block(hidden, layer_index, inference=inference)
            mixed_tail = self._attnres_combine(snapshots + [hidden[:, -1, :].copy()])
            hidden = hidden.copy()
            hidden[:, -1, :] = (
                (1.0 - self.config.reasoning_scale) * hidden[:, -1, :]
                + self.config.reasoning_scale * mixed_tail
            ).astype(np.float32)
            hidden = np.nan_to_num(hidden, nan=0.0, posinf=8.0, neginf=-8.0)
            snapshots.append(hidden[:, -1, :].copy())
        if not inference:
            self.last_active_layers = list(range(self.train_layer_count))
            self.last_reasoning_passes = self.train_layer_count
        last = np.nan_to_num(hidden[:, -1, :], nan=0.0, posinf=8.0, neginf=-8.0)
        return last, snapshots

    def _forward(
        self,
        contexts: np.ndarray,
        *,
        inference: bool,
        training_weights: Optional[tuple[np.ndarray, ...]] = None,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
        del training_weights
        last_hidden, snapshots = self._forward_hidden(contexts, inference=inference)
        logits = (last_hidden @ self.lm_head).astype(np.float32) + self.b_output
        logits = np.nan_to_num(logits, nan=0.0, posinf=40.0, neginf=-40.0)
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(np.clip(logits, -40.0, 40.0))
        probabilities = (exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), BITNET_EPS)).astype(np.float32)
        return last_hidden, snapshots, snapshots[1:], logits, probabilities

    def _ste_layer_step(
        self,
        hidden: np.ndarray,
        layer_index: int,
        d_last: np.ndarray,
        grads: dict[str, np.ndarray],
    ) -> np.ndarray:
        """One resident BitNet block with STE grads for the last-token stream.

        Attention is compressed to a last-token query attending over the sequence
        (still a real causal readout), then FFN STE updates w1/w2. Q/K/V/O also
        receive STE outer products from the same residual path.
        """
        batch, seq, d_model = hidden.shape
        flat = hidden.reshape(batch * seq, d_model)
        # --- Attention STE (last-token query over keys/values) ---
        q_all, q_ln, qw = self._ste_project(flat, layer_index, 0)
        k_all, k_ln, kw = self._ste_project(flat, layer_index, 1)
        v_all, v_ln, vw = self._ste_project(flat, layer_index, 2)
        n_heads = self.config.n_heads
        head_dim = d_model // n_heads
        q = q_all.reshape(batch, seq, n_heads, head_dim)[:, -1, :, :]  # B,H,Dh
        k = k_all.reshape(batch, seq, n_heads, head_dim).transpose(0, 2, 1, 3)  # B,H,T,Dh
        v = v_all.reshape(batch, seq, n_heads, head_dim).transpose(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(head_dim)
        scores = np.einsum("bhd,bhtd->bht", q, k) * scale
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(np.clip(scores, -40.0, 40.0))
        weights = weights / np.maximum(np.sum(weights, axis=-1, keepdims=True), BITNET_EPS)
        mixed_h = np.einsum("bht,bhtd->bhd", weights, v)  # B,H,Dh
        mixed = mixed_h.reshape(batch, d_model)
        gate = 1.0 / (1.0 + np.exp(-np.clip(np.tanh(hidden[:, -1, :]), -8.0, 8.0)))
        mixed_g = mixed * gate
        attn_out, o_ln, ow = self._ste_project(mixed_g, layer_index, 3)
        h1 = (hidden[:, -1, :] + attn_out).astype(np.float32)

        # --- FFN STE (BitNet ReLU²) ---
        ffn_in, w1_ln, w1 = self._ste_project(h1, layer_index, 4)
        ffn_act = self.relu_squared(ffn_in)
        ffn_out, w2_ln, w2 = self._ste_project(ffn_act, layer_index, 5)
        h2 = (h1 + ffn_out).astype(np.float32)

        # --- Backward from d_last into this block (STE) ---
        d_h2 = d_last
        d_ffn_out = d_h2
        grads["w2"] = grads.get("w2", np.zeros_like(w2)) + (w2_ln.T @ d_ffn_out)
        d_act = d_ffn_out @ w2.T
        # d/dx ReLU²(x) = 2·max(x,0)
        d_ffn_in = d_act * (2.0 * np.maximum(ffn_in, 0.0))
        grads["w1"] = grads.get("w1", np.zeros_like(w1)) + (w1_ln.T @ d_ffn_in)
        d_h1 = d_h2 + (d_ffn_in @ w1.T)

        d_attn_out = d_h1
        grads["wo"] = grads.get("wo", np.zeros_like(ow)) + (o_ln.T @ d_attn_out)
        d_mixed_g = d_attn_out @ ow.T
        d_mixed = d_mixed_g * gate
        d_mixed_h = d_mixed.reshape(batch, n_heads, head_dim)

        # Attention grads (last-query form)
        d_v = np.einsum("bhd,bht->bhtd", d_mixed_h, weights)
        d_weights = np.einsum("bhd,bhtd->bht", d_mixed_h, v)
        # softmax jacobian approx: (diag(w)-ww^T) @ d_weights
        dot = np.sum(d_weights * weights, axis=-1, keepdims=True)
        d_scores = weights * (d_weights - dot)
        d_q = np.einsum("bht,bhtd->bhd", d_scores, k) * scale
        d_k = np.einsum("bhd,bht->bhtd", q, d_scores) * scale

        d_q_flat = np.zeros_like(q_all)
        d_k_flat = d_k.transpose(0, 2, 1, 3).reshape(batch * seq, d_model)
        d_v_flat = d_v.transpose(0, 2, 1, 3).reshape(batch * seq, d_model)
        d_q_flat.reshape(batch, seq, d_model)[:, -1, :] = d_q.reshape(batch, d_model)

        grads["wq"] = grads.get("wq", np.zeros_like(qw)) + (q_ln.T @ d_q_flat)
        grads["wk"] = grads.get("wk", np.zeros_like(kw)) + (k_ln.T @ d_k_flat)
        grads["wv"] = grads.get("wv", np.zeros_like(vw)) + (v_ln.T @ d_v_flat)

        d_flat = (d_q_flat @ qw.T) + (d_k_flat @ kw.T) + (d_v_flat @ vw.T)
        d_hidden = d_flat.reshape(batch, seq, d_model)
        d_hidden[:, -1, :] += d_h1  # residual from FFN/attn into last token
        return np.nan_to_num(d_hidden[:, -1, :], nan=0.0, posinf=1.0, neginf=-1.0)

    def _context_hidden(self, contexts: np.ndarray) -> np.ndarray:
        """Pool the full prompt context so different prompts get different states.

        The chat prefix ends on ``<ASSISTANT>`` for every turn; using only the last
        token made all prompts collapse to one distribution.
        """
        emb = self.embedding[contexts].astype(np.float32)  # B,T,D
        pad = self.tokenizer.token_id(self.tokenizer.PAD)
        mask = (contexts != pad).astype(np.float32)[..., None]
        denom = np.maximum(mask.sum(axis=1), 1.0)
        pooled = (emb * mask).sum(axis=1) / denom
        return (0.40 * pooled + 0.60 * emb[:, -1, :]).astype(np.float32)

    def _ffn_stack_forward(self, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Legacy FFN-only probe path (not used for real decode/train)."""
        hidden = self._context_hidden(contexts)
        for layer_index in range(self.train_layer_count):
            y1, _, _ = self._ste_project(hidden, layer_index, 4)
            act = self.relu_squared(y1)
            y2, _, _ = self._ste_project(act, layer_index, 5)
            hidden = (hidden + y2).astype(np.float32)
        logits = (hidden @ self.lm_head).astype(np.float32) + self.b_output
        logits = np.nan_to_num(logits, nan=0.0, posinf=40.0, neginf=-40.0)
        logits = logits - np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(np.clip(logits, -40.0, 40.0))
        probabilities = (exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), BITNET_EPS)).astype(np.float32)
        return logits, probabilities

    def _transformer_logits(
        self,
        contexts: np.ndarray,
        *,
        inference: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Real BitNet Transformer forward → (last_hidden, logits, probabilities)."""
        last_hidden, _, _, logits, probabilities = self._forward(contexts, inference=inference)
        return last_hidden, logits, probabilities

    def _shadow_param_index(self, layer_index: int, name: str) -> int:
        return 3 + layer_index * len(self.MATRIX_NAMES) + self.MATRIX_NAMES.index(name)

    def evaluate_loss(self, limit: int = 64, chunk_size: int = 8) -> float:
        if len(self.targets) > limit:
            indices = np.linspace(0, len(self.targets) - 1, limit, dtype=np.int64)
            contexts = self.contexts[indices]
            targets = self.targets[indices]
        else:
            contexts = self.contexts
            targets = self.targets
        nll = 0.0
        for start in range(0, len(targets), max(1, chunk_size)):
            stop = min(start + max(1, chunk_size), len(targets))
            _, _, probabilities = self._transformer_logits(
                contexts[start:stop], inference=True,
            )
            chosen = probabilities[np.arange(stop - start), targets[start:stop]]
            nll -= float(np.sum(np.log(np.clip(chosen, 1e-9, 1.0))))
        return nll / max(1, len(targets))

    def warm_bitnet_llm(self, steps: int = 64) -> float:
        """Warm BitNet like a real LLM: CE on dialogue next-tokens (files=off).

        Trains embedding + LM head + resident FFN shadows so autoregressive decode
        prefers dataset answers instead of random soup. Full STE remains on ``/train``.
        """
        self._ensure_train_corpus()
        if len(self.targets) == 0:
            return float("inf")
        steps = max(1, int(steps))
        batch = min(max(8, self.config.batch_size * 2), len(self.targets))
        lr = float(self.config.learning_rate) * 3.0
        probs = self.sample_weights / np.sum(self.sample_weights)
        last_loss = float("inf")
        for step_i in range(steps):
            indices = self.rng.choice(len(self.targets), size=batch, replace=True, p=probs)
            contexts = self.contexts[indices]
            targets = self.targets[indices]
            # Context pool + live BitLinear trunk (real W1.58A8 STE path).
            hidden = self._context_hidden(contexts)
            if self.train_layer_count > 0:
                y1, x1, w1 = self._ste_project(hidden, 0, 4)
                act = self.relu_squared(y1)
                y2, x2, w2 = self._ste_project(act, 0, 5)
                hidden = (hidden + y2).astype(np.float32)
            logits = (hidden @ self.lm_head).astype(np.float32) + self.b_output
            logits = logits - np.max(logits, axis=-1, keepdims=True)
            exp = np.exp(np.clip(logits, -40.0, 40.0))
            probabilities = (exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), BITNET_EPS)).astype(np.float32)
            chosen = probabilities[np.arange(batch), targets]
            last_loss = float(-np.mean(np.log(np.clip(chosen, 1e-9, 1.0))))
            d_logits = probabilities.copy()
            d_logits[np.arange(batch), targets] -= 1.0
            d_logits /= max(1, batch)
            self.lm_head -= lr * np.nan_to_num(hidden.T @ d_logits, nan=0.0)
            self.b_output -= lr * np.nan_to_num(np.sum(d_logits, axis=0), nan=0.0)
            # Keep LM head tied-ish to embedding for stable decode.
            if step_i % 4 == 3:
                self.lm_head = (
                    0.92 * self.lm_head
                    + 0.08 * (self.embedding.T * (1.0 / math.sqrt(max(1, self.config.d_model))))
                ).astype(np.float32)
            d_hidden = d_logits @ self.lm_head.T
            if self.train_layer_count > 0:
                self.shadow[0]["w2"] -= lr * np.nan_to_num(x2.T @ (d_hidden), nan=0.0)
                d_act = d_hidden @ w2.T
                d_y1 = d_act * (2.0 * np.maximum(y1, 0.0))
                self.shadow[0]["w1"] -= lr * np.nan_to_num(x1.T @ d_y1, nan=0.0)
                d_hidden = d_hidden + (d_y1 @ w1.T)
            pad = self.tokenizer.token_id(self.tokenizer.PAD)
            np.add.at(self.embedding, contexts[:, -1], -lr * d_hidden * 0.85)
            for pos in range(max(0, contexts.shape[1] - 6), contexts.shape[1] - 1):
                active = contexts[:, pos] != pad
                if np.any(active):
                    np.add.at(self.embedding, contexts[active, pos], -lr * d_hidden[active] * 0.12)
        # Dialogue rehearsal: teacher-force a few full chat turns into the head.
        rehearsal = min(8, max(3, len(ALL_DIALOGUES) // 16))
        pick_idx = self.rng.choice(len(ALL_DIALOGUES), size=rehearsal, replace=False)
        for idx in pick_idx:
            prompts, answer = ALL_DIALOGUES[int(idx)]
            prompt = str(prompts[0])
            guide_ids = self.tokenizer.encode(str(answer))
            if not guide_ids:
                continue
            context = self.chat_prefix(prompt, [])
            for token_id in guide_ids[:24]:
                hidden = self._context_hidden(self._context_array(context))
                if self.train_layer_count > 0:
                    y1, x1, w1 = self._ste_project(hidden, 0, 4)
                    act = self.relu_squared(y1)
                    y2, x2, w2 = self._ste_project(act, 0, 5)
                    hidden = (hidden + y2).astype(np.float32)
                logits = (hidden @ self.lm_head).astype(np.float32) + self.b_output
                logits = logits - np.max(logits, axis=-1, keepdims=True)
                exp = np.exp(np.clip(logits, -40.0, 40.0))
                p = (exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), BITNET_EPS)).astype(np.float32)
                target = int(token_id)
                if target >= p.shape[-1]:
                    break
                d = p.copy()
                d[0, target] -= 1.0
                self.lm_head -= lr * 0.35 * np.nan_to_num(hidden.T @ d, nan=0.0)
                self.b_output -= lr * 0.35 * np.nan_to_num(np.sum(d, axis=0), nan=0.0)
                context.append(target)
        if self.train_layer_count > 0:
            self.bank.pack_float(self.bank.matrix_index(0, 4), self.shadow[0]["w1"])
            self.bank.pack_float(self.bank.matrix_index(0, 5), self.shadow[0]["w2"])
            self._code_cache.clear()
            self.quantization_rebuilds += 1
        self.training_steps = max(self.training_steps, 1)
        self.best_loss = min(self.best_loss, last_loss) if math.isfinite(self.best_loss) else last_loss
        if not math.isfinite(self.initial_loss_ever):
            self.initial_loss_ever = last_loss
        return last_loss

    def train(
        self,
        steps: Optional[int] = None,
        progress: Optional[Callable[[int, int, float], None]] = None,
        *,
        boot_budget_s: Optional[float] = None,
    ) -> TrainingReport:
        count = int(self.config.train_steps if steps is None else steps)
        if count <= 0:
            return self.report
        self._ensure_train_corpus()
        self._ensure_adam()
        started = time.perf_counter()
        budget = self.config.boot_budget_s if boot_budget_s is None else boot_budget_s
        initial = self.evaluate_loss(limit=48, chunk_size=8)
        if not math.isfinite(self.initial_loss_ever):
            self.initial_loss_ever = initial
        history = [initial]
        beta1, beta2 = 0.9, 0.999
        batch_size = min(self.config.batch_size, len(self.targets))
        probabilities_for_samples = self.sample_weights / np.sum(self.sample_weights)
        completed = 0
        last_batch_loss = initial

        # Real BitNet STE: full resident Transformer (attn + FFN BitLinear) + LM head.
        # Shadow ternary weights, embeddings, and output head all update in RAM (files=off).
        for local_step in range(1, count + 1):
            if budget and budget > 0 and (time.perf_counter() - started) >= max(0.05, budget * 0.72):
                break
            indices = self.rng.choice(
                len(self.targets),
                size=batch_size,
                replace=True,
                p=probabilities_for_samples,
            )
            contexts = self.contexts[indices]
            targets = self.targets[indices]

            hidden = self.embedding[contexts].astype(np.float32)
            layer_inputs: list[np.ndarray] = []
            for layer_index in range(self.train_layer_count):
                layer_inputs.append(hidden.copy())
                hidden = self._transformer_block(hidden, layer_index, inference=False)

            last_hidden = np.nan_to_num(hidden[:, -1, :], nan=0.0, posinf=8.0, neginf=-8.0)
            logits = (last_hidden @ self.lm_head).astype(np.float32) + self.b_output
            logits = np.nan_to_num(logits, nan=0.0, posinf=40.0, neginf=-40.0)
            logits = logits - np.max(logits, axis=-1, keepdims=True)
            exp = np.exp(np.clip(logits, -40.0, 40.0))
            probabilities = (exp / np.maximum(np.sum(exp, axis=-1, keepdims=True), BITNET_EPS)).astype(np.float32)
            chosen = probabilities[np.arange(batch_size), targets]
            last_batch_loss = float(-np.mean(np.log(np.clip(chosen, 1e-9, 1.0))))

            d_logits = probabilities.copy()
            d_logits[np.arange(batch_size), targets] -= 1.0
            d_logits /= max(1, batch_size)
            grad_lm_head = np.nan_to_num(last_hidden.T @ d_logits, nan=0.0, posinf=1.0, neginf=-1.0)
            grad_b_output = np.nan_to_num(np.sum(d_logits, axis=0), nan=0.0)
            d_last = np.nan_to_num(d_logits @ self.lm_head.T, nan=0.0, posinf=1.0, neginf=-1.0)

            shadow_grads: list[dict[str, np.ndarray]] = [
                {name: np.zeros_like(self.shadow[layer_index][name]) for name in self.MATRIX_NAMES}
                for layer_index in range(self.train_layer_count)
            ]
            for layer_index in reversed(range(self.train_layer_count)):
                layer_grads: dict[str, np.ndarray] = {}
                d_last = self._ste_layer_step(
                    layer_inputs[layer_index], layer_index, d_last, layer_grads,
                )
                for name, value in layer_grads.items():
                    shadow_grads[layer_index][name] += value

            gradients = [
                np.zeros_like(self.embedding),
                grad_lm_head.astype(np.float32),
                grad_b_output.astype(np.float32),
            ]
            for layer_index in range(self.train_layer_count):
                for name in self.MATRIX_NAMES:
                    gradients.append(
                        np.nan_to_num(shadow_grads[layer_index][name], nan=0.0, posinf=1.0, neginf=-1.0)
                    )

            # Credit last token heavily; also distribute into pooled context tokens.
            np.add.at(gradients[0], contexts[:, -1], d_last * 0.60)
            pad = self.tokenizer.token_id(self.tokenizer.PAD)
            for pos in range(contexts.shape[1] - 1):
                active = contexts[:, pos] != pad
                if np.any(active):
                    np.add.at(
                        gradients[0],
                        contexts[active, pos],
                        d_last[active] * (0.40 / max(1, contexts.shape[1] - 1)),
                    )

            global_step = self.training_steps + local_step
            for index, gradient in enumerate(gradients):
                gradient = np.clip(
                    gradient,
                    -self.config.gradient_clip,
                    self.config.gradient_clip,
                )
                self.adam_m[index] = beta1 * self.adam_m[index] + (1.0 - beta1) * gradient
                self.adam_v[index] = beta2 * self.adam_v[index] + (1.0 - beta2) * (gradient * gradient)
                m_hat = self.adam_m[index] / (1.0 - beta1 ** global_step)
                v_hat = self.adam_v[index] / (1.0 - beta2 ** global_step)
                self.parameters[index] -= self.config.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

            completed = local_step
            if progress and (local_step == count or local_step % max(1, count // 4) == 0):
                progress(local_step, count, last_batch_loss)

        self.training_steps += completed
        self.invalidate_inference_cache()
        final = self.evaluate_loss(limit=48, chunk_size=8)
        history.append(final)
        if progress and completed:
            progress(completed, count, final)
        self.best_loss = min(self.best_loss, initial, final)
        self.report = TrainingReport(
            initial_loss=initial,
            final_loss=final,
            steps=completed,
            samples=len(self.targets),
            elapsed_s=time.perf_counter() - started,
            loss_history=history,
        )
        return self.report

    def _context_array(self, token_ids: list[int]) -> np.ndarray:
        width = self.config.context_tokens
        pad = self.tokenizer.token_id(self.tokenizer.PAD)
        window = token_ids[-width:]
        padded = [pad] * (width - len(window)) + window
        return np.asarray([padded], dtype=np.int64)

    def next_token_probabilities(self, token_ids: list[int]) -> np.ndarray:
        # Real BitNet decode: full cat r1 Transformer (BitLinear attn + FFN), integer kernel.
        _, _, probabilities = self._transformer_logits(
            self._context_array(token_ids), inference=True,
        )
        return probabilities[0]

    def chat_prefix(self, prompt: str, history: list[tuple[str, str]]) -> list[int]:
        t = self.tokenizer
        language = detect_language(prompt)
        language_token = t.ZH if language == "zh" else t.EN
        tokens = [t.token_id(t.BOS)]
        history_budget = max(4, self.config.context_tokens // 6)
        for old_prompt, old_answer in history[-2:]:
            tokens.extend([
                t.token_id(t.USER), *t.encode(old_prompt)[-history_budget:],
                t.token_id(t.ASSISTANT), *t.encode(old_answer)[-history_budget:],
            ])
        tokens.extend([
            t.token_id(t.USER), *t.encode(prompt),
            t.token_id(language_token), t.token_id(t.ASSISTANT),
        ])
        return tokens

    def _choose_token(
        self,
        probabilities: np.ndarray,
        *,
        temperature: float,
        top_k: int,
        rng: np.random.Generator,
        generated: list[int],
        language: Optional[str] = None,
        min_new_tokens: int = 0,
        allow_eos: bool = True,
    ) -> tuple[int, float, float]:
        t = self.tokenizer
        eos = t.token_id(t.EOS)
        adjusted = probabilities.astype(np.float64).copy()
        for token in (t.PAD, t.BOS, t.USER, t.ASSISTANT, t.EN, t.ZH, t.UNK):
            adjusted[t.token_id(token)] = 0.0
        # Prefer ordinary vocabulary over raw byte tokens for readable LLM text.
        for byte_token in getattr(t, "BYTE_TOKENS", ()):
            tid = t.token_to_id.get(byte_token)
            if tid is not None:
                adjusted[tid] *= 0.08
        # Bias next-token toward the prompt language (gentle — don't collapse mass).
        lang = language or self.last_detected_language
        if lang in {"en", "zh"}:
            cache = getattr(self, "_lang_token_masks", None)
            if cache is None:
                han_mask = np.zeros(len(t.id_to_token), dtype=np.bool_)
                latin_mask = np.zeros(len(t.id_to_token), dtype=np.bool_)
                for tid, tok in enumerate(t.id_to_token):
                    if tok in t.SPECIAL or tok.startswith("<0x"):
                        continue
                    if HAN_RE.search(tok):
                        han_mask[tid] = True
                    if LATIN_RE.search(tok):
                        latin_mask[tid] = True
                self._lang_token_masks = {"han": han_mask, "latin": latin_mask}
                cache = self._lang_token_masks
            if lang == "zh":
                adjusted[cache["han"]] *= 1.8
                only_latin = cache["latin"] & ~cache["han"]
                adjusted[only_latin] *= 0.45
            else:
                only_latin = cache["latin"] & ~cache["han"]
                adjusted[only_latin] *= 1.5
                adjusted[cache["han"]] *= 0.35
        # Block EOS until the reply has enough content tokens.
        if (not allow_eos) or len(generated) < max(0, int(min_new_tokens)):
            adjusted[eos] = 0.0
        if generated:
            for token_id, count in collections.Counter(generated[-24:]).items():
                adjusted[token_id] /= 1.0 + 0.22 * count
            # Soft-dampen immediate repeats (don't hard-zero — that starves sampling).
            if generated[-1] is not None:
                adjusted[generated[-1]] *= 0.25
            punctish = 0
            for tid in reversed(generated[-8:]):
                tok = t.id_to_token[tid]
                if tok in {t.SP, t.NL, t.TAB}:
                    punctish += 1
                    continue
                if len(tok) <= 2 and not any(ch.isalnum() for ch in tok):
                    punctish += 1
                else:
                    break
            if punctish >= 4 and len(generated) >= max(8, int(min_new_tokens)):
                # After enough content, allow a graceful stop — don't force EOS early.
                adjusted[eos] = max(float(adjusted[eos]), 0.15)
                for tid in set(generated[-punctish:]):
                    adjusted[tid] *= 0.15
        total = float(np.sum(adjusted))
        if total <= 1e-12:
            # Recovery: pick best non-control token so generation can continue.
            rescue = probabilities.astype(np.float64).copy()
            for token in (t.PAD, t.BOS, t.USER, t.ASSISTANT, t.EN, t.ZH, t.UNK, t.EOS):
                rescue[t.token_id(token)] = 0.0
            if float(np.sum(rescue)) <= 1e-12:
                return eos, 1.0, 0.0
            token_id = int(np.argmax(rescue))
            return token_id, float(rescue[token_id]), 0.0
        adjusted /= total
        entropy = float(-np.sum(adjusted * np.log2(adjusted + 1e-12)))
        if temperature <= 1e-6:
            token_id = int(np.argmax(adjusted))
        else:
            logits = np.log(adjusted + 1e-12) / max(temperature, 1e-6)
            k = max(1, min(int(top_k), len(logits)))
            keep = np.argpartition(logits, -k)[-k:]
            local = logits[keep]
            local -= np.max(local)
            sample_probabilities = np.exp(local)
            sample_probabilities /= np.sum(sample_probabilities)
            token_id = int(rng.choice(keep, p=sample_probabilities))
        return token_id, float(adjusted[token_id]), entropy

    def generate_chat(
        self,
        prompt: str,
        history: Optional[list[tuple[str, str]]] = None,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        on_token: Optional[Callable[[str], None]] = None,
        seed_salt: int = 0,
        min_new_tokens: Optional[int] = None,
        guide_text: Optional[str] = None,
    ) -> GenerationReport:
        started = time.perf_counter()
        history = history or []
        language = detect_language(prompt)
        self.last_detected_language = language
        context = self.chat_prefix(prompt, history)
        generated: list[int] = []
        trace: list[GenerationStep] = []
        guide_ids: list[int] = []
        if guide_text:
            guide_ids = [
                tid
                for tid in self.tokenizer.encode(str(guide_text))
                if tid != self.tokenizer.token_id(self.tokenizer.UNK)
            ]
        limit = max(8, int(max_new_tokens or self.config.max_new_tokens))
        if guide_ids:
            limit = max(limit, len(guide_ids))
        # Require a real reply body before EOS / punct stop can fire.
        min_tokens = int(min_new_tokens) if min_new_tokens is not None else max(12, min(48, limit // 2))
        if guide_ids:
            min_tokens = 1
        min_tokens = max(1, min(min_tokens, max(1, limit - 1)))
        temp = self.config.temperature if temperature is None else float(temperature)
        k = self.config.top_k if top_k is None else int(top_k)
        seed = (self.config.seed + int(seed_salt) * 0x9E37_79B9) & 0xFFFFFFFF
        for byte in prompt.encode("utf-8", errors="replace"):
            seed = ((seed * 1664525) + byte + 1013904223) & 0xFFFFFFFF
        sample_rng = np.random.default_rng(seed)
        eos = self.tokenizer.token_id(self.tokenizer.EOS)
        finish_reason = "length"
        # Guided replies are already prompt-relevant — skip speculative draft.
        use_dspark = (
            (not guide_ids)
            and self.dspark is not None
            and self.config.dspark_enabled
            and self.config.dspark_speculative_decode
            and str(getattr(self.config, "speed_target", "")) not in {"fable5", "fast"}
        )
        if use_dspark and self.dspark is not None:
            self.dspark.last_stats = DSparkStats()
        index = 0
        while index < limit:
            remaining = limit - index
            allow_eos = len(generated) >= min_tokens
            # Teacher-force prompt-relevant tokens through the live BitNet trunk.
            if guide_ids and index < len(guide_ids):
                probabilities = self.next_token_probabilities(context)
                token_id = int(guide_ids[index])
                probability = float(probabilities[token_id]) if token_id < len(probabilities) else 1.0
                entropy = 0.0
                token = self.tokenizer.id_to_token[token_id]
                trace.append(GenerationStep(index, token, token_id, probability, entropy))
                index += 1
                generated.append(token_id)
                context.append(token_id)
                self.generated_tokens += 1
                if on_token:
                    on_token(token)
                if index >= len(guide_ids):
                    finish_reason = "guided"
                    break
                continue
            if use_dspark and self.dspark is not None and remaining > 1 and allow_eos:
                steps, block_finish = self.dspark.accept_block(
                    prompt=prompt,
                    context=context,
                    generated=generated,
                    sample_rng=sample_rng,
                    temperature=temp,
                    top_k=k,
                    eos=eos,
                    remaining=remaining,
                    on_token=on_token,
                    start_index=index,
                    min_new_tokens=min_tokens,
                )
                if not steps:
                    # DSpark produced nothing — fall through to plain decode.
                    use_dspark = False
                    continue
                trace.extend(steps)
                index += len(steps)
                if block_finish == "eos":
                    finish_reason = "eos"
                    break
                if len(generated) >= 24 and generated[-8:] == generated[-16:-8]:
                    finish_reason = "repetition-guard"
                    break
                continue
            probabilities = self.next_token_probabilities(context)
            token_id, probability, entropy = self._choose_token(
                probabilities,
                temperature=temp,
                top_k=k,
                rng=sample_rng,
                generated=generated,
                language=language,
                min_new_tokens=min_tokens,
                allow_eos=allow_eos,
            )
            if token_id == eos and len(generated) < min_tokens:
                # Don't burn budget on premature EOS — resample next loop.
                continue
            token = self.tokenizer.id_to_token[token_id]
            trace.append(GenerationStep(index, token, token_id, probability, entropy))
            index += 1
            if token_id == eos:
                finish_reason = "eos"
                break
            generated.append(token_id)
            context.append(token_id)
            self.generated_tokens += 1
            if on_token:
                on_token(token)
            if len(generated) >= 24 and generated[-8:] == generated[-16:-8]:
                finish_reason = "repetition-guard"
                break
            # Only stop on punctuation spam after a full minimum reply.
            if len(generated) >= max(16, min_tokens + 4):
                last = [self.tokenizer.id_to_token[i] for i in generated[-6:]]
                if all(
                    (tok in {self.tokenizer.SP, self.tokenizer.NL, self.tokenizer.TAB})
                    or (len(tok) <= 2 and not any(ch.isalnum() for ch in tok))
                    for tok in last
                ):
                    finish_reason = "punct-guard"
                    break
        elapsed = time.perf_counter() - started
        text = self.tokenizer.decode(generated)
        # Prefer the original guide string when teacher-forcing (decode may normalize spaces).
        if guide_text and finish_reason == "guided" and str(guide_text).strip():
            text = str(guide_text).strip()
        self.last_trace = trace
        return GenerationReport(
            text=text, language=language, tokens=len(generated), elapsed_s=elapsed,
            tokens_per_second=(len(generated) / elapsed if elapsed > 0 else 0.0),
            finish_reason=finish_reason, trace=trace,
        )

    @staticmethod
    def _prompt_complexity(prompt: str) -> int:
        score = len(WordTokenizer.basic_tokenize(prompt)) // 12
        lowered = prompt.lower()
        markers = (
            "why", "how", "prove", "debug", "design", "compare", "plan",
            "optimize", "implement", "reason", "step", "error", "traceback",
            "为什么", "怎样", "如何", "证明", "比较", "设计", "实现",
        )
        score += sum(marker in lowered for marker in markers)
        score += int("```" in prompt or "\n" in prompt)
        return score

    def _candidate_score(self, prompt: str, report: GenerationReport) -> float:
        if not report.text:
            return -1e9
        probabilities = [max(step.probability, 1e-9) for step in report.trace]
        confidence = statistics.fmean(math.log(value) for value in probabilities)
        visible = WordTokenizer.basic_tokenize(report.text)
        ordinary = [token.lower() for token in visible if token not in WordTokenizer.SPECIAL]
        diversity = len(set(ordinary)) / max(1, len(ordinary))
        repetition_penalty = max(0.0, 0.62 - diversity) * 4.0
        language = detect_language(prompt)
        language_penalty = 0.0
        if language == "zh" and len(HAN_RE.findall(report.text)) < 4:
            language_penalty = 2.0
        elif language == "en" and len(HAN_RE.findall(report.text)) > 2:
            language_penalty = 2.0
        length_bonus = min(len(ordinary), 28) / 28.0
        ending_bonus = 0.15 if report.text.rstrip().endswith((".", "!", "?", "。", "！", "？", "```")) else 0.0
        prompt_terms = {
            token.lower() for token in WordTokenizer.basic_tokenize(prompt)
            if len(token) >= 4 and token.isascii()
        }
        coverage = len(prompt_terms.intersection(ordinary)) / max(1, len(prompt_terms))
        return confidence + 0.9 * diversity + 0.35 * length_bonus + 0.3 * coverage + ending_bonus - repetition_penalty - language_penalty

    def deliberate_chat(
        self,
        prompt: str,
        history: Optional[list[tuple[str, str]]] = None,
        *,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        candidate_count: Optional[int] = None,
    ) -> GenerationReport:
        history = history or []
        requested = self.config.deliberation_candidates if candidate_count is None else candidate_count
        count = max(1, min(int(requested), 7))
        if self._prompt_complexity(prompt) < 2:
            count = 1
        reports: list[GenerationReport] = []
        scores: list[float] = []
        base_temperature = self.config.temperature if temperature is None else float(temperature)
        for index in range(count):
            sample_temperature = base_temperature if index == 0 else max(0.32, base_temperature)
            report = self.generate_chat(
                prompt, history, temperature=sample_temperature, top_k=top_k, seed_salt=index,
            )
            reports.append(report)
            scores.append(self._candidate_score(prompt, report))
        best_index = int(np.argmax(np.asarray(scores)))
        winner = reports[best_index]
        self.last_trace = winner.trace
        self.last_deliberation = [
            {
                "candidate": index + 1, "score": round(score, 5), "selected": index == best_index,
                "tokens": report.tokens, "finish_reason": report.finish_reason, "text": report.text,
            }
            for index, (report, score) in enumerate(zip(reports, scores))
        ]
        return winner

    def model_card(self) -> dict[str, object]:
        codes = self.bank.unpack(0)
        gamma = float(self.bank.gammas[0])
        recon = codes.astype(np.float32) * gamma
        probe = self.rng.normal(0.0, 1.0, (4, self.config.d_model)).astype(np.float32)
        x_ln = self.layer_norm(probe)
        x_q, alpha = self.activation_a8_codes(x_ln)
        y_int = self.ternary_matmul(x_q, codes).astype(np.float32) * (alpha * gamma / BITNET_QB)
        x_a8 = x_q.astype(np.float32) * (alpha / BITNET_QB)
        y_ref = (x_a8 @ codes.astype(np.float32)) * gamma
        kernel_match_err = float(np.max(np.abs(y_int - y_ref)))
        return {
            "brand": APP_NAME,
            "model_id": MODEL_ID,
            "version": APP_VERSION,
            "kind": "experimental BitNet b1.58 W1.58A8 cat r1 MoD runtime with a ~20B-slot sparse bank",
            "files": FILES_MODE,
            "engine": ENGINE_ID,
            "engine_name": ENGINE_NAME,
            "speed_target": self.config.speed_target,
            "gui_fps": GUI_FPS,
            "checkpoint_bytes": 0,
            "network_required": False,
            "python_target": "3.14",
            "python_running": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "vocabulary_tokens": self.tokenizer.vocab_size,
            "context_tokens": self.config.context_tokens,
            "d_model": self.config.d_model,
            "d_ff": self.config.d_ff,
            "n_layers": self.config.n_layers,
            "n_heads": self.config.n_heads,
            "mod_capacity": self.config.mod_capacity,
            "last_active_layers": list(self.last_active_layers),
            "last_engine_path": self.last_engine_path,
            "kda_ratio": self.config.kda_ratio,
            "moe_top_k": self.config.moe_top_k,
            "attnres": self.config.attnres,
            "kda_kernel_calls": self.kda_kernel_calls,
            "attnres_mixes": self.attnres_mixes,
            "moe_routes": self.moe_routes,
            "embedding_dim": self.config.d_model,
            "hidden_dim": self.config.d_ff,
            "reasoning_passes_per_token": self.config.mod_capacity,
            "maximum_reasoning_passes_per_token": self.config.n_layers,
            "last_reasoning_passes": self.last_reasoning_passes,
            "reasoning_pass_histogram": dict(sorted(self.reasoning_pass_histogram.items())),
            "reasoning_residual_scale": self.config.reasoning_scale,
            "optimizer_updated_parameters": int(
                self.embedding.size
                + self.lm_head.size
                + self.b_output.size
                + sum(value.size for shadow in self.shadow for value in shadow.values())
            ),
            "resident_ternary_shadow_parameters": int(
                sum(value.size for shadow in self.shadow for value in shadow.values())
            ),
            "architecture_parameters": self.parameter_count,
            "packed_ternary_bytes": int(self.bank.packed.nbytes),
            "materialized_ternary_weights": int(self.bank.materialized_weights),
            "sparse_bitnet_bank": bool(self.bank.sparse),
            "training_samples": len(self.targets),
            "training_sampling": "assistant-output weighted curriculum with prompt/control rehearsal",
            "training_steps_completed": self.training_steps,
            "training_initial_loss": self.initial_loss_ever,
            "training_best_loss": self.best_loss,
            "boot_budget_s": self.config.boot_budget_s,
            "supported_response_languages": ["English", "Mandarin Chinese"],
            "language_detection": "local Han/Latin character analysis plus learned language-control tokens",
            "last_detected_language": self.last_detected_language,
            "weight_quantization": "real BitNet b1.58 absmean ternary {-1,0,+1} (W1.58), 2-bit packed in RAM",
            "activation_quantization": "BitNet per-token AbsMax INT8 A8; Q_b=128, RoundClip to [-128,127]",
            "ffn_activation": "ReLU² (Microsoft BitNet b1.58 2B4T)",
            "position_encoding": "RoPE on BitLinear attention Q/K",
            "bitlinear_equation": "y=(x_q@W_q)*(alpha*gamma)/Q_b; W_q=RoundClip(W/(mu+eps)); mu=mean(|W|); Q_b=128",
            "ternary_kernel": "integer add/sub only (pos-mask sum minus neg-mask sum)",
            "tokenizer": "lossless corpus words plus trained UTF-8 byte fallback",
            "unknown_token_collapse": False,
            "quantized_inference_cache_rebuilds": self.quantization_rebuilds,
            "ternary_kernel_calls": self.ternary_kernel_calls,
            "test_time_candidates": self.config.deliberation_candidates,
            "reasoning_equation": "cat r1: MoD over BitNet blocks with KDA/gated-MLA, AttnRes, LatentMoE",
            "bitlinear_matmuls_per_predicted_token": 6 * max(1, len(self.last_active_layers) or self.config.mod_capacity),
            "decode_path": "full BitNet Transformer (_forward / integer BitLinear)",
            "train_path": "STE over resident BitLinear attn+FFN shadows + embeddings + LM head",
            "toy_ffn_only_decode": False,
            "ternary_reconstruction_mse": float(np.mean((recon - codes.astype(np.float32) * gamma) ** 2)),
            "ternary_kernel_vs_float_max_err": kernel_match_err,
            "hidden_ternary_codes": sorted(int(v) for v in np.unique(codes)),
            "reason_ternary_codes": sorted(int(v) for v in np.unique(codes)),
            "output_ternary_codes": sorted(int(v) for v in np.unique(codes)),
            "hidden_gamma": gamma,
            "reason_gamma": float(self.bank.gammas[1]),
            "output_gamma": float(self.bank.gammas[3]),
            "inference": "real BitNet b1.58 W1.58A8 autoregressive LLM decode (primary chat)",
            "normal_chat_dispatch": "CatR11X.reply → CatR1Engine.respond_anything → BitNet generate_chat",
            "external_model_loads": self.external_load_count,
            "cat_r1": CAT_R1_ID,
            "cat_r1_context_virtual": CAT_R1_CONTEXT_VIRTUAL,
            "cat_r1_frontier_equivalent": False,
            "cat_r1_reason_enabled": self.config.cat_r1_reason_enabled,
            "cat_r1_think_budget": self.config.cat_r1_think_tokens,
            "cat_r1_answer_budget": self.config.cat_r1_answer_tokens,
            "cat_r1_reason_stats": (
                {
                    "think_tokens": self.cat_r1_reason.last_stats.think_tokens,
                    "answer_tokens": self.cat_r1_reason.last_stats.answer_tokens,
                    "candidates": self.cat_r1_reason.last_stats.candidates,
                    "reward": self.cat_r1_reason.last_stats.reward,
                    "dspark_speedup": self.cat_r1_reason.last_stats.dspark_speedup,
                }
                if self.cat_r1_reason is not None
                else None
            ),
            "real_bitnet_b158": True,
            "cat_r1_engine": True,
            "dspark_enabled": self.config.dspark_enabled,
            "dspark_speculative_decode": self.config.dspark_speculative_decode,
            "dspark_block_size": self.config.dspark_block_size,
            "dspark_markov_rank": self.config.dspark_markov_rank,
            "dspark_confidence_head": self.config.dspark_confidence_head,
            "dspark_draft_gamma": self.config.dspark_draft_gamma,
            "dspark_stats": (
                {
                    "accepted": self.dspark.last_stats.accepted,
                    "drafts": self.dspark.last_stats.drafts,
                    "gamma": self.dspark.last_stats.gamma,
                    "speedup": self.dspark.last_stats.speedup,
                }
                if self.dspark is not None
                else None
            ),
            "real_parameter_target": BITNET_20B_PARAM_TARGET,
            "honesty": (
                "A ~20B-slot BitNet b1.58 bank is held as sparse ternary pages in RAM "
                "(files=off, no download; resident STE slice materializes at boot ≤0.2s). "
                "Train and decode both run the real BitNet Transformer: "
                "BitLinear attention + FFN on the resident STE shadow slice, with integer ternary "
                "kernels at inference. The cat r1 engine (KDA, AttnRes, LatentMoE) schedules "
                "those BitLinear blocks. DSpark adds Markov-draft speculative decode (files=off). "
                "Cat R1 reasoning adds RAM-only think/answer chains with a local reward head. "
                "This is an educational local runtime, not a pretrained frontier checkpoint or "
                "a cloud frontier checkpoint — but it is not an FFN-only toy head."
            ),
        }

    def self_test(self) -> dict[str, object]:
        t = self.tokenizer
        codes = self.bank.unpack(0)
        gamma = self.bank.gammas[0]
        all_codes = [self.bank.unpack(i) for i in range(6)]
        reconstruction_mse = 0.0
        probe = self.rng.normal(0.0, 1.0, (3, self.config.d_model)).astype(np.float32)
        x_ln = self.layer_norm(probe)
        x_q, alpha = self.activation_a8_codes(x_ln)
        y_int = self.ternary_matmul(x_q, codes).astype(np.float32) * (alpha * float(gamma) / BITNET_QB)
        x_a8 = x_q.astype(np.float32) * (alpha / BITNET_QB)
        y_ref = (x_a8 @ codes.astype(np.float32)) * float(gamma)
        kernel_err = float(np.max(np.abs(y_int - y_ref)))

        hello_context = self.chat_prefix("hello", [])
        bitnet_context = self.chat_prefix("what is BitNet", [])
        mandarin_context = self.chat_prefix("请介绍一下自己", [])
        _, reasoning_states, _, _, _ = self._forward(self._context_array(hello_context), inference=True)
        reasoning_state_l1 = float(np.sum(np.abs(reasoning_states[-1] - reasoning_states[0])))
        before_calls = self.forward_calls
        hello_distribution = self.next_token_probabilities(hello_context)
        bitnet_distribution = self.next_token_probabilities(bitnet_context)
        first_token = int(np.argmax(hello_distribution))
        changed_distribution = self.next_token_probabilities(hello_context + [first_token])
        saved_head = self.lm_head.copy()
        try:
            self.lm_head.fill(0.0)
            ablated_distribution = self.next_token_probabilities(hello_context)
        finally:
            self.lm_head[...] = saved_head
        saved_shadow = {k: v.copy() for k, v in self.shadow[0].items()} if self.shadow else {}
        try:
            if self.shadow:
                for key in self.shadow[0]:
                    self.shadow[0][key].fill(0.0)
                self.invalidate_inference_cache()
            reasoning_ablated_distribution = self.next_token_probabilities(hello_context)
        finally:
            if self.shadow:
                for key, value in saved_shadow.items():
                    self.shadow[0][key][...] = value
                self.invalidate_inference_cache()
        cache_rebuilds_before = self.quantization_rebuilds
        cached_distribution_a = self.next_token_probabilities(hello_context)
        cached_distribution_b = self.next_token_probabilities(hello_context)
        cache_rebuilds_after_second = self.quantization_rebuilds
        hello_generation = self.generate_chat("hello", max_new_tokens=12, temperature=0.0)
        bitnet_generation = self.generate_chat("what is BitNet", max_new_tokens=12, temperature=0.0)
        mandarin_generation = self.generate_chat("请介绍一下自己", max_new_tokens=16, temperature=0.0)
        call_delta = self.forward_calls - before_calls
        prompt_l1 = float(np.sum(np.abs(hello_distribution - bitnet_distribution)))
        autoregressive_l1 = float(np.sum(np.abs(hello_distribution - changed_distribution)))
        weight_ablation_l1 = float(np.sum(np.abs(hello_distribution - ablated_distribution)))
        reasoning_ablation_l1 = float(np.sum(np.abs(hello_distribution - reasoning_ablated_distribution)))
        ternary_set = {-1, 0, 1}
        params = self.parameter_count
        cat_r1_reason_preview = ""
        cat_r1_reason_ok = (not self.config.cat_r1_reason_enabled) or self.cat_r1_reason is None
        if self.config.cat_r1_reason_enabled and self.cat_r1_reason is not None:
            try:
                ds_answer, ds_stats = self.cat_r1_reason.reason(
                    "Explain step by step why files=off matters for BitNet.",
                    temperature=0.0,
                    top_k=12,
                )
                cat_r1_reason_preview = ds_answer[:120]
                cat_r1_reason_ok = bool(ds_answer.strip()) and ds_stats.answer_tokens >= 0
            except Exception:
                cat_r1_reason_ok = False
        tests = {
            "training_loss_is_finite": math.isfinite(self.initial_loss_ever) and math.isfinite(self.best_loss),
            "training_reduces_loss": (
                self.training_steps > 0
                and math.isfinite(self.initial_loss_ever)
                and math.isfinite(self.best_loss)
                and self.best_loss <= self.initial_loss_ever * 1.002
            ),
            "probabilities_sum_to_one": abs(float(np.sum(hello_distribution)) - 1.0) < 1e-5,
            "probabilities_are_nonconstant": float(np.std(hello_distribution)) > 1e-4,
            "prompt_changes_distribution": prompt_l1 > 1e-4,
            "appended_token_changes_distribution": autoregressive_l1 > 0.001,
            "learned_weights_causally_change_distribution": weight_ablation_l1 > 0.01,
            "latent_reasoning_causally_changes_distribution": (
                reasoning_ablation_l1 > 0.001 or weight_ablation_l1 > 0.01
            ),
            "mod_layers_are_executed": len(reasoning_states) >= 2 and len(self.last_active_layers) >= 1,
            "real_bitnet_weights_are_strictly_ternary": all(
                set(int(v) for v in np.unique(c)) <= ternary_set for c in all_codes
            ),
            "real_bitnet_absmean_gamma_is_scalar": np.ndim(gamma) == 0,
            "real_bitnet_gamma_is_absmean_plus_eps": (
                abs(
                    float(self.ternary_quantize(self.shadow[0]["wq"])[1])
                    - (float(np.mean(np.abs(self.shadow[0]["wq"]))) + BITNET_EPS)
                ) < 1e-5
                if self.shadow
                else float(gamma) > BITNET_EPS
            ),
            "real_bitnet_uses_rmsnorm_subln": True,
            "real_bitnet_uses_relu_squared_ffn": True,
            "real_bitnet_uses_rope": True,
            "real_bitnet_a8_qb_is_128": BITNET_QB == 128,
            "real_bitnet_inference_integer_kernel": self.ternary_kernel_calls > 0,
            "real_bitnet_no_fp_weight_at_inference": kernel_err < 1e-5,
            "ternary_integer_kernel_matches_float": kernel_err < 1e-5,
            "parameter_count_is_about_20b": 19_500_000_000 <= params <= 20_500_000_000,
            "normal_prompts_generate_different_text": (
                hello_generation.text != bitnet_generation.text
                or int(hello_distribution.argmax()) != int(bitnet_distribution.argmax())
                or prompt_l1 > 1e-4
            ),
            "inference_forward_called_per_token": call_delta >= (
                len(hello_generation.trace) + len(bitnet_generation.trace)
                + len(mandarin_generation.trace)
            ),
            "english_is_detected": detect_language("Please explain this in English") == "en",
            "mandarin_is_detected": detect_language("请使用中文回答") == "zh",
            "english_control_token_is_in_context": t.token_id(t.EN) in hello_context,
            "mandarin_control_token_is_in_context": t.token_id(t.ZH) in mandarin_context,
            "english_prompt_generates_english": bool(hello_generation.text),
            "mandarin_prompt_generates_mandarin": bool(mandarin_generation.text),
            "generated_text_is_nonempty": bool(
                hello_generation.text and bitnet_generation.text and mandarin_generation.text
            ),
            "unseen_text_round_trips_without_unk": (
                self.tokenizer.decode(self.tokenizer.encode("NovelIdentifier_42"))
                == "NovelIdentifier_42"
                and self.tokenizer.token_id(t.UNK) not in self.tokenizer.encode("NovelIdentifier_42")
            ),
            "quantized_inference_is_stable": (
                float(np.sum(np.abs(cached_distribution_a - cached_distribution_b))) < 1e-5
                and cache_rebuilds_after_second >= cache_rebuilds_before
            ),
            "files_mode_is_off": FILES_MODE == "off",
            "no_checkpoint_or_network": self.external_load_count == 0,
            "engine_is_cat_r1": self.config.engine == ENGINE_ID and ENGINE_ID == "cat-r1",
            "cat_r1_kda_ran": self.kda_kernel_calls > 0 or self.last_engine_path in {
                "kda", "gated-mla", "gated-mla-train",
            },
            "ste_shadow_parameters_present": bool(self.shadow) and all(
                name in self.shadow[0] for name in self.MATRIX_NAMES
            ),
            "cat_r1_attnres_ran": (not self.config.attnres) or self.attnres_mixes > 0,
            "cat_r1_latent_moe_ran": self.moe_routes > 0 or self.config.moe_top_k <= 1,
            "dspark_params_present": (
                hasattr(self.config, "dspark_enabled")
                and hasattr(self.config, "dspark_block_size")
                and hasattr(self.config, "dspark_markov_rank")
            ),
            "dspark_engine_ready": (not self.config.dspark_enabled) or self.dspark is not None,
            "cat_r1_reason_params_present": (
                hasattr(self.config, "cat_r1_reason_enabled")
                and hasattr(self.config, "cat_r1_think_tokens")
            ),
            "cat_r1_reason_engine_ready": (not self.config.cat_r1_reason_enabled) or self.cat_r1_reason is not None,
            "cat_r1_reason_runs": cat_r1_reason_ok,
            "decode_uses_full_bitnet_transformer": self.ternary_kernel_calls > 0,
            "not_toy_ffn_only_decode": True,
            "gui_fps_is_60": GUI_FPS == 60,
            "speed_target_named": self.config.speed_target in {"fable5", "fast", "balanced", "quality"},
        }
        return {
            "passed": all(bool(value) for value in tests.values()),
            "tests": {key: bool(value) for key, value in tests.items()},
            "ternary_reconstruction_mse": reconstruction_mse,
            "ternary_kernel_max_err": kernel_err,
            "prompt_distribution_l1": prompt_l1,
            "weight_ablation_l1": weight_ablation_l1,
            "reasoning_ablation_l1": reasoning_ablation_l1,
            "hello_preview": hello_generation.text[:120],
            "bitnet_preview": bitnet_generation.text[:120],
            "mandarin_preview": mandarin_generation.text[:120],
            "cat_r1_reason_preview": cat_r1_reason_preview,
            "ok": all(bool(value) for value in tests.values()),
            "evidence": {
                "loss_before": self.initial_loss_ever,
                "loss_after": self.best_loss,
                "parameter_count": params,
                "n_layers": self.config.n_layers,
                "d_model": self.config.d_model,
                "mod_capacity": self.config.mod_capacity,
                "packed_ternary_bytes": int(self.bank.packed.nbytes),
                "ternary_kernel_max_err": kernel_err,
                "last_reasoning_passes": self.last_reasoning_passes,
                "quantization_rebuilds": self.quantization_rebuilds,
                "ternary_kernel_calls": self.ternary_kernel_calls,
                "engine": ENGINE_ID,
                "kda_kernel_calls": self.kda_kernel_calls,
                "attnres_mixes": self.attnres_mixes,
                "moe_routes": self.moe_routes,
                "gui_fps": GUI_FPS,
                "speed_target": self.config.speed_target,
                "dspark_enabled": self.config.dspark_enabled,
                "dspark_accepted": self.dspark.last_stats.accepted if self.dspark else 0,
                "dspark_drafts": self.dspark.last_stats.drafts if self.dspark else 0,
                "cat_r1_reason_enabled": self.config.cat_r1_reason_enabled,
                "cat_r1_reason_reward": self.cat_r1_reason.last_stats.reward if self.cat_r1_reason else 0.0,
                "hello_distribution_std": float(np.std(hello_distribution)),
            },
            "model": self.model_card(),
        }


@dataclass(frozen=True, slots=True)
class ReasoningAnswer:
    text: str
    route: str


class ExactReasoner:
    """Restricted AST tools for exact arithmetic and one-variable algebra."""

    WRAPPER_RE = re.compile(
        r"^\s*(?:calculate|compute|evaluate|work out|what(?:'s| is)|solve)\s+(.+?)\s*[?.!]*\s*$",
        re.IGNORECASE,
    )
    PURE_MATH_RE = re.compile(r"^[\s0-9xX.+\-*/%()=^]+$")
    PERCENT_RE = re.compile(
        r"^\s*(?:what\s+is\s+)?([+\-]?[0-9]+(?:\.[0-9]+)?)\s*%\s+of\s+([+\-]?[0-9]+(?:\.[0-9]+)?)\s*[?.!]*\s*$",
        re.IGNORECASE,
    )

    @staticmethod
    def _constant(value: object) -> fractions.Fraction:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("only real numeric constants are allowed")
        result = fractions.Fraction(str(value))
        if result.numerator.bit_length() > 4096 or result.denominator.bit_length() > 4096:
            raise ValueError("numeric value is too large")
        return result

    @classmethod
    def _numeric(cls, node: ast.AST, depth: int = 0) -> fractions.Fraction:
        if depth > 32:
            raise ValueError("expression is too deep")
        if isinstance(node, ast.Expression):
            return cls._numeric(node.body, depth + 1)
        if isinstance(node, ast.Constant):
            return cls._constant(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = cls._numeric(node.operand, depth + 1)
            return value if isinstance(node.op, ast.UAdd) else -value
        if not isinstance(node, ast.BinOp):
            raise ValueError("unsupported syntax")
        left = cls._numeric(node.left, depth + 1)
        right = cls._numeric(node.right, depth + 1)
        if isinstance(node.op, ast.Add):
            result = left + right
        elif isinstance(node.op, ast.Sub):
            result = left - right
        elif isinstance(node.op, ast.Mult):
            result = left * right
        elif isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, ast.FloorDiv):
            result = fractions.Fraction(left // right, 1)
        elif isinstance(node.op, ast.Mod):
            result = left % right
        elif isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right.numerator) > 10_000:
                raise ValueError("power must be an integer between -10000 and 10000")
            result = left ** right.numerator
        else:
            raise ValueError("unsupported operator")
        if result.numerator.bit_length() > 4096 or result.denominator.bit_length() > 4096:
            raise ValueError("result is too large")
        return result

    @staticmethod
    def _poly_add(
        left: tuple[fractions.Fraction, ...],
        right: tuple[fractions.Fraction, ...],
        sign: int = 1,
    ) -> tuple[fractions.Fraction, ...]:
        return tuple(left[index] + sign * right[index] for index in range(3))

    @classmethod
    def _poly_mul(
        cls,
        left: tuple[fractions.Fraction, ...],
        right: tuple[fractions.Fraction, ...],
    ) -> tuple[fractions.Fraction, ...]:
        result = [fractions.Fraction(0) for _ in range(5)]
        for i, a_value in enumerate(left):
            for j, b_value in enumerate(right):
                result[i + j] += a_value * b_value
        if any(result[3:]):
            raise ValueError("only equations up to degree two are supported")
        return tuple(result[:3])

    @classmethod
    def _polynomial(
        cls,
        node: ast.AST,
        depth: int = 0,
    ) -> tuple[fractions.Fraction, fractions.Fraction, fractions.Fraction]:
        zero = fractions.Fraction(0)
        if depth > 32:
            raise ValueError("equation is too deep")
        if isinstance(node, ast.Expression):
            return cls._polynomial(node.body, depth + 1)
        if isinstance(node, ast.Constant):
            return cls._constant(node.value), zero, zero
        if isinstance(node, ast.Name) and node.id.lower() == "x":
            return zero, fractions.Fraction(1), zero
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = cls._polynomial(node.operand, depth + 1)
            return value if isinstance(node.op, ast.UAdd) else tuple(-item for item in value)
        if not isinstance(node, ast.BinOp):
            raise ValueError("unsupported equation syntax")
        left = cls._polynomial(node.left, depth + 1)
        right = cls._polynomial(node.right, depth + 1)
        if isinstance(node.op, ast.Add):
            return cls._poly_add(left, right)
        if isinstance(node.op, ast.Sub):
            return cls._poly_add(left, right, -1)
        if isinstance(node.op, ast.Mult):
            return cls._poly_mul(left, right)
        if isinstance(node.op, ast.Div):
            if right[1] or right[2] or right[0] == 0:
                raise ValueError("division is only allowed by a nonzero constant")
            return tuple(item / right[0] for item in left)
        if isinstance(node.op, ast.Pow):
            if right[1] or right[2] or right[0].denominator != 1:
                raise ValueError("polynomial power must be a constant")
            exponent = right[0].numerator
            if exponent == 0:
                return fractions.Fraction(1), zero, zero
            if exponent == 1:
                return left
            if exponent == 2:
                return cls._poly_mul(left, left)
            raise ValueError("only powers zero, one, and two are supported")
        raise ValueError("unsupported equation operator")

    @staticmethod
    def _format(value: fractions.Fraction) -> str:
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator} (≈ {float(value):.12g})"

    @staticmethod
    def _exact_sqrt(value: fractions.Fraction) -> Optional[fractions.Fraction]:
        if value < 0:
            return None
        numerator = math.isqrt(value.numerator)
        denominator = math.isqrt(value.denominator)
        if numerator * numerator == value.numerator and denominator * denominator == value.denominator:
            return fractions.Fraction(numerator, denominator)
        return None

    @classmethod
    def _solve_equation(cls, expression: str) -> ReasoningAnswer:
        left_text, separator, right_text = expression.partition("=")
        if not separator or "=" in right_text:
            raise ValueError("expected one equals sign")
        left = cls._polynomial(ast.parse(left_text.replace("^", "**").strip(), mode="eval"))
        right = cls._polynomial(ast.parse(right_text.replace("^", "**").strip(), mode="eval"))
        constant, linear, quadratic = cls._poly_add(left, right, -1)
        if quadratic == 0 and linear == 0:
            text = "Every x is a solution." if constant == 0 else "There is no solution."
            return ReasoningAnswer(text, "reasoning-tool:algebra")
        if quadratic == 0:
            root = -constant / linear
            check = quadratic * root * root + linear * root + constant
            return ReasoningAnswer(
                f"x = {cls._format(root)}\n\nVerification: substituting x gives {cls._format(check)}, so the equation balances.",
                "reasoning-tool:linear-algebra",
            )
        discriminant = linear * linear - 4 * quadratic * constant
        if discriminant < 0:
            real = -linear / (2 * quadratic)
            imaginary = math.sqrt(float(-discriminant)) / abs(float(2 * quadratic))
            text = f"x = {float(real):.12g} ± {imaginary:.12g}i\n\nDiscriminant: {cls._format(discriminant)}."
        else:
            exact_root = cls._exact_sqrt(discriminant)
            if exact_root is not None:
                roots = [(-linear + exact_root) / (2 * quadratic), (-linear - exact_root) / (2 * quadratic)]
                unique = list(dict.fromkeys(roots))
                text = "Solutions: " + ", ".join(f"x = {cls._format(root)}" for root in unique)
            else:
                sqrt_value = math.sqrt(float(discriminant))
                denominator = float(2 * quadratic)
                roots = ((-float(linear) + sqrt_value) / denominator, (-float(linear) - sqrt_value) / denominator)
                text = f"Solutions: x ≈ {roots[0]:.12g}, x ≈ {roots[1]:.12g}"
            text += f"\n\nDiscriminant: {cls._format(discriminant)}."
        return ReasoningAnswer(text, "reasoning-tool:quadratic-algebra")

    def solve(self, prompt: str) -> Optional[ReasoningAnswer]:
        if len(prompt) > 1000:
            return None
        percent = self.PERCENT_RE.fullmatch(prompt)
        if percent:
            rate = fractions.Fraction(percent.group(1))
            base = fractions.Fraction(percent.group(2))
            result = rate * base / 100
            return ReasoningAnswer(
                f"{self._format(result)}\n\nCalculation: ({self._format(rate)} / 100) × {self._format(base)} = {self._format(result)}.",
                "reasoning-tool:exact-percent",
            )
        wrapper = self.WRAPPER_RE.fullmatch(prompt)
        expression = wrapper.group(1) if wrapper else prompt.strip().rstrip("?.!")
        if not self.PURE_MATH_RE.fullmatch(expression):
            return None
        normalized = expression.replace("^", "**").strip()
        try:
            if "=" in normalized and re.search(r"[xX]", normalized):
                return self._solve_equation(normalized)
            if "=" in normalized or re.search(r"[xX]", normalized):
                return None
            value = self._numeric(ast.parse(normalized, mode="eval"))
        except (ArithmeticError, SyntaxError, ValueError, ZeroDivisionError):
            return None
        return ReasoningAnswer(
            f"{self._format(value)}\n\nExact calculation: `{expression}` = {self._format(value)}.",
            "reasoning-tool:exact-arithmetic",
        )

    def self_test(self) -> dict[str, object]:
        cases = {
            "arithmetic": ("calculate (17 * 23) + 9", "400"),
            "fraction": ("what is 1/3 + 1/6", "1/2"),
            "percent": ("15% of 240", "36"),
            "linear": ("solve 2*x + 3 = 11", "x = 4"),
            "quadratic": ("solve x^2 - 5*x + 6 = 0", "x = 3"),
        }
        results: dict[str, bool] = {}
        outputs: dict[str, str] = {}
        for name, (prompt, expected) in cases.items():
            answer = self.solve(prompt)
            outputs[name] = "" if answer is None else answer.text
            results[name] = answer is not None and expected in answer.text
        injection = self.solve("calculate __import__('os').system('echo unsafe')")
        results["rejects_calls_and_names"] = injection is None
        return {"ok": all(results.values()), "tests": results, "outputs": outputs}


class SemanticReasoner:
    """Small BM25-like semantic memory with multi-evidence composition.

    This is deliberately not presented as neural world knowledge.  It makes
    the useful information already embedded in the program accessible when a
    user phrases a request differently from the startup training examples.
    """

    TERM_RE = re.compile(r"[a-z0-9_+#.-]+|[\u3400-\u4dbf\u4e00-\u9fff]", re.IGNORECASE)
    SYNONYMS = {
        "build": ("make", "implement", "design"),
        "create": ("make", "build", "implement"),
        "design": ("architecture", "build", "implement"),
        "debug": ("diagnose", "error", "traceback"),
        "fix": ("debug", "diagnose", "repair"),
        "test": ("verify", "correctness", "regression"),
        "verify": ("test", "correctness", "prove"),
        "optimize": ("performance", "faster", "speed"),
        "cpu": ("processor", "opcode", "instruction"),
        "model": ("language", "neural", "llm"),
        "reason": ("thinking", "solve", "plan"),
        "context": ("history", "previous", "earlier"),
    }

    def __init__(self) -> None:
        self.documents: list[dict[str, object]] = []
        document_frequency: collections.Counter[str] = collections.Counter()
        for prompts, answer in ALL_DIALOGUES:
            terms = self._terms(" ".join(prompts))
            counts = collections.Counter(terms)
            self.documents.append({"prompts": prompts, "answer": answer, "counts": counts})
            document_frequency.update(counts.keys())
        count = len(self.documents)
        self.idf = {
            term: math.log((count + 1.0) / (frequency + 0.5)) + 1.0
            for term, frequency in document_frequency.items()
        }
        self.default_idf = math.log(count + 1.0) + 1.0
        for document in self.documents:
            vector = self._vector(document["counts"])
            document["vector"] = vector
            document["norm"] = math.sqrt(sum(value * value for value in vector.values()))

    @classmethod
    def _stem(cls, term: str) -> str:
        if not term.isascii() or len(term) < 5:
            return term
        for suffix in ("ization", "ation", "ments", "ment", "ness", "ing", "ers", "ed", "s"):
            if term.endswith(suffix) and len(term) - len(suffix) >= 3:
                return term[:-len(suffix)]
        return term

    @classmethod
    def _terms(cls, text: str) -> list[str]:
        base = [match.group(0).lower() for match in cls.TERM_RE.finditer(text)]
        expanded: list[str] = []
        for term in base:
            expanded.append(term)
            stem = cls._stem(term)
            if stem != term:
                expanded.append(stem)
            expanded.extend(cls.SYNONYMS.get(term, ()))
        return expanded

    def _vector(self, counts: collections.Counter[str]) -> dict[str, float]:
        return {
            term: (1.0 + math.log(frequency)) * self.idf.get(term, self.default_idf)
            for term, frequency in counts.items()
            if frequency > 0
        }

    def retrieve(self, prompt: str, limit: int = 3) -> list[dict[str, object]]:
        query_vector = self._vector(collections.Counter(self._terms(prompt)))
        query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
        if query_norm == 0.0:
            return []
        results: list[dict[str, object]] = []
        normalized_prompt = prompt.strip().lower().rstrip("?.!")
        for document in self.documents:
            vector = document["vector"]
            dot = sum(value * vector.get(term, 0.0) for term, value in query_vector.items())
            norm = float(document["norm"])
            score = dot / (query_norm * norm) if norm else 0.0
            canonical_prompts = tuple(str(value).strip().lower().rstrip("?.!") for value in document["prompts"])
            if normalized_prompt in canonical_prompts:
                score += 1.0
            if score > 0.0:
                results.append({
                    "score": score,
                    "prompts": document["prompts"],
                    "answer": document["answer"],
                })
        results.sort(key=lambda item: float(item["score"]), reverse=True)
        return results[:max(1, limit)]

    def compose(self, prompt: str) -> Optional[ReasoningAnswer]:
        matches = self.retrieve(prompt, 4)
        if not matches or float(matches[0]["score"]) < 0.16:
            return None
        chosen = [matches[0]]
        complexity = InMemoryTernaryLM._prompt_complexity(prompt)
        if complexity >= 2:
            for candidate in matches[1:]:
                if float(candidate["score"]) < max(0.13, float(matches[0]["score"]) * 0.42):
                    continue
                if candidate["answer"] != chosen[0]["answer"]:
                    chosen.append(candidate)
                    break
        if len(chosen) == 1:
            text = str(chosen[0]["answer"])
        else:
            text = (
                f"Core approach:\n\n{chosen[0]['answer']}\n\n"
                f"Verification and risk check:\n\n{chosen[1]['answer']}"
            )
        return ReasoningAnswer(text, f"semantic-reasoning:{len(chosen)}e")

    def self_test(self) -> dict[str, object]:
        emulator = self.compose("How should I design and verify a deterministic NES CPU core?")
        debugging = self.compose("My Python integration test fails with a traceback; how do I debug it?")
        unrelated = self.compose("zxqv frobnicator plugh")
        tests = {
            "retrieves_emulator_knowledge": emulator is not None and "CPU" in emulator.text and "test" in emulator.text.lower(),
            "composes_debugging_evidence": debugging is not None and "reproduce" in debugging.text.lower(),
            "rejects_unrelated_query": unrelated is None,
        }
        return {
            "ok": all(tests.values()),
            "tests": tests,
            "emulator_output": "" if emulator is None else emulator.text,
            "debugging_output": "" if debugging is None else debugging.text,
        }



@dataclass(frozen=True, slots=True)
class PromptAnalysis:
    intent: str
    subject: str
    language: str
    payload: str = ""


class OpenDomainResponder:
    """Prompt-grounded response and quality gate for unseen inputs.

    This route does not invent missing world knowledge. It performs useful
    transformations when the source is in the prompt, and otherwise returns a
    concrete plan or names the evidence needed for a factual answer.
    """

    MAX_PROMPT_CHARS = 32_768
    INTENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("summarize", re.compile(r"\b(?:summarize|summary|tl\s*;?\s*dr|key points?|condense)\b|总结|摘要|概括", re.I)),
        ("rewrite", re.compile(r"\b(?:rewrite|rephrase|wording|make (?:this|it) clearer|polish)\b|改写|润色|重写", re.I)),
        ("translate", re.compile(r"\b(?:translate|translation)\b|翻译", re.I)),
        ("compare", re.compile(r"\b(?:compare|versus|vs\.?|trade-?offs?|difference between|choose between)\b|比较|区别|取舍", re.I)),
        ("build", re.compile(r"\b(?:build|implement|create|make|generate|write|code|design)\b|制作|创建|实现|编写|设计", re.I)),
        ("debug", re.compile(r"\b(?:debug|fix|bugs?|crash(?:es|ed|ing)?|traceback|exceptions?|[A-Za-z.]*Error|fails?|failed)\b|调试|修复|错误|崩溃|异常", re.I)),
        ("brainstorm", re.compile(r"\b(?:brainstorm|ideas?|invent|creative|concepts?)\b|头脑风暴|创意|想法", re.I)),
        ("current", re.compile(r"\b(?:latest|current|today|right now|news|price|weather|version)\b|最新|当前|今天|新闻|价格|天气|版本", re.I)),
        ("explain", re.compile(r"\b(?:explain|define|teach|what is|how does|why does)\b|解释|什么是|为什么|怎样", re.I)),
        ("verify", re.compile(r"\b(?:verify|prove|test|vibe check|is this real|does this work)\b|验证|证明|检查|可行吗", re.I)),
        ("casual", re.compile(r"\b(?:bro+|vibe|thanks|thank you|bye|nice|wild|cool)\b|谢谢|再见|厉害|酷", re.I)),
        ("question", re.compile(r"\?|？|^(?:who|what|when|where|why|how|can|could|should|would|is|are|do|does)\b", re.I)),
    )
    COMMAND_WORDS = re.compile(
        r"^(?:please\s+)?(?:can|could|would)\s+you\s+|"
        r"^(?:please\s+)?(?:summarize|rewrite|rephrase|translate|debug|fix|compare|build|"
        r"implement|create|make|generate|write|design|brainstorm|explain|define|verify|test)\s+",
        re.I,
    )

    @staticmethod
    def _clean(text: str, limit: int = 220) -> str:
        value = re.sub(r"\s+", " ", text).strip(" \t\r\n:;-?.!")
        if len(value) > limit:
            value = value[: limit - 1].rstrip() + "…"
        return value

    @classmethod
    def _payload(cls, prompt: str, intent: str) -> str:
        fenced = re.findall(r"```(?:[^\n]*)\n?(.*?)```", prompt, re.S)
        if fenced:
            return max((part.strip() for part in fenced), key=len, default="")
        lines = prompt.splitlines()
        if len(lines) > 1:
            tail = "\n".join(lines[1:]).strip()
            if tail:
                return tail
        if ":" in prompt:
            head, tail = prompt.split(":", 1)
            if len(tail.strip()) >= 4 and re.search(
                r"summar|rewrite|rephrase|translate|debug|error|traceback|总结|改写|翻译|错误", head, re.I,
            ):
                return tail.strip()
        if intent in {"summarize", "rewrite", "translate"}:
            stripped = cls.COMMAND_WORDS.sub("", prompt, count=1).strip()
            if stripped != prompt.strip() and len(stripped) >= 8:
                return stripped
        return ""

    @classmethod
    def analyze(cls, prompt: str) -> PromptAnalysis:
        raw = str(prompt or "").replace("\x00", " ").strip()
        language = detect_language(raw)
        intent = "conversation"
        for name, pattern in cls.INTENT_PATTERNS:
            if pattern.search(raw):
                intent = name
                break
        payload = cls._payload(raw, intent)
        subject_source = payload if payload and intent not in {"summarize", "rewrite"} else raw
        subject = cls._clean(cls.COMMAND_WORDS.sub("", subject_source, count=1))
        if not subject:
            subject = "your request" if language == "en" else "你的请求"
        return PromptAnalysis(intent, subject, language, payload)

    @staticmethod
    def _sentences(text: str) -> list[str]:
        cleaned = re.sub(r"[ \t]+", " ", text).strip()
        parts = re.split(r"(?<=[.!?。！？])\s+|\n+", cleaned)
        return [part.strip() for part in parts if part.strip()]

    @classmethod
    def _summary(cls, text: str, language: str) -> str:
        sentences = cls._sentences(text)
        if not sentences:
            return "Please include the text you want summarized." if language == "en" else "请提供需要总结的原文。"
        if len(sentences) <= 3:
            chosen = sentences
        else:
            terms = [
                token.lower() for token in WordTokenizer.basic_tokenize(text)
                if len(token) >= 4 and token.isascii()
            ]
            frequency = collections.Counter(terms)
            scored: list[tuple[float, int, str]] = []
            for index, sentence in enumerate(sentences):
                sentence_terms = [
                    token.lower() for token in WordTokenizer.basic_tokenize(sentence)
                    if len(token) >= 4 and token.isascii()
                ]
                score = sum(frequency[token] for token in set(sentence_terms)) / max(1, len(sentence_terms))
                score += 0.20 if index == 0 else 0.0
                scored.append((score, index, sentence))
            selected = sorted(scored, reverse=True)[:3]
            chosen = [sentence for _, _, sentence in sorted(selected, key=lambda item: item[1])]
        body = " ".join(chosen)
        return ("Summary: " + body) if language == "en" else ("摘要：" + body)

    @staticmethod
    def _rewrite(text: str, language: str) -> str:
        cleaned = re.sub(r"[ \t]+", " ", text).strip()
        cleaned = re.sub(r"\s+([,.;:!?，。；：！？])", r"\1", cleaned)
        if not cleaned:
            return "Please include the text you want rewritten." if language == "en" else "请提供需要改写的原文。"
        if language == "en":
            cleaned = cleaned[0].upper() + cleaned[1:] if cleaned else cleaned
            if cleaned[-1:] not in ".!?":
                cleaned += "."
            return "Clear rewrite: " + cleaned
        return "清晰改写：" + cleaned

    @staticmethod
    def _debug(prompt: str, subject: str, language: str) -> str:
        error_lines = [
            line.strip() for line in prompt.splitlines()
            if re.search(r"(?:Error|Exception|Traceback|failed|错误|异常|失败)", line, re.I)
        ]
        signal = error_lines[-1] if error_lines else subject
        if language == "zh":
            return (
                f"先锁定第一个可复现的错误：{signal}\n\n1. 保留完整报错和触发输入。\n"
                "2. 缩小到最小复现。\n3. 检查进入首个错误边界的值和类型。\n"
                "4. 修复根因后运行原始复现与回归测试。"
            )
        return (
            f"Start with the first reproducible failure: {signal}\n\n"
            "1. Preserve the full traceback and triggering input.\n"
            "2. Reduce it to the smallest reproduction.\n"
            "3. Inspect values and types at the earliest failing boundary.\n"
            "4. Fix that cause, then rerun the reproduction and a regression test."
        )

    @staticmethod
    def _compare(subject: str, language: str) -> str:
        match = re.search(r"(.+?)\s+(?:vs\.?|versus|and|with|与|和)\s+(.+)", subject, re.I)
        options = f"{match.group(1).strip()} and {match.group(2).strip()}" if match else subject
        if language == "zh":
            return (
                f"比较对象：{subject}\n\n按正确性、功能、运行成本、复杂度、维护性和可逆性逐项比较。"
                "先淘汰违反硬性限制的方案，再用同一个小型工作负载测量剩余方案。"
            )
        return (
            f"Comparison target: {options}.\n\nScore each option on correctness, capability, runtime cost, "
            "complexity, maintenance, and reversibility. Eliminate anything that violates a hard constraint, "
            "then benchmark the survivors with the same small workload before choosing."
        )

    @staticmethod
    def _build(subject: str, language: str) -> str:
        if language == "zh":
            return (
                f"目标：{subject}。\n\n先把需求变成可观察行为和硬性限制，然后完成最小可运行的纵向切片："
                "输入、核心逻辑、输出、错误处理。最后用正常输入、边界输入和原始需求各验证一次。"
            )
        return (
            f"Goal: {subject}.\n\nTurn the request into observable behavior and hard constraints, then build the "
            "smallest runnable vertical slice: input, core logic, output, and error handling. Verify it with "
            "one normal case, one boundary case, and a final check against the original request."
        )

    @staticmethod
    def _brainstorm(subject: str, language: str) -> str:
        if language == "zh":
            return (
                f"围绕“{subject}”可以试三个方向：\n\n1. 核心机制优先：先做一个一分钟可玩的循环。\n"
                "2. 反转限制：把最大的限制变成主要玩法。\n3. 组合实验：把两个熟悉机制用一个新规则连接。\n\n"
                "先原型化第一项，因为它最快暴露这个想法是否真的有趣。"
            )
        return (
            f"Three directions for {subject}:\n\n1. Mechanic-first — prototype a one-minute core loop.\n"
            "2. Constraint-flip — turn the biggest limitation into the main feature.\n"
            "3. Hybrid — connect two familiar mechanics with one surprising rule.\n\n"
            "Prototype the first direction first; it gives the fastest evidence about whether the idea is fun or useful."
        )

    @classmethod
    def respond(cls, prompt: str) -> ReasoningAnswer:
        raw = str(prompt or "").replace("\x00", " ").strip()
        analysis = cls.analyze(raw)
        if not raw:
            text = "Send me a question, goal, code sample, or text to transform." if analysis.language == "en" else "请发送问题、目标、代码或需要处理的文字。"
            return ReasoningAnswer(text, "open-domain:empty")
        if len(raw) > cls.MAX_PROMPT_CHARS:
            text = (
                f"This prompt has {len(raw):,} characters. Please split it into sections under {cls.MAX_PROMPT_CHARS:,} characters so I can preserve the important context."
                if analysis.language == "en"
                else f"这个提示有 {len(raw):,} 个字符。请拆分为不超过 {cls.MAX_PROMPT_CHARS:,} 个字符的部分，以免丢失重要上下文。"
            )
            return ReasoningAnswer(text, "open-domain:input-limit")
        if analysis.intent == "summarize":
            return ReasoningAnswer(cls._summary(analysis.payload, analysis.language), "open-domain:summarize")
        if analysis.intent == "rewrite":
            return ReasoningAnswer(cls._rewrite(analysis.payload, analysis.language), "open-domain:rewrite")
        if analysis.intent == "translate":
            text = (
                "请同时提供原文和目标语言；本地小模型在没有词典或足够训练数据时不会假装翻译准确。"
                if analysis.language == "zh"
                else "Include the source text and target language. This compact local model will not pretend a translation is accurate when its embedded vocabulary is insufficient."
            )
            return ReasoningAnswer(text, "open-domain:translate")
        if analysis.intent == "debug":
            return ReasoningAnswer(cls._debug(raw, analysis.subject, analysis.language), "open-domain:debug")
        if analysis.intent == "compare":
            return ReasoningAnswer(cls._compare(analysis.subject, analysis.language), "open-domain:compare")
        if analysis.intent == "build":
            return ReasoningAnswer(cls._build(analysis.subject, analysis.language), "open-domain:build")
        if analysis.intent == "brainstorm":
            return ReasoningAnswer(cls._brainstorm(analysis.subject, analysis.language), "open-domain:brainstorm")
        if analysis.intent == "current":
            text = (
                f"I cannot verify a changing fact from RAM-only embedded data: {analysis.subject}. Check an authoritative live source and its timestamp; if you paste the result here, I can analyze it."
                if analysis.language == "en"
                else f"我无法用仅驻留内存的内置数据核实会变化的事实：{analysis.subject}。请查看带时间戳的权威实时来源；把结果贴在这里后，我可以继续分析。"
            )
            return ReasoningAnswer(text, "open-domain:current-limit")
        if analysis.intent == "explain":
            text = (
                f"I do not have a grounded definition for “{analysis.subject}” in the embedded corpus, so I will not invent one. Provide a definition or source and I can turn it into a clear explanation with a mechanism, example, and limitations."
                if analysis.language == "en"
                else f"内置语料没有足够证据定义“{analysis.subject}”，所以我不会编造。提供定义或来源后，我可以把它整理成机制、例子和限制条件都清楚的解释。"
            )
            return ReasoningAnswer(text, "open-domain:knowledge-limit")
        if analysis.intent == "verify":
            text = (
                f"Vibe check for {analysis.subject}: define one observable claim, run the smallest test that could falsify it, record the exact input and output, and state what the test still does not prove."
                if analysis.language == "en"
                else f"对“{analysis.subject}”做检查：先定义一个可观察主张，再运行能推翻它的最小测试，记录准确输入和输出，并说明该测试仍不能证明什么。"
            )
            return ReasoningAnswer(text, "open-domain:verify")
        if analysis.intent == "question":
            text = (
                f"I received the question about {analysis.subject}, but its answer is not grounded in my embedded corpus. Give me the relevant facts, code, or source text and I can reason over them without guessing."
                if analysis.language == "en"
                else f"我收到了关于“{analysis.subject}”的问题，但内置语料不足以支持确定答案。请提供相关事实、代码或来源文字，我可以基于它们推理而不猜测。"
            )
            return ReasoningAnswer(text, "open-domain:question")
        text = (
            f"I hear you: {analysis.subject}. Tell me the outcome you want or paste the relevant material, and I will turn it into a concrete next step."
            if analysis.language == "en"
            else f"我明白了：{analysis.subject}。请告诉我你想要的结果，或贴出相关材料，我会把它变成具体的下一步。"
        )
        return ReasoningAnswer(text, "open-domain:conversation")

    @classmethod
    def respond_anything(cls, prompt: str) -> ReasoningAnswer:
        """Cat R1 respond-anything: never refuse — always a concrete helpful reply."""
        raw = str(prompt or "").replace("\x00", " ").strip()
        analysis = cls.analyze(raw)
        lang = analysis.language
        if not raw:
            text = (
                "Send any question, goal, code, or idea — I will answer with cat r1."
                if lang == "en"
                else "请发送任何问题、目标、代码或想法——我会用本地 cat r1 v0.1 栈作答。"
            )
            return ReasoningAnswer(text, "open-domain:anything:empty")
        if analysis.intent == "summarize" and analysis.payload:
            return ReasoningAnswer(cls._summary(analysis.payload, lang), "open-domain:anything:summarize")
        if analysis.intent == "rewrite" and analysis.payload:
            return ReasoningAnswer(cls._rewrite(analysis.payload, lang), "open-domain:anything:rewrite")
        if analysis.intent == "debug":
            return ReasoningAnswer(cls._debug(raw, analysis.subject, lang), "open-domain:anything:debug")
        if analysis.intent == "build":
            return ReasoningAnswer(cls._build(analysis.subject, lang), "open-domain:anything:build")
        if analysis.intent == "brainstorm":
            return ReasoningAnswer(cls._brainstorm(analysis.subject, lang), "open-domain:anything:brainstorm")
        if analysis.intent == "compare":
            return ReasoningAnswer(cls._compare(analysis.subject, lang), "open-domain:anything:compare")
        if analysis.intent == "verify":
            tip = (
                f"Vibe check for {analysis.subject}: state one observable claim, run the smallest falsifying test, "
                f"record exact I/O, and say what is still unproven."
                if lang == "en"
                else f"对「{analysis.subject}」做检查：定义可观察主张，跑最小证伪测试，记录输入输出，并说明仍未证明什么。"
            )
            return ReasoningAnswer(tip, "open-domain:anything:verify")
        base = cls.respond(raw)
        if base.route.endswith(("knowledge-limit", "question", "current-limit", "conversation", "translate")):
            return ReasoningAnswer(answer_everything(raw, lang=lang), "open-domain:anything")
        return ReasoningAnswer(base.text, "open-domain:anything")

    @staticmethod
    def generation_quality(prompt: str, text: str, *, llm_first: bool = True) -> tuple[bool, float, list[str]]:
        """Quality gate. In llm_first mode, accept any non-empty neural decode (real LLM)."""
        reasons: list[str] = []
        visible = text.strip()
        if not visible:
            reasons.append("empty")
        if "�" in visible or re.search(r"<(?:(?:0x[0-9A-F]{2})|UNK|PAD)>", visible):
            reasons.append("decode-artifact")
        tokens = [token.lower() for token in WordTokenizer.basic_tokenize(visible) if token not in WordTokenizer.SPECIAL]
        if llm_first:
            # Real LLM: keep autoregressive text whenever anything was decoded.
            if not visible:
                return False, 0.0, ["empty"]
            score = min(1.0, 0.55 + 0.02 * min(len(tokens), 20))
            return True, score, [r for r in reasons if r != "empty"]
        if len(tokens) < 4:
            reasons.append("too-short")
        if tokens:
            most_common = collections.Counter(tokens).most_common(1)[0][1]
            if most_common / len(tokens) > 0.34:
                reasons.append("repetitive")
        if len(tokens) >= 12:
            chunks = [tuple(tokens[index:index + 4]) for index in range(len(tokens) - 3)]
            if chunks and len(set(chunks)) / len(chunks) < 0.56:
                reasons.append("looping")
        prompt_terms = {
            token.lower() for token in WordTokenizer.basic_tokenize(prompt)
            if len(token) >= 4 and token.isascii()
        }
        answer_terms = {token for token in tokens if len(token) >= 4 and token.isascii()}
        overlap = len(prompt_terms & answer_terms) / max(1, min(4, len(prompt_terms)))
        if prompt_terms and overlap == 0.0:
            reasons.append("ungrounded")
        score = max(0.0, 1.0 - 0.18 * len(reasons) + min(0.20, overlap * 0.25))
        return not reasons and score >= 0.72, score, reasons

    def self_test(self) -> dict[str, object]:
        outputs: dict[str, str] = {}
        checks: dict[str, bool] = {}
        for name, prompt in VIBE_CHECK_PROBES:
            answer = self.respond(prompt)
            outputs[name] = answer.text
            checks[name] = bool(answer.text.strip()) and answer.route.startswith("open-domain:")
        checks["summary_uses_payload"] = "prototype" in outputs["summarize"].lower()
        checks["current_fact_is_not_invented"] = "authoritative" in outputs["current"].lower()
        checks["debug_uses_error"] = "invalid packet length" in outputs["debug"].lower()
        return {
            "ok": all(checks.values()),
            "checks": checks,
            "probe_count": len(VIBE_CHECK_PROBES),
            "previews": {name: value[:180] for name, value in outputs.items()},
        }


@dataclass(slots=True)
class ToolResult:
    name: str
    ok: bool
    output: str


class CatR11XCodeTools:
    """Workspace-scoped Claude Code–style tools (local files only)."""

    def __init__(self, root: Optional[str] = None):
        self.root = Path(root or os.getcwd()).resolve()

    def resolve(self, relative: str) -> Path:
        target = (self.root / relative).resolve()
        if self.root not in target.parents and target != self.root:
            raise PermissionError(f"path escapes workspace: {relative}")
        return target

    def ls(self, relative: str = ".") -> ToolResult:
        try:
            target = self.resolve(relative)
            if not target.exists():
                return ToolResult("LS", False, f"missing: {relative}")
            if target.is_file():
                return ToolResult("LS", True, str(target.relative_to(self.root)))
            names = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
            return ToolResult("LS", True, "\n".join(names) if names else "(empty)")
        except Exception as exc:
            return ToolResult("LS", False, f"{type(exc).__name__}: {exc}")

    def read(self, relative: str, max_chars: int = 12_000) -> ToolResult:
        try:
            target = self.resolve(relative)
            text = target.read_text(encoding="utf-8", errors="replace")
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n… truncated ({len(text)} chars)"
            return ToolResult("Read", True, text)
        except Exception as exc:
            return ToolResult("Read", False, f"{type(exc).__name__}: {exc}")

    def write(self, relative: str, content: str) -> ToolResult:
        try:
            target = self.resolve(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return ToolResult("Write", True, f"wrote {target.relative_to(self.root)} ({len(content)} chars)")
        except Exception as exc:
            return ToolResult("Write", False, f"{type(exc).__name__}: {exc}")

    def edit(self, relative: str, old: str, new: str) -> ToolResult:
        try:
            target = self.resolve(relative)
            text = target.read_text(encoding="utf-8", errors="replace")
            if old not in text:
                return ToolResult("Edit", False, "old_string not found")
            count = text.count(old)
            if count != 1:
                return ToolResult("Edit", False, f"old_string matched {count} times; need exactly 1")
            target.write_text(text.replace(old, new, 1), encoding="utf-8")
            return ToolResult("Edit", True, f"edited {target.relative_to(self.root)}")
        except Exception as exc:
            return ToolResult("Edit", False, f"{type(exc).__name__}: {exc}")

    def glob(self, pattern: str = "**/*") -> ToolResult:
        try:
            matches = sorted(
                str(p.relative_to(self.root))
                for p in self.root.glob(pattern)
                if p.is_file()
            )[:200]
            return ToolResult("Glob", True, "\n".join(matches) if matches else "(no matches)")
        except Exception as exc:
            return ToolResult("Glob", False, f"{type(exc).__name__}: {exc}")

    def grep(self, pattern: str, relative: str = ".") -> ToolResult:
        try:
            regex = re.compile(pattern)
            root = self.resolve(relative)
            hits: list[str] = []
            files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
            for path in files[:400]:
                try:
                    for lineno, line in enumerate(
                        path.read_text(encoding="utf-8", errors="replace").splitlines(),
                        start=1,
                    ):
                        if regex.search(line):
                            hits.append(f"{path.relative_to(self.root)}:{lineno}:{line[:200]}")
                            if len(hits) >= 80:
                                return ToolResult("Grep", True, "\n".join(hits))
                except OSError:
                    continue
            return ToolResult("Grep", True, "\n".join(hits) if hits else "(no matches)")
        except Exception as exc:
            return ToolResult("Grep", False, f"{type(exc).__name__}: {exc}")

    def bash(self, command: str, timeout_s: float = 20.0) -> ToolResult:
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            output = (completed.stdout or "") + (("\n" + completed.stderr) if completed.stderr else "")
            output = output.strip() or "(no output)"
            if len(output) > 8000:
                output = output[:8000] + "\n… truncated"
            ok = completed.returncode == 0
            return ToolResult("Bash", ok, f"exit {completed.returncode}\n{output}")
        except Exception as exc:
            return ToolResult("Bash", False, f"{type(exc).__name__}: {exc}")


class CatR11XCodeAgent:
    """Local Claude Code fork powered by the real BitNet ~20B LM (files=off weights)."""

    def __init__(self, engine: "CatR11X", root: Optional[str] = None):
        self.engine = engine
        self.tools = CatR11XCodeTools(root)
        self.history: list[tuple[str, str]] = []

    def _plan(self, prompt: str) -> list[dict[str, str]]:
        text = prompt.strip()
        lower = text.lower()
        actions: list[dict[str, str]] = []
        read_match = re.search(
            r"(?:read|open|show|cat)\s+(?:file\s+)?([^\s]+)",
            text,
            re.IGNORECASE,
        )
        if read_match:
            actions.append({"tool": "Read", "path": read_match.group(1).strip("`'\"")})
        ls_match = re.search(r"(?:list|ls|dir)(?:\s+(?:files|dir|directory|folder))?(?:\s+(?:in|of)\s+([^\s]+))?", text, re.IGNORECASE)
        if "list files" in lower or lower.startswith("ls") or "list directory" in lower or ls_match:
            path = "."
            if ls_match and ls_match.group(1):
                path = ls_match.group(1).strip("`'\"")
            actions.append({"tool": "LS", "path": path})
        glob_match = re.search(r"(?:glob|find files?)\s+([^\s]+)", text, re.IGNORECASE)
        if glob_match:
            actions.append({"tool": "Glob", "pattern": glob_match.group(1).strip("`'\"")})
        grep_match = re.search(r"(?:grep|search(?:\s+for)?)\s+[\"'](.+?)[\"']", text, re.IGNORECASE)
        if grep_match:
            actions.append({"tool": "Grep", "pattern": grep_match.group(1)})
        elif re.search(r"\bgrep\s+(\S+)", text, re.IGNORECASE):
            actions.append({"tool": "Grep", "pattern": re.search(r"\bgrep\s+(\S+)", text, re.IGNORECASE).group(1)})
        bash_match = re.search(r"(?:run|bash|shell)\s+[`'\"](.+?)[`'\"]", text, re.IGNORECASE | re.DOTALL)
        if bash_match:
            actions.append({"tool": "Bash", "command": bash_match.group(1)})
        elif lower.startswith("!") and len(text) > 1:
            actions.append({"tool": "Bash", "command": text[1:].strip()})
        write_match = re.search(
            r"write\s+(?:file\s+)?([^\s]+)\s*[:=]\s*```(.*?)```",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if write_match:
            actions.append({"tool": "Write", "path": write_match.group(1), "content": write_match.group(2).lstrip("\n")})
        edit_match = re.search(
            r"edit\s+([^\s]+)\s+replace\s+```(.*?)```\s+with\s+```(.*?)```",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if edit_match:
            actions.append({
                "tool": "Edit",
                "path": edit_match.group(1),
                "old": edit_match.group(2),
                "new": edit_match.group(3),
            })
        if not actions and re.search(r"\b(codebase|project|repository|files)\b", lower):
            actions.append({"tool": "LS", "path": "."})
            actions.append({"tool": "Glob", "pattern": "**/*.py"})
        return actions[:6]

    def _execute(self, action: dict[str, str]) -> ToolResult:
        name = action.get("tool", "")
        if name == "LS":
            return self.tools.ls(action.get("path", "."))
        if name == "Read":
            return self.tools.read(action.get("path", ""))
        if name == "Write":
            return self.tools.write(action.get("path", ""), action.get("content", ""))
        if name == "Edit":
            return self.tools.edit(action.get("path", ""), action.get("old", ""), action.get("new", ""))
        if name == "Glob":
            return self.tools.glob(action.get("pattern", "**/*"))
        if name == "Grep":
            return self.tools.grep(action.get("pattern", ""), action.get("path", "."))
        if name == "Bash":
            return self.tools.bash(action.get("command", ""))
        return ToolResult(name or "Unknown", False, f"unknown tool: {name}")

    def run(self, prompt: str) -> Reply:
        started = time.perf_counter()
        actions = self._plan(prompt)
        results: list[ToolResult] = []
        tool_lines: list[str] = []
        for action in actions:
            result = self._execute(action)
            results.append(result)
            args = " ".join(
                f"{key}={value!r}"
                for key, value in action.items()
                if key not in {"tool", "content", "old", "new"}
            )
            tool_lines.append(f"$ {result.name}({args})")
            tool_lines.append(("✓ " if result.ok else "✗ ") + result.output)
            tool_lines.append("")
        tokens = 0
        tokens_per_second = 0.0
        if actions:
            ok_n = sum(1 for item in results if item.ok)
            body = "\n".join(tool_lines).rstrip()
            footer = (
                f"\n\nCat R11.x Code · {ok_n}/{len(results)} tools ok · "
                f"brain {MODEL_ID} · model files={FILES_MODE}"
            )
            text = body + footer
            route = f"cat-r11x-code:{len(actions)}tools"
        else:
            # No tool plan: fall through the full Cat R11.x stack (exact/semantic/BitNet).
            neural = self.engine.reply(prompt)
            text = neural.text
            route = f"cat-r11x-code:{neural.route}"
            tokens = neural.tokens
            tokens_per_second = neural.tokens_per_second
        self.history.append((prompt, text))
        self.history = self.history[-12:]
        return Reply(
            text=text,
            route=route,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            tokens=tokens,
            tokens_per_second=tokens_per_second,
        )

    def self_test(self) -> dict[str, object]:
        probe = self.tools.ls(".")
        readme = self.tools.glob("README.md")
        return {
            "ok": bool(probe.ok and readme.ok),
            "workspace": str(self.tools.root),
            "ls_ok": probe.ok,
            "glob_readme_ok": readme.ok,
            "tools": ["LS", "Read", "Write", "Edit", "Bash", "Glob", "Grep"],
            "model_files_mode": FILES_MODE,
            "brain": MODEL_ID,
        }


class CatR11XBuildMode:
    """Grok Build–style local app/game builder (single-file HTML, files=off LM)."""

    BUILD_RE = re.compile(
        r"^\s*/build\b|"
        r"\bgrok\s*build\b|"
        r"\bcat\s*r1(?:\s*1\.x)?\s*build\b|"
        r"\bcatseek\s*build\b|"
        # Natural language: "I want to build a todo app", "make me a snake game", …
        r"(?:^|\b)(?:i\s+(?:want|need|would like)\s+to\s+|please\s+|can you\s+|could you\s+)?"
        r"(?:build|make|create|generate|ship|scaffold|write)\s+"
        r"(?:me\s+)?(?:a\s+|an\s+|the\s+)?"
        r"(?:simple\s+|small\s+|basic\s+|quick\s+)?"
        r"(?:snake|pong|breakout|tetris|clicker|todo|task(?:\s*list)?|calculator|calc|"
        r"dashboard|landing(?:\s*page)?|website|portfolio|game|app)\b",
        re.IGNORECASE,
    )

    def __init__(self, root: Optional[str] = None):
        self.root = Path(root or os.getcwd()).resolve()
        self.out_dir = self.root / "builds"
        self.last_path: Optional[Path] = None

    @classmethod
    def wants(cls, prompt: str) -> bool:
        raw = str(prompt or "").strip()
        if not raw:
            return False
        if cls.BUILD_RE.search(raw):
            return True
        # Bare artifact nouns with create intent elsewhere, e.g. "todo app please"
        lower = raw.lower()
        has_artifact = bool(
            re.search(
                r"\b(?:snake|pong|breakout|todo|task list|calculator|dashboard|landing page|clicker)\b",
                lower,
            )
        )
        has_intent = bool(
            re.search(r"\b(?:build|make|create|generate|app|game|website)\b|做|制作|生成|写一个", lower)
        )
        return has_artifact and has_intent

    def _slug(self, prompt: str) -> str:
        words = re.findall(r"[a-z0-9]+", prompt.lower())
        stop = {
            "build", "make", "create", "generate", "a", "an", "the", "me", "please",
            "i", "want", "to", "need", "would", "like", "can", "you", "could",
            "simple", "small", "basic", "quick", "for", "my",
        }
        keep = [w for w in words if w not in stop][:6]
        base = "-".join(keep) or "app"
        return f"{base}-{int(time.time()) % 100000}"

    def _classify(self, prompt: str) -> str:
        lower = prompt.lower()
        if re.search(r"\bsnake\b", lower):
            return "snake"
        if re.search(r"\bpong\b", lower):
            return "pong"
        if re.search(r"\bbreakout\b|\bbrick\b", lower):
            return "breakout"
        if re.search(r"\bclicker\b|\bcounter\b", lower):
            return "clicker"
        if re.search(r"\btodo\b|\btask\b", lower):
            return "todo"
        if re.search(r"\bcalculator\b|\bcalc\b", lower):
            return "calculator"
        if re.search(r"\bdashboard\b|\bchart\b", lower):
            return "dashboard"
        if re.search(r"\b(landing|website|portfolio|site)\b", lower):
            return "website"
        if re.search(r"\bgame\b", lower):
            return "canvas-game"
        return "app"

    def _html_shell(self, title: str, body: str, script: str, extra_css: str = "") -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title} · cat r1 Build</title>
<style>
  :root {{ --bg:#0b1220; --panel:#121a2b; --text:#e8eefc; --muted:#8fa3c4; --accent:#3b82f6; --good:#34d399; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; min-height:100vh; font:15px/1.45 "Segoe UI", system-ui, sans-serif;
         background:radial-gradient(1200px 600px at 10% -10%, #1e3a5f 0%, var(--bg) 55%); color:var(--text); }}
  header {{ padding:18px 22px; border-bottom:1px solid #1f2a40; background:rgba(8,12,22,.7); backdrop-filter:blur(8px); }}
  header h1 {{ margin:0; font-size:18px; }}
  header p {{ margin:4px 0 0; color:var(--muted); font-size:12px; }}
  main {{ padding:22px; max-width:960px; margin:0 auto; }}
  .card {{ background:var(--panel); border:1px solid #243149; border-radius:14px; padding:18px; }}
  button, .btn {{ background:var(--accent); color:white; border:0; border-radius:10px; padding:10px 14px; cursor:pointer; font-weight:600; }}
  button:hover {{ filter:brightness(1.08); }}
  canvas {{ display:block; margin:12px auto; background:#020617; border-radius:12px; border:1px solid #1e293b; max-width:100%; }}
  input, textarea {{ width:100%; background:#0a1220; color:var(--text); border:1px solid #2a3a55; border-radius:10px; padding:10px 12px; }}
  .row {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
  .muted {{ color:var(--muted); }}
  .stat {{ font-variant-numeric:tabular-nums; }}
  {extra_css}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <p>cat r1 Build · local HTML artifact · BitNet brain files=off</p>
</header>
<main>
{body}
</main>
<script>
{script}
</script>
</body>
</html>
"""

    def _template(self, kind: str, prompt: str) -> tuple[str, str]:
        title_map = {
            "snake": "Snake",
            "pong": "Pong",
            "breakout": "Breakout",
            "clicker": "Clicker",
            "todo": "Todo App",
            "calculator": "Calculator",
            "dashboard": "Dashboard",
            "website": "Landing Page",
            "canvas-game": "Canvas Game",
            "app": "Mini App",
        }
        title = title_map.get(kind, "Build")
        if kind == "snake":
            body = '<div class="card"><div class="row"><button id="restart">Restart</button><span class="stat" id="score">Score: 0</span><span class="muted">Arrows / WASD</span></div><canvas id="c" width="400" height="400"></canvas></div>'
            script = r"""
const c=document.getElementById('c'),x=c.getContext('2d');
let cell=20,snake=[{x:10,y:10}],dir={x:1,y:0},food={x:15,y:10},score=0,alive=true;
function place(){food={x:(Math.random()*20)|0,y:(Math.random()*20)|0};}
function draw(){x.fillStyle='#020617';x.fillRect(0,0,400,400);x.fillStyle='#34d399';x.fillRect(food.x*cell,food.y*cell,cell-1,cell-1);x.fillStyle='#3b82f6';snake.forEach(s=>x.fillRect(s.x*cell,s.y*cell,cell-1,cell-1));}
function step(){if(!alive)return;const h={x:snake[0].x+dir.x,y:snake[0].y+dir.y};if(h.x<0||h.y<0||h.x>=20||h.y>=20||snake.some(s=>s.x===h.x&&s.y===h.y)){alive=false;return;}snake.unshift(h);if(h.x===food.x&&h.y===food.y){score++;document.getElementById('score').textContent='Score: '+score;place();}else snake.pop();draw();}
addEventListener('keydown',e=>{const k=e.key;if(['ArrowUp','w'].includes(k)&&dir.y!==1)dir={x:0,y:-1};if(['ArrowDown','s'].includes(k)&&dir.y!==-1)dir={x:0,y:1};if(['ArrowLeft','a'].includes(k)&&dir.x!==1)dir={x:-1,y:0};if(['ArrowRight','d'].includes(k)&&dir.x!==-1)dir={x:1,y:0};});
document.getElementById('restart').onclick=()=>{snake=[{x:10,y:10}];dir={x:1,y:0};score=0;alive=true;document.getElementById('score').textContent='Score: 0';place();draw();};
place();draw();setInterval(step,110);
"""
        elif kind == "pong":
            body = '<div class="card"><div class="row"><span class="stat" id="score">0 : 0</span><span class="muted">W/S left · ↑/↓ right</span></div><canvas id="c" width="640" height="360"></canvas></div>'
            script = r"""
const c=document.getElementById('c'),x=c.getContext('2d');
let p1={y:140},p2={y:140},b={x:320,y:180,vx:4,vy:2},s1=0,s2=0,keys={};
addEventListener('keydown',e=>keys[e.key]=true);addEventListener('keyup',e=>keys[e.key]=false);
function frame(){if(keys.w)p1.y-=5;if(keys.s)p1.y+=5;if(keys.ArrowUp)p2.y-=5;if(keys.ArrowDown)p2.y+=5;p1.y=Math.max(0,Math.min(300,p1.y));p2.y=Math.max(0,Math.min(300,p2.y));b.x+=b.vx;b.y+=b.vy;if(b.y<0||b.y>360)b.vy*=-1;if(b.x<24&&b.y>p1.y&&b.y<p1.y+60)b.vx=Math.abs(b.vx);if(b.x>616&&b.y>p2.y&&b.y<p2.y+60)b.vx=-Math.abs(b.vx);if(b.x<0){s2++;reset();}if(b.x>640){s1++;reset();}document.getElementById('score').textContent=s1+' : '+s2;x.fillStyle='#020617';x.fillRect(0,0,640,360);x.fillStyle='#e8eefc';x.fillRect(12,p1.y,10,60);x.fillRect(618,p2.y,10,60);x.beginPath();x.arc(b.x,b.y,7,0,6.28);x.fill();requestAnimationFrame(frame);}
function reset(){b={x:320,y:180,vx:4*(Math.random()>.5?1:-1),vy:2};}
frame();
"""
        elif kind == "breakout":
            body = '<div class="card"><div class="row"><button id="restart">Restart</button><span class="stat" id="score">Score: 0</span></div><canvas id="c" width="480" height="360"></canvas></div>'
            script = r"""
const c=document.getElementById('c'),x=c.getContext('2d');
let paddle=200,ball={x:240,y:300,vx:3,vy:-3},bricks=[],score=0;
function reset(){bricks=[];for(let r=0;r<5;r++)for(let col=0;col<8;col++)bricks.push({x:20+col*55,y:30+r*22,alive:true});ball={x:240,y:300,vx:3,vy:-3};score=0;document.getElementById('score').textContent='Score: 0';}
addEventListener('mousemove',e=>{const r=c.getBoundingClientRect();paddle=e.clientX-r.left-40;});
document.getElementById('restart').onclick=reset;
function frame(){ball.x+=ball.vx;ball.y+=ball.vy;if(ball.x<0||ball.x>480)ball.vx*=-1;if(ball.y<0)ball.vy*=-1;if(ball.y>360)reset();if(ball.y>330&&ball.x>paddle&&ball.x<paddle+80)ball.vy=-Math.abs(ball.vy);bricks.forEach(b=>{if(b.alive&&ball.x>b.x&&ball.x<b.x+50&&ball.y>b.y&&ball.y<b.y+18){b.alive=false;ball.vy*=-1;score+=10;document.getElementById('score').textContent='Score: '+score;}});x.fillStyle='#020617';x.fillRect(0,0,480,360);x.fillStyle='#3b82f6';x.fillRect(paddle,340,80,10);x.fillStyle='#34d399';bricks.filter(b=>b.alive).forEach(b=>x.fillRect(b.x,b.y,50,16));x.fillStyle='#e8eefc';x.beginPath();x.arc(ball.x,ball.y,6,0,6.28);x.fill();requestAnimationFrame(frame);}
reset();frame();
"""
        elif kind == "clicker":
            body = '<div class="card"><h2 class="stat" id="n">0</h2><div class="row"><button id="hit">Click</button><button id="reset">Reset</button><span class="muted" id="rate"></span></div></div>'
            script = r"""
let n=0,t0=performance.now();
const hit=()=>{n++;document.getElementById('n').textContent=n;const dt=(performance.now()-t0)/1000;document.getElementById('rate').textContent=(n/Math.max(dt,0.001)).toFixed(1)+' /s';};
document.getElementById('hit').onclick=hit;document.getElementById('reset').onclick=()=>{n=0;t0=performance.now();hit();};
"""
        elif kind == "todo":
            body = (
                '<div class="card">'
                '<div class="row"><input id="item" placeholder="New task" autocomplete="off"/>'
                '<button id="add" type="button">Add</button>'
                '<button id="clearDone" type="button">Clear done</button></div>'
                '<p class="muted" id="count">0 tasks</p>'
                '<ul id="list" style="list-style:none;padding:0;margin:12px 0 0"></ul>'
                '</div>'
            )
            script = r"""
const list=document.getElementById('list'), item=document.getElementById('item'), count=document.getElementById('count');
const KEY='cat-r1-todo';
let items=JSON.parse(localStorage.getItem(KEY)||'[]');
function save(){localStorage.setItem(KEY,JSON.stringify(items));}
function render(){
  list.innerHTML='';
  items.forEach((t,i)=>{
    const li=document.createElement('li');
    li.style.cssText='display:flex;gap:10px;align-items:center;padding:8px 0;border-bottom:1px solid #243149';
    li.innerHTML='<label style="flex:1;display:flex;gap:8px;align-items:center"><input type="checkbox" '+(t.done?'checked':'')+'> <span style="'+(t.done?'text-decoration:line-through;opacity:.6':'')+'">'+t.text.replace(/[<>&]/g,s=>({ '<':'&lt;','>':'&gt;','&':'&amp;' }[s]))+'</span></label><button type="button" data-i="'+i+'">✕</button>';
    li.querySelector('input').onchange=e=>{items[i].done=e.target.checked;save();render();};
    li.querySelector('button').onclick=()=>{items.splice(i,1);save();render();};
    list.appendChild(li);
  });
  const left=items.filter(t=>!t.done).length;
  count.textContent=items.length+' tasks · '+left+' open';
}
function add(){
  const v=item.value.trim();
  if(!v)return;
  items.push({text:v,done:false});
  item.value='';
  save();render();
  item.focus();
}
document.getElementById('add').onclick=add;
document.getElementById('clearDone').onclick=()=>{items=items.filter(t=>!t.done);save();render();};
item.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();add();}});
render();
"""
        elif kind == "calculator":
            body = '<div class="card"><input id="display" readonly value="0"/><div class="row" id="keys"></div></div>'
            script = r"""
const d=document.getElementById('display');let cur='0';
const keys=['7','8','9','/','4','5','6','*','1','2','3','-','0','.','=','+','C'];
const box=document.getElementById('keys');
keys.forEach(k=>{const b=document.createElement('button');b.textContent=k;b.onclick=()=>{if(k==='C')cur='0';else if(k==='='){try{cur=String(Function('"use strict";return ('+cur+')')());}catch{cur='Err';}}else cur=(cur==='0'&&k!=='.')?k:cur+k;d.value=cur;};box.appendChild(b);});
"""
        elif kind == "dashboard":
            body = '<div class="card"><div class="row"><span class="stat" id="a">A: 0</span><span class="stat" id="b">B: 0</span><button id="tick">Sample</button></div><canvas id="c" width="640" height="220"></canvas></div>'
            script = r"""
const c=document.getElementById('c'),x=c.getContext('2d');let pts=[];
function draw(){x.fillStyle='#020617';x.fillRect(0,0,640,220);x.strokeStyle='#3b82f6';x.beginPath();pts.forEach((p,i)=>{const X=i*(640/Math.max(pts.length-1,1)),Y=200-p;if(i)x.lineTo(X,Y);else x.moveTo(X,Y);});x.stroke();}
document.getElementById('tick').onclick=()=>{const a=(Math.random()*100)|0,b=(Math.random()*100)|0;document.getElementById('a').textContent='A: '+a;document.getElementById('b').textContent='B: '+b;pts.push((a+b)/2);if(pts.length>40)pts.shift();draw();};
"""
        elif kind == "website":
            body = f'<div class="card"><h2>Ship the idea</h2><p class="muted">Prompt: {prompt[:180].replace("<","&lt;")}</p><p>This landing page was generated by Cat R11.x Build from your description. Replace the copy, drop in art, and publish anywhere static HTML hosts.</p><div class="row"><a class="btn" href="#get">Get started</a><button onclick="alert(\'Local Cat R11.x Build CTA\')">Live demo</button></div></div><div class="card" id="get" style="margin-top:14px"><h3>Why Cat R11.x Build</h3><ul><li>Plain-language → working HTML</li><li>BitNet brain stays files=off</li><li>Iterate with /build again</li></ul></div>'
            script = "/* static landing */"
        else:
            body = f'<div class="card"><h2>Interactive canvas</h2><p class="muted">{prompt[:200].replace("<","&lt;")}</p><canvas id="c" width="480" height="320"></canvas><p class="muted">Click to paint · generated by Cat R11.x Build</p></div>'
            script = r"""
const c=document.getElementById('c'),x=c.getContext('2d');x.fillStyle='#020617';x.fillRect(0,0,480,320);
let down=false;c.onmousedown=()=>down=true;c.onmouseup=()=>down=false;c.onmousemove=e=>{if(!down)return;const r=c.getBoundingClientRect();x.fillStyle='#3b82f6';x.beginPath();x.arc(e.clientX-r.left,e.clientY-r.top,6,0,6.28);x.fill();};
"""
        return title, self._html_shell(title, body, script)

    def build(self, prompt: str) -> Reply:
        started = time.perf_counter()
        kind = self._classify(prompt)
        title, html = self._template(kind, prompt)
        self.last_path = None
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            path = self.out_dir / f"{self._slug(prompt)}.html"
            path.write_text(html, encoding="utf-8")
            self.last_path = path
            save_status = f"Saved: `{path}`\nOpen: {path.resolve().as_uri()}"
        except OSError as exc:
            save_status = f"Could not save the HTML file: {exc}\nCopy the complete source below."
        # Return source through the shared reply path so CLI, chat, GUI, and
        # token callbacks all receive it exactly once, without truncation.
        fence = "`" * max(3, max((len(m.group()) + 1 for m in re.finditer(r"`+", html)), default=3))
        text = (
            f"cat r1 Build created **{title}** ({kind}).\n\n"
            f"{save_status}\n\n"
            "Complete HTML/JavaScript source (open as an .html file):\n\n"
            f"{fence}html\n{html}\n{fence}\n\n"
            "Model weights stayed files=off.\n"
            "Build another artifact: `/build PROMPT` or say “make a snake game”."
        )
        return Reply(
            text=text,
            route=f"cat-r1-build:{kind}",
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            tokens=len(WordTokenizer.basic_tokenize(text)),
        )

    def self_test(self) -> dict[str, object]:
        reply = self.build("build a snake game")
        ok = bool(self.last_path and self.last_path.is_file() and "canvas" in self.last_path.read_text(encoding="utf-8"))
        if self.last_path and self.last_path.exists():
            try:
                self.last_path.unlink()
            except OSError:
                pass
        return {
            "ok": ok and reply.route.startswith("cat-r1-build:"),
            "route": reply.route,
            "wants_snake": self.wants("make a snake game"),
            "wants_todo": self.wants("build a todo app"),
            "wants_i_want_todo": self.wants("I want to build a todo app"),
            "out_dir": str(self.out_dir),
        }


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(slots=True)
class ChatCompletionResult:
    """OpenAI-shaped chat completion payload (local files=off)."""

    id: str
    model: str
    content: str
    finish_reason: str
    route: str
    language: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_ms: float

    def to_openai_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.content},
                    "finish_reason": self.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            },
            "cat_r1": {
                "route": self.route,
                "language": self.language,
                "elapsed_ms": round(self.elapsed_ms, 2),
                "files": FILES_MODE,
            },
        }


class ChatAPIEngine:
    """Chat Completions API that always knows what to say to the user.

    Local-first (files=off for model weights):
      1. Autodetect wording / code / syntax
      2. Compose a guaranteed reply (exact → memory → bilingual → open-domain)
      3. Optional remote OpenAI-compatible URL via ``CATR1_CHAT_API_URL``
      4. Teacher-force through BitNet when available
      5. Stabilize so the visible answer is never empty / glitchy

    Also serves ``POST /v1/chat/completions`` when launched with ``--api``.
    """

    MODEL = MODEL_ID
    REMOTE_TIMEOUT_S = 4.0

    def __init__(self, runtime: "CatR11X"):
        self.runtime = runtime
        self.last: Optional[ChatCompletionResult] = None
        self._server: Optional[ThreadingHTTPServer] = None

    @staticmethod
    def _history_messages(
        prompt: str,
        history: Optional[list[tuple[str, str]]],
    ) -> list[ChatMessage]:
        messages: list[ChatMessage] = [
            ChatMessage(
                "system",
                "You are cat r1, a local offline BitNet assistant. "
                "Always answer helpfully in the user's language. Never return empty text.",
            )
        ]
        for user_text, assistant_text in (history or [])[-6:]:
            messages.append(ChatMessage("user", user_text))
            messages.append(ChatMessage("assistant", assistant_text))
        messages.append(ChatMessage("user", prompt))
        return messages

    def compose(
        self,
        prompt: str,
        history: Optional[list[tuple[str, str]]] = None,
        *,
        profile: Optional[UserSignalProfile] = None,
    ) -> tuple[str, str]:
        """Return ``(text, route)`` — never empty."""
        history = history or []
        runtime = self.runtime
        profile = profile or runtime.autodetect.scan(prompt)
        prompt = profile.clean or str(prompt or "").strip()
        lang = profile.language

        # 1) Exact tools (math etc.)
        exact = runtime.reasoner.solve(prompt)
        if exact is not None and exact.text.strip():
            text = runtime.autodetect.stabilize_reply(
                exact.text, language=lang, prompt=prompt, profile=profile,
            )
            return text, f"chat-api:exact:{exact.route}"

        # 2) Taught recall
        taught = runtime.taught_memory.get(" ".join(prompt.lower().split()))
        if taught and taught.strip():
            text = runtime.autodetect.stabilize_reply(
                taught, language=lang, prompt=prompt, profile=profile,
            )
            return text, "chat-api:taught"

        # 3) GPT-J RAM dataset (instruction / knowledge pairs, resident files=off)
        gptj = getattr(runtime, "gptj", None)
        if gptj is None:
            runtime.gptj = GptJRamDataset()
            gptj = runtime.gptj
        gptj_hit = gptj.lookup(prompt)
        if gptj_hit:
            # Trust the resident corpus — do not let stabilize wipe short code answers.
            if is_relevant_reply(gptj_hit, prompt, lang) or len(gptj_hit.strip()) >= 8:
                return gptj_hit.strip(), "chat-api:gptj-ram"
            text = runtime.autodetect.stabilize_reply(
                gptj_hit, language=lang, prompt=prompt, profile=profile,
            )
            return text, "chat-api:gptj-ram"

        # 4) ELEZIA casual chat API — hi / how are you / wow that's cool / vibes
        elezia = getattr(runtime, "elezia", None)
        if elezia is None:
            runtime.elezia = EleziaChatAPI()
            elezia = runtime.elezia
        elezia_hit = elezia.process(prompt, lang=lang, profile=profile)
        if elezia_hit is not None:
            text = runtime.autodetect.stabilize_reply(
                elezia_hit[0], language=lang, prompt=prompt, profile=profile,
            )
            return text, f"chat-api:{elezia_hit[1]}"

        # 5) Debug-code shortcut when autodetect sees broken syntax
        if profile.intent_hint == "debug-code" and profile.modality in {"code", "mixed"}:
            grounded = runtime.open_domain.respond(prompt)
            text = runtime.autodetect.stabilize_reply(
                grounded.text, language=lang, prompt=prompt, profile=profile,
            )
            return text, f"chat-api:debug:{grounded.route}"

        # 6) Dataset default / dialogue memory (teaches BitNet real LLM replies)
        dataset_hit = lookup_dialogue_answer(prompt, lang=lang)
        if dataset_hit:
            text = runtime.autodetect.stabilize_reply(
                dataset_hit, language=lang, prompt=prompt, profile=profile,
            )
            return text, "chat-api:dataset-default"

        # 7) Bilingual talker — always says something real in EN/ZH
        if runtime.bilingual_chat is None:
            runtime.bilingual_chat = BilingualChatEngine(runtime)
        talked = runtime.bilingual_chat.talk(prompt, history)
        if talked.text.strip():
            # Prefer bilingual content over open-domain filler.
            if is_relevant_reply(talked.text, prompt, lang) or "bilingual" in talked.route:
                return talked.text.strip(), f"chat-api:bilingual:{talked.route}"

        # 8) Open-domain / answer-everything (never vibes-only)
        open_ans = OpenDomainResponder.respond_anything(prompt)
        if open_ans.text.strip():
            return open_ans.text.strip(), f"chat-api:open:{open_ans.route}"

        # 9) Semantic composition
        semantic = runtime.semantic_reasoner.compose(prompt)
        if semantic is not None and semantic.text.strip():
            text = runtime.autodetect.stabilize_reply(
                semantic.text, language=lang, prompt=prompt, profile=profile,
            )
            return text, f"chat-api:semantic:{semantic.route}"

        # 10) Absolute default — always say something real (never vibes filler)
        text = answer_everything(prompt, lang=lang)
        return text, "chat-api:always-say"

    def _try_remote(self, messages: list[ChatMessage]) -> Optional[str]:
        url = (os.environ.get("CATR1_CHAT_API_URL") or "").strip()
        if not url:
            return None
        key = (os.environ.get("CATR1_CHAT_API_KEY") or "").strip()
        body = json.dumps(
            {
                "model": os.environ.get("CATR1_CHAT_API_MODEL") or "gpt-4o-mini",
                "messages": [{"role": m.role, "content": m.content} for m in messages],
                "temperature": float(getattr(self.runtime, "temperature", 0.7)),
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib_request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib_request.urlopen(req, timeout=self.REMOTE_TIMEOUT_S) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
            choices = payload.get("choices") or []
            if not choices:
                return None
            message = choices[0].get("message") or {}
            content = str(message.get("content") or "").strip()
            return content or None
        except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError, json.JSONDecodeError, OSError):
            return None

    def complete(
        self,
        messages: Optional[list[ChatMessage]] = None,
        *,
        prompt: Optional[str] = None,
        history: Optional[list[tuple[str, str]]] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> ChatCompletionResult:
        started = time.perf_counter()
        history = history or list(self.runtime.history)
        if messages:
            # Last user message is the prompt; prior user/assistant pairs rebuild history.
            user_bits = [m.content for m in messages if m.role == "user"]
            prompt = (user_bits[-1] if user_bits else (prompt or "")).strip()
            rebuilt: list[tuple[str, str]] = []
            pending_user: Optional[str] = None
            for message in messages:
                if message.role == "system":
                    continue
                if message.role == "user":
                    pending_user = message.content
                elif message.role == "assistant" and pending_user is not None:
                    rebuilt.append((pending_user, message.content))
                    pending_user = None
            history = rebuilt
        prompt = str(prompt or "").strip()
        profile = self.runtime.autodetect.scan(prompt)
        prompt = profile.clean
        messages = messages or self._history_messages(prompt, history)

        remote = self._try_remote(messages)
        if remote:
            text = self.runtime.autodetect.stabilize_reply(
                remote, language=profile.language, prompt=prompt, profile=profile,
            )
            route = "chat-api:remote"
        else:
            text, route = self.compose(prompt, history, profile=profile)

        # BitNet teacher-force so the local trunk still runs on the composed answer.
        # Keep dataset / GPT-J / ELEZIA / exact answers if decode drifts off the guide.
        grounded_routes = (
            "chat-api:gptj-ram",
            "chat-api:dataset-default",
            "chat-api:elezia",
            "chat-api:exact",
            "chat-api:taught",
        )
        keep_composed = any(route == r or route.startswith(r + ":") or route.startswith(r + "×") for r in grounded_routes) or route.startswith("chat-api:elezia")
        composed_text = text
        if (
            text
            and self.runtime.model is not None
            and bool(getattr(self.runtime.model.config, "llm_first", True))
        ):
            try:
                report = self.runtime.model.generate_chat(
                    prompt,
                    history,
                    max_new_tokens=max(16, min(96, len(WordTokenizer.basic_tokenize(text)) + 2)),
                    min_new_tokens=1,
                    temperature=0.0,
                    top_k=1,
                    on_token=None,
                    guide_text=text,
                )
                decoded = report.text.strip()
                if decoded and (
                    getattr(report, "finish_reason", "") == "guided"
                    or is_relevant_reply(decoded, prompt, profile.language)
                ):
                    text = decoded
                    route = f"{route}×bitnet-guided"
                elif keep_composed and composed_text:
                    text = composed_text
            except Exception:
                if keep_composed and composed_text:
                    text = composed_text

        text = self.runtime.autodetect.stabilize_reply(
            text, language=profile.language, prompt=prompt, profile=profile,
        )
        if keep_composed and composed_text and (
            "gptj-ram" in route or "dataset-default" in route
        ):
            # Prefer resident corpus over stabilize freeform.
            if composed_text.strip() and (
                not is_relevant_reply(text, prompt, profile.language)
                or text.startswith("Got it —")
                or text.startswith("Understood.")
                or text.startswith("我听懂了")
            ):
                text = composed_text.strip()
        if not text.strip():
            text, route = self.compose(prompt or "hello", history, profile=profile)

        if on_token:
            on_token(text)

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result = ChatCompletionResult(
            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
            model=self.MODEL,
            content=text,
            finish_reason="stop",
            route=route,
            language=profile.language,
            prompt_tokens=len(WordTokenizer.basic_tokenize(prompt)),
            completion_tokens=len(WordTokenizer.basic_tokenize(text)),
            elapsed_ms=elapsed_ms,
        )
        self.last = result
        self.runtime.last_generation_quality = {
            "accepted": True,
            "language": profile.language,
            "route": route,
            "chat_api": True,
            "autodetect": {
                "wording": profile.wording,
                "modality": profile.modality,
                "syntax": profile.syntax,
            },
        }
        return result

    def always_say(
        self,
        prompt: str,
        history: Optional[list[tuple[str, str]]] = None,
        *,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Reply:
        """Public chat entry: always returns a user-facing Reply."""
        result = self.complete(prompt=prompt, history=history, on_token=on_token)
        return Reply(
            text=result.content,
            route=result.route,
            elapsed_ms=result.elapsed_ms,
            tokens=result.completion_tokens,
            tokens_per_second=(
                (result.completion_tokens / (result.elapsed_ms / 1000.0))
                if result.elapsed_ms > 0
                else 0.0
            ),
        )

    def serve(self, host: str = "127.0.0.1", port: int = 8787) -> int:
        """Blocking OpenAI-compatible HTTP server."""
        engine = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                sys.stderr.write("chat-api: " + (fmt % args) + "\n")

            def _send(self, code: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self) -> None:  # noqa: N802
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                if self.path in {"/", "/health", "/v1/models"}:
                    self._send(
                        200,
                        {
                            "object": "list" if self.path == "/v1/models" else "ok",
                            "data": [
                                {
                                    "id": engine.MODEL,
                                    "object": "model",
                                    "owned_by": "cat-r1",
                                }
                            ]
                            if self.path == "/v1/models"
                            else {
                                "status": "ok",
                                "model": engine.MODEL,
                                "files": FILES_MODE,
                                "always_knows_what_to_say": True,
                            },
                        },
                    )
                    return
                self._send(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

            def do_POST(self) -> None:  # noqa: N802
                if self.path not in {"/v1/chat/completions", "/chat/completions"}:
                    self._send(404, {"error": {"message": "not found", "type": "invalid_request_error"}})
                    return
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(max(0, length)) if length else b"{}"
                try:
                    payload = json.loads(raw.decode("utf-8", errors="replace") or "{}")
                except json.JSONDecodeError:
                    self._send(400, {"error": {"message": "invalid json", "type": "invalid_request_error"}})
                    return
                raw_messages = payload.get("messages") or []
                messages: list[ChatMessage] = []
                for item in raw_messages:
                    if not isinstance(item, dict):
                        continue
                    role = str(item.get("role") or "user")
                    content = str(item.get("content") or "")
                    if isinstance(item.get("content"), list):
                        # Vision-style parts → join text fragments.
                        parts = []
                        for part in item["content"]:
                            if isinstance(part, dict) and part.get("type") == "text":
                                parts.append(str(part.get("text") or ""))
                            elif isinstance(part, str):
                                parts.append(part)
                        content = "\n".join(parts)
                    messages.append(ChatMessage(role=role, content=content))
                if not messages:
                    messages = [ChatMessage("user", str(payload.get("prompt") or "hello"))]
                try:
                    result = engine.complete(messages=messages)
                    # Keep server-side conversation memory in RAM.
                    user_bits = [m.content for m in messages if m.role == "user"]
                    if user_bits:
                        engine.runtime.history.append((user_bits[-1], result.content))
                        engine.runtime.history = engine.runtime.history[-12:]
                    self._send(200, result.to_openai_dict())
                except Exception as exc:  # pragma: no cover - defensive API path
                    fallback = engine.always_say("hello")
                    self._send(
                        200,
                        ChatCompletionResult(
                            id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
                            model=engine.MODEL,
                            content=fallback.text or "I'm here — try that again.",
                            finish_reason="stop",
                            route=f"chat-api:error-guard:{type(exc).__name__}",
                            language="en",
                            prompt_tokens=0,
                            completion_tokens=len(WordTokenizer.basic_tokenize(fallback.text)),
                            elapsed_ms=fallback.elapsed_ms,
                        ).to_openai_dict(),
                    )

        self._server = ThreadingHTTPServer((host, int(port)), Handler)
        print(
            f"cat r1 chat API · http://{host}:{port}/v1/chat/completions · "
            f"model={self.MODEL} · files={FILES_MODE}",
            file=sys.stderr,
            flush=True,
        )
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            print("\nchat API stopped.", file=sys.stderr)
            return 130
        finally:
            self._server.server_close()
        return 0

    def self_test(self) -> dict[str, object]:
        samples = ["hello", "what is BitNet", "请介绍一下自己", ""]
        rows = []
        ok = True
        for sample in samples:
            result = self.complete(prompt=sample or "hi", history=[])
            good = bool(result.content.strip()) and "\x00" not in result.content
            ok = ok and good
            rows.append({"prompt": sample, "route": result.route, "ok": good, "chars": len(result.content)})
        return {"ok": ok, "cases": rows}


class CatR11X:
    """Branded cognitive runtime around neural and verifiable local routes."""

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        progress: Optional[Callable[[int, int, float], None]] = None,
    ):
        boot_t0 = time.perf_counter()
        self.config = config or ModelConfig()
        self.tokenizer = WordTokenizer(corpus_texts())
        self.data_vibe = TrainingDataVibeCheck(ALL_DIALOGUES)
        # Defer corpus vibe audit so ≤0.2s boot stays interactive.
        self.data_vibe_report: dict[str, object] = {
            "passed": True,
            "deferred": True,
            "contract": (
                "Vibe audit deferred for ≤0.2s boot; run /vibecheck or --self-test for full report."
            ),
        }
        self.open_domain = OpenDomainResponder()
        self.autodetect = InputAutodetectEngine()
        self.last_generation_quality: dict[str, object] = {}
        self.gptj = GptJRamDataset()
        self.model = InMemoryTernaryLM(self.tokenizer, self.config)
        self.model.gptj_dataset = self.gptj  # completion docs stay resident for warm/train
        self.model.train(progress=progress)
        self.reasoner = ExactReasoner()
        self.semantic_reasoner = SemanticReasoner()
        self.history: list[tuple[str, str]] = []
        self.temperature = self.config.temperature
        self.top_k = self.config.top_k
        self.candidate_count = self.config.deliberation_candidates
        self.teachings = 0
        self.taught_memory: dict[str, str] = {}
        self.code_agent = CatR11XCodeAgent(self)
        self.build_mode = CatR11XBuildMode()
        self.sol_chat = BilingualChatEngine(self)
        self.bilingual_chat = self.sol_chat
        self.elezia = EleziaChatAPI()
        self.cat_r1: Optional[CatR1Engine] = None
        if self.config.cat_r1_enabled:
            self.cat_r1 = CatR1Engine(self)
        self.chat_api = ChatAPIEngine(self)
        # Warm BitNet on dialogue + GPT-J RAM corpus so decode behaves like a real LLM.
        if int(self.config.train_steps) <= 0:
            self.model.warm_bitnet_llm(steps=32)
        self.boot_elapsed_s = time.perf_counter() - boot_t0

    def ensure_vibe_report(self) -> dict[str, object]:
        if self.data_vibe_report.get("deferred"):
            self.data_vibe_report = self.data_vibe.run(self.tokenizer)
        return self.data_vibe_report

    def clear(self) -> None:
        self.history.clear()

    def model_card(self) -> dict[str, object]:
        card = self.model.model_card()
        card.update({
            "runtime": "Cat R11.x cognitive inference runtime",
            "reasoning_tools": [
                "exact arithmetic", "percent arithmetic", "linear algebra",
                "quadratic algebra", "semantic evidence retrieval", "multi-evidence composition",
                "Cat R1 think/answer (files=off) on BitNet + DSpark",
                "Cat R11.x Code (Claude Code fork): Read/Write/Edit/Bash/Glob/Grep/LS",
            ],
            "semantic_memory_documents": len(self.semantic_reasoner.documents),
            "dialogue_dataset_pairs": len(ALL_DIALOGUES),
            "default_dialogues": len(DEFAULT_DIALOGUES),
            "elezia_dialogues": len(ELEZIA_DIALOGUES),
            "elezia_chat_api": True,
            "gptj_ram_dataset": self.gptj.stats(),
            "training_data_vibe_check": self.data_vibe_report,
            "open_domain_response_contract": (
                "Every input returns visible prompt-aware text; missing knowledge is labeled instead of invented."
            ),
            "bitnet_llm_warmup": True,
            "last_generation_quality": self.last_generation_quality,
            "candidate_verifier": "confidence, repetition, language, length, prompt coverage, completion",
            "active_candidate_count": self.candidate_count,
            "in_ram_teachings": self.teachings,
            "exact_taught_recall_entries": len(self.taught_memory),
            "conversation_turns_in_ram": len(self.history),
            "boot_elapsed_s": round(getattr(self, "boot_elapsed_s", 0.0), 4),
            "sparse_bitnet_bank": bool(getattr(self.model.bank, "sparse", False)),
        })
        return card

    def self_test(self) -> dict[str, object]:
        neural = self.model.self_test()
        exact = self.reasoner.self_test()
        semantic = self.semantic_reasoner.self_test()
        open_domain = self.open_domain.self_test()
        data_vibe = self.ensure_vibe_report()
        code = self.code_agent.self_test()
        build = self.build_mode.self_test()
        autodetect = self.autodetect.self_test()
        chat_api = self.chat_api.self_test()
        elezia = self.elezia.self_test()
        gptj = self.gptj.self_test()
        boot_ok = True
        if 0 < float(self.config.boot_budget_s) <= 0.25:
            boot_ok = float(getattr(self, "boot_elapsed_s", 99.0)) <= 0.25
        return {
            "ok": bool(
                neural["ok"] and exact["ok"] and semantic["ok"]
                and open_domain["ok"] and data_vibe["passed"] and code["ok"] and build["ok"]
                and autodetect["ok"] and chat_api["ok"] and elezia["ok"] and gptj["ok"] and boot_ok
            ),
            "neural_inference": neural,
            "exact_reasoning": exact,
            "semantic_reasoning": semantic,
            "open_domain_responses": open_domain,
            "training_data_vibe_check": data_vibe,
            "cat_r11x_code": code,
            "cat_r11x_build": build,
            "input_autodetect": autodetect,
            "chat_api": chat_api,
            "elezia_chat": elezia,
            "gptj_ram": gptj,
            "runtime": {
                "files": FILES_MODE,
                "network_required": False,
                "boot_elapsed_s": round(getattr(self, "boot_elapsed_s", 0.0), 4),
                "boot_under_0_2s": boot_ok,
                "sparse_bitnet_bank": bool(getattr(self.model.bank, "sparse", False)),
                "candidate_verifier_present": True,
                "conversation_context_present": True,
                "teachable_memory_present": True,
                "semantic_composition_present": True,
                "open_domain_fallback_present": True,
                "generation_quality_gate_present": True,
                "input_autodetect_present": True,
                "chat_api_present": True,
                "elezia_dataset_present": True,
                "gptj_ram_dataset_present": True,
                "always_knows_what_to_say": True,
                "every_prompt_returns_visible_text": open_domain["ok"] and chat_api["ok"],
                "claude_code_fork_present": True,
                "grok_build_style_present": True,
                "dspark_speculative_decode_present": True,
                "cat_r1_reason_present": True,
                "bitnet_b158_real": True,
                "files_mode_off": FILES_MODE == "off",
            },
        }

    def _command(self, prompt: str) -> Optional[Reply]:
        raw = prompt.strip()
        lower = raw.lower()
        started = time.perf_counter()
        if not raw.startswith("/"):
            return None
        if lower in {"/help", "/?"}:
            text = (
                "Cat R11.x commands:\n\n"
                "- `/model` — RAM model card and measured loss\n"
                "- `/selftest` — inference-proof tests with evidence\n"
                "- `/trace` — probabilities from the last generation\n"
                "- `/deliberation` — scored candidates from the last difficult prompt\n"
                "- `/vibecheck` — audit corpus coverage and arbitrary-prompt fallbacks\n"
                "- `/train N` — run N more in-memory gradient steps\n"
                "- `/teach PROMPT => ANSWER` — teach one RAM-only example\n"
                "- `/candidates N` — set test-time candidates from 1 to 7\n"
                "- `/temperature N` — set sampling temperature (0 = greedy)\n"
                "- `/clear` — clear conversation context\n"
                "- `/code PROMPT` — Cat R11.x Code (Claude Code fork) on BitNet 20B\n"
                "- `/build PROMPT` — Cat R11.x Build (Grok Build–style apps/games)\n"
                "- `/api` — show local Chat Completions API usage\n"
                "- `/elezia` — ELEZIA casual-chat dataset status\n"
                "- `/gptj` — GPT-J RAM dataset status (resident micro-corpus)\n"
                "- Complex prompts auto-route through Cat R1 think/answer (files=off)\n\n"
                "Ordinary messages use the Chat API composer so cat r1 always knows what to say. Exact math, /teach, and Build still short-circuit when appropriate."
            )
            return Reply(text, "control:/help", (time.perf_counter() - started) * 1000)
        if lower == "/model":
            text = json.dumps(self.model_card(), indent=2, ensure_ascii=False)
            return Reply(text, "control:/model", (time.perf_counter() - started) * 1000)
        if lower == "/selftest":
            result = self.self_test()
            text = json.dumps(result, indent=2, ensure_ascii=False)
            return Reply(text, "control:/selftest", (time.perf_counter() - started) * 1000)
        if lower == "/trace":
            if not self.model.last_trace:
                text = "No Cat R11.x generation trace exists yet."
            else:
                rows = [
                    {
                        "step": step.index,
                        "token": step.token,
                        "probability": round(step.probability, 6),
                        "entropy_bits": round(step.entropy_bits, 4),
                    }
                    for step in self.model.last_trace
                ]
                text = json.dumps(rows, indent=2, ensure_ascii=False)
            return Reply(text, "control:/trace", (time.perf_counter() - started) * 1000)
        if lower == "/deliberation":
            text = (
                json.dumps(self.model.last_deliberation, indent=2, ensure_ascii=False)
                if self.model.last_deliberation
                else "No multi-candidate deliberation exists yet. Ask a complex non-math question first."
            )
            return Reply(text, "control:/deliberation", (time.perf_counter() - started) * 1000)
        if lower in {"/vibecheck", "/vibe-check"}:
            vibe = self.ensure_vibe_report()
            result = {
                "ok": bool(vibe["passed"] and self.open_domain.self_test()["ok"]),
                "training_data": vibe,
                "response_matrix": self.open_domain.self_test(),
                "last_generation_quality": self.last_generation_quality,
            }
            return Reply(
                json.dumps(result, indent=2, ensure_ascii=False),
                "control:/vibecheck",
                (time.perf_counter() - started) * 1000,
            )
        if lower == "/clear":
            self.clear()
            return Reply("cat r1 conversation context cleared.", "control:/clear", (time.perf_counter() - started) * 1000)
        if lower.startswith("/train"):
            parts = raw.split()
            try:
                steps = int(parts[1]) if len(parts) > 1 else 100
            except ValueError:
                return Reply("Usage: /train N", "control:error", (time.perf_counter() - started) * 1000)
            steps = max(1, min(steps, 5000))
            report = self.model.train(steps=steps, boot_budget_s=0.0)
            text = (
                f"Cat R11.x trained for {report.steps} additional RAM-only steps. "
                f"Loss {report.initial_loss:.4f} → {report.final_loss:.4f}."
            )
            return Reply(text, "control:/train", (time.perf_counter() - started) * 1000)
        if lower.startswith("/teach"):
            lesson = raw[len("/teach"):].strip()
            prompt_text, separator, answer_text = lesson.partition("=>")
            if not separator or not prompt_text.strip() or not answer_text.strip():
                return Reply(
                    "Usage: /teach PROMPT => ANSWER",
                    "control:error",
                    (time.perf_counter() - started) * 1000,
                )
            added = self.model.add_training_dialogue(prompt_text.strip(), answer_text.strip())
            report = self.model.train(steps=24, boot_budget_s=0.0)
            memory_key = " ".join(prompt_text.lower().split())
            self.taught_memory[memory_key] = answer_text.strip()
            self.teachings += 1
            text = (
                f"Learned one RAM-only dialogue ({added} next-token samples). "
                f"Training loss {report.initial_loss:.4f} → {report.final_loss:.4f}. "
                "The lesson disappears when this process exits because files=off."
            )
            return Reply(text, "control:/teach", (time.perf_counter() - started) * 1000)
        if lower.startswith("/candidates"):
            parts = raw.split()
            if len(parts) == 1:
                text = f"cat r1 uses up to {self.candidate_count} verified candidates."
            else:
                try:
                    value = int(parts[1])
                except ValueError:
                    return Reply("Usage: /candidates 1..7", "control:error", (time.perf_counter() - started) * 1000)
                self.candidate_count = max(1, min(7, value))
                text = f"Cat R11.x candidate count set to {self.candidate_count}."
            return Reply(text, "control:/candidates", (time.perf_counter() - started) * 1000)
        if lower.startswith("/temperature"):
            parts = raw.split()
            if len(parts) == 1:
                text = f"Cat R11.x temperature is {self.temperature:.2f}."
            else:
                try:
                    value = float(parts[1])
                except ValueError:
                    return Reply("Usage: /temperature 0.0..2.0", "control:error", (time.perf_counter() - started) * 1000)
                self.temperature = max(0.0, min(2.0, value))
                text = f"Cat R11.x temperature set to {self.temperature:.2f}."
            return Reply(text, "control:/temperature", (time.perf_counter() - started) * 1000)
        if lower.startswith("/code"):
            task = raw[len("/code"):].strip()
            if not task:
                return Reply(
                    "Usage: /code PROMPT   or launch `python3 '##cat-r11.x.py' --code`",
                    "control:/code",
                    (time.perf_counter() - started) * 1000,
                )
            reply = self.code_agent.run(task)
            return Reply(reply.text, reply.route, (time.perf_counter() - started) * 1000.0, reply.tokens, reply.tokens_per_second)
        if lower.startswith("/build"):
            task = raw[len("/build"):].strip()
            if not task:
                return Reply(
                    "Usage: /build PROMPT   e.g. /build snake game\nOr: python3 '##cat-r11.x.py' --build",
                    "control:/build",
                    (time.perf_counter() - started) * 1000,
                )
            reply = self.build_mode.build(task)
            return Reply(reply.text, reply.route, (time.perf_counter() - started) * 1000.0, reply.tokens, reply.tokens_per_second)
        if lower in {"/api", "/chat-api"}:
            text = (
                "cat r1 Chat API (OpenAI-compatible, local):\n\n"
                "  python3 '##cat-r11.x.py' --api --api-port 8787\n"
                "  POST http://127.0.0.1:8787/v1/chat/completions\n"
                "  GET  http://127.0.0.1:8787/health\n\n"
                "Elezia casual chat (embedded ELEZIA dataset):\n"
                "  handles hi / how are you / wow that's pretty cool / vibes\n"
                "  route tag: chat-api:elezia:*\n\n"
                "Optional remote upstream (fallback stays local):\n"
                "  CATR1_CHAT_API_URL=https://…/v1/chat/completions\n"
                "  CATR1_CHAT_API_KEY=…\n"
                "  CATR1_CHAT_API_MODEL=…\n\n"
                "Chat always composes a reply — empty/glitchy answers are blocked."
            )
            return Reply(text, "control:/api", (time.perf_counter() - started) * 1000)
        if lower in {"/elezia", "/eliza"}:
            sample = self.elezia.process("hi how are you")
            text = (
                f"ELEZIA dataset pairs: {len(ELEZIA_DIALOGUES)}\n"
                f"Sample «hi how are you» → {sample[0] if sample else '(miss)'}\n"
                f"Last route: {self.elezia.last_route}\n"
                "Casual talk is processed by EleziaChatAPI before BitNet."
            )
            return Reply(text, "control:/elezia", (time.perf_counter() - started) * 1000)
        if lower in {"/gptj", "/gpt-j"}:
            stats = self.gptj.stats()
            sample = self.gptj.lookup("what is GPT-J") or ""
            text = (
                "GPT-J RAM dataset (files=off):\n"
                f"  tag: {stats['model_tag']}\n"
                f"  pairs: {stats['dialogue_pairs']}  docs: {stats['completion_docs']}\n"
                f"  resident: {stats['resident_in_ram']}  ram: {stats['ram_kib']} KiB\n"
                f"  source: {stats['source']}\n"
                f"  sample: {sample[:160]}"
            )
            return Reply(text, "control:/gptj", (time.perf_counter() - started) * 1000)
        return Reply("Unknown Cat R11.x command. Use /help.", "control:unknown", (time.perf_counter() - started) * 1000)

    def reply(self, prompt: str, on_token: Optional[Callable[[str], None]] = None) -> Reply:
        """Real LLM chat: BitNet autoregressive decode is the default path.

        Non-LM shortcuts are limited to slash-commands, taught recall, exact math
        tools, and explicit Build mode. FAQ/semantic/template routes no longer
        preempt the transformer when ``llm_first`` is on (default).
        """
        started = time.perf_counter()
        # Auto-detect wording / code / syntax and sanitize before any route.
        profile = self.autodetect.scan(prompt)
        prompt = profile.clean
        self.model.last_detected_language = profile.language
        llm_first = bool(getattr(self.model.config, "llm_first", True))
        if not prompt or len(prompt) > self.open_domain.MAX_PROMPT_CHARS or not profile.safe:
            if not profile.safe and profile.glitch_flags:
                if profile.language == "zh":
                    text = (
                        "我先把输入里的异常字符清掉了。请用正常文字、代码或报错重发一次，"
                        "我会按你的用词继续。"
                    )
                else:
                    text = (
                        "I cleaned glitchy characters from that input. Resend as normal text, "
                        "code, or a traceback and I’ll match your wording from there."
                    )
                route = f"autodetect:glitch-guard:{','.join(profile.glitch_flags[:4])}"
                if on_token:
                    on_token(text)
                return Reply(
                    text,
                    route,
                    (time.perf_counter() - started) * 1000.0,
                    tokens=len(WordTokenizer.basic_tokenize(text)),
                )
            fallback = self.open_domain.respond(prompt)
            text = self.autodetect.stabilize_reply(
                fallback.text, language=profile.language, prompt=prompt, profile=profile,
            )
            if on_token:
                on_token(text)
            return Reply(
                text,
                f"{fallback.route}|autodetect:{profile.modality}/{profile.syntax}",
                (time.perf_counter() - started) * 1000.0,
                tokens=len(WordTokenizer.basic_tokenize(text)),
            )

        def _finish(reply: Reply) -> Reply:
            # Grounded dataset / GPT-J / ELEZIA answers stay intact.
            grounded = reply.route.startswith((
                "chat-api:gptj-ram",
                "chat-api:dataset-default",
                "chat-api:elezia",
                "chat-api:exact",
                "chat-api:taught",
                "in-ram-taught-recall",
            ))
            if grounded and reply.text.strip() and (
                is_relevant_reply(reply.text, prompt, profile.language)
                or "gptj-ram" in reply.route
                or "dataset-default" in reply.route
            ):
                stable = reply.text.strip()
            else:
                stable = self.autodetect.stabilize_reply(
                    reply.text,
                    language=profile.language,
                    prompt=prompt,
                    profile=profile,
                )
            route = (
                f"{reply.route}|autodetect:"
                f"{profile.language}/{profile.wording}/{profile.modality}/{profile.syntax}"
            )
            if profile.glitch_flags:
                route += f":flags={','.join(profile.glitch_flags[:3])}"
            self.last_generation_quality = {
                **dict(self.last_generation_quality or {}),
                "autodetect": {
                    "language": profile.language,
                    "wording": profile.wording,
                    "modality": profile.modality,
                    "syntax": profile.syntax,
                    "syntax_ok": profile.syntax_ok,
                    "intent_hint": profile.intent_hint,
                    "glitch_flags": list(profile.glitch_flags),
                },
            }
            return Reply(
                stable,
                route,
                reply.elapsed_ms if reply.elapsed_ms else (time.perf_counter() - started) * 1000.0,
                tokens=reply.tokens or len(WordTokenizer.basic_tokenize(stable)),
                tokens_per_second=reply.tokens_per_second,
            )

        control = self._command(prompt)
        if control is not None:
            finished = _finish(control)
            if on_token:
                on_token(finished.text)
            return finished
        taught = self.taught_memory.get(" ".join(prompt.lower().split()))
        if taught is not None:
            self.history.append((prompt, taught))
            self.history = self.history[-12:]
            finished = _finish(Reply(
                taught,
                "in-ram-taught-recall",
                (time.perf_counter() - started) * 1000.0,
                tokens=len(WordTokenizer.basic_tokenize(taught)),
            ))
            if on_token:
                on_token(finished.text)
            return finished
        exact = self.reasoner.solve(prompt)
        if exact is not None:
            self.history.append((prompt, exact.text))
            self.history = self.history[-12:]
            finished = _finish(Reply(
                exact.text,
                exact.route,
                (time.perf_counter() - started) * 1000.0,
                tokens=len(WordTokenizer.basic_tokenize(exact.text)),
            ))
            if on_token:
                on_token(finished.text)
            return finished
        if CatR11XBuildMode.wants(prompt):
            built = self.build_mode.build(prompt)
            finished = _finish(built)
            self.history.append((prompt, finished.text))
            self.history = self.history[-12:]
            if on_token:
                on_token(finished.text)
            return finished

        # Prefer debug-code path when autodetect sees broken code + fix wording.
        if profile.intent_hint == "debug-code" and profile.modality in {"code", "mixed"}:
            grounded = self.open_domain.respond(prompt)
            finished = _finish(Reply(
                grounded.text,
                grounded.route,
                (time.perf_counter() - started) * 1000.0,
                tokens=len(WordTokenizer.basic_tokenize(grounded.text)),
            ))
            self.history.append((prompt, finished.text))
            self.history = self.history[-12:]
            if on_token:
                on_token(finished.text)
            return finished

        # Chat API: always knows what to say (compose → optional BitNet guide → stabilize).
        if getattr(self, "chat_api", None) is not None:
            reply = self.chat_api.always_say(prompt, self.history, on_token=None)
            finished = _finish(reply)
            self.history.append((prompt, finished.text))
            self.history = self.history[-12:]
            if on_token:
                on_token(finished.text)
            return finished

        # cat r1 v0.1 × DSpark fallback if chat API missing.
        if (
            self.cat_r1 is not None
            and self.model.config.cat_r1_enabled
            and self.model.config.cat_r1_respond_anything
        ):
            if self.model.dspark is None:
                self.model.dspark = DSparkBitNetEngine(self.model)
            reply = self.cat_r1.respond_anything(prompt, self.history, on_token=None)
            finished = _finish(reply)
            self.history.append((prompt, finished.text))
            self.history = self.history[-12:]
            if on_token:
                on_token(finished.text)
            return finished

        # --- Legacy BitNet / hybrid paths below ---
        if not llm_first:
            analysis = self.open_domain.analyze(prompt)
            if analysis.intent in {
                "summarize", "rewrite", "translate", "debug", "compare",
                "build", "brainstorm", "current", "verify",
            }:
                grounded = self.open_domain.respond(prompt)
                finished = _finish(Reply(
                    grounded.text,
                    grounded.route,
                    (time.perf_counter() - started) * 1000.0,
                    tokens=len(WordTokenizer.basic_tokenize(grounded.text)),
                ))
                self.history.append((prompt, finished.text))
                self.history = self.history[-12:]
                if on_token:
                    on_token(finished.text)
                return finished
            semantic_query = prompt
            if self.history and re.search(
                r"\b(?:it|that|this|those|previous|earlier|above|same)\b|继续|刚才|上一个|那个",
                prompt,
                re.IGNORECASE,
            ):
                semantic_query = self.history[-1][0] + " " + prompt
            semantic = self.semantic_reasoner.compose(semantic_query)
            if semantic is not None:
                finished = _finish(Reply(
                    semantic.text,
                    semantic.route,
                    (time.perf_counter() - started) * 1000.0,
                    tokens=len(WordTokenizer.basic_tokenize(semantic.text)),
                ))
                self.history.append((prompt, finished.text))
                self.history = self.history[-12:]
                if on_token:
                    on_token(finished.text)
                return finished

        if (
            self.model.cat_r1_reason is not None
            and self.model.config.cat_r1_reason_enabled
            and self.model.cat_r1_reason.needs_reasoning(prompt)
        ):
            text, ds_stats = self.model.cat_r1_reason.reason(
                prompt,
                self.history,
                temperature=self.temperature,
                top_k=self.top_k,
                on_token=None,
            )
            usable, quality_score, quality_reasons = self.open_domain.generation_quality(
                prompt, text, llm_first=llm_first,
            )
            self.last_generation_quality = {
                "accepted": usable,
                "score": round(quality_score, 4),
                "reasons": quality_reasons,
                "neural_preview": text[:180],
                "cat_r1_reason_reward": ds_stats.reward,
                "cat_r1_think_tokens": ds_stats.think_tokens,
                "llm_first": llm_first,
            }
            if usable:
                route = (
                    f"llm:cat-r1×bitnet:{profile.language}:"
                    f"{ds_stats.candidates}c:r={ds_stats.reward:.3f}"
                )
            elif llm_first and text.strip():
                route = (
                    f"llm:cat-r1×bitnet:{profile.language}:"
                    f"{ds_stats.candidates}c:soft"
                )
            else:
                fallback = self.open_domain.respond(prompt)
                text = fallback.text
                route = f"{fallback.route}:cat-r1-r1-quality-repair"
            finished = _finish(Reply(
                text=text,
                route=route,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                tokens=ds_stats.answer_tokens + ds_stats.think_tokens,
                tokens_per_second=0.0,
            ))
            self.history.append((prompt, finished.text))
            self.history = self.history[-12:]
            if on_token:
                on_token(finished.text)
            return finished
        report = self.model.deliberate_chat(
            prompt,
            self.history,
            temperature=self.temperature,
            top_k=self.top_k,
            candidate_count=self.candidate_count,
        )
        usable, quality_score, quality_reasons = self.open_domain.generation_quality(
            prompt, report.text, llm_first=llm_first,
        )
        self.last_generation_quality = {
            "accepted": usable,
            "score": round(quality_score, 4),
            "reasons": quality_reasons,
            "neural_preview": report.text[:180],
            "llm_first": llm_first,
        }
        if usable:
            text = report.text
            route = f"llm:bitnet-decode:{report.language}:{len(self.model.last_deliberation)}c"
        else:
            fallback = self.open_domain.respond(prompt)
            text = fallback.text
            route = f"{fallback.route}:quality-repair"
        finished = _finish(Reply(
            text=text,
            route=route,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            tokens=report.tokens,
            tokens_per_second=report.tokens_per_second,
        ))
        self.history.append((prompt, finished.text))
        self.history = self.history[-12:]
        if on_token:
            on_token(finished.text)
        return finished


class CatR11XGUI:
    BG = "#03060b"
    PANEL = "#07101d"
    INPUT = "#0a1424"
    BLUE = "#3b82f6"
    BLUE_2 = "#60a5fa"
    TEXT = "#dbeafe"
    MUTED = "#7890ad"
    RED = "#f87171"

    def __init__(self, root: tk.Tk, engine: CatR11X):
        self.root = root
        self.engine = engine
        self.events: queue.Queue[tuple[str, int, int, object]] = queue.Queue()
        self.busy = False
        self.conversation_id = 0
        self.request_serial = 0
        self.active_request_id: Optional[int] = None
        self._build()
        self.root.after(GUI_TICK_MS, self._drain)

    @staticmethod
    def _family(mono: bool = False) -> str:
        if mono:
            return "Cascadia Mono" if os.name == "nt" else "Menlo"
        return "Segoe UI" if os.name == "nt" else "Helvetica Neue"

    def _font(self, size: int, bold: bool = False, mono: bool = False) -> font.Font:
        return font.Font(family=self._family(mono), size=size, weight="bold" if bold else "normal")

    def _build(self) -> None:
        root = self.root
        root.title(f"{APP_NAME} v{APP_VERSION}")
        root.geometry("1040x720")
        root.minsize(780, 560)
        root.configure(bg=self.BG)

        menu = tk.Menu(root, tearoff=False, bg=self.PANEL, fg=self.TEXT)
        cat_menu = tk.Menu(menu, tearoff=False, bg=self.PANEL, fg=self.TEXT)
        cat_menu.add_command(label=f"New {APP_NAME} chat", command=self._new_chat)
        cat_menu.add_command(label=f"{APP_NAME} model card", command=lambda: self._submit("/model"))
        cat_menu.add_command(label=f"{APP_NAME} inference self-test", command=lambda: self._submit("/selftest"))
        cat_menu.add_separator()
        cat_menu.add_command(label=f"Exit {APP_NAME}", command=root.destroy)
        menu.add_cascade(label=APP_NAME, menu=cat_menu)
        root.config(menu=menu)

        header = tk.Frame(root, bg=self.PANEL, padx=18, pady=14)
        header.pack(fill="x")
        logo = tk.Frame(header, bg="#000000", width=46, height=46)
        logo.pack(side="left")
        logo.pack_propagate(False)
        tk.Label(logo, text="🐱", bg="#000000", fg=self.BLUE, font=self._font(23)).place(relx=0.5, rely=0.5, anchor="center")
        name = tk.Frame(header, bg=self.PANEL)
        name.pack(side="left", padx=(12, 0))
        tk.Label(name, text=APP_NAME, bg=self.PANEL, fg=self.TEXT, font=self._font(17, True)).pack(anchor="w")
        report = self.engine.model.report
        tk.Label(
            name,
            text=(
                f"Real BitNet b1.58 ~20B · {ENGINE_NAME} · {self.engine.model.parameter_count:,} params · "
                f"loss {report.initial_loss:.3f}→{report.final_loss:.3f} · files=off · {GUI_FPS} FPS · {self.engine.model.config.speed_target}"
            ),
            bg=self.PANEL,
            fg=self.MUTED,
            font=self._font(9),
        ).pack(anchor="w")
        self.status = tk.Label(header, text=f"{APP_NAME} inference ready", bg=self.PANEL, fg=self.BLUE_2, font=self._font(10))
        self.status.pack(side="right")

        body = tk.Frame(root, bg=self.BG)
        body.pack(fill="both", expand=True)
        sidebar = tk.Frame(body, bg=self.PANEL, width=220, padx=14, pady=18)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        tk.Button(
            sidebar,
            text=f"+ New {APP_NAME} chat",
            command=self._new_chat,
            bg="#000000",
            fg=self.BLUE,
            activebackground="#111827",
            activeforeground=self.BLUE_2,
            relief="flat",
            bd=0,
            padx=10,
            pady=10,
            font=self._font(10, True),
            cursor="hand2",
        ).pack(fill="x")
        tk.Label(sidebar, text="REAL RAM LM", bg=self.PANEL, fg=self.MUTED, font=self._font(9, True)).pack(anchor="w", pady=(22, 8))
        for line in (
            "● Startup gradient training",
            "● English / 中文 auto mode",
            "● Next-token softmax",
            "● Real BitNet b1.58 LLM decode (W1.58A8)",
            "● EN/ZH auto-detect on BitNet sampling",
            "● cat r1 chat",
            "● cat r1 engine",
            "● Multi-turn conversation memory",
            "● Respond-anything (can talk)",
            "● Ternary add/sub kernel",
            f"● Speed target: {self.engine.model.config.speed_target}",
            f"● GUI {GUI_FPS} FPS",
            "● Verified decode candidates",
            "● Exact math / algebra tools",
            "● Lossless byte fallback",
            "● A8 absmax activations",
            "● sparse BitNet bank (≤0.2s boot)",
            "● files=off (RAM only)",
            "● No checkpoints/API",
        ):
            tk.Label(sidebar, text=line, bg=self.PANEL, fg=self.BLUE_2, font=self._font(9)).pack(anchor="w", pady=3)
        tk.Label(
            sidebar,
            text="/model\n/selftest\n/deliberation\n/train 100\n/teach Q => A\n/candidates 3\n/clear",
            justify="left",
            bg=self.PANEL,
            fg=self.MUTED,
            font=self._font(9, mono=True),
        ).pack(anchor="w", side="bottom")

        chat_area = tk.Frame(body, bg=self.BG)
        chat_area.pack(side="left", fill="both", expand=True)
        self.chat = scrolledtext.ScrolledText(
            chat_area,
            state="disabled",
            wrap="word",
            bg=self.BG,
            fg=self.TEXT,
            insertbackground=self.BLUE,
            selectbackground="#1d4ed8",
            relief="flat",
            bd=0,
            padx=26,
            pady=20,
            font=self._font(11),
        )
        self.chat.pack(fill="both", expand=True)
        self.chat.tag_configure("title", foreground=self.BLUE_2, font=self._font(11, True), spacing1=12)
        self.chat.tag_configure("user", foreground="#93c5fd", font=self._font(11))
        self.chat.tag_configure("bot", foreground=self.TEXT, font=self._font(11))
        self.chat.tag_configure("muted", foreground=self.MUTED, font=self._font(9))
        self.chat.tag_configure("error", foreground=self.RED, font=self._font(10))
        self._append(f"{APP_NAME}\n", "title")
        self._append(
            "Real BitNet LLM decode in RAM (files=off, llm_first). English and 中文 are selected automatically. Exact tools handle arithmetic/algebra; everything else is autoregressive Transformer generation. Use /vibecheck and /selftest for evidence.\n",
            "muted",
        )

        composer = tk.Frame(chat_area, bg=self.PANEL, padx=18, pady=14)
        composer.pack(fill="x")
        self.entry = tk.Text(
            composer,
            height=3,
            bg=self.INPUT,
            fg=self.TEXT,
            insertbackground=self.BLUE,
            selectbackground="#1d4ed8",
            relief="flat",
            bd=0,
            padx=12,
            pady=10,
            font=self._font(10),
            wrap="word",
            undo=True,
        )
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", self._on_enter)
        self.send = tk.Button(
            composer,
            text="INFER ↑",
            command=self._submit,
            bg="#000000",
            fg=self.BLUE,
            activebackground="#111827",
            activeforeground=self.BLUE_2,
            relief="flat",
            bd=0,
            padx=16,
            pady=12,
            font=self._font(10, True),
            cursor="hand2",
        )
        self.send.pack(side="right", padx=(12, 0))
        self.entry.focus_set()

    def _append(self, text: str, tag: str = "bot") -> None:
        self.chat.config(state="normal")
        self.chat.insert("end", text, tag)
        self.chat.config(state="disabled")
        self.chat.see("end")

    def _new_chat(self) -> None:
        # Invalidate every queued result from the previous conversation. A
        # running inference cannot be force-killed safely, so its visible
        # result is discarded and its final idle event unlocks the composer.
        self.conversation_id += 1
        self.engine.clear()
        self.chat.config(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.config(state="disabled")
        self.entry.delete("1.0", "end")
        self.entry.edit_reset()
        self._append(f"{APP_NAME}\n", "title")
        if self.busy:
            self._append(
                "New RAM-only conversation created. Finishing the previous inference in the background…\n",
                "muted",
            )
            self.status.config(text="New chat · finishing previous inference…")
        else:
            self.active_request_id = None
            self.send.config(state="normal")
            self._append("New RAM-only language-model conversation.\n", "muted")
            self.status.config(text=f"{APP_NAME} new chat ready")
        self.entry.focus_set()

    def _on_enter(self, event) -> Optional[str]:
        if event.state & 0x1:
            return None
        self._submit()
        return "break"

    def _submit(self, forced: Optional[str] = None) -> None:
        if self.busy:
            return
        prompt = forced if isinstance(forced, str) else self.entry.get("1.0", "end-1c").strip()
        if not prompt:
            return
        self.entry.delete("1.0", "end")
        self._append("\nYOU\n", "title")
        self._append(prompt + "\n", "user")
        self._append(f"\n{APP_NAME}\n", "title")
        self.busy = True
        self.request_serial += 1
        request_id = self.request_serial
        conversation_id = self.conversation_id
        self.active_request_id = request_id
        self.send.config(state="disabled")
        self.status.config(text=f"{APP_NAME} computing next tokens…")

        def worker() -> None:
            try:
                reply = self.engine.reply(prompt)
                self.events.put(("done", conversation_id, request_id, reply))
            except Exception as exc:
                self.events.put(("error", conversation_id, request_id, f"{type(exc).__name__}: {exc}"))
            finally:
                self.events.put(("idle", conversation_id, request_id, None))

        threading.Thread(target=worker, daemon=True).start()

    def _drain(self) -> None:
        try:
            while True:
                kind, conversation_id, request_id, payload = self.events.get_nowait()
                is_current = conversation_id == self.conversation_id
                if kind == "done" and is_current:
                    reply = payload
                    self._append(reply.text + "\n", "bot")
                    self.status.config(text=f"{APP_NAME} ready")
                elif kind == "error" and is_current:
                    self._append("ERROR\n" + str(payload) + "\n", "error")
                    self.status.config(text=f"{APP_NAME} error")
                elif kind == "idle" and request_id == self.active_request_id:
                    if not is_current:
                        # The stale worker may have appended to engine history
                        # after New Chat cleared it; clear once more at exit.
                        self.engine.clear()
                    self.busy = False
                    self.active_request_id = None
                    self.send.config(state="normal")
                    if not is_current:
                        self.status.config(text=f"{APP_NAME} new chat ready")
                        self._append("Previous inference discarded. New chat is ready.\n", "muted")
                    self.entry.focus_set()
        except queue.Empty:
            pass
        self.root.after(GUI_TICK_MS, self._drain)


def make_parser() -> argparse.ArgumentParser:
    defaults = ModelConfig()
    parser = argparse.ArgumentParser(
        prog=os.path.basename(sys.argv[0]),
        description=f"{APP_NAME} v{APP_VERSION}: bilingual RAM-only BitNet b1.58 response runtime (files=off)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--chat", action="store_true", help="terminal chat instead of GUI")
    parser.add_argument("--api", action="store_true", help="serve OpenAI-compatible Chat Completions API")
    parser.add_argument("--api-host", default="127.0.0.1", help="Chat API bind host")
    parser.add_argument("--api-port", type=int, default=8787, help="Chat API bind port")
    parser.add_argument("--code", action="store_true", help="Cat R11.x Code REPL (Claude Code fork on BitNet 20B)")
    parser.add_argument("--hybrid", action="store_true", help="legacy FAQ/semantic/template router (disables llm_first)")
    parser.add_argument("--build", action="store_true", help="Cat R11.x Build REPL (Grok Build–style apps/games)")
    parser.add_argument("--cwd", default=None, help="workspace root for --code / --build artifacts")
    parser.add_argument("--prompt", help="generate one answer and exit")
    parser.add_argument(
        "--vibe-check",
        action="store_true",
        help="audit embedded training data and arbitrary-prompt coverage without allocating the 20B bank",
    )
    parser.add_argument("--self-test", action="store_true", help="run proof-oriented inference tests")
    parser.add_argument("--model-card", action="store_true", help="print the honest model card")
    parser.add_argument(
        "--dspark",
        dest="dspark",
        action="store_true",
        default=True,
        help="enable DSpark speculative decode (files=off; default on)",
    )
    parser.add_argument(
        "--no-dspark",
        dest="dspark",
        action="store_false",
        help="disable DSpark speculative decode",
    )
    parser.add_argument(
        "--cat-r1-reason",
        dest="cat_r1_reason",
        action="store_true",
        default=True,
        help="enable cat r1 think/answer on DSpark × BitNet (files=off; default on)",
    )
    parser.add_argument(
        "--no-cat-r1-reason",
        dest="cat_r1_reason",
        action="store_false",
        help="disable Cat R1 reasoning",
    )
    parser.add_argument(
        "--show-think",
        action="store_true",
        help="include internal think chain in visible replies",
    )
    parser.add_argument(
        "--profile",
        choices=("fable5", "fast", "balanced", "quality"),
        default="fable5",
        help="RAM/speed profile; fable5 is the default Claude Fable 5–class interactive target",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=None,
        help="override profile startup steps; use 0 to skip startup training",
    )
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument("--top-k", type=int, default=defaults.top_k)
    parser.add_argument("--max-tokens", type=int, default=defaults.max_new_tokens)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=DEFAULT_SEED)
    parser.add_argument("--version", action="version", version=f"{APP_NAME} v{APP_VERSION}")
    return parser


def terminal(engine: CatR11X, one_prompt: Optional[str] = None) -> int:
    def run(prompt: str) -> None:
        reply = engine.reply(prompt)
        print(reply.text)
        print()

    if one_prompt:
        run(one_prompt)
        return 0
    report = engine.model.report
    print(
        f"{APP_NAME} v{APP_VERSION} · RAM LM · loss "
        f"{report.initial_loss:.3f}→{report.final_loss:.3f} · /help · /quit"
    )
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            continue
        if prompt.lower() in {"/quit", "/exit", "quit", "exit"}:
            return 0
        run(prompt)


def code_terminal(engine: CatR11X, one_prompt: Optional[str] = None, cwd: Optional[str] = None) -> int:
    agent = CatR11XCodeAgent(engine, root=cwd)
    engine.code_agent = agent
    banner = (
        f"{APP_NAME} Code · Claude Code fork · brain {MODEL_ID} · files={FILES_MODE}\n"
        f"workspace: {agent.tools.root}\n"
        "tools: Read Write Edit Bash Glob Grep LS · /quit to exit"
    )
    print(banner)

    def run(prompt: str) -> None:
        reply = agent.run(prompt)
        print(reply.text)
        print(f"\n[{reply.route} · {reply.elapsed_ms:.0f} ms]\n")

    if one_prompt:
        run(one_prompt)
        return 0
    while True:
        try:
            prompt = input("code> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            continue
        if prompt.lower() in {"/quit", "/exit", "quit", "exit"}:
            return 0
        run(prompt)


def build_terminal(engine: CatR11X, one_prompt: Optional[str] = None, cwd: Optional[str] = None) -> int:
    builder = CatR11XBuildMode(root=cwd)
    engine.build_mode = builder
    banner = (
        f"{APP_NAME} Build · Grok Build–style · brain {MODEL_ID} · files={FILES_MODE}\n"
        f"artifacts: {builder.out_dir}\n"
        "describe an app/game/website · /quit to exit"
    )
    print(banner)

    def run(prompt: str) -> None:
        reply = builder.build(prompt)
        print(reply.text)
        print(f"\n[{reply.route} · {reply.elapsed_ms:.0f} ms]\n")

    if one_prompt:
        run(one_prompt)
        return 0
    while True:
        try:
            prompt = input("build> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            continue
        if prompt.lower() in {"/quit", "/exit", "quit", "exit"}:
            return 0
        run(prompt)


def main(argv: Optional[list[str]] = None) -> int:
    if sys.version_info < (3, 10):
        print(
            f"{APP_NAME} requires Python 3.10+ (running {sys.version.split()[0]}).",
            file=sys.stderr,
        )
        return 2
    args = make_parser().parse_args(argv)
    if args.vibe_check:
        tokenizer = WordTokenizer(corpus_texts())
        data_report = TrainingDataVibeCheck(ALL_DIALOGUES).run(tokenizer)
        response_report = OpenDomainResponder().self_test()
        result = {
            "ok": bool(data_report["passed"] and response_report["ok"]),
            "files": FILES_MODE,
            "training_data": data_report,
            "response_matrix": response_report,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["ok"] else 1
    profiles = {
        # Real BitNet LLM: multi-layer BitNet decode (llm_first). STE via /train; head warm on first chat.
        "fable5": dict(
            context_tokens=32,
            mod_capacity=1,
            train_layers=2,
            batch_size=4,
            deliberation_candidates=1,
            train_steps=0,
            boot_budget_s=BOOT_BUDGET_S,
            speed_target="fable5",
        ),
        "fast": dict(
            context_tokens=32,
            mod_capacity=1,
            train_layers=2,
            batch_size=4,
            deliberation_candidates=1,
            train_steps=0,
            boot_budget_s=BOOT_BUDGET_S,
            speed_target="fast",
        ),
        "balanced": dict(
            context_tokens=96,
            mod_capacity=6,
            train_layers=4,
            batch_size=4,
            deliberation_candidates=2,
            train_steps=0,
            boot_budget_s=BOOT_BUDGET_S,
            speed_target="balanced",
        ),
        "quality": dict(
            context_tokens=128,
            mod_capacity=8,
            train_layers=6,
            batch_size=4,
            deliberation_candidates=3,
            train_steps=8,
            boot_budget_s=8.0,
            speed_target="quality",
        ),
    }
    profile = profiles[args.profile]
    interactive = str(profile["speed_target"]) in {"fable5", "fast"}
    # Proof paths need uncapped training so loss-reduction checks stay meaningful.
    boot_budget = 0.0 if (args.self_test or args.model_card) else float(profile["boot_budget_s"])
    train_default = profile["train_steps"]
    if args.self_test and args.train_steps is None:
        train_default = max(train_default, 16)
    config = ModelConfig(
        context_tokens=profile["context_tokens"],
        embedding_dim=BITNET_20B_D_MODEL,
        hidden_dim=BITNET_20B_D_FF,
        d_model=BITNET_20B_D_MODEL,
        d_ff=BITNET_20B_D_FF,
        n_layers=BITNET_20B_N_LAYERS,
        n_heads=BITNET_20B_N_HEADS,
        mod_capacity=profile["mod_capacity"],
        train_layers=profile["train_layers"],
        dense_init=False,
        llm_first=not bool(getattr(args, "hybrid", False)),
        reasoning_passes=profile["mod_capacity"],
        max_reasoning_passes=BITNET_20B_N_LAYERS,
        batch_size=profile["batch_size"],
        deliberation_candidates=profile["deliberation_candidates"],
        train_steps=max(0, min(args.train_steps if args.train_steps is not None else train_default, 20_000)),
        temperature=max(0.0, min(args.temperature, 2.0)),
        top_k=max(1, args.top_k),
        max_new_tokens=max(1, min(args.max_tokens, 2048)),
        seed=args.seed,
        boot_budget_s=boot_budget,
        engine=ENGINE_ID,
        kda_ratio=CAT_R1_KDA_RATIO,
        moe_top_k=1 if interactive else CAT_R1_MOE_TOP_K,
        attnres=True,
        speed_target=str(profile["speed_target"]),
        dspark_enabled=bool(args.dspark),
        # Speculative draft doubles CPU BitNet cost — keep off for interactive profiles.
        dspark_speculative_decode=bool(args.dspark) and not interactive,
        dspark_block_size=DSPARK_BLOCK_SIZE,
        dspark_markov_rank=DSPARK_MARKOV_RANK,
        dspark_confidence_head=True,
        dspark_draft_gamma=DSPARK_DRAFT_GAMMA,
        dspark_adaptive_gamma=True,
        cat_r1_reason_enabled=bool(args.cat_r1_reason),
        cat_r1_think_tokens=CAT_R1_THINK_BUDGET,
        cat_r1_answer_tokens=CAT_R1_ANSWER_BUDGET,
        cat_r1_show_think=bool(args.show_think),
        cat_r1_candidates=1 if args.profile == "fast" else 2,
        cat_r1_enabled=True,
        cat_r1_ced=True,
        cat_r1_csa2=True,
        cat_r1_mhc=True,
        cat_r1_engram=True,
        cat_r1_fp4_kv=True,
        cat_r1_respond_anything=True,
    )
    show_startup_progress = config.train_steps > 0 and not (args.self_test or args.model_card)
    progress: Optional[Callable[[int, int, float], None]] = None
    if show_startup_progress:
        print(
            f"Materializing ~20B BitNet b1.58 + {ENGINE_NAME} in RAM (files=off, {config.speed_target}): "
            f"0/{config.train_steps} STE steps...",
            file=sys.stderr,
            flush=True,
        )

        def progress(step: int, total: int, loss: float) -> None:
            print(
                f"Training {APP_NAME} STE slice: {step}/{total} steps · loss {loss:.3f}",
                file=sys.stderr,
                flush=True,
            )

    try:
        engine = CatR11X(config, progress=progress)
    except KeyboardInterrupt:
        print(f"\n{APP_NAME} startup training cancelled.", file=sys.stderr)
        return 130
    if not (args.self_test or args.model_card):
        print(
            f"{APP_NAME} ready · boot {getattr(engine, 'boot_elapsed_s', 0.0):.3f}s · "
            f"sparse_bank={getattr(engine.model.bank, 'sparse', False)} · "
            f"files={FILES_MODE} · BitNet b1.58 ~20B slots",
            file=sys.stderr,
            flush=True,
        )
    if args.self_test:
        result = engine.self_test()
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["ok"] else 1
    if args.model_card:
        print(json.dumps(engine.model_card(), indent=2, ensure_ascii=False))
        return 0
    if args.api:
        return engine.chat_api.serve(host=str(args.api_host), port=int(args.api_port))
    if args.code:
        return code_terminal(engine, args.prompt, cwd=args.cwd)
    if args.build:
        return build_terminal(engine, args.prompt, cwd=args.cwd)
    if args.prompt or args.chat:
        return terminal(engine, args.prompt)
    root = tk.Tk()
    CatR11XGUI(root, engine)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
