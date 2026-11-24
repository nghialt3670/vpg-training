#!/usr/bin/env python3
"""Train one logit per FLUX.2 Klein attention head (all heads) from compbench-style edit samples.

For each sample:
  1. Denoise with Euler steps (default 30) and classifier-free guidance. During the first
     `--guided-steps` steps (default 5) the latent is nudged along the gradient of a coverage
     loss that compares the head-aggregated object attention with the visual prompt density.
  2. Decode the final latent and compute an L1 image loss inside and outside the edit mask.
  3. Backpropagate through the whole trajectory exactly. Each step is recomputed from its
     stored input latent, so only one step's second-order graph is alive at a time.
  4. Take an AdamW step on the head logits.

The head logits are the only trainable parameters. The transformer and VAE are frozen.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import Flux2KleinPipeline
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
from PIL import Image, ImageDraw
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint

PATCH = 16  # pixels per token-grid cell (VAE 8x and 2x2 packing)
TEXT_TOKENS = 256  # text sequence length used for text embeddings and object-token lookup
TEMPERATURE = 1.0  # softmax temperature over head logits
POINT_SIGMA = 12.0  # Gaussian sigma for point prompts, in working-resolution pixels
EPS = 1e-12
EXPECTED_HEADS = 600
SPLITS = ("add", "remove", "replace")
VISUALS = ("box", "point", "scribble")
DTYPE = torch.bfloat16


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="black-forest-labs/FLUX.2-klein-base-4B")
    p.add_argument("--data-root", type=Path, default=Path("data/compbench-vpg"))
    p.add_argument("--splits", nargs="+", default=["add"], choices=SPLITS)
    p.add_argument("--visuals", nargs="+", default=["scribble"], choices=VISUALS)
    p.add_argument("--limit", type=int, default=0, help="Use only the first N samples in manifest order (0 = all).")
    p.add_argument("--output", type=Path, default=Path("outputs/flux2_heads/head_logits.json"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=30, help="Denoising steps.")
    p.add_argument("--guided-steps", type=int, default=5, help="Leading steps that get the coverage guidance.")
    p.add_argument("--max-side", type=int, default=512)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--strength", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--query-chunk", type=int, default=64, help="Query rows per attention-probability chunk.")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    if not 0 <= args.guided_steps <= args.steps:
        p.error("--guided-steps must be between 0 and --steps")
    return args


# ----------------------------------------------------------------------------- data


def load_samples(args: argparse.Namespace) -> list[dict]:
    samples: list[dict] = []
    for split in args.splits:
        for visual in args.visuals:
            manifest = args.data_root / split / f"manifest_{visual}.json"
            data = json.loads(manifest.read_text())
            items = data["samples"] if isinstance(data, dict) else data
            for item in items:
                item = dict(item)
                item["root"] = manifest.parent
                item["split"] = split
                samples.append(item)
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("No samples found. Check --data-root, --splits and --visuals.")
    return samples


def working_size(width: int, height: int, max_side: int) -> tuple[int, int]:
    scale = max_side / max(width, height)
    w = max(PATCH, int(round(width * scale / PATCH)) * PATCH)
    h = max(PATCH, int(round(height * scale / PATCH)) * PATCH)
    return w, h


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB -> float32 tensor [1, 3, H, W] in [0, 1]."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def object_token_indices(tokenizer, prompt: str, phrase: str) -> torch.Tensor | None:
    """Token indices of the object phrase inside the chat-templated prompt the text encoder sees."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    base = text.find(prompt)
    phrase_start = prompt.lower().find(phrase.lower())
    if base < 0 or phrase_start < 0:
        return None
    start = base + phrase_start
    end = start + len(phrase)
    enc = tokenizer(text, padding="max_length", truncation=True, max_length=TEXT_TOKENS, return_offsets_mapping=True)
    inds = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > s and s < end and e > start]
    return torch.tensor(inds, dtype=torch.long) if inds else None


def visual_density(
    visual: dict,
    src_w: int,
    src_h: int,
    width: int,
    height: int,
    gh: int,
    gw: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalized density [1, 1, gh, gw] (sums to 1) from a box, point or scribble prompt.

    Coordinates are in source-image pixels and are rescaled to the working resolution.
    """
    sx = width / src_w
    sy = height / src_h
    kind = visual["type"]

    if kind == "point":
        cx, cy = visual["x"] * sx, visual["y"] * sy
        ys = (torch.arange(gh, device=device, dtype=torch.float32) + 0.5) * PATCH
        xs = (torch.arange(gw, device=device, dtype=torch.float32) + 0.5) * PATCH
        d2 = (ys[:, None] - cy) ** 2 + (xs[None, :] - cx) ** 2
        dens = torch.exp(-d2 / (2 * POINT_SIGMA**2))
    elif kind == "box":
        x0, x1 = visual["x0"] * sx, visual["x1"] * sx
        y0, y1 = visual["y0"] * sy, visual["y1"] * sy
        ys = (torch.arange(gh, device=device, dtype=torch.float32) + 0.5) * PATCH
        xs = (torch.arange(gw, device=device, dtype=torch.float32) + 0.5) * PATCH
        inside = (ys[:, None] >= y0) & (ys[:, None] <= y1) & (xs[None, :] >= x0) & (xs[None, :] <= x1)
        if not inside.any():
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            d2 = (ys[:, None] - cy) ** 2 + (xs[None, :] - cx) ** 2
            inside = d2 == d2.min()
        dens = inside.float()
    elif kind == "scribble":
        scale = (sx + sy) / 2
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        pts = [(x * sx, y * sy) for x, y in visual["path"]]
        line_w = max(1, int(round(float(visual.get("width", 6.0)) * scale)))
        if len(pts) == 1:
            x, y = pts[0]
            r = line_w / 2
            draw.ellipse([x - r, y - r, x + r, y + r], fill=255)
        else:
            draw.line(pts, fill=255, width=line_w, joint="curve")
        m = torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0)[None, None].to(device)
        dens = F.adaptive_max_pool2d(m, (gh, gw))[0, 0]
    else:
        raise ValueError(f"unknown visual type: {kind}")

    total = dens.sum()
    if float(total) <= 0:
        raise ValueError("empty density")
    return (dens / total).reshape(1, 1, gh, gw)


def edit_mask_tensor(sample: dict, width: int, height: int, device: torch.device) -> torch.Tensor | None:
    if not sample.get("edit_mask"):
        return None
    mask = Image.open(sample["root"] / sample["edit_mask"]).convert("L").resize((width, height), Image.NEAREST)
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    return torch.from_numpy(arr > 0.5).float()[None, None].to(device)


# ----------------------------------------------------------------------------- math helpers


def pack_tokens_to_grid(tokens: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
    """[B, N, C] -> [B, C, gh, gw]."""
    b, _, c = tokens.shape
    return tokens.transpose(1, 2).reshape(b, c, gh, gw)


def unpatchify(z: torch.Tensor) -> torch.Tensor:
    """[B, C, h, w] -> [B, C//4, 2h, 2w] (inverse of the diffusers patchify)."""
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3)
    return z.reshape(b, c // 4, h * 2, w * 2)


def coverage_loss(agg: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between the visual density and the normalized aggregated object attention."""
    p = agg.clamp_min(EPS)
    p = p / p.sum((-2, -1), keepdim=True).clamp_min(EPS)
    return -(density * p.clamp_min(EPS).log()).sum()


def image_loss(pred: torch.Tensor, target: torch.Tensor, edit: torch.Tensor | None) -> torch.Tensor:
    diff = (pred - target).abs()
    if edit is None:
        return diff.mean()
    inside = edit.expand_as(diff)
    outside = 1.0 - inside
    l_in = (diff * inside).sum() / inside.sum().clamp_min(1.0)
    l_out = (diff * outside).sum() / outside.sum().clamp_min(1.0)
    return l_in + l_out


def sdpa_math():
    """Math SDPA kernel for the guidance pass; the fused kernels lack second-order grads."""
    return sdpa_kernel([SDPBackend.MATH])


def head_labels(transformer) -> list[str]:
    labels = []
    for i, _ in enumerate(transformer.transformer_blocks):
        labels += [f"D{i}:H{h:02d}" for h in range(transformer.config.num_attention_heads)]
    for i, _ in enumerate(transformer.single_transformer_blocks):
        labels += [f"S{i}:H{h:02d}" for h in range(transformer.config.num_attention_heads)]
    return labels


# ----------------------------------------------------------------------------- head capture


class HeadCapture:
    """Pre-hooks on every attention module that compute per-head object attention probabilities.

    For each head, prob = sum over object tokens of softmax(q k^T / sqrt(d)), computed on the
    image-query rows against all keys. Queries and keys are rebuilt from the attention inputs
    with RoPE applied, so the probabilities match the forward pass.
    """

    def __init__(self, transformer, chunk: int):
        self.transformer = transformer
        self.chunk = chunk
        self.enabled = False
        self.maps: dict[str, torch.Tensor] = {}
        self.text_len = TEXT_TOKENS
        self.n_img = 0
        self.inds: torch.Tensor | None = None
        self.labels = head_labels(transformer)
        for i, block in enumerate(transformer.transformer_blocks):
            block.attn.register_forward_pre_hook(self._make_hook(f"D{i}", double=True), with_kwargs=True)
        for i, block in enumerate(transformer.single_transformer_blocks):
            block.attn.register_forward_pre_hook(self._make_hook(f"S{i}", double=False), with_kwargs=True)

    def _make_hook(self, prefix: str, double: bool):
        def hook(module, args, kwargs):
            if self.enabled:
                probs = self._probs(module, kwargs, double)  # [B, H, n]
                self.maps[prefix] = probs
            return None

        return hook

    def _probs(self, attn, kwargs: dict, double: bool) -> torch.Tensor:
        h = kwargs["hidden_states"]
        rope = kwargs["image_rotary_emb"]
        hd = attn.head_dim
        if double:
            enc = kwargs["encoder_hidden_states"]
            q_img = attn.norm_q(attn.to_q(h).unflatten(-1, (-1, hd)))
            k_img = attn.norm_k(attn.to_k(h).unflatten(-1, (-1, hd)))
            q_txt = attn.norm_added_q(attn.add_q_proj(enc).unflatten(-1, (-1, hd)))
            k_txt = attn.norm_added_k(attn.add_k_proj(enc).unflatten(-1, (-1, hd)))
            q = torch.cat([q_txt, q_img], dim=1)
            k = torch.cat([k_txt, k_img], dim=1)
        else:
            inner = attn.inner_dim
            w = attn.to_qkv_mlp_proj.weight[: 3 * inner]
            b = attn.to_qkv_mlp_proj.bias
            b = b[: 3 * inner] if b is not None else None
            q, k, _ = F.linear(h, w, b).chunk(3, dim=-1)
            q = attn.norm_q(q.unflatten(-1, (-1, hd)))
            k = attn.norm_k(k.unflatten(-1, (-1, hd)))
        q = apply_rotary_emb(q, rope, sequence_dim=1)
        k = apply_rotary_emb(k, rope, sequence_dim=1)

        t0, n = self.text_len, self.n_img
        q_img = q[:, t0 : t0 + n].transpose(1, 2).float()  # [B, H, n, D]
        k_t = k.transpose(1, 2).float().transpose(-1, -2)  # [B, H, D, L]
        scale = 1.0 / math.sqrt(q.shape[-1])
        nums, dens = [], []
        for s in range(0, n, self.chunk):
            logits = torch.matmul(q_img[:, :, s : s + self.chunk], k_t) * scale  # [B, H, c, L]
            nums.append(logits.index_select(-1, self.inds).logsumexp(-1))
            dens.append(logits.logsumexp(-1))
        return torch.exp(torch.cat(nums, -1) - torch.cat(dens, -1))

    def aggregate(self, weights: torch.Tensor) -> torch.Tensor:
        """Weighted sum of all head probabilities -> [B, n]."""
        probs = torch.cat([self.maps[prefix] for prefix in self._block_prefixes()], dim=1)  # [B, 600, n]
        return (probs * weights.view(1, -1, 1)).sum(1)

    def _block_prefixes(self) -> list[str]:
        t = self.transformer
        return [f"D{i}" for i in range(len(t.transformer_blocks))] + [
            f"S{i}" for i in range(len(t.single_transformer_blocks))
        ]


# ----------------------------------------------------------------------------- trainer


class HeadTrainer:
    def __init__(self, pipe, args: argparse.Namespace):
        self.pipe = pipe
        self.args = args
        self.device = torch.device(args.device)
        self.tr = pipe.transformer
        self.vae = pipe.vae
        self.n_heads_total = len(head_labels(self.tr))
        if self.n_heads_total != EXPECTED_HEADS:
            raise SystemExit(f"Expected {EXPECTED_HEADS} heads, transformer config gives {self.n_heads_total}.")
        self.labels = head_labels(self.tr)

        self.tr.requires_grad_(False).eval()
        self.vae.requires_grad_(False).eval()
        self.tr.enable_gradient_checkpointing(gradient_checkpointing_func=self._checkpoint)

        self.logits = torch.nn.Parameter(torch.zeros(self.n_heads_total, device=self.device))
        self.optim = torch.optim.AdamW([self.logits], lr=args.lr, weight_decay=0.0)
        self.capture = HeadCapture(self.tr, args.query_chunk)
        self.prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.neg_emb: torch.Tensor | None = None
        self.neg_ids: torch.Tensor | None = None
        self.gen = torch.Generator(device=self.device).manual_seed(args.seed)
        self.in_channels = self.tr.config.in_channels // 4
        self.gh = self.gw = 0

    # --- text ---------------------------------------------------------------

    @torch.no_grad()
    def precompute_text(self, samples: list[dict]) -> list[dict]:
        """Embed every unique prompt and the empty negative prompt, then drop the text encoder."""
        kept: list[dict] = []
        tokenizer = self.pipe.tokenizer
        for s in samples:
            inds = object_token_indices(tokenizer, s["prompt"], s["object_phrase"])
            if inds is None:
                print(f"skip {s['id']}: object phrase not found in prompt")
                continue
            s["inds"] = inds
            kept.append(s)
            if s["prompt"] not in self.prompt_cache:
                emb, ids = self.pipe.encode_prompt(
                    prompt=s["prompt"], device=self.device, max_sequence_length=TEXT_TOKENS
                )
                self.prompt_cache[s["prompt"]] = (emb.to(DTYPE).cpu(), ids.cpu())
        self.neg_emb, self.neg_ids = self.pipe.encode_prompt(
            prompt="", device=self.device, max_sequence_length=TEXT_TOKENS
        )
        self.neg_emb = self.neg_emb.to(DTYPE)
        self.pipe.text_encoder = None  # the text encoder is no longer needed
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return kept

    # --- per-sample state ---------------------------------------------------

    @torch.no_grad()
    def prepare(self, s: dict) -> dict:
        inp = Image.open(s["root"] / s["input_image"]).convert("RGB")
        tgt = Image.open(s["root"] / s["target_image"]).convert("RGB")
        src_w, src_h = inp.size
        width, height = working_size(src_w, src_h, self.args.max_side)
        gh, gw = height // PATCH, width // PATCH

        density = visual_density(s["visual"], src_w, src_h, width, height, gh, gw, self.device)
        edit = edit_mask_tensor(s, width, height, self.device)
        if edit is None:
            edit = F.interpolate((density > 0).float(), size=(height, width), mode="nearest")

        ref_img = pil_to_tensor(inp.resize((width, height), Image.BICUBIC)).to(self.device) * 2 - 1
        target = pil_to_tensor(tgt.resize((width, height), Image.BICUBIC)).to(self.device)

        ref_tokens, ref_ids = self.pipe.prepare_image_latents(
            images=[ref_img], batch_size=1, generator=self.gen, device=self.device, dtype=DTYPE
        )
        latents, latent_ids = self.pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=self.in_channels,
            height=height,
            width=width,
            dtype=DTYPE,
            device=self.device,
            generator=self.gen,
        )
        n_img = latents.shape[1]
        if n_img != gh * gw:
            raise RuntimeError(f"token count {n_img} != grid {gh}x{gw}")

        mu = compute_empirical_mu(image_seq_len=n_img, num_steps=self.args.steps)
        scheduler = self.pipe.scheduler
        sigmas_np = None if getattr(scheduler.config, "use_flow_sigmas", False) else np.linspace(
            1.0, 1.0 / self.args.steps, self.args.steps
        )
        timesteps, _ = retrieve_timesteps(
            scheduler, self.args.steps, self.device, sigmas=sigmas_np, mu=mu
        )
        sigmas = scheduler.sigmas.to(self.device, torch.float32)
        if len(sigmas) != self.args.steps + 1:
            raise RuntimeError(f"expected {self.args.steps + 1} sigmas, got {len(sigmas)}")

        emb, text_ids = self.prompt_cache[s["prompt"]]
        return {
            "gh": gh,
            "gw": gw,
            "n_img": n_img,
            "latents": latents.float(),
            "img_ids": torch.cat([latent_ids, ref_ids], dim=1),
            "ref": ref_tokens.to(DTYPE),
            "cond_emb": emb.to(self.device),
            "cond_ids": text_ids.to(self.device),
            "density": density,
            "edit": edit,
            "target": target,
            "inds": s["inds"].to(self.device),
            "timesteps": timesteps.to(self.device, torch.float32),
            "sigmas": sigmas,
        }

    def set_sample(self, st: dict) -> None:
        self.st = st
        self.gh, self.gw = st["gh"], st["gw"]
        self.capture.text_len = TEXT_TOKENS
        self.capture.n_img = st["n_img"]
        self.capture.inds = st["inds"]

    # --- model calls --------------------------------------------------------

    def _checkpoint(self, fn, *args):
        """Non-reentrant block checkpoint. Recomputation reuses the forward's capture flag and SDPA kernel."""
        on = self.capture.enabled

        def run(*inputs):
            prev = self.capture.enabled
            self.capture.enabled = on
            try:
                with sdpa_math() if on else nullcontext():
                    return fn(*inputs)
            finally:
                self.capture.enabled = prev

        return checkpoint(run, *args, use_reentrant=False)

    def _model(self, x: torch.Tensor, k: int, emb: torch.Tensor, txt_ids: torch.Tensor) -> torch.Tensor:
        """Batch size is taken from `emb`; `x` is broadcast to it."""
        st = self.st
        b = emb.shape[0]
        t = st["timesteps"][k].expand(b).to(DTYPE) / 1000
        hidden = torch.cat([x.to(DTYPE).expand(b, -1, -1), st["ref"].expand(b, -1, -1)], dim=1)
        out = self.tr(
            hidden_states=hidden,
            timestep=t,
            guidance=None,
            encoder_hidden_states=emb,
            txt_ids=txt_ids,
            img_ids=st["img_ids"].expand(b, -1, -1),
            return_dict=False,
        )[0]
        return out[:, : st["n_img"]].float()

    def _denoise(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """One Euler step with classifier-free guidance. Returns the next latent (float32)."""
        st = self.st
        emb = torch.cat([st["cond_emb"], self.neg_emb])
        ids = torch.cat([st["cond_ids"], self.neg_ids])
        cond, neg = self._model(x, k, emb, ids).chunk(2)
        v = neg + self.args.cfg * (cond - neg)
        dt = st["sigmas"][k + 1] - st["sigmas"][k]
        return x + dt * v

    def _guide(self, x: torch.Tensor, k: int, create_graph: bool) -> torch.Tensor:
        """Coverage-gradient nudge for step k.

        With create_graph=False, x is detached and the result is a plain tensor.
        With create_graph=True, the result stays differentiable with respect to x and the logits.
        """
        x_leaf = x if create_graph else x.detach().requires_grad_(True)
        with torch.enable_grad(), sdpa_math():
            weights = torch.softmax(self.logits / TEMPERATURE, dim=0)
            self.capture.enabled = True
            self.capture.maps = {}
            try:
                self._model(x_leaf, k, self.st["cond_emb"], self.st["cond_ids"])
            finally:
                self.capture.enabled = False
            agg = self.capture.aggregate(weights).reshape(1, 1, self.gh, self.gw)
            self.capture.maps = {}
            loss = coverage_loss(agg, self.st["density"])
            (g,) = torch.autograd.grad(loss, x_leaf, create_graph=create_graph)
            scale = g.square().mean().clamp_min(1e-16).sqrt()
            x_new = x_leaf - self.args.strength * g / scale
        return x_new if create_graph else x_new.detach()

    def _step(self, x: torch.Tensor, k: int, guided: bool, create_graph: bool) -> torch.Tensor:
        if guided:
            x = self._guide(x, k, create_graph=create_graph)
        return self._denoise(x, k)

    def _decode(self, lat: torch.Tensor) -> torch.Tensor:
        """Final latent [1, N, C] (float32) -> RGB image [1, 3, H, W] in [0, 1]."""
        z = pack_tokens_to_grid(lat.to(DTYPE), self.gh, self.gw)
        bn = self.vae.bn
        mean = bn.running_mean.view(1, -1, 1, 1).to(z)
        std = torch.sqrt(bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(z)
        z = unpatchify(z * std + mean)
        img = self.vae.decode(z, return_dict=False)[0]
        return ((img.float() / 2) + 0.5).clamp(0, 1)

    # --- training -----------------------------------------------------------

    def forward_trajectory(self, st: dict) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Denoise without a graph. Returns the input of every step and the final latent."""
        traj: list[torch.Tensor] = []
        x = st["latents"]
        with torch.no_grad():
            for k in range(self.args.steps):
                traj.append(x)
                x = self._step(x, k, guided=k < self.args.guided_steps, create_graph=False)
        return traj, x.detach()

    def image_grad(self, final: torch.Tensor, st: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Image loss (tensor) and its gradient with respect to the final latent."""
        lat = final.requires_grad_(True)
        with torch.enable_grad():
            loss = image_loss(self._decode(lat), st["target"], st["edit"])
            (g,) = torch.autograd.grad(loss, lat)
        return loss.detach(), g.detach()

    def backward_trajectory(self, traj: list[torch.Tensor], g: torch.Tensor) -> None:
        """Accumulate logit gradients, recomputing one step at a time from its stored input."""
        for k in reversed(range(self.args.steps)):
            x_in = traj[k].detach().requires_grad_(True)
            with torch.enable_grad():
                x_next = self._step(x_in, k, guided=k < self.args.guided_steps, create_graph=True)
                torch.autograd.backward(x_next, g)
            g = x_in.grad
            traj[k] = None
            del x_next, x_in  # release this step's graph before the next one is built

    def train_sample(self, st: dict) -> tuple[float, float]:
        self.set_sample(st)
        self.optim.zero_grad(set_to_none=True)
        traj, final = self.forward_trajectory(st)
        loss_img, g = self.image_grad(final, st)
        self.backward_trajectory(traj, g)

        grad_norm = torch.nn.utils.clip_grad_norm_([self.logits], self.args.grad_clip)
        if torch.isfinite(grad_norm):
            self.optim.step()
        else:
            print("  non-finite gradient, step skipped")
        return float(loss_img), float(grad_norm)

    def save(self, path: Path, epoch: int, history: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        logits = self.logits.detach().float().cpu()
        payload = {
            "labels": self.labels,
            "logits": logits.tolist(),
            "weights": torch.softmax(logits / TEMPERATURE, 0).tolist(),
            "temperature": TEMPERATURE,
            "epoch": epoch,
            "args": {k: str(v) for k, v in vars(self.args).items()},
            "history": history,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(path)

    def train(self, samples: list[dict]) -> None:
        history: list[dict] = []
        total = len(samples)
        for epoch in range(self.args.epochs):
            order = list(range(total))
            random.Random(self.args.seed + epoch).shuffle(order)
            for n, i in enumerate(order):
                s = samples[i]
                t0 = time.time()
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                try:
                    st = self.prepare(s)
                    loss, grad_norm = self.train_sample(st)
                except ValueError as exc:
                    print(f"skip {s['id']}: {exc}")
                    continue
                peak = torch.cuda.max_memory_allocated(self.device) / 2**30 if self.device.type == "cuda" else 0.0
                dt = time.time() - t0
                record = {
                    "epoch": epoch,
                    "id": s["id"],
                    "split": s.get("split"),
                    "visual": s["visual"]["type"],
                    "loss": loss,
                    "grad_norm": grad_norm,
                    "peak_gib": peak,
                    "seconds": dt,
                }
                history.append(record)
                print(
                    f"[epoch {epoch + 1}/{self.args.epochs}] {n + 1}/{total} {s['id']} "
                    f"loss={loss:.4f} gnorm={grad_norm:.3e} peak={peak:.2f}GiB time={dt:.1f}s",
                    flush=True,
                )
                self.save(self.args.output, epoch, history)
                del st
        self.save(self.args.output, self.args.epochs - 1, history)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    samples = load_samples(args)
    print(f"loaded {len(samples)} samples")

    pipe = Flux2KleinPipeline.from_pretrained(args.model, torch_dtype=DTYPE)
    pipe.to(device)

    trainer = HeadTrainer(pipe, args)
    samples = trainer.precompute_text(samples)
    if not samples:
        raise SystemExit("No usable samples after text preprocessing.")
    print(f"{len(samples)} samples, {trainer.n_heads_total} heads, output {args.output}")
    trainer.train(samples)


if __name__ == "__main__":
    main()
