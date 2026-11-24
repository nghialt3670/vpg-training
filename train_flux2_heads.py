#!/usr/bin/env python3
"""Learn 600 shared FLUX.2 Klein attention-head logits using local image supervision.

At each guided step, compute the original coverage-gradient latent update. For
selected guided steps, estimate the clean latent as z_guided - sigma * velocity,
decode it, and optimize the masked image loss. Earlier states are detached:
this is a local surrogate, NOT the gradient of the full final-image objective.
Separate VJPs keep the image and second-order guidance graphs out of memory at
the same time. Optional final-image evaluation uses the actual Euler trajectory.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import time
import zlib
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
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=20, help="Denoising steps.")
    p.add_argument("--guided-steps", type=int, default=5, help="Leading steps that get the coverage guidance.")
    p.add_argument("--max-side", type=int, default=512)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--strength", type=float, default=0.03)
    p.add_argument(
        "--image-loss",
        choices=("l1", "lpips", "mix"),
        default="mix",
        help="l1: masked L1 inside and outside. lpips: masked LPIPS inside. mix: LPIPS inside plus L1 outside.",
    )
    p.add_argument("--bg-weight", type=float, default=1.0, help="Weight of the outside L1 term in mix mode.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--query-chunk", type=int, default=64, help="Query rows per attention-probability chunk.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--local-start", type=int, default=None,
                   help="First supervised step, zero-indexed; default last 3 guided steps.")
    p.add_argument("--grad-accum", type=int, default=5,
                   help="Average sample gradients before an AdamW update.")
    p.add_argument("--guidance-rms-floor", type=float, default=1e-6,
                   help="Lower bound on the RMS denominator for latent guidance.")
    p.add_argument("--eval-every", type=int, default=1,
                   help="Measure final-image loss every N epochs, 0 disables.")
    p.add_argument("--diagnose", action="store_true",
                   help="First sample: repeat fixed-state loss and check a directional derivative.")
    p.add_argument("--fd-eps", type=float, default=0.01,
                   help="RMS perturbation of logits for central finite differences.")
    args = p.parse_args()
    if not 1 <= args.guided_steps <= args.steps:
        p.error("--guided-steps must be between 1 and --steps")
    if args.local_start is None:
        args.local_start = max(0, args.guided_steps - 3)
    if not 0 <= args.local_start < args.guided_steps:
        p.error("--local-start must be in [0, guided-steps)")
    if (args.grad_accum < 1 or args.guidance_rms_floor <= 0 or args.fd_eps <= 0
            or args.eval_every < 0 or args.epochs < 1 or args.lr <= 0 or args.grad_clip <= 0):
        p.error("Invalid optimization or diagnostic argument")
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
# crc32 instead of hash(): hash() is salted per process.
def sample_seed(base: int, sample_id: str) -> int:
    return (base * 1_000_003 + zlib.crc32(sample_id.encode())) % 2**31
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
def build_lpips(device: torch.device) -> torch.nn.Module:
    import lpips  # imported here so `--image-loss l1` does not need the package
    net = lpips.LPIPS(net="alex", spatial=True, verbose=False)
    return net.to(device).eval().requires_grad_(False)
def masked_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0)
def image_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    edit: torch.Tensor | None,
    mode: str,
    bg_weight: float,
    lpips_net: torch.nn.Module | None = None,
) -> torch.Tensor:
    mask = edit if edit is not None else torch.ones_like(pred[:, :1])
    inside = mask.expand_as(pred)
    outside = 1.0 - inside
    if mode == "l1":
        diff = (pred - target).abs()
        return masked_mean(diff, inside) + masked_mean(diff, outside)
    d = lpips_net(pred * 2 - 1, target * 2 - 1)
    if d.shape[-2:] != mask.shape[-2:]:
        d = F.interpolate(d, size=mask.shape[-2:], mode="bilinear", align_corners=False)
    loss = masked_mean(d, mask)
    if mode == "mix":
        loss = loss + bg_weight * masked_mean((pred - target).abs(), outside)
    return loss
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
        self.lpips_net: torch.nn.Module | None = None
        self.capture = HeadCapture(self.tr, args.query_chunk)
        self.prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.neg_emb: torch.Tensor | None = None
        self.neg_ids: torch.Tensor | None = None
        self.cov_log: list[torch.Tensor] = []
        self.rms_log: list[torch.Tensor] = []
        self.did_diagnose = False
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
        gen = torch.Generator(device=self.device).manual_seed(sample_seed(self.args.seed, s["id"]))
        ref_tokens, ref_ids = self.pipe.prepare_image_latents(
            images=[ref_img], batch_size=1, generator=gen, device=self.device, dtype=DTYPE
        )
        latents, latent_ids = self.pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=self.in_channels,
            height=height,
            width=width,
            dtype=DTYPE,
            device=self.device,
            generator=gen,
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
    def _velocity(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Classifier-free guided velocity in FP32."""
        st = self.st
        emb = torch.cat([st["cond_emb"], self.neg_emb])
        ids = torch.cat([st["cond_ids"], self.neg_ids])
        cond, neg = self._model(x, k, emb, ids).chunk(2)
        return neg + self.args.cfg * (cond - neg)
    def _denoise(self, x: torch.Tensor, k: int) -> torch.Tensor:
        dt = self.st["sigmas"][k + 1] - self.st["sigmas"][k]
        return x + dt * self._velocity(x, k)
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
            if not create_graph:
                self.cov_log.append(loss.detach())
            (g,) = torch.autograd.grad(loss, x_leaf, create_graph=create_graph)
            mean_sq = g.float().square().mean()
            scale = mean_sq.clamp_min(self.args.guidance_rms_floor ** 2).sqrt()
            if not create_graph:
                self.rms_log.append(mean_sq.detach().sqrt())
            x_new = x_leaf - self.args.strength * g / scale
            # Checkpoint recomputation may repopulate the capture dictionary.
            self.capture.maps.clear()
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
    def _loss(self, latent: torch.Tensor) -> torch.Tensor:
        if self.args.image_loss != "l1" and self.lpips_net is None:
            self.lpips_net = build_lpips(self.device)
        return image_loss(self._decode(latent), self.st["target"], self.st["edit"],
                          self.args.image_loss, self.args.bg_weight, self.lpips_net)

    def _local_image_grad(self, guided: torch.Tensor, k: int):
        """Exact local VJP, split into VAE and velocity passes to reduce peak memory."""
        sigma = self.st["sigmas"][k]
        with torch.enable_grad():
            y = guided.detach().requires_grad_(True)
            v = self._velocity(y, k)
            clean = y - sigma * v
            clean_leaf = clean.detach().requires_grad_(True)
            loss = self._loss(clean_leaf)
            (g_clean,) = torch.autograd.grad(loss, clean_leaf)
            result = loss.detach()
            del loss, clean_leaf
            (g_y,) = torch.autograd.grad(clean, y, grad_outputs=g_clean)
        return result, g_y.detach(), v.detach()

    def _local_head_grad(self, x: torch.Tensor, k: int, g_y: torch.Tensor):
        """Mixed derivative of the original guidance update wrt head logits."""
        try:
            with torch.enable_grad():
                x_leaf = x.detach().requires_grad_(True)
                guided = self._guide(x_leaf, k, create_graph=True)
                (g_logits,) = torch.autograd.grad(guided, self.logits, grad_outputs=g_y)
            return g_logits.detach()
        finally:
            self.capture.maps.clear()

    @torch.no_grad()
    def _fixed_state_loss(self, states: list[tuple[int, torch.Tensor]]) -> float:
        # Only the head logits change. Hold earlier latent states fixed, matching the
        # stop-gradient local objective used by train_sample.
        vals = []
        self.cov_log.clear()
        self.rms_log.clear()
        try:
            for k, x in states:
                y = self._guide(x, k, create_graph=False)
                clean = y - self.st["sigmas"][k] * self._velocity(y, k)
                vals.append(float(self._loss(clean)))
            return sum(vals) / len(vals)
        finally:
            self.cov_log.clear()
            self.rms_log.clear()

    @torch.no_grad()
    def _diagnose(self, states, grad):
        original = self.logits.detach().clone()
        try:
            base = self._fixed_state_loss(states)
            repeat = self._fixed_state_loss(states)
            print(f"  diagnostic: fixed loss={base:.7f}, repeat={repeat:.7f}, "
                  f"abs_diff={abs(base-repeat):.3e}", flush=True)
            if not torch.isfinite(grad).all() or float(grad.norm()) == 0:
                print("  diagnostic: invalid or zero gradient; skipping finite difference", flush=True)
                return
            direction = grad / grad.square().mean().sqrt()
            predicted = float((grad * direction).sum())
            for eps in (self.args.fd_eps, self.args.fd_eps * 3):
                self.logits.copy_(original + eps * direction)
                plus = self._fixed_state_loss(states)
                self.logits.copy_(original - eps * direction)
                minus = self._fixed_state_loss(states)
                fd = (plus - minus) / (2 * eps)
                print(f"  diagnostic: eps={eps:g} auto={predicted:.6g} fd={fd:.6g} "
                      f"loss(-eps)={minus:.7f} loss(+eps)={plus:.7f}", flush=True)
        finally:
            self.logits.copy_(original)

    def train_sample(self, st: dict, evaluate_final: bool):
        self.set_sample(st)
        self.cov_log.clear()
        self.rms_log.clear()
        x = st["latents"].detach()
        gradients = torch.zeros_like(self.logits)
        losses, states = [], []
        n_supervised = self.args.guided_steps - self.args.local_start
        diagnose = self.args.diagnose and not self.did_diagnose
        for k in range(self.args.guided_steps):
            # The forward update is the same as the original inference guidance.
            with torch.no_grad():
                y = self._guide(x, k, create_graph=False)
            if k >= self.args.local_start:
                if diagnose:
                    states.append((k, x.detach()))
                loss, g_y, v = self._local_image_grad(y, k)
                g_logits = self._local_head_grad(x, k, g_y)
                gradients.add_(g_logits / n_supervised)
                losses.append(float(loss))
                del g_y, g_logits, loss
            else:
                with torch.no_grad():
                    v = self._velocity(y, k)
            with torch.no_grad():
                dt = st["sigmas"][k + 1] - st["sigmas"][k]
                x = (y + dt * v).detach()
            del y, v
        cov = float(torch.stack(self.cov_log).mean())
        rms = float(torch.stack(self.rms_log).mean())
        final_loss = None
        if evaluate_final:
            with torch.no_grad():
                for k in range(self.args.guided_steps, self.args.steps):
                    x = self._denoise(x, k)
                final_loss = float(self._loss(x))
        if diagnose:
            self._diagnose(states, gradients)
            self.did_diagnose = True
        return sum(losses) / len(losses), gradients, cov, rms, final_loss, losses

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
            batch_grad = torch.zeros_like(self.logits)
            batch_count = 0
            epoch_records = []
            evaluate = self.args.eval_every > 0 and epoch % self.args.eval_every == 0
            def update():
                nonlocal batch_count
                if not batch_count:
                    return
                self.optim.zero_grad(set_to_none=True)
                self.logits.grad = batch_grad / batch_count
                norm = torch.nn.utils.clip_grad_norm_([self.logits], self.args.grad_clip)
                if torch.isfinite(norm):
                    self.optim.step()
                else:
                    print("  non-finite batch gradient, update skipped", flush=True)
                self.optim.zero_grad(set_to_none=True)
                batch_grad.zero_()
                batch_count = 0
            for n, i in enumerate(order):
                s = samples[i]
                t0 = time.time()
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(self.device)
                try:
                    st = self.prepare(s)
                    loss, grad, cov, rms, final, per_step = self.train_sample(st, evaluate)
                except ValueError as exc:
                    print(f"skip {s['id']}: {exc}", flush=True)
                    continue
                norm = float(grad.norm())
                valid = math.isfinite(loss) and bool(torch.isfinite(grad).all())
                if valid:
                    batch_grad.add_(grad)
                    batch_count += 1
                else:
                    print("  non-finite sample loss/gradient, skipped", flush=True)
                if batch_count >= self.args.grad_accum or n == total - 1:
                    update()
                peak = torch.cuda.max_memory_allocated(self.device) / 2**30 if self.device.type == "cuda" else 0.0
                dt = time.time() - t0
                w = torch.softmax(self.logits.detach() / TEMPERATURE, 0)
                spread = float(w.max() / w.min())
                record = dict(epoch=epoch, id=s["id"], split=s.get("split"),
                              visual=s["visual"]["type"], loss=loss, local_loss=loss,
                              local_losses=per_step, final_loss=final, coverage=cov,
                              guidance_rms=rms, weight_spread=spread, grad_norm=norm,
                              peak_gib=peak, seconds=dt, valid=valid)
                history.append(record)
                if valid:
                    epoch_records.append(record)
                final_str = f"{final:.4f}" if final is not None else "n/a"
                print(f"[epoch {epoch+1}/{self.args.epochs}] {n+1}/{total} {s['id']} "
                      f"local={loss:.4f} final={final_str} cov={cov:.4f} "
                      f"rms={rms:.3e} wspread={spread:.4f} gnorm={norm:.3e} "
                      f"peak={peak:.2f}GiB time={dt:.1f}s", flush=True)
                self.save(self.args.output, epoch, history)
                del st, grad
            update()  # also flush when the final sample was skipped
            if epoch_records:
                avg = sum(r["local_loss"] for r in epoch_records) / len(epoch_records)
                finals = [r["final_loss"] for r in epoch_records if r["final_loss"] is not None]
                final_avg = f"{sum(finals)/len(finals):.4f}" if finals else "n/a"
                print(f"[epoch {epoch+1} mean] local={avg:.5f} final={final_avg} "
                      f"valid={len(epoch_records)}/{total}", flush=True)
            self.save(self.args.output, epoch, history)

def main() -> None:
    global DTYPE
    args = parse_args()
    DTYPE = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    if args.dtype == "fp32":
        torch.set_float32_matmul_precision("highest")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
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
    print(f"Local supervision: steps {args.local_start}..{args.guided_steps-1} (zero-indexed), dtype={args.dtype}, accumulation={args.grad_accum}", flush=True)
    trainer.train(samples)
if __name__ == "__main__":
    main()
