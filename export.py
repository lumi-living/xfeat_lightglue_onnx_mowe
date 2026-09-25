#!/usr/bin/env python3
"""Reproducible XFeat + LighterGlue ONNX export for the Mow-e VIO front-end (T-0109).

    export.py --model xfeat       --hw 640x384 --k 1024 --out ../../../out/onnx
    export.py --model lighterglue --k 512              --out ../../../out/onnx

Writes ``<out>/xfeat_<W>x<H>_k<K>.onnx`` / ``<out>/lighterglue_k<K>.onnx`` and
updates ``manifest.json`` (filename -> sha256, resolution, K, opset, exporter
git hash + sha256, date) next to this script.

Contracts (mowe-nav-kb 02 §XFeat output, 03 §static shapes + padding mask):

* **XFeat extractor** — input ``images[2,1,H,W]`` float32, raw 0..255 mono, NCHW.
  Batch is fixed at 2: the stereo pair goes through one enqueue (KB 03 §batching).
  XFeat applies no /255 or mean/std (InstanceNorm inside the net handles scale)
  and its first conv is ``Conv2d(1, 24, ...)`` after ``x.mean(dim=1)``, so a
  single channel is loss-free for the OV9281 (mono, 1280x800; ADR-0005).
  H and W must be multiples of 32 so XFeat's internal /32 resize is a no-op and
  the keypoint scale correction is unity.  Outputs, all static:
  ``keypoints[2,K,2]`` int32 pixel (x, y), ``descriptors[2,K,64]`` float32
  L2-normalised, ``scores[2,K]`` float32 with **-1 in padding slots** (fewer
  than K local maxima above the detection threshold).  Keypoints are int32, not
  float, on purpose: the flat top-K index runs to H*W ≈ 1.02 M and an fp16
  TensorRT engine overflows a float decode to ±inf (observed on device); an
  integer Gather from a baked coordinate LUT is exact in any precision
  (ADR-0040).  No ``NonZero`` anywhere: NMS is max-pool suppression followed by
  one bounded ``TopK`` (K ≤ 3840, TensorRT's hard TopK limit).  Ranking follows
  upstream ``detectAndCompute``: reliability-weighted score
  (keypoint heatmap × bilinearly-upsampled reliability map), threshold 0.05.

* **LighterGlue split matcher** — inputs ``kpts0[1,K,2]``, ``kpts1[1,K,2]``
  float32 keypoints normalised as kornia's ``normalize_keypoints`` does
  (``(xy - size/2) / (max(size)/2)``, i.e. roughly [-1, 1]);
  ``desc0/desc1[1,K,64]`` float32; ``scores0/scores1[1,K]`` float32 as emitted
  by the extractor.  Validity is ``score > 0``: invalid (padding) points are
  masked out of every self-/cross-attention key set and out of the assignment
  matrix, so padding cannot corrupt real matches (KB 03 §padding).  Outputs
  ``matches0[1,K]`` int32 (index into set 1, or -1) and ``mscores0[1,K]``
  float32 — fixed shape, mutual-nearest + threshold applied inside the graph.
  Full 6-layer weights (``xfeat-lighterglue.pt`` has layers 0..5; the old L3
  export silently ran a truncated net).  Early exit / point pruning do not
  survive export; the graph runs full depth.

Opset 18 throughout (LighterGlue needs ≥ 18; TensorRT 10.3 reads it natively).
"""
import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from modules.xfeat import XFeat  # noqa: E402
from modules.interpolator import InterpolateSparse2d  # noqa: E402
from modules.lighterglue import LighterGlue  # noqa: E402
from modules import lightglue as _lg  # noqa: E402

OPSET = 18
TRT_TOPK_MAX = 3840
STEREO_BATCH = 2  # KB 03 §batching: left+right in one enqueue


class XFeatExtractorStatic(nn.Module):
    """XFeat detect+describe with static ``[B,K,*]`` outputs (see module doc)."""

    def __init__(self, xfeat: XFeat):
        super().__init__()
        self.xfeat = xfeat
        self._nearest = InterpolateSparse2d("nearest")
        self._bilinear = InterpolateSparse2d("bilinear")

    def forward(self, x):
        xf = self.xfeat
        # H,W % 32 == 0 is enforced by the CLI, so preprocess_tensor is a no-op.
        x, _, _ = xf.preprocess_tensor(x)
        B, _, H, W = x.shape

        M1, K1, H1 = xf.net(x)
        M1 = F.normalize(M1, dim=1)
        K1h = xf.get_kpts_heatmap(K1)  # [B,1,H,W]

        # Static NMS: max-pool suppression + detection threshold (upstream NMS),
        # ranked by the same reliability-weighted score upstream sorts on.
        local_max = F.max_pool2d(K1h, kernel_size=5, stride=1, padding=2)
        is_max = (K1h == local_max) & (K1h > xf.detection_threshold)
        H1_up = F.interpolate(H1, (H, W), mode="bilinear", align_corners=False)
        ranked = torch.where(is_max, K1h * H1_up, torch.zeros_like(K1h))
        rank_vals, idx = torch.topk(ranked.reshape(B, -1), xf.top_k, dim=-1)

        # Flat index -> (x, y) by Gather from a constant LUT (no div/mod: ADR-0040).
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        xs = xx.reshape(-1).to(torch.int32)[idx]
        ys = yy.reshape(-1).to(torch.int32)[idx]
        mkpts = torch.stack([xs, ys], dim=-1).to(K1h.dtype)  # [B,K,2]

        scores = (self._nearest(K1h, mkpts, H, W) *
                  self._bilinear(H1, mkpts, H, W)).squeeze(-1)
        scores = torch.where(rank_vals > 0, scores, torch.full_like(scores, -1.0))
        feats = F.normalize(xf.interpolator(M1, mkpts, H=H, W=W), dim=-1)
        keypoints = torch.stack([xs, ys], dim=-1)  # int32 px
        return keypoints, feats, scores


class MaskedAttention(nn.Module):
    """Drop-in for lightglue.Attention that masks padded keys (KB 03 §padding).

    LightGlue reuses one attention module for both directions of a block
    (self: set 0 then set 1; cross: keys of set 1 then set 0), so the masks are
    consumed in call order; ``reset`` is called once per forward.
    """

    def __init__(self):
        super().__init__()
        self.masks, self.i = [None, None], 0

    def reset(self, first, second):
        self.masks, self.i = [first, second], 0

    def forward(self, q, k, v):
        mask = self.masks[self.i % 2]
        self.i += 1
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class LighterGlueSplit(nn.Module):
    """LighterGlue over two padded sets with a score-derived validity mask."""

    def __init__(self, weights: str):
        super().__init__()
        self.net = LighterGlue(n_layers=6, weights=weights).net.eval()
        for layer in self.net.transformers:
            layer.self_attn.inner_attn = MaskedAttention()
            layer.cross_attn.inner_attn = MaskedAttention()
        self.threshold = self.net.conf.filter_threshold

    def forward(self, kpts0, kpts1, desc0, desc1, scores0, scores1):
        net = self.net
        valid0, valid1 = scores0 > 0, scores1 > 0
        m0, m1 = valid0[:, None, None, :], valid1[:, None, None, :]  # [B,1,1,N] key masks
        d0, d1 = net.input_proj(desc0), net.input_proj(desc1)
        e0, e1 = net.posenc(kpts0), net.posenc(kpts1)
        for layer in net.transformers:
            layer.self_attn.inner_attn.reset(m0, m1)
            layer.cross_attn.inner_attn.reset(m1, m0)
            d0, d1 = layer(d0, d1, e0, e1)
        scores = net.log_assignment[-1](d0, d1)  # [B,M,N] log assignment
        invalid = ~(valid0[:, :, None] & valid1[:, None, :])
        scores = scores.masked_fill(invalid, -1e9)
        # Mutual nearest neighbour + threshold, all static shape (no NonZero).
        max0_val, m0i = scores.max(2)          # [B,M]
        _, m1i = scores.max(1)                 # [B,N]
        idx = torch.arange(m0i.shape[1])[None]
        mutual = (idx == m1i.gather(1, m0i)) & valid0
        mscores0 = torch.where(mutual, max0_val.exp(), torch.zeros_like(max0_val))
        ok = mscores0 > self.threshold
        matches0 = torch.where(ok, m0i, torch.full_like(m0i, -1)).to(torch.int32)
        return matches0, mscores0


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_hash():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "export.py", "modules"],
                               cwd=HERE).returncode != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def _export(model, args_tuple, path, input_names, output_names, opset):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(model, args_tuple, path, input_names=input_names,
                          output_names=output_names, opset_version=opset,
                          do_constant_folding=True, dynamo=False)
    import onnx
    from onnxsim import simplify  # folds the static-shape residue (If/Shape/Range) — deterministic
    m, ok = simplify(onnx.load(path))
    if not ok:
        raise SystemExit(f"{path}: onnxsim could not validate the simplified graph")
    onnx.checker.check_model(m)
    onnx.save(m, path)
    bad = sorted({n.op_type for n in m.graph.node if n.op_type == "NonZero"})
    if bad:
        raise SystemExit(f"{path}: graph contains {bad} — data-dependent shape, refusing")


def _update_manifest(path, entry):
    mpath = os.path.join(HERE, "manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    manifest[os.path.basename(path)] = dict(
        sha256=_sha256(path), size_bytes=os.path.getsize(path), opset=OPSET,
        exporter_git=_git_hash(), exporter_sha256=_sha256(os.path.abspath(__file__)),
        torch=torch.__version__, date=_dt.date.today().isoformat(), **entry)
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[manifest] {os.path.basename(path)} sha256={manifest[os.path.basename(path)]['sha256'][:12]}…")


def build_xfeat(k, weights_dir=os.path.join(HERE, "weights")):
    xf = XFeat(weights=os.path.join(weights_dir, "xfeat.pt"), top_k=k).eval()
    return XFeatExtractorStatic(xf).eval()


def build_lighterglue(weights_dir=os.path.join(HERE, "weights")):
    return LighterGlueSplit(os.path.join(weights_dir, "xfeat-lighterglue.pt")).eval()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("xfeat", "lighterglue"), required=True)
    p.add_argument("--hw", default="640x384", help="WIDTHxHEIGHT, multiples of 32 (xfeat only)")
    p.add_argument("--k", type=int, required=True, help="fixed top-K (≤ 3840)")
    p.add_argument("--out", default="onnx", help="output directory for .onnx")
    p.add_argument("--weights-dir", default=os.path.join(HERE, "weights"))
    a = p.parse_args(argv)
    if a.k > TRT_TOPK_MAX:
        p.error(f"--k must be <= {TRT_TOPK_MAX} (TensorRT TopK limit)")
    torch.manual_seed(0)

    if a.model == "xfeat":
        w, h = (int(v) for v in a.hw.lower().split("x"))
        if w % 32 or h % 32:
            p.error("--hw must be multiples of 32 (XFeat /32 resize must be a no-op)")
        path = os.path.join(a.out, f"xfeat_{w}x{h}_k{a.k}.onnx")
        model = build_xfeat(a.k, a.weights_dir)
        dummy = torch.rand(STEREO_BATCH, 1, h, w) * 255
        _export(model, (dummy,), path, ["images"], ["keypoints", "descriptors", "scores"], OPSET)
        _update_manifest(path, dict(model="xfeat", width=w, height=h, k=a.k, batch=STEREO_BATCH,
                                    input="images[2,1,H,W] float32 raw 0..255 mono",
                                    outputs="keypoints[2,K,2] int32 px, descriptors[2,K,64], scores[2,K] (-1 = padding)",
                                    weights_sha256=_sha256(os.path.join(a.weights_dir, "xfeat.pt"))))
    else:
        path = os.path.join(a.out, f"lighterglue_k{a.k}.onnx")
        model = build_lighterglue(a.weights_dir)
        kp = torch.rand(1, a.k, 2) * 2 - 1
        de = F.normalize(torch.randn(1, a.k, 64), dim=-1)
        sc = torch.rand(1, a.k)
        sc[:, -a.k // 8:] = -1  # exercise the padding path in the traced graph
        _export(model, (kp, kp.clone(), de, de.clone(), sc, sc.clone()), path,
                ["kpts0", "kpts1", "desc0", "desc1", "scores0", "scores1"],
                ["matches0", "mscores0"], OPSET)
        _update_manifest(path, dict(model="lighterglue", k=a.k, layers=6, threshold=model.threshold,
                                    input="kpts[1,K,2] normalised (kornia normalize_keypoints), desc[1,K,64], scores[1,K] (>0 valid)",
                                    outputs="matches0[1,K] int32 (-1 = none), mscores0[1,K]",
                                    weights_sha256=_sha256(os.path.join(a.weights_dir, "xfeat-lighterglue.pt"))))
    print(f"[export] wrote {path}")


if __name__ == "__main__":
    main()
