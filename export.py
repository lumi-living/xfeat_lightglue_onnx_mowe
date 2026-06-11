#!/usr/bin/env python3
"""Export XFeat (+ optional LighterGlue) to ONNX for the Mow-e VIO frontend.

Tailored to the on-device TensorRT path
(tools/test-board/okvis2-mowe/okvis_xfeat_frontend):

  * MONO input (1, 1, H, W). The OV9281 is an 8-bit mono global-shutter sensor;
    XFeat averages channels to one immediately (model.py: ``x.mean(dim=1)``) and
    its first conv is ``Conv2d(1, 24, ...)``, so a single-channel input is
    loss-free and lets the CUDA preprocess kernel emit a one-plane buffer with no
    RGB expansion. (RGB still works via --channels 3 if ever needed.)

  * STATIC output shapes. Upstream ``detectAndCompute`` ends with
    ``mkpts[scores > 0]`` — a boolean mask that compiles to a ``NonZero`` op with
    a data-dependent output length, which TensorRT handles poorly. We export a
    wrapper that keeps the fixed ``top_k`` set (keypoints/descriptors/scores all
    length ``top_k``); the C++ frontend thresholds on score instead. With a
    static image size this yields a fully static engine — TensorRT's happy path.

  * Output bindings: ``keypoints`` (1, K, 2), ``descriptors`` (1, K, 64),
    ``scores`` (1, K) — the names/shapes okvis_xfeat_frontend::TensorRTEngine
    binds. Keeping the batch dim means they drop straight into LighterGlue.

C++ preprocessing contract (matches what this graph expects): XFeat does NOT
divide by 255 and applies no mean/std normalisation (InstanceNorm inside the net
handles scale). The CUDA kernel must emit float pixels in the raw 0..255 range,
NCHW, single channel, at exactly (H, W) below (a multiple of 32 so XFeat's
internal /32 resize is a no-op and the keypoint scale correction is unity).
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.xfeat import XFeat
from modules.interpolator import InterpolateSparse2d
from modules.lighterglue import LighterGlue


# opset 18: LighterGlue needs ≥18, and TensorRT 10.3 supports it natively — no
# fragile down-conversion. The extractor is fine at 18 too.
OPSET_DEFAULT = 18


class XFeatExtractorStatic(nn.Module):
    """XFeat detect+describe with static, fixed-``top_k`` outputs for TensorRT.

    A faithful copy of ``XFeat.detectAndCompute`` with two export-oriented
    changes only:
      * the final ``valid = scores > 0`` boolean mask is dropped (no NonZero);
      * the in-place ``scores[...] = -1`` becomes ``torch.where`` (trace-safe).
    Invalid slots keep score -1 and keypoint (0, 0); the consumer filters them.
    """

    def __init__(self, xfeat: XFeat):
        super().__init__()
        self.xfeat = xfeat

    def forward(self, x):
        xf = self.xfeat
        x, rh1, rw1 = xf.preprocess_tensor(x)
        _, _, _H1, _W1 = x.shape

        M1, K1, H1 = xf.net(x)
        M1 = F.normalize(M1, dim=1)

        K1h = xf.get_kpts_heatmap(K1)
        mkpts = xf.NMS(K1h)

        _nearest = InterpolateSparse2d("nearest")
        _bilinear = InterpolateSparse2d("bilinear")
        scores = (_nearest(K1h, mkpts, _H1, _W1) *
                  _bilinear(H1, mkpts, _H1, _W1)).squeeze(-1)
        # Mark padding keypoints (0,0) invalid — torch.where keeps this exportable.
        scores = torch.where(
            torch.all(mkpts == 0, dim=-1),
            torch.tensor(-1.0, dtype=scores.dtype, device=scores.device),
            scores,
        )

        # Fixed top_k selection — output length is constant, no masking.
        idxs = torch.argsort(-scores)
        mkpts_x = torch.gather(mkpts[..., 0], -1, idxs)[:, : xf.top_k]
        mkpts_y = torch.gather(mkpts[..., 1], -1, idxs)[:, : xf.top_k]
        mkpts = torch.cat([mkpts_x[..., None], mkpts_y[..., None]], dim=-1)
        scores = torch.gather(scores, -1, idxs)[:, : xf.top_k]

        feats = xf.interpolator(M1, mkpts, H=_H1, W=_W1)
        feats = F.normalize(feats, dim=-1)

        # Map keypoints back to original-image pixel coords (unity when H,W % 32 == 0).
        mkpts = mkpts * torch.tensor(
            [rw1, rh1], device=mkpts.device).view(1, 1, -1)

        return mkpts, feats, scores  # (1,K,2) (1,K,64) (1,K)


def _try_simplify(path: str) -> None:
    """Best-effort onnx-simplifier. Skipped (with a note) if unavailable/fails —
    the raw graph is still valid; simplification is an optimisation only."""
    try:
        import onnx
        from onnxsim import simplify
    except Exception as e:  # noqa: BLE001
        print(f"  [sim] skipped (onnxsim unavailable: {e})")
        return
    try:
        model_simp, ok = simplify(onnx.load(path))
        if ok:
            onnx.save(model_simp, path)
            print("  [sim] simplified OK")
        else:
            print("  [sim] simplifier could not validate; keeping raw graph")
    except Exception as e:  # noqa: BLE001
        print(f"  [sim] failed ({e}); keeping raw graph")


def _verify(path: str, model: nn.Module, height: int, width: int,
            channels: int) -> None:
    """Faithfulness check: torch wrapper vs onnxruntime on a REAL image, comparing
    the *set* of valid keypoints. Positional comparison is meaningless here —
    argsort ties (esp. the score=-1 padding) get ordered differently across
    backends, so identical keypoint sets can still differ slot-by-slot. Set-IoU
    is the right invariant. Needs a real image (random noise has too many ties)."""
    try:
        import cv2
        import numpy as np
        import onnxruntime as ort
    except Exception as e:  # noqa: BLE001
        print(f"  [verify] skipped (missing dep: {e})")
        return

    sample = next((c for c in ("assets/ref.png", "assets/tgt.png")
                   if os.path.exists(c)), None)
    if sample is None:
        print("  [verify] no sample image under assets/ — skipped")
        return
    im = cv2.resize(cv2.imread(sample, cv2.IMREAD_GRAYSCALE),
                    (width, height)).astype(np.float32)  # raw 0..255 mono
    t = torch.from_numpy(im)[None, None]
    if channels == 3:
        t = t.repeat(1, 3, 1, 1)

    with torch.no_grad():
        tk, _, ts = (a.cpu().numpy() for a in model(t))
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ok, _, osc = sess.run(None, {"images": t.numpy()})
    names = [o.name for o in sess.get_outputs()]
    print("  [verify] onnx outputs:",
          ", ".join(f"{n}{list(o.shape)}" for n, o in zip(names, (ok, _, osc))))

    def kset(k, s):
        return {(int(round(a)), int(round(b)))
                for (a, b), sc in zip(k[0], s[0]) if sc > 0}

    A, B = kset(tk, ts), kset(ok, osc)
    iou = len(A & B) / max(len(A | B), 1)
    sdiff = float(np.max(np.abs(np.sort(ts[0]) - np.sort(osc[0]))))
    flag = "OK" if iou > 0.99 else "WARN"
    print(f"  [verify] {flag} valid kpts torch={len(A)} onnx={len(B)}  "
          f"IoU={iou:.4f}  scores max|Δ|={sdiff:.2e}")


def export_extractor(args) -> str:
    xfeat = XFeat(weights=args.weights, top_k=args.top_k,
                  detection_threshold=args.detection_threshold).eval()
    model = XFeatExtractorStatic(xfeat).eval()

    dummy = torch.randn(1, args.channels, args.height, args.width)
    print(f"[extractor] input {tuple(dummy.shape)}  top_k={args.top_k}  "
          f"{'dynamic' if args.dynamic else 'static'}")

    output_names = ["keypoints", "descriptors", "scores"]
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "images": {2: "height", 3: "width"},
            "keypoints": {1: "num_keypoints"},
            "descriptors": {1: "num_keypoints"},
            "scores": {1: "num_keypoints"},
        }
        fname = f"xfeat_mono_dynamic_{args.top_k}.onnx"
    else:
        fname = f"xfeat_mono_{args.top_k}_{args.height}x{args.width}.onnx"
    if args.channels != 1:
        fname = fname.replace("mono", f"c{args.channels}")

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, fname)

    torch.onnx.export(
        model, dummy, path,
        input_names=["images"], output_names=output_names,
        opset_version=args.opset, do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )
    print(f"[extractor] wrote {path}")
    _try_simplify(path)
    _verify(path, model, args.height, args.width, args.channels)
    return path


def export_matcher(args) -> str:
    matcher = LighterGlue(n_layers=args.lighterglue_layers).eval()
    k = args.top_k
    kpts = torch.rand(1, k, 2, dtype=torch.float32) * 2 - 1
    desc = torch.rand(1, k, 64, dtype=torch.float32)

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "kpts0": {1: "num_keypoints0"}, "kpts1": {1: "num_keypoints1"},
            "desc0": {1: "num_keypoints0"}, "desc1": {1: "num_keypoints1"},
            "matches": {0: "num_matches"}, "scores": {0: "num_matches"},
        }

    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir,
                        f"lighterglue_L{args.lighterglue_layers}.onnx")
    print(f"[matcher] LighterGlue L{args.lighterglue_layers}  kpts={k}")
    torch.onnx.export(
        matcher, (kpts, kpts, desc, desc), path,
        input_names=["kpts0", "kpts1", "desc0", "desc1"],
        output_names=["matches", "scores"],
        opset_version=args.opset, do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )
    print(f"[matcher] wrote {path}")
    _try_simplify(path)
    return path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="weights/xfeat.pt",
                   help="XFeat .pt weights")
    p.add_argument("--output-dir", default="onnx")
    p.add_argument("--height", type=int, default=800,
                   help="input height (multiple of 32; OV9281 native is 800)")
    p.add_argument("--width", type=int, default=1280,
                   help="input width (multiple of 32; OV9281 native is 1280)")
    p.add_argument("--channels", type=int, default=1, choices=(1, 3),
                   help="1 = mono (OV9281, default), 3 = RGB")
    p.add_argument("--top-k", type=int, default=2048)
    p.add_argument("--detection-threshold", type=float, default=0.05)
    p.add_argument("--opset", type=int, default=OPSET_DEFAULT)
    p.add_argument("--dynamic", action="store_true",
                   help="export dynamic shapes (default: static — better for TRT)")
    p.add_argument("--lighterglue-layers", type=int, default=3)
    p.add_argument("--no-extractor", action="store_true")
    p.add_argument("--no-matcher", action="store_true")
    args = p.parse_args()
    if not args.dynamic and (args.height % 32 or args.width % 32):
        p.error("static export needs height & width to be multiples of 32")
    return args


if __name__ == "__main__":
    args = parse_args()
    if not args.no_extractor:
        export_extractor(args)
    if not args.no_matcher:
        export_matcher(args)
