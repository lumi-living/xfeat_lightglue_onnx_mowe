"""T-0109: exported ONNX must reproduce the PyTorch wrapper on real images.

Run the four ``export.py`` commands first (they write ``out/onnx``), then
``MOWE_TICKET=T-0109 pytest -q tests``.  Writes ``out/agent/<ticket>/iou.json``.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(ROOT, "..", "..", ".."))
sys.path.insert(0, ROOT)
import export  # noqa: E402

ONNX_DIR = os.environ.get("MOWE_ONNX_DIR", os.path.join(REPO, "out", "onnx"))
TICKET = os.environ.get("MOWE_TICKET", "T-0109")
IOU_JSON = os.path.join(REPO, "out", "agent", TICKET, "iou.json")
RADIUS_PX = 2.0
RESULTS = {}

XFEAT_FILES = ["xfeat_640x384_k1024.onnx", "xfeat_1280x800_k1024.onnx", "xfeat_512x512_k1024.onnx"]
IMAGES = ["room1_c0_a", "room1_c0_c", "room1_c0_d"]
PAIRS = [("room1_c0_a", "room1_c1_a"), ("room1_c0_a", "room1_c0_b"), ("room1_c0_c", "room1_c0_d")]


def load(name, w, h):
    im = cv2.imread(os.path.join(HERE, "fixtures", name + ".png"), cv2.IMREAD_UNCHANGED)
    if im.dtype == np.uint16:
        im = (im >> 8).astype(np.uint8)
    return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)  # raw 0..255


def valid_sets(kp, sc):
    return kp[sc > 0].astype(np.float32), sc > 0


def set_iou(a, b, r=RADIUS_PX):
    """Jaccard over keypoint sets, a point counts as shared if the other set has one within r px."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    inter = min(int((d.min(1) <= r).sum()), int((d.min(0) <= r).sum()))
    return inter / (len(a) + len(b) - inter)


def desc_cosine_at_shared(kp_a, de_a, kp_b, de_b):
    d = np.linalg.norm(kp_a[:, None, :] - kp_b[None, :, :], axis=-1)
    j = d.argmin(1)
    ok = d[np.arange(len(kp_a)), j] == 0  # identical pixel → same keypoint
    return (de_a[ok] * de_b[j[ok]]).sum(-1)


def _session(fname):
    path = os.path.join(ONNX_DIR, fname)
    if not os.path.exists(path):
        pytest.skip(f"{path} missing — run export.py first")
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def manifest():
    return json.load(open(os.path.join(ROOT, "manifest.json")))


@pytest.mark.parametrize("fname", XFEAT_FILES)
def test_xfeat_onnx_matches_torch(fname, manifest):
    m = manifest[fname]
    w, h, k = m["width"], m["height"], m["k"]
    sess = _session(fname)
    assert [i.shape for i in sess.get_inputs()] == [[2, 1, h, w]]
    assert [o.shape for o in sess.get_outputs()] == [[2, k, 2], [2, k, 64], [2, k]]
    model = export.build_xfeat(k)
    ious, cos_min, cos_mean, counts, ious_up = [], [], [], [], []
    for i, name in enumerate(IMAGES):
        # batch = (this image, next image): each slot must be independent of the other.
        other = IMAGES[(i + 1) % len(IMAGES)]
        x = np.stack([load(name, w, h), load(other, w, h)])[:, None]
        with torch.no_grad():
            tk, td, ts = (t.numpy() for t in model(torch.from_numpy(x)))
        ok, od, osc = sess.run(None, {"images": x})
        assert ok.dtype == np.int32 and osc.dtype == np.float32
        for b in range(2):
            ka, ma = valid_sets(tk[b], ts[b])
            kb, mb = valid_sets(ok[b], osc[b])
            iou = set_iou(ka, kb)
            cos = desc_cosine_at_shared(ka, td[b][ma], kb, od[b][mb])
            ious.append(iou); cos_min.append(float(cos.min())); cos_mean.append(float(cos.mean()))
            counts.append((int(ma.sum()), int(mb.sum())))
            assert (ka[:, 0] < w).all() and (ka[:, 1] < h).all()
        # Informational: how close the static wrapper is to upstream detectAndCompute.
        up = model.xfeat.detectAndCompute(torch.from_numpy(x[:1]))
        ious_up.append(set_iou(valid_sets(tk[0], ts[0])[0], up["keypoints"].numpy()))
    key = fname.rsplit("_k", 1)[0]
    RESULTS[f"{key}_iou"] = min(ious)
    RESULTS[f"{key}_desc_cosine_min"] = min(cos_min)
    RESULTS[f"{key}_detail"] = dict(iou=ious, desc_cosine_mean=cos_mean, valid_counts_torch_onnx=counts,
                                    iou_vs_upstream_detectAndCompute=ious_up, images=IMAGES)
    assert min(ious) >= 0.98, ious
    assert min(cos_min) >= 0.99, cos_min


def _normalise(kp, w, h):
    size = np.array([w, h], np.float32)
    return (kp - size / 2) / (size.max() / 2)


def _match_set(m0):
    return {(i, int(j)) for i, j in enumerate(m0[0]) if j >= 0}


def test_lighterglue_onnx_matches_torch():
    fname = "lighterglue_k512.onnx"
    sess = _session(fname)
    k = 512
    assert [o.shape for o in sess.get_outputs()] == [[1, k], [1, k]]
    w, h = 512, 512
    ext, matcher = export.build_xfeat(k), export.build_lighterglue()
    agree, n_matches = [], []
    for a, b in PAIRS:
        x = np.stack([load(a, w, h), load(b, w, h)])[:, None]
        with torch.no_grad():
            kp, de, sc = (t.numpy() for t in ext(torch.from_numpy(x)))
        kp = _normalise(kp.astype(np.float32), w, h)
        # Exercise the padding path: drop the weakest 1/8 of set 1 by score.
        sc1 = sc[1].copy(); sc1[np.argsort(sc1)[: k // 8]] = -1.0
        feed = dict(kpts0=kp[:1], kpts1=kp[1:], desc0=de[:1], desc1=de[1:],
                    scores0=sc[:1], scores1=sc1[None])
        with torch.no_grad():
            tm, tms = (t.numpy() for t in matcher(*(torch.from_numpy(v) for v in feed.values())))
        om, oms = sess.run(None, feed)
        assert om.dtype == np.int32
        A, B = _match_set(tm), _match_set(om)
        assert all(sc1[j] > 0 for _, j in A | B), "match into a padded slot"
        assert all(sc[0][i] > 0 for i, _ in A | B)
        agree.append(len(A & B) / max(len(A | B), 1)); n_matches.append((len(A), len(B)))
    RESULTS["lighterglue_k512_match_agreement"] = min(agree)
    RESULTS["lighterglue_k512_detail"] = dict(agreement=agree, n_matches_torch_onnx=n_matches, pairs=PAIRS)
    assert n_matches[0][0] >= 100, "stereo pair should yield many matches (keypoint normalisation?)"
    assert min(agree) >= 0.95, agree


def test_manifest_complete(manifest):
    for f in XFEAT_FILES + ["lighterglue_k512.onnx"]:
        assert {"sha256", "opset", "k", "exporter_git", "date"} <= set(manifest[f]), f
        assert manifest[f]["opset"] == 18


@pytest.fixture(scope="session", autouse=True)
def _write_results():
    yield
    os.makedirs(os.path.dirname(IOU_JSON), exist_ok=True)
    RESULTS["onnx_dir"] = ONNX_DIR
    RESULTS["radius_px"] = RADIUS_PX
    json.dump(RESULTS, open(IOU_JSON, "w"), indent=2)
