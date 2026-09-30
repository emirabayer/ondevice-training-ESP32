#!/usr/bin/env python3
"""Prepare the collected gesture CSV for on-device training.

Pipeline:
  1. Load data/gestures.csv (long format) -> windows X[N, 64, 4], labels y[N].
  2. Downsample each window in time (mean-pool by --downsample) to cut input
     size / noise while preserving the swipe time-order, then flatten to
     [N, n_features].
  3. Normalize by a single global scale (so a touch of typical strength maps
     to ~[-1, 1]); the same scale is emitted for the device.
  4. Stratified train/test split.
  5. HOST LEARNABILITY CHECK: train a small numpy MLP (n_features -> hidden ->
     n_classes) and report train/test accuracy, this is the realistic target
     the on-device trainers should approach, and confirms the task is
     separable before we invest in firmware.
  6. Emit include/gesture_dataset.h with normalized float arrays + labels +
     metadata, shared by both the FP32 and INT8 training firmwares.

Run with system python (has numpy):  python3 tools/prepare_dataset.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys

import numpy as np

CHANNELS = ("d7", "d8", "d9", "d12")
WINDOW = 64  # samples per take (from the capture firmware @ 50 Hz)


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    takes = collections.OrderedDict()
    for r in rows:
        t = int(r["take_id"])
        takes.setdefault(t, {"label": r["label"], "samples": {}})
        takes[t]["samples"][int(r["sample_idx"])] = [int(r[c]) for c in CHANNELS]
    X, y_labels = [], []
    for t, d in takes.items():
        if len(d["samples"]) != WINDOW:
            print(f"  (skipping take {t}: {len(d['samples'])} samples != {WINDOW})")
            continue
        win = [d["samples"][i] for i in range(WINDOW)]
        X.append(win)
        y_labels.append(d["label"])
    return np.asarray(X, dtype=np.float64), y_labels


def stratified_split(y, test_frac, rng):
    idx_train, idx_test = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_frac)))
        idx_test += list(idx[:n_test])
        idx_train += list(idx[n_test:])
    rng.shuffle(idx_train)
    rng.shuffle(idx_test)
    return np.asarray(idx_train), np.asarray(idx_test)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_mlp(Xtr, ytr, Xte, yte, n_classes, hidden=16, epochs=1500, lr=0.05, seed=0):
    """Tiny full-batch MLP (ReLU) with Adam, mirrors the on-device model."""
    rng = np.random.default_rng(seed)
    n_in = Xtr.shape[1]
    W1 = rng.normal(0, 1 / np.sqrt(n_in), (n_in, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 1 / np.sqrt(hidden), (hidden, n_classes))
    b2 = np.zeros(n_classes)
    params = [W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    b1c, b2c, eps = 0.9, 0.999, 1e-8

    def forward(X):
        h = np.maximum(X @ W1 + b1, 0)
        return h, softmax(h @ W2 + b2)

    Ytr = np.eye(n_classes)[ytr]
    for t in range(1, epochs + 1):
        h, p = forward(Xtr)
        dz2 = (p - Ytr) / len(Xtr)
        gW2 = h.T @ dz2
        gb2 = dz2.sum(0)
        dh = (dz2 @ W2.T) * (h > 0)
        gW1 = Xtr.T @ dh
        gb1 = dh.sum(0)
        grads = [gW1, gb1, gW2, gb2]
        for i, (pm, g) in enumerate(zip(params, grads)):
            m[i] = b1c * m[i] + (1 - b1c) * g
            v[i] = b2c * v[i] + (1 - b2c) * g * g
            pm -= lr * (m[i] / (1 - b1c**t)) / (np.sqrt(v[i] / (1 - b2c**t)) + eps)

    tr_acc = (forward(Xtr)[1].argmax(1) == ytr).mean()
    te_acc = (forward(Xte)[1].argmax(1) == yte).mean()
    return tr_acc, te_acc, forward(Xte)[1].argmax(1), (W1, b1, W2, b2)


def train_head_only(Xtr, ytr, Xte, yte, W1, b1, n_classes, quantize_w1=False,
                    epochs=1500, lr=0.05, seed=0):
    """Freeze the pre-trained feature extractor (W1,b1) and train ONLY the head
    on the resulting hidden features, mirrors the INT8 sparse-update path.
    If quantize_w1, simulate INT8 by rounding W1 and the hidden activations to
    ~8-bit, to predict the effect of quantization on the frozen backbone."""
    W1f, b1f = W1.copy(), b1.copy()
    if quantize_w1:
        s = np.abs(W1f).max() / 127.0
        W1f = np.round(W1f / s) * s
    Htr = np.maximum(Xtr @ W1f + b1f, 0)
    Hte = np.maximum(Xte @ W1f + b1f, 0)
    if quantize_w1:
        hs = np.abs(Htr).max() / 127.0
        Htr = np.round(Htr / hs) * hs
        Hte = np.round(Hte / hs) * hs
    rng = np.random.default_rng(seed)
    hidden = W1.shape[1]
    W2 = rng.normal(0, 1 / np.sqrt(hidden), (hidden, n_classes))
    b2 = np.zeros(n_classes)
    m = [np.zeros_like(W2), np.zeros_like(b2)]
    v = [np.zeros_like(W2), np.zeros_like(b2)]
    Ytr = np.eye(n_classes)[ytr]
    for t in range(1, epochs + 1):
        p = softmax(Htr @ W2 + b2)
        dz = (p - Ytr) / len(Htr)
        grads = [Htr.T @ dz, dz.sum(0)]
        for i, (pm, g) in enumerate(zip((W2, b2), grads)):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            pm -= lr * (m[i] / (1 - 0.9**t)) / (np.sqrt(v[i] / (1 - 0.999**t)) + 1e-8)
    return (softmax(Hte @ W2 + b2).argmax(1) == yte).mean()


def emit_header(path, Xtr, ytr, Xte, yte, classes, scale, downsample, W1, b1):
    n_feat = Xtr.shape[1]
    hidden = W1.shape[1]

    def arr2d(name, A):
        lines = [f"static const float {name}[{A.shape[0]}][{A.shape[1]}] = {{"]
        for row in A:
            lines.append("  {" + ",".join(f"{x:.6f}f" for x in row) + "},")
        lines.append("};")
        return "\n".join(lines)

    def arr1d_i(name, a):
        return f"static const int32_t {name}[{len(a)}] = {{" + ",".join(str(int(x)) for x in a) + "};"

    with open(path, "w") as f:
        f.write("// Auto-generated by tools/prepare_dataset.py, do not edit by hand.\n")
        f.write("// Gesture dataset: normalized (~[-1,1]) downsampled touch windows.\n")
        f.write("#pragma once\n#include <cstdint>\n\n")
        f.write(f"#define GDS_N_TRAIN {Xtr.shape[0]}\n")
        f.write(f"#define GDS_N_TEST  {Xte.shape[0]}\n")
        f.write(f"#define GDS_N_FEATURES {n_feat}\n")
        f.write(f"#define GDS_HIDDEN {hidden}\n")
        f.write(f"#define GDS_N_CLASSES {len(classes)}\n")
        f.write(f"#define GDS_INPUT_SCALE {scale:.6f}f  // raw_delta = normalized * scale\n")
        f.write(f"#define GDS_DOWNSAMPLE {downsample}\n\n")
        names = ",".join(f'"{c}"' for c in classes)
        f.write(f"static const char *const GDS_CLASS_NAMES[{len(classes)}] = {{{names}}};\n\n")
        f.write(arr2d("GDS_X_TRAIN", Xtr) + "\n\n")
        f.write(arr1d_i("GDS_Y_TRAIN", ytr) + "\n\n")
        f.write(arr2d("GDS_X_TEST", Xte) + "\n\n")
        f.write(arr1d_i("GDS_Y_TEST", yte) + "\n\n")
        f.write("// Pre-trained feature extractor (host-trained); FROZEN on-device\n")
        f.write("// for the INT8 QAS + sparse-update path. Layout: W1[n_features][hidden].\n")
        f.write(arr2d("GDS_W1", W1) + "\n\n")
        f.write("static const float GDS_B1[" + str(len(b1)) + "] = {" +
                ",".join(f"{x:.6f}f" for x in b1) + "};\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/gestures.csv")
    ap.add_argument("--out-header", default="include/gesture_dataset.h")
    ap.add_argument("--downsample", type=int, default=4, help="time mean-pool factor (64/ds timesteps)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X, y_labels = load(args.csv)
    classes = sorted(set(y_labels))
    cls_idx = {c: i for i, c in enumerate(classes)}
    y = np.asarray([cls_idx[l] for l in y_labels])
    print(f"loaded {len(X)} windows, {len(classes)} classes: {classes}")

    # Downsample in time then flatten (preserves per-channel time-order).
    ds = args.downsample
    n_t = WINDOW // ds
    Xd = X[:, : n_t * ds, :].reshape(len(X), n_t, ds, 4).mean(axis=2)  # [N, n_t, 4]
    Xf = Xd.reshape(len(X), n_t * 4)  # flatten, channel-interleaved per timestep
    print(f"downsampled window 64 -> {n_t} timesteps; feature vector = {Xf.shape[1]}")

    rng = np.random.default_rng(args.seed)
    itr, ite = stratified_split(y, args.test_frac, rng)

    # Normalize by 99th percentile of |x| on TRAIN only (robust to outliers).
    scale = np.percentile(np.abs(Xf[itr]), 99)
    scale = float(scale) if scale > 1 else 1.0
    Xn = np.clip(Xf / scale, -1.0, 1.0)
    Xtr, ytr, Xte, yte = Xn[itr], y[itr], Xn[ite], y[ite]
    print(f"normalization scale (99th pct |delta|): {scale:.1f}")
    print(f"train/test split: {len(Xtr)} / {len(Xte)}")

    tr_acc, te_acc, te_pred, (W1, b1, W2, b2) = train_mlp(
        Xtr, ytr, Xte, yte, len(classes), hidden=args.hidden, seed=args.seed)
    print("\n=== HOST LEARNABILITY CHECK (full MLP = naive FP32 baseline proxy) ===")
    print(f"  train accuracy: {tr_acc*100:.1f}%")
    print(f"  test  accuracy: {te_acc*100:.1f}%   <-- target for the FP32 firmware")
    # confusion on test
    cm = np.zeros((len(classes), len(classes)), int)
    for t, p in zip(yte, te_pred):
        cm[t, p] += 1
    print("  test confusion (rows=true, cols=pred):")
    print("      " + " ".join(f"{c[:6]:>6}" for c in classes))
    for i, c in enumerate(classes):
        print(f"  {c[:6]:>6} " + " ".join(f"{cm[i,j]:>6}" for j in range(len(classes))))

    fh = train_head_only(Xtr, ytr, Xte, yte, W1, b1, len(classes),
                         quantize_w1=False, seed=args.seed)
    fhq = train_head_only(Xtr, ytr, Xte, yte, W1, b1, len(classes),
                          quantize_w1=True, seed=args.seed)
    print("\n=== SPARSE-PATH PREVIEW (frozen feature extractor + trained head) ===")
    print(f"  frozen W1 + trained head (float):   {fh*100:.1f}%")
    print(f"  + INT8-simulated W1/activations:    {fhq*100:.1f}%   <-- predicts INT8 sparse firmware")

    emit_header(args.out_header, Xtr, ytr, Xte, yte, classes, scale, ds, W1, b1)
    print(f"\nwrote {args.out_header}  "
          f"(train {Xtr.shape}, test {Xte.shape}, {len(classes)} classes, hidden={args.hidden})")


if __name__ == "__main__":
    sys.exit(main())
