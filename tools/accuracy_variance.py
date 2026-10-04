#!/usr/bin/env python3
"""Accuracy spread across random train/test splits.

The test set is only 20 examples, so one split is noisy. This runs the two
host models (the full FP32 MLP and the int8 + sparse head) over several seeds
and prints the mean and range, which shows the two are about equal on accuracy.

Run: python3 tools/accuracy_variance.py
"""

import sys

import numpy as np

sys.path.insert(0, "tools")
import prepare_dataset as P

SEEDS = 10
DS = 4
WINDOW = 64


def main():
    X, labels = P.load("data/gestures.csv")
    classes = sorted(set(labels))
    ci = {c: i for i, c in enumerate(classes)}
    y = np.array([ci[l] for l in labels])

    nt = WINDOW // DS
    Xf = X[:, : nt * DS, :].reshape(len(X), nt, DS, 4).mean(axis=2).reshape(len(X), nt * 4)

    full, sparse, test_n = [], [], 0
    for seed in range(SEEDS):
        r = np.random.default_rng(seed)
        itr, ite = P.stratified_split(y, 0.2, r)
        test_n = len(ite)
        s = np.percentile(np.abs(Xf[itr]), 99)
        s = float(s) if s > 1 else 1.0
        Xn = np.clip(Xf / s, -1.0, 1.0)
        Xtr, ytr, Xte, yte = Xn[itr], y[itr], Xn[ite], y[ite]
        _, te, _, (W1, b1, _, _) = P.train_mlp(Xtr, ytr, Xte, yte, len(classes),
                                               hidden=32, seed=seed)
        fh = P.train_head_only(Xtr, ytr, Xte, yte, W1, b1, len(classes),
                               quantize_w1=True, seed=seed)
        full.append(te * 100)
        sparse.append(fh * 100)

    full, sparse = np.array(full), np.array(sparse)
    print(f"{SEEDS} random splits, test set = {test_n} examples each")
    print(f"full FP32 MLP:      mean {full.mean():.1f}%  range {full.min():.0f}-{full.max():.0f}")
    print(f"int8 + sparse head: mean {sparse.mean():.1f}%  range {sparse.min():.0f}-{sparse.max():.0f}")


if __name__ == "__main__":
    main()
