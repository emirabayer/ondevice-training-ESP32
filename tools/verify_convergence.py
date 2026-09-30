#!/usr/bin/env python3
"""Verification script.

Generates a synthetic classification problem with the same shape as the
C++ engine's demo (src/main.cpp): a frozen random-projection "feature
extractor" (Linear(12, 24) + ReLU) feeding a trainable linear head
(Linear(24, 3)), trained with Adam on cross-entropy. It then:

  1. Trains a full-precision (float32) reference model of that exact
     architecture and reports its convergence (loss/accuracy per epoch).
  2. Builds and runs the actual C++ INT8 training engine (the `train`
     target) and parses its convergence log.
  3. Compares the two: both should converge to comparably high accuracy on
     this easy synthetic problem,
     which is the intended verification, that quantization-aware training
     with an INT8 forward pass and a float-shadow/STE backward pass
     converges essentially as well as the unquantized equivalent, not that
     the two are bit-for-bit identical.

Uses PyTorch if it's importable; otherwise falls back to an equivalent
hand-rolled NumPy reference implementation of the same architecture and
optimizer, since a
verification script should not hard-fail an environment that simply
doesn't have PyTorch installed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    import torch

    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

INPUT_DIM = 12
HIDDEN_DIM = 24
NUM_CLASSES = 3
SAMPLES_PER_CLASS = 100
NUM_TRAIN = 240
NUM_EPOCHS = 60
BATCH_SIZE = 20
LEARNING_RATE = 0.08


def generate_dataset(seed: int):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-3.0, 3.0, size=(NUM_CLASSES, INPUT_DIM)).astype(np.float32)

    features = np.empty((NUM_CLASSES * SAMPLES_PER_CLASS, INPUT_DIM), dtype=np.float32)
    labels = np.empty(NUM_CLASSES * SAMPLES_PER_CLASS, dtype=np.int64)
    idx = 0
    for c in range(NUM_CLASSES):
        noise = rng.uniform(-0.6, 0.6, size=(SAMPLES_PER_CLASS, INPUT_DIM)).astype(np.float32)
        features[idx : idx + SAMPLES_PER_CLASS] = centers[c] + noise
        labels[idx : idx + SAMPLES_PER_CLASS] = c
        idx += SAMPLES_PER_CLASS

    perm = rng.permutation(len(labels))
    features, labels = features[perm], labels[perm]
    return (
        features[:NUM_TRAIN],
        labels[:NUM_TRAIN],
        features[NUM_TRAIN:],
        labels[NUM_TRAIN:],
    )


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_numpy_reference(seed: int):
    """Full-precision equivalent: frozen random Linear+ReLU -> trainable
    Linear head, trained with a hand-rolled Adam step over analytic
    softmax+cross-entropy gradients, deliberately mirroring the same
    backprop math as edge::DenseLayer::Backward / Softmax_CrossEntropy_Backward
    in the C++ engine, just without any quantization in the loop."""
    x_train, y_train, x_test, y_test = generate_dataset(seed)
    rng = np.random.default_rng(seed + 1)

    limit1 = 1.0 / np.sqrt(INPUT_DIM)
    w1 = rng.uniform(-limit1, limit1, size=(INPUT_DIM, HIDDEN_DIM)).astype(np.float32)
    b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)

    limit2 = 1.0 / np.sqrt(HIDDEN_DIM)
    w2 = rng.uniform(-limit2, limit2, size=(HIDDEN_DIM, NUM_CLASSES)).astype(np.float32)
    b2 = np.zeros(NUM_CLASSES, dtype=np.float32)

    m_w2 = np.zeros_like(w2)
    v_w2 = np.zeros_like(w2)
    m_b2 = np.zeros_like(b2)
    v_b2 = np.zeros_like(b2)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    def forward(x):
        hidden = np.maximum(x @ w1 + b1, 0.0)
        logits = hidden @ w2 + b2
        return hidden, logits

    def evaluate(x, y):
        _, logits = forward(x)
        probs = softmax(logits)
        loss = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean()
        acc = (probs.argmax(axis=1) == y).mean()
        return float(loss), float(acc)

    history = []
    n = len(x_train)
    for epoch in range(1, NUM_EPOCHS + 1):
        order = rng.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            batch_idx = order[start : start + BATCH_SIZE]
            xb, yb = x_train[batch_idx], y_train[batch_idx]
            b = len(yb)

            hidden, logits = forward(xb)
            probs = softmax(logits)
            y_onehot = np.zeros_like(probs)
            y_onehot[np.arange(b), yb] = 1.0
            dz = (probs - y_onehot) / b

            grad_w2 = hidden.T @ dz
            grad_b2 = dz.sum(axis=0)

            t += 1
            for param, grad, m, v in (
                (w2, grad_w2, m_w2, v_w2),
                (b2, grad_b2, m_b2, v_b2),
            ):
                m *= beta1
                m += (1 - beta1) * grad
                v *= beta2
                v += (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)
                param -= LEARNING_RATE * m_hat / (np.sqrt(v_hat) + eps)

        train_loss, train_acc = evaluate(x_train, y_train)
        test_loss, test_acc = evaluate(x_test, y_test)
        history.append((epoch, train_loss, train_acc, test_loss, test_acc))

    return history


def train_torch_reference(seed: int):
    x_train, y_train, x_test, y_test = generate_dataset(seed)
    torch.manual_seed(seed + 1)

    feature_extractor = torch.nn.Linear(INPUT_DIM, HIDDEN_DIM)
    for p in feature_extractor.parameters():
        p.requires_grad_(False)
    head = torch.nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    opt = torch.optim.Adam(head.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    xt_train = torch.from_numpy(x_train)
    yt_train = torch.from_numpy(y_train)
    xt_test = torch.from_numpy(x_test)
    yt_test = torch.from_numpy(y_test)

    def forward(x):
        return head(torch.relu(feature_extractor(x)))

    def evaluate(x, y):
        with torch.no_grad():
            logits = forward(x)
            loss = loss_fn(logits, y).item()
            acc = (logits.argmax(dim=1) == y).float().mean().item()
        return loss, acc

    history = []
    n = len(x_train)
    generator = torch.Generator().manual_seed(seed + 2)
    for epoch in range(1, NUM_EPOCHS + 1):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            opt.zero_grad()
            logits = forward(xt_train[idx])
            loss = loss_fn(logits, yt_train[idx])
            loss.backward()
            opt.step()

        train_loss, train_acc = evaluate(xt_train, yt_train)
        test_loss, test_acc = evaluate(xt_test, yt_test)
        history.append((epoch, train_loss, train_acc, test_loss, test_acc))

    return history


def find_train_binary() -> Path | None:
    for candidate in ("build/train", "build-release/train"):
        p = Path(candidate)
        if p.is_file():
            return p
    return None


def run_cpp_engine(binary: Path):
    result = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    stdout = result.stdout
    print(stdout)
    if result.returncode != 0:
        print(f"[warn] {binary} exited with status {result.returncode}", file=sys.stderr)

    acc_match = re.search(r"final held-out test accuracy:\s*([0-9.]+)%", stdout)
    peak_match = re.search(r"peak.*?:\s*(\d+) bytes.*?([0-9.]+)% of the 256KB budget", stdout)
    return {
        "test_accuracy": float(acc_match.group(1)) / 100.0 if acc_match else None,
        "peak_bytes": int(peak_match.group(1)) if peak_match else None,
        "peak_pct": float(peak_match.group(2)) if peak_match else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--train-binary", type=Path, default=None,
                         help="path to the compiled `train` executable")
    parser.add_argument("--skip-cpp", action="store_true",
                         help="only run the float reference model")
    args = parser.parse_args()

    backend = "PyTorch" if HAVE_TORCH else "NumPy (PyTorch not installed; using equivalent fallback)"
    print(f"=== Full-precision reference model ({backend}) ===")
    history = train_torch_reference(args.seed) if HAVE_TORCH else train_numpy_reference(args.seed)
    for epoch, train_loss, train_acc, test_loss, test_acc in history:
        if epoch == 1 or epoch % 5 == 0 or epoch == NUM_EPOCHS:
            print(f"epoch {epoch:3d}  train_loss={train_loss:.5f} train_acc={train_acc:.4f}  "
                  f"test_loss={test_loss:.5f} test_acc={test_acc:.4f}")
    ref_final_acc = history[-1][4]
    print(f"\nreference final test accuracy: {ref_final_acc * 100:.2f}%\n")

    if args.skip_cpp:
        return 0 if ref_final_acc >= 0.8 else 1

    binary = args.train_binary or find_train_binary()
    if binary is None:
        print("[error] could not find the compiled `train` binary "
              "(expected build/train or build-release/train; build it first, "
              "or pass --train-binary)", file=sys.stderr)
        return 2

    print(f"=== C++ INT8 training engine ({binary}) ===")
    cpp_result = run_cpp_engine(binary)

    print("=== Comparison ===")
    ref_pct = ref_final_acc * 100.0
    cpp_pct = cpp_result["test_accuracy"] * 100.0 if cpp_result["test_accuracy"] is not None else float("nan")
    print(f"reference (float32) test accuracy: {ref_pct:6.2f}%")
    print(f"C++ engine (INT8)    test accuracy: {cpp_pct:6.2f}%")
    if cpp_result["peak_bytes"] is not None:
        print(f"C++ engine peak arena usage:        {cpp_result['peak_bytes']} bytes "
              f"({cpp_result['peak_pct']:.2f}% of 256KB)")

    ok = (
        ref_final_acc >= 0.8
        and cpp_result["test_accuracy"] is not None
        and cpp_result["test_accuracy"] >= 0.8
        and abs(ref_pct - cpp_pct) <= 15.0
    )
    print("\nVERDICT:", "PASS, both models converge and agree within tolerance" if ok
          else "FAIL, convergence or agreement criteria not met")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
