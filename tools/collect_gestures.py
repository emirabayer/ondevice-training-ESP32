#!/usr/bin/env python3
"""Interactive gesture-data collector for the YD-ESP32-S3 touch pad.

Reads the 4-channel touch stream produced by firmware_gesture_capture
(lines: "D <ms> <d7> <d8> <d9> <d12>") over serial, and walks you through
recording labeled examples of each gesture. Each recorded example ("take")
is a fixed-length window of samples captured right after a countdown.

Data is appended to a long-format CSV (crash-safe, every sample is flushed
as it's read), one row per sample:

    take_id,label,sample_idx,ms,d7,d8,d9,d12

Downstream (training-prep) reshapes by take_id into windows of shape
[window, 4] with one label each.

Only dependency beyond the stdlib is pyserial (already present in the
ESP-IDF Python environment). Run it from an ESP-IDF shell:

    source ~/esp/get_idf.fish
    python tools/collect_gestures.py            # defaults: 15 takes/gesture

Useful flags:
    --takes N         examples to record per gesture (default 15)
    --gestures a,b,c  gesture set (default swipe_left,swipe_right,hold,two_tap,idle)
    --window N        samples per take (default 64 == 1.28 s at 50 Hz)
    --port PATH       serial port (default /dev/ttyACM0)
    --out PATH        dataset CSV (default data/gestures.csv, appends)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

try:
    import serial  # pyserial
except ImportError:
    sys.exit("pyserial not found. Run inside the ESP-IDF env "
             "(`source ~/esp/get_idf.fish`) or `pip install pyserial`.")

CHANNELS = (7, 8, 9, 12)
DEFAULT_GESTURES = ["swipe_left", "swipe_right", "hold", "two_tap", "idle"]

GESTURE_HINTS = {
    "swipe_left":  "slide one finger across the pads 12 -> 9 -> 8 -> 7",
    "swipe_right": "slide one finger across the pads 7 -> 8 -> 9 -> 12",
    "hold":        "press and hold one pad steadily for the whole window",
    "two_tap":     "tap the two end pads (7 and 12) together with two fingers",
    "idle":        "do NOT touch anything (captures the resting/no-gesture class)",
}


def read_sample(ser: serial.Serial):
    """Read one 'D ...' data line; return (ms, [d7,d8,d9,d12]) or None."""
    raw = ser.readline().decode("ascii", errors="replace").strip()
    if not raw.startswith("D "):
        return None
    parts = raw.split()
    if len(parts) != 2 + len(CHANNELS):
        return None
    try:
        ms = int(parts[1])
        deltas = [int(x) for x in parts[2:]]
    except ValueError:
        return None
    return ms, deltas


def wait_for_stream(ser: serial.Serial, timeout=8.0) -> bool:
    """Confirm the firmware is actually streaming samples."""
    print("Waiting for touch stream ...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if read_sample(ser) is not None:
            print(" OK")
            return True
    print(" NONE")
    return False


def live_preview(ser: serial.Serial, seconds=4.0):
    """Show live per-channel deltas so you can confirm contact before recording."""
    print(f"\nLive preview ({seconds:.0f}s), touch the pads to check they respond:")
    ser.reset_input_buffer()
    deadline = time.time() + seconds
    while time.time() < deadline:
        s = read_sample(ser)
        if s is None:
            continue
        _, d = s
        bars = "  ".join(f"G{ch}:{val:+5d}{'*' if val > 800 else ' '}"
                         for ch, val in zip(CHANNELS, d))
        print("\r  " + bars, end="", flush=True)
    print("\n")


def countdown(n=3):
    for i in range(n, 0, -1):
        print(f"  {i}...", end="", flush=True)
        time.sleep(0.7)
    print("  GO!", flush=True)


def record_take(ser: serial.Serial, window: int):
    """Capture `window` fresh samples immediately (call right after GO)."""
    ser.reset_input_buffer()
    samples = []
    misses = 0
    while len(samples) < window:
        s = read_sample(ser)
        if s is None:
            misses += 1
            if misses > window * 4:
                break  # stream stalled
            continue
        samples.append(s)
    return samples


def summarize_take(samples):
    """Peak absolute delta per channel, quick 'did it register?' feedback."""
    peaks = [0] * len(CHANNELS)
    for _, d in samples:
        for i, v in enumerate(d):
            if abs(v) > abs(peaks[i]):
                peaks[i] = v
    return peaks


def next_take_id(path: str) -> int:
    if not os.path.exists(path):
        return 0
    last = -1
    with open(path) as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = max(last, int(line.split(",")[0]))
            except (ValueError, IndexError):
                pass
    return last + 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--window", type=int, default=64, help="samples per take (50 Hz)")
    ap.add_argument("--takes", type=int, default=15, help="examples per gesture")
    ap.add_argument("--gestures", default=",".join(DEFAULT_GESTURES))
    ap.add_argument("--out", default="data/gestures.csv")
    args = ap.parse_args()

    gestures = [g.strip() for g in args.gestures.split(",") if g.strip()]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.5)
    except serial.SerialException as e:
        return f"Could not open {args.port}: {e}"

    # Opening the port resets the board; give it a moment to boot and stream.
    time.sleep(1.5)
    if not wait_for_stream(ser):
        return ("No samples arrived. Is the capture firmware flashed and running? "
                f"(port {args.port})")

    take_id = next_take_id(args.out)
    new_file = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    out = open(args.out, "a", buffering=1)  # line-buffered -> crash-safe
    if new_file:
        out.write("take_id,label,sample_idx,ms,d7,d8,d9,d12\n")

    print("=" * 64)
    print(f"Collecting {args.takes} takes each for: {', '.join(gestures)}")
    print(f"Window: {args.window} samples (~{args.window/50:.2f}s)   "
          f"Appending to: {args.out}   (starting take_id={take_id})")
    print("Controls at each prompt: [Enter]=record  r=redo last  s=skip  q=quit")
    print("=" * 64)

    live_preview(ser)

    counts = {g: 0 for g in gestures}
    for g in gestures:
        print(f"\n########## GESTURE: {g.upper()}, {GESTURE_HINTS.get(g,'')} ##########")
        take = 0
        while take < args.takes:
            cmd = input(f"[{g}] take {take+1}/{args.takes}, Enter=record, s=skip, q=quit: ").strip().lower()
            if cmd == "q":
                print("Quitting early; data so far is saved.")
                out.close()
                _print_summary(args.out)
                return 0
            if cmd == "s":
                print("  skipped.")
                break
            countdown()
            samples = record_take(ser, args.window)
            if len(samples) < args.window:
                print(f"  ! only got {len(samples)}/{args.window} samples, stream hiccup, retrying")
                continue
            peaks = summarize_take(samples)
            peak_str = "  ".join(f"G{ch}:{p:+5d}" for ch, p in zip(CHANNELS, peaks))
            touched = any(abs(p) > 800 for p in peaks)
            flag = "" if (touched or g == "idle") else "  <-- weak! (no clear touch)"
            print(f"  recorded peaks: {peak_str}{flag}")
            ans = input("  keep this take? [Enter=keep, r=discard & redo]: ").strip().lower()
            if ans == "r":
                print("  discarded, redoing this take (nothing written).")
                continue
            for idx, (ms, d) in enumerate(samples):
                out.write(f"{take_id},{g},{idx},{ms},{d[0]},{d[1]},{d[2]},{d[3]}\n")
            take_id += 1
            take += 1
            counts[g] += 1

    out.close()
    print("\nDone.")
    _print_summary(args.out)
    return 0


def _print_summary(path: str):
    if not os.path.exists(path):
        return
    per_label = {}
    with open(path) as f:
        f.readline()
        seen = set()
        for line in f:
            p = line.split(",")
            if len(p) < 2:
                continue
            tid, label = p[0], p[1]
            key = (tid, label)
            if key in seen:
                continue
            seen.add(key)
            per_label[label] = per_label.get(label, 0) + 1
    print("Dataset totals (takes per gesture):")
    for label, n in sorted(per_label.items()):
        print(f"  {label:12s} {n}")
    print(f"Saved to {path}")


if __name__ == "__main__":
    sys.exit(main())
