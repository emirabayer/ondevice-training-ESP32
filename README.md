# INT8 on-device training on ESP32-S3

A small neural-network training library written in C++17 that runs inside a
fixed 256 KB buffer with no dynamic allocation, plus a demo that trains a
touch-gesture classifier directly on an ESP32-S3.

I wrote the training engine first and tested it on a laptop, then used it to
train a gesture recognizer on data I recorded from the chip's own capacitive
touch pins. All the device numbers below were printed by the firmware running
on the board, not simulated.

## Results

Board: YD-ESP32-S3 (N16R8). Same task, same 100 recorded examples, trained two
ways on the device:

| | FP32, whole network | INT8 + sparse (head only) |
|---|---|---|
| Test accuracy | 95.0% (19/20) | 90.0% (18/20) |
| Time per epoch | 45.1 ms | 8.1 ms |
| Total, 800 epochs | 36.1 s | 6.5 s |
| Trainable state (params + grads + Adam) | 35.1 KB, 2245 params | 2.8 KB, 165 params |
| Peak RAM while training | 35.1 KB | 29.3 KB (11.5% of the arena) |

The test set is 20 examples, 4 per class, so those two numbers are 19/20 and
18/20, one example apart. That gap is noise at this size. Running the same two
models over 10 random train/test splits on the host, the full FP32 network
averages 93.5% (range 85 to 100) and the int8 + sparse head averages 93.0%
(range 85 to 100), so on this dataset their accuracy is the same within the
spread. The differences that hold up are speed and trainable memory, not
accuracy. `tools/accuracy_variance.py` reproduces the split study.

The INT8 + sparse run trains about 5.6x faster. Two separate things cause that,
and it is worth keeping them apart:

* The memory drop comes from training only the classifier head (165 params)
  instead of the whole network (2245). It is not from quantization. The engine
  keeps a float shadow of each trained weight plus its int8 copy, so int8
  training actually costs a little more per parameter (17 bytes vs 16).
* The speed comes from both the int8 integer matmul and from only updating the
  head, over features from the frozen layer that are computed once and reused.

I did not benchmark against TFLite Micro. It is an inference runtime and cannot
train on the device, so it is not a baseline for on-device training.

## Hardware

A YD-ESP32-S3 dev board (16 MB flash, 8 MB PSRAM). Four of the capacitive touch
pins (GPIO 7, 8, 9, 12) act as a gesture pad, touched directly with a finger.
I picked those four after probing all 14 touch channels on the board and
finding which ones were exposed and responsive (GPIO5 is dead on my unit).

Five gestures: swipe left, swipe right, hold, two-finger tap, and idle.

## How it works

The engine (`include/`, `src/`):

* `arena.hpp` / `arena.cpp`: one fixed `uint8_t[262144]` buffer with a bump
  pointer and mark/rewind for scratch. It aborts if an allocation would go over
  256 KB and tracks a high-water mark. Every weight, gradient, optimizer value
  and activation buffer comes from here.
* `tensor.hpp`: tensor views, int8 quantization, and the int8 matmul kernel.
  Requantization uses a Q31 fixed-point multiplier and a shift, so there is no
  floating point in the inner loop.
* `layers.hpp` / `layers.cpp`: a dense layer (int8 forward, float-shadow
  backward using a straight-through estimator), int8 ReLU, and a fused
  softmax + cross-entropy backward. A frozen layer allocates no gradient or
  shadow buffers.
* `qas.hpp`: picks each layer's output scale from the range of activations it
  actually produces.
* `optimizer.hpp` / `optimizer.cpp`: SGD (with momentum) and Adam over the
  float shadow weights, and a selector that marks which layers get trained.

The application:

1. `firmware_gesture_capture/` streams the four touch channels over serial at
   50 Hz.
2. `tools/collect_gestures.py` records labeled windows into `data/gestures.csv`.
3. `tools/prepare_dataset.py` downsamples and normalizes the windows, splits
   train/test, checks the task is learnable with a small numpy MLP, and writes
   `include/gesture_dataset.h` (the data plus a feature-extractor layer trained
   on the laptop).
4. `firmware_gesture_train_fp32/` trains the whole 64->32->5 network from
   scratch in float32 on the board.
5. `firmware_gesture_train_int8/` loads the frozen extractor, computes its int8
   features once, and trains only the int8 head with the engine.

## Layout

```
include/                       engine headers + generated gesture_dataset.h
src/                           engine code + a synthetic-data demo (main.cpp)
tests/                         host unit tests
tools/                         data collection, dataset prep, host check
firmware/                      synthetic-data training demo (ESP-IDF)
firmware_gesture_capture/      touch capture firmware
firmware_gesture_train_fp32/   float32 trainer
firmware_gesture_train_int8/   int8 + sparse trainer
data/gestures.csv              the recorded dataset
```

## Building and running

Host build and tests (needs CMake and a C++17 compiler):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
./build/train
```

On the board (ESP-IDF v5.x, Python 3.10 to 3.13):

```sh
. $IDF_PATH/export.sh

cd firmware_gesture_capture && idf.py set-target esp32s3 && idf.py -p /dev/ttyACM0 flash
cd .. && python tools/collect_gestures.py --takes 20
python3 tools/prepare_dataset.py

cd firmware_gesture_train_int8 && idf.py set-target esp32s3 && idf.py -p /dev/ttyACM0 flash monitor
```

The CH343 USB-UART on this board shows up as `/dev/ttyACM0`.

## Notes and limitations

* The dataset is 100 examples from one person, split 80 train / 20 test (4 per
  class). A 20-example test set is small, so single-split accuracy is noisy;
  see the split study in the Results section.
* The int8 path fine-tunes: the feature extractor is trained on the laptop and
  frozen, and only the head is trained on the board. That is the realistic case
  for adapting a model on a device, not training a whole network from scratch
  in int8.
* Swipe left vs swipe right is the hardest pair, since the direction is in the
  order the channels light up rather than in their peak values. It is where
  almost all of the test error is in both models.

## Constraints

No `malloc`, `new`, `std::vector` or `std::string` in the tensor, math or
training code. A hard 256 KB budget. int8 weights and activations with int32
accumulators, float only for the shadow weights and scales. No external matrix
library.
