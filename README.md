# cnn-doordash

A from-scratch convolutional neural network (PyTorch) that predicts the
**price in dollars** of a restaurant menu item from a single photo of the
food. Trained on the public DoorDash menu-changes dataset.

The motivating use case: give a restaurant operator instant feedback on how
"premium" a candidate menu photo looks, so they can pick the most appetizing
shot before publishing it.

> **Task type:** image regression (one scalar output, dollars).
> **Backbone:** custom 5-block CNN (~4.8M parameters, no pretraining).
> **Result:** ~**$4.59 test MAE** vs **$5.88 naive baseline** after just 3 epochs on a M-series Mac (MPS).

---

## Repository layout

```
cnn-doordash/
├── cnn.ipynb                  # exploratory notebook (EDA + sanity checks)
├── train.py                   # CLI entry point: trains and evaluates the CNN
├── cnn_doordash/
│   ├── __init__.py
│   ├── data.py                # CSV cleaning, image cache, Dataset, transforms, splits
│   └── model.py               # PriceCNN architecture + device picker
├── restaurantmenuchanges.csv  # raw data (5,000 menu-change events)
├── images/                    # local image cache (gitignored, populated on first run)
├── runs/<name>/
│   ├── best.pt                # best-on-val checkpoint + training history
│   └── training_curves.png    # loss + MAE curves
├── pyproject.toml             # uv-managed dependencies
└── uv.lock
```

---

## Quickstart

The project uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
# install / sync dependencies into .venv
uv sync

# train with sensible defaults (15 epochs, batch size 32, AdamW + cosine LR)
uv run python train.py

# quick smoke test
uv run python train.py --epochs 3

# see all knobs
uv run python train.py --help
```

The first run downloads ~2,000 menu-item images into `images/` (cached, so
subsequent runs are instant). Training writes `runs/default/best.pt` and
`runs/default/training_curves.png`.

To work in the notebook instead, the kernel `Python (cnn-doordash)` is
already registered against `.venv/bin/python` — just open `cnn.ipynb` and
pick that kernel.

---

## Data processing & augmentation

### Source data

`restaurantmenuchanges.csv` is a 5,000-row log of DoorDash menu-change events
(create, update, delete) covering items in several US cities. Each row has
~20 columns; we use only:


| column                 | role                            |
| ---------------------- | ------------------------------- |
| `menuItemImageUrl`     | input — points at a hosted JPEG |
| `menuItemCurrentPrice` | target — string like `"$2.19"`  |


### Cleaning pipeline (`cnn_doordash.data.load_and_clean`)

1. Drop rows with no image URL or no price.
2. Strip the `$` and `,`, cast to `float`.
3. **Dedupe by image URL using the median price per URL.** Many items
  appear repeatedly in the changelog at slightly different prices; the
   median is robust to promotional `$0` rows and one-off typos.
4. Drop rows priced ≤ `$0` or ≥ `$150` (data glitches and outliers that
  would dominate a regression loss).

After cleaning we keep ~**1,975 unique image/price pairs**.

### Image cache (`cnn_doordash.data.cache_images`)

- Filename = `md5(url).hex + ".jpg"` so the same URL always maps to the same
file (idempotent re-runs).
- 16-thread `ThreadPoolExecutor` with a 10-second timeout per request.
- Re-encoded to JPEG (`quality=90`, `RGB` mode) so every cached file has a
known format and color space.
- Failures (404s, timeouts, corrupted bytes) are silently dropped from the
returned DataFrame. ~3% of rows fail to download.

### Splits (`cnn_doordash.data.stratified_split`)

- **70% train / 15% val / 15% test**, fixed seed (`42`).
- Stratified on price **quintiles** so all three splits have a similar price
distribution. This matters for regression: an unlucky split could put
every >$50 item in the test set and make the model look much worse than
it is.
- Dedupe-by-URL upstream guarantees **no image leakage** across splits.

### Transforms (`cnn_doordash.data.build_transforms`)


| stage               | transforms                                                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **train**           | `Resize(256)` → `RandomCrop(224)` → `RandomHorizontalFlip()` → `ColorJitter(0.2, 0.2, 0.2)` → `ToTensor()` → `Normalize(ImageNet mean/std)` |
| **eval (val/test)** | `Resize(256)` → `CenterCrop(224)` → `ToTensor()` → `Normalize(ImageNet mean/std)`                                                           |


- **Input size 224×224** matches the standard `torchvision` pretrained
backbones, so we can swap in ResNet/EfficientNet later with no other
changes.
- **ImageNet normalization** for the same reason.
- **Augmentation** (random crop / flip / mild color jitter) is intentionally
conservative — extreme distortion would change the visual cues that
correlate with price (portion size, plating, garnishes).

---

## Network architecture

`cnn_doordash.model.PriceCNN` — ~**4.78M trainable parameters**.

### Layer sequence

A "conv block" is `Conv3×3 → BN → ReLU → Conv3×3 → BN → ReLU → MaxPool2×2`.
Five blocks halve the spatial size and double the channel count:


| stage   | op                                                                                 | output shape (C × H × W) |
| ------- | ---------------------------------------------------------------------------------- | ------------------------ |
| input   | image                                                                              | 3 × 224 × 224            |
| block 1 | conv block, 32 channels                                                            | 32 × 112 × 112           |
| block 2 | conv block, 64 channels                                                            | 64 × 56 × 56             |
| block 3 | conv block, 128 channels                                                           | 128 × 28 × 28            |
| block 4 | conv block, 256 channels                                                           | 256 × 14 × 14            |
| block 5 | conv block, 512 channels                                                           | 512 × 7 × 7              |
| pool    | `AdaptiveAvgPool2d(1)`                                                             | 512 × 1 × 1              |
| head    | `Flatten → Dropout(0.3) → Linear(512, 128) → ReLU → Dropout(0.3) → Linear(128, 1)` | (B,)                     |


### Architecture diagram

```
 RGB image                                Per-block sub-graph
 (3, 224, 224)                            ──────────────────────────
       │                                  ┌──────────────────────┐
       ▼                                  │ Conv3×3 (no bias)    │
 ┌─────────────┐                          │ BatchNorm2d          │
 │  block 1    │  → 32 × 112 × 112        │ ReLU(inplace)        │
 ├─────────────┤                          │ Conv3×3 (no bias)    │
 │  block 2    │  → 64 × 56 × 56          │ BatchNorm2d          │
 ├─────────────┤                          │ ReLU(inplace)        │
 │  block 3    │  → 128 × 28 × 28         │ MaxPool2d(2)         │
 ├─────────────┤                          └──────────────────────┘
 │  block 4    │  → 256 × 14 × 14
 ├─────────────┤
 │  block 5    │  → 512 × 7 × 7
 └─────────────┘
       │
       ▼
 AdaptiveAvgPool2d(1)  → 512
       │
       ▼
 Dropout(0.3) → Linear(512, 128) → ReLU → Dropout(0.3) → Linear(128, 1)
       │
       ▼
   price ($)
```

### Hyperparameters per layer

- **All convs:** `kernel_size=3`, `padding=1`, `bias=False` (BN absorbs the
bias term). Stride 1 — spatial downsampling happens only at the
`MaxPool2d(2)` at the end of each block.
- **All BNs:** default momentum `0.1`, default eps `1e-5`.
- **Activations:** `ReLU(inplace=True)` everywhere — `inplace=True` saves
memory because the gradient flowing back doesn't need the pre-activation
values.
- **Pooling:** `MaxPool2d(2)` (5×) and `AdaptiveAvgPool2d(1)` (1×). Adaptive
pooling means the head doesn't care about input resolution — feeding 256×256
images at inference time would still work.
- **Dropout:** 0.3 on both fully-connected layers in the head, off in the
convolutional trunk (BatchNorm provides enough regularization there).

### Weight initialization

Done explicitly in `PriceCNN._init_weights()`:

- `nn.Conv2d` → Kaiming normal, `mode="fan_out"`, `nonlinearity="relu"`
- `nn.BatchNorm2d` → weight = 1, bias = 0
- `nn.Linear` → Kaiming normal, `nonlinearity="relu"`, bias = 0

This was added after the default PyTorch init produced occasional
`O(10⁶)`-magnitude predictions on the first batch, which then poisoned the
mean MAE for an entire epoch. Kaiming init keeps activation magnitudes
stable through all five blocks.

---

## Training procedure

### Loss function

`nn.HuberLoss(delta=1.0)`. Behaves like MSE near zero (smooth gradients,
faster convergence) but like MAE in the tails (a single $30-off prediction
no longer dominates the gradient). Empirically more stable than plain MSE
on this long-tailed price distribution.

### Optimizer

`AdamW(lr=1e-3, weight_decay=1e-4)`. AdamW decouples weight decay from the
adaptive moment estimates, which is the recommended default for
modern training of CNNs.

### Learning-rate schedule

`CosineAnnealingLR(T_max=epochs)` — LR follows half a cosine from `1e-3` at
epoch 0 down to `0` at the final epoch. Smoother than step decay and
removes one hyperparameter (no decay milestones to tune).

### Regularization

- `weight_decay=1e-4` (via AdamW).
- `Dropout(p=0.3)` on both head FC layers.
- `BatchNorm2d` after every conv (acts as implicit regularizer).
- `RandomCrop`, `RandomHorizontalFlip`, `ColorJitter` augmentation only on
the training split.
- **Gradient clipping:** `clip_grad_norm_(max_norm=1.0)` after every
backward pass — prevents one pathological batch from flinging weights
to infinity.
- **Early stopping (snapshot variant):** we don't *halt* training early,
but we deep-copy the model state every time val MAE improves and restore
the best snapshot before evaluating on the test set.

### Defensive checks

`run_epoch` raises `RuntimeError` immediately if the model produces a
non-finite prediction. This makes silent NaN propagation impossible and
catches numerical issues (e.g. MPS quirks) on the very first bad batch.

### Batch size, epochs, and runtime


| setting            | default | flag                 |
| ------------------ | ------- | -------------------- |
| epochs             | 15      | `--epochs`           |
| batch size         | 32      | `--batch-size`       |
| learning rate      | `1e-3`  | `--lr`               |
| weight decay       | `1e-4`  | `--weight-decay`     |
| dropout            | 0.3     | `--dropout`          |
| grad clip          | 1.0     | `--grad-clip`        |
| DataLoader workers | 4       | `--num-workers`      |
| download workers   | 16      | `--download-workers` |
| seed               | 42      | `--seed`             |


On an M-series Mac (`mps` device), one epoch over the ~1,400 training
images takes ~40 seconds, so the full 15-epoch run finishes in roughly
**10 minutes**. On CPU it's roughly 4× slower; on a CUDA GPU, 5× faster.

---

## Performance evaluation

### Metrics

- **Huber loss (delta=1.0)** — what we minimize. Reported per-epoch.
- **MAE in dollars** — the human-interpretable metric. Reported per-epoch
and on the test set.
- **Naive baseline:** always predict `train_df["menuItemCurrentPrice"].mean()`.
The CNN has to beat this MAE to be useful at all.

### Result snapshot (3-epoch smoke test, seed 42, MPS)

```
epoch 01/3  train MAE $5.40   val MAE $5.83
epoch 02/3  train MAE $5.01   val MAE $4.92
epoch 03/3  train MAE $4.81   val MAE $4.77
Test MAE: $4.59
Naive baseline (predict $10.49): MAE $5.88
```

- Train and val MAE stay within ~$1 of each other → no obvious overfitting
yet at 3 epochs.
- CNN beats the naive baseline by **$1.29 / item** on held-out test data.
- `runs/default/training_curves.png` plots both Huber loss and MAE for both
splits.

### Why no confusion matrix

Confusion matrices are for classification. This is a continuous regression
problem (price in dollars), so we use:

- **Predicted-vs-true scatter plot** with the `y = x` line (in `cnn.ipynb`,
test-set evaluation cell).
- **MAE in dollars** as the headline metric.
- **Naive baseline** for sanity.

### Visualization (planned)

- Saliency maps and Grad-CAM overlays to see *where* on the plate the
model looks when predicting price. Implementation hook: easy to add via
`torch.autograd.grad` on the input tensor against the model output, or
via the `pytorch-grad-cam` package targeting the final conv block
(`model.features[4]`).

---

## Model deployment details

### Weights

Saved to `runs/<out-dir>/best.pt` after every training run. The checkpoint
is a dict with:


| key          | value                                              |
| ------------ | -------------------------------------------------- |
| `state_dict` | `model.state_dict()` from the best-on-val epoch    |
| `val_mae`    | best validation MAE (dollars)                      |
| `test_mae`   | test MAE on the restored best weights              |
| `history`    | per-epoch train/val loss + MAE lists               |
| `args`       | the CLI args used for the run, for reproducibility |


### Inference code

```python
from PIL import Image
import torch

from cnn_doordash.data import build_transforms
from cnn_doordash.model import PriceCNN, pick_device

device = pick_device()
model = PriceCNN().to(device).eval()
ckpt = torch.load("runs/default/best.pt", map_location=device)
model.load_state_dict(ckpt["state_dict"])

_, eval_tf = build_transforms()

def predict_price(image_path: str) -> float:
    img = eval_tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        return float(model(img).item())

print(predict_price("images/some_dish.jpg"))
```

---

## Roadmap

- **Pretrained backbone.** Swap `PriceCNN` for `torchvision.models.resnet18(weights="DEFAULT")`
with `model.fc = nn.Linear(512, 1)`. Should cut MAE substantially with no
other pipeline changes — input size, normalization, and
Dataset/DataLoader API are already compatible.
- **Log-target training.** Predict `log1p(price)` and exponentiate at
inference. Better-behaved gradients on the long tail of expensive items.
- **Per-restaurant features.** Concatenate restaurant metadata
(`market`, `restaurantPriceRange`, `restaurantAverageRating`) into the
MLP head as a multi-modal model.
- **Grad-CAM cell** in the notebook so we can see what the CNN actually
pays attention to.

---

## Installation (full)

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
# 1. clone
git clone <this-repo>
cd cnn-doordash

# 2. install deps into .venv (reads pyproject.toml + uv.lock)
uv sync

# 3. (optional) register a Jupyter kernel for the notebook
uv run python -m ipykernel install --user --name cnn-doordash --display-name "Python (cnn-doordash)"

# 4. train
uv run python train.py
```

### Adding / updating dependencies

```bash
uv add <pkg>           # add a new dependency, updates pyproject.toml + uv.lock
uv remove <pkg>        # remove one
uv sync                # reinstall exactly what's in uv.lock (CI / fresh clones)
uv run <cmd>           # run any command inside the project venv
```

