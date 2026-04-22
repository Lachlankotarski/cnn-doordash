"""Train the menu-item price CNN.

Run from the project root:

    uv run python train.py

Every run starts from fresh weights, so re-running is always a clean restart.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from cnn_doordash.data import (
    DEFAULT_CSV,
    DEFAULT_IMAGE_DIR,
    PRICE_COL,
    build_dataloaders,
    cache_images,
    load_and_clean,
    stratified_split,
)
from cnn_doordash.model import PriceCNN, count_parameters, pick_device


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: optim.Optimizer | None = None,
    grad_clip: float | None = 1.0,
) -> tuple[float, float]:
    """One pass through `loader`.

    Set `optimizer=None` for eval. Returns (avg_huber_loss, avg_MAE_dollars).
    """
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    total_abs_err = 0.0
    total_n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, prices in loader:
            images = images.to(device, non_blocking=True)
            prices = prices.to(device, non_blocking=True)

            preds = model(images)

            if not torch.isfinite(preds).all():
                raise RuntimeError(
                    "Model produced non-finite predictions. "
                    f"min={preds.min().item()} max={preds.max().item()}"
                )

            loss = criterion(preds, prices)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            batch_n = prices.size(0)
            total_loss += loss.item() * batch_n
            total_abs_err += (preds - prices).abs().sum().item()
            total_n += batch_n

    return total_loss / total_n, total_abs_err / total_n


def train(args: argparse.Namespace) -> None:
    device = pick_device()
    print(f"Using device: {device}")

    df = load_and_clean(args.csv)
    print(f"Cleaned dataset: {len(df)} rows")

    df = cache_images(df, args.image_dir, max_workers=args.download_workers)
    print(f"Rows with image on disk: {len(df)}")

    train_df, val_df, test_df = stratified_split(df, seed=args.seed)
    print(
        f"Splits -> train {len(train_df)} | val {len(val_df)} | test {len(test_df)}"
    )
    for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
        p = d[PRICE_COL]
        print(
            f"  {name:5s}  mean ${p.mean():5.2f}  std ${p.std():5.2f}  "
            f"min ${p.min():5.2f}  max ${p.max():6.2f}"
        )

    pin = device.type == "cuda"
    train_loader, val_loader, test_loader = build_dataloaders(
        train_df,
        val_df,
        test_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    model = PriceCNN(dropout=args.dropout).to(device)
    n_params, n_trainable = count_parameters(model)
    print(f"Model params: {n_params:,} total  ({n_trainable:,} trainable)")

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "val_loss": [], "train_mae": [], "val_mae": []}
    best_val_mae = float("inf")
    best_state: dict | None = None

    print()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_mae = run_epoch(
            model, train_loader, criterion, device,
            optimizer=optimizer, grad_clip=args.grad_clip,
        )
        va_loss, va_mae = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_mae"].append(tr_mae)
        history["val_mae"].append(va_mae)

        improved = va_mae < best_val_mae
        if improved:
            best_val_mae = va_mae
            best_state = copy.deepcopy(model.state_dict())

        dt = time.time() - t0
        flag = "  <- best" if improved else ""
        print(
            f"epoch {epoch:02d}/{args.epochs}  "
            f"train loss {tr_loss:7.4f} / MAE ${tr_mae:5.2f}   "
            f"val loss {va_loss:7.4f} / MAE ${va_mae:5.2f}   "
            f"({dt:4.1f}s){flag}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nRestored best-on-val weights. Best val MAE: ${best_val_mae:.2f}")

    test_loss, test_mae = run_epoch(model, test_loader, criterion, device)
    train_mean = float(train_df[PRICE_COL].mean())
    naive_mae = (test_df[PRICE_COL] - train_mean).abs().mean()
    print(f"Test Huber loss: {test_loss:.4f}")
    print(f"Test MAE:        ${test_mae:.2f}")
    print(f"Naive baseline (predict ${train_mean:.2f}): MAE ${naive_mae:.2f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "val_mae": best_val_mae,
            "test_mae": test_mae,
            "history": history,
            "args": vars(args),
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(history["train_loss"], label="train")
    ax1.plot(history["val_loss"], label="val")
    ax1.set_title("Huber loss"); ax1.set_xlabel("epoch"); ax1.legend()
    ax2.plot(history["train_mae"], label="train")
    ax2.plot(history["val_mae"], label="val")
    ax2.set_title("MAE ($)"); ax2.set_xlabel("epoch"); ax2.legend()
    plt.tight_layout()
    fig_path = out_dir / "training_curves.png"
    fig.savefig(fig_path, dpi=120)
    print(f"Saved training curves to {fig_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=DEFAULT_CSV)
    p.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    p.add_argument("--out-dir", default="runs/default")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--download-workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
