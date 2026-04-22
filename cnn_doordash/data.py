"""Data loading, image caching, Dataset, transforms, and split helpers."""

from __future__ import annotations

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

IMAGE_COL = "menuItemImageUrl"
PRICE_COL = "menuItemCurrentPrice"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224
RESIZE_TO = 256

DEFAULT_CSV = "restaurantmenuchanges.csv"
DEFAULT_IMAGE_DIR = "images"


def load_and_clean(csv_path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    """Read the CSV and return one row per image URL with a sane price."""
    df = pd.read_csv(csv_path, escapechar="\\")
    df = df.dropna(subset=[IMAGE_COL, PRICE_COL]).copy()
    df = df[df[IMAGE_COL].str.strip().astype(bool)]
    df[PRICE_COL] = (
        df[PRICE_COL].replace(r"[\$,]", "", regex=True).astype(float)
    )

    # One row per image URL; use median price to shrug off promo/$0 noise.
    other_cols = [c for c in df.columns if c not in (IMAGE_COL, PRICE_COL)]
    df = (
        df.groupby(IMAGE_COL, as_index=False)
        .agg({PRICE_COL: "median", **{c: "first" for c in other_cols}})
        .reset_index(drop=True)
    )

    # Drop non-positive or absurdly-priced items so they don't warp the loss.
    df = df[(df[PRICE_COL] > 0) & (df[PRICE_COL] < 150)].reset_index(drop=True)
    return df


def _url_to_path(url: str, image_dir: Path) -> Path:
    return image_dir / (hashlib.md5(url.encode("utf-8")).hexdigest() + ".jpg")


def _download_one(url: str, image_dir: Path, timeout: float) -> tuple[str, Path | None]:
    path = _url_to_path(url, image_dir)
    if path.exists() and path.stat().st_size > 0:
        return url, path
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        Image.open(io.BytesIO(resp.content)).convert("RGB").save(
            path, format="JPEG", quality=90
        )
        return url, path
    except (requests.RequestException, UnidentifiedImageError, OSError):
        return url, None


def cache_images(
    df: pd.DataFrame,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    *,
    max_workers: int = 16,
    timeout: float = 10.0,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Download every URL in `df[IMAGE_COL]` into `image_dir`.

    Returns a new DataFrame restricted to rows whose image we have on disk,
    with an added `imagePath` column pointing at that file.
    """
    image_dir = Path(image_dir)
    image_dir.mkdir(exist_ok=True)

    urls = df[IMAGE_COL].tolist()
    results: dict[str, Path | None] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_download_one, u, image_dir, timeout) for u in urls]
        it = as_completed(futures)
        if show_progress:
            it = tqdm(it, total=len(futures), desc="downloading")
        for fut in it:
            url, path = fut.result()
            results[url] = path

    df = df.copy()
    df["imagePath"] = df[IMAGE_COL].map(lambda u: results.get(u))
    df = df.dropna(subset=["imagePath"]).reset_index(drop=True)
    df["imagePath"] = df["imagePath"].astype(str)
    return df


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize(RESIZE_TO),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(RESIZE_TO),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def denormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (img_tensor.cpu() * std + mean).clamp(0, 1)


class MenuItemImageDataset(Dataset):
    """Yields (image_tensor, price_tensor) pairs for a CNN regression model."""

    def __init__(
        self,
        frame: pd.DataFrame,
        transform: transforms.Compose,
        price_column: str = PRICE_COL,
    ) -> None:
        self.paths = frame["imagePath"].tolist()
        self.prices = frame[price_column].astype("float32").tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        price = torch.tensor(self.prices[idx], dtype=torch.float32)
        return img, price


def stratified_split(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15 (by default) split stratified on price quintiles."""
    bins = pd.qcut(df[PRICE_COL], q=5, labels=False, duplicates="drop")

    tmp_frac = val_size + test_size
    train_df, temp_df, _, temp_bins = train_test_split(
        df, bins, test_size=tmp_frac, random_state=seed, stratify=bins
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / tmp_frac,
        random_state=seed,
        stratify=temp_bins,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_tf, eval_tf = build_transforms()
    train_ds = MenuItemImageDataset(train_df, train_tf)
    val_ds = MenuItemImageDataset(val_df, eval_tf)
    test_ds = MenuItemImageDataset(test_df, eval_tf)

    loader_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
