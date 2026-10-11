"""
5-fold cross-validation: fine-tune ResNet50 on HAM10000 using HAM10000Dataset (dataset.py).

Folds are stratified by diagnosis (dx) and grouped by lesion_id, so images of the
same lesion never appear in both the training and validation side of a fold.

Usage:
    python train.py --csv HAM10000_metadata.csv --img_dir path/to/all_images
    python train.py --csv ... --img_dir ... --only_fold 0     # run a single fold (quick test)
"""
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from torchvision import models, transforms

from dataset import HAM10000Dataset


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="HAM10000_metadata.csv")
    p.add_argument("--img_dir", required=True, help="folder containing all .jpg images")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--only_fold", type=int, default=None, help="run just this fold index (0-based)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_prefix", default="resnet50_ham10000", help="checkpoints: <prefix>_fold<k>.pt")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms():
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        norm,
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm,
    ])
    return train_tf, val_tf


def run_fold(fold, train_df, val_df, classes, args, device):
    """Train a fresh ResNet50 on one fold. Returns metrics and predictions at the best epoch."""
    train_tf, val_tf = get_transforms()
    train_ds = HAM10000Dataset(train_df, args.img_dir, train_tf)
    val_ds = HAM10000Dataset(val_df, args.img_dir, val_tf)
    assert train_ds.classes == classes and val_ds.classes == classes, "class mapping mismatch"

    use_cuda = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True,
                          persistent_workers=args.workers > 0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0)

    # Fresh model for every fold (no information carries over between folds)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    # Class-weighted loss, computed from this fold's training data only
    counts = train_df["dx"].value_counts().reindex(classes).values.astype(np.float32)
    weights = counts.sum() / (len(classes) * counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    best = {"bal_acc": -1.0}
    ckpt_path = f"{args.out_prefix}_fold{fold}.pt"

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        running = 0.0
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_cuda):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
        scheduler.step()

        # ---- Validate ----
        model.eval()
        preds, labels = [], []
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=use_cuda):
            for x, y in val_dl:
                out = model(x.to(device, non_blocking=True))
                preds.append(out.argmax(1).cpu())
                labels.append(y)
        preds = torch.cat(preds).numpy()
        labels = torch.cat(labels).numpy()

        acc = float((preds == labels).mean())
        bal_acc = float(balanced_accuracy_score(labels, preds))
        f1 = float(f1_score(labels, preds, average="macro"))
        print(f"[Fold {fold}] Epoch {epoch:02d}/{args.epochs} | loss {running / len(train_ds):.4f} | "
              f"acc {acc:.3f} | bal-acc {bal_acc:.3f} | macro-F1 {f1:.3f}")

        if bal_acc > best["bal_acc"]:
            best = {"epoch": epoch, "acc": acc, "bal_acc": bal_acc, "f1": f1,
                    "preds": preds, "labels": labels}
            torch.save({"model": model.state_dict(), "classes": classes, "fold": fold}, ckpt_path)

    print(f"[Fold {fold}] best epoch {best['epoch']}: acc {best['acc']:.3f} | "
          f"bal-acc {best['bal_acc']:.3f} | macro-F1 {best['f1']:.3f}  -> {ckpt_path}")

    # Free GPU memory before the next fold
    del model, optimizer, scaler, train_dl, val_dl
    if use_cuda:
        torch.cuda.empty_cache()
    return best


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    torch.backends.cudnn.benchmark = True

    df = pd.read_csv(args.csv)
    classes = sorted(df["dx"].unique())
    print("Classes:", classes)

    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(skf.split(df, df["dx"], groups=df["lesion_id"]))

    results, oof_rows = [], []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        if args.only_fold is not None and fold != args.only_fold:
            continue
        print(f"\n{'=' * 20} Fold {fold + 1}/{args.folds} "
              f"(train {len(tr_idx)}, val {len(va_idx)}) {'=' * 20}")
        set_seed(args.seed + fold)
        train_df, val_df = df.iloc[tr_idx], df.iloc[va_idx]

        best = run_fold(fold, train_df, val_df, classes, args, device)
        results.append({"fold": fold, "best_epoch": best["epoch"], "acc": best["acc"],
                        "bal_acc": best["bal_acc"], "macro_f1": best["f1"]})

        # Out-of-fold predictions (val_df order matches best["preds"])
        oof_rows.append(pd.DataFrame({
            "image_id": val_df["image_id"].values,
            "fold": fold,
            "true": [classes[i] for i in best["labels"]],
            "pred": [classes[i] for i in best["preds"]],
        }))

    # ---- Summary across folds ----
    res = pd.DataFrame(results)
    res.to_csv(f"{args.out_prefix}_cv_results.csv", index=False)
    print("\n" + "=" * 25 + " Cross-validation summary " + "=" * 25)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for m in ["acc", "bal_acc", "macro_f1"]:
        print(f"{m:>9}: {res[m].mean():.3f} ± {res[m].std(ddof=1) if len(res) > 1 else 0:.3f}")

    # ---- Per-class report on pooled out-of-fold predictions ----
    oof = pd.concat(oof_rows, ignore_index=True)
    oof.to_csv(f"{args.out_prefix}_oof_predictions.csv", index=False)
    print("\nPer-class report (pooled out-of-fold predictions):")
    print(classification_report(oof["true"], oof["pred"], labels=classes, digits=3, zero_division=0))


if __name__ == "__main__":  # required on Windows for DataLoader workers
    main()