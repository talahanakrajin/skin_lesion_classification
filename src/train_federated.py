"""
Federated learning simulation: fine-tune ResNet50 on HAM10000 across simulated clinics.

Replaces the 5-fold CV of train.py with a federated setup:

  * The data is split by lesion_id (no lesion ever spans two splits/clients) into
        - a global TEST set  (server side, never used for training or model selection)
        - a global VAL set   (server side, used to pick the best round)
        - a TRAIN pool       (partitioned among the simulated clients/clinics)
  * The train pool is partitioned across N clients. With --alpha > 0 the partition is
    non-IID (Dirichlet label skew: small alpha = very different clinics);
    with --alpha 0 it is an IID random split.
  * Each round: the server samples a fraction of clients, each trains locally on its own
    images, and the server aggregates their weights (FedAvg, or FedProx with --mu > 0).
    Raw images never "leave" a client; only model weights are exchanged.

Usage:
    python train_federated.py --csv HAM10000_metadata.csv --img_dir path/to/all_images
    python train_federated.py --csv ... --img_dir ... --alpha 0          # IID baseline
    python train_federated.py --csv ... --img_dir ... --alpha 0.3 --mu 0.01   # non-IID + FedProx
    python train_federated.py --csv ... --img_dir ... --rounds 2 --clients 3  # quick smoke test
"""
import argparse
import math
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
    # federation
    p.add_argument("--clients", type=int, default=5, help="number of simulated clinics")
    p.add_argument("--rounds", type=int, default=30, help="communication rounds")
    p.add_argument("--local_epochs", type=int, default=2, help="local epochs per round")
    p.add_argument("--frac", type=float, default=1.0, help="fraction of clients sampled per round")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Dirichlet concentration for label skew (<=0 means IID split)")
    p.add_argument("--mu", type=float, default=0.0,
                   help="FedProx proximal coefficient (0 = plain FedAvg)")
    p.add_argument("--min_client_size", type=int, default=100,
                   help="minimum images per client when partitioning")
    # local training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_prefix", default="fl_resnet50_ham10000",
                   help="outputs: <prefix>_best.pt, <prefix>_history.csv, ...")
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


# --------------------------------------------------------------------------- #
# Data splitting / partitioning
# --------------------------------------------------------------------------- #
def split_server_and_pool(df, seed):
    """Lesion-grouped, diagnosis-stratified split into test (10%), val (10%), train pool (80%)."""
    skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = [va for _, va in skf.split(df, df["dx"], groups=df["lesion_id"])]
    test_df = df.iloc[folds[0]]
    val_df = df.iloc[folds[1]]
    pool_df = df.iloc[np.concatenate(folds[2:])]
    return pool_df, val_df, test_df


def partition_clients(pool_df, n_clients, alpha, seed, min_size):
    """
    Assign whole lesions to clients so no lesion is shared between clients.
    alpha <= 0 : IID random split.
    alpha  > 0 : per-diagnosis Dirichlet(alpha) split -> label skew across clients.
    Returns a list of DataFrames (one per client).
    """
    rng = np.random.default_rng(seed)
    lesions = pool_df.drop_duplicates("lesion_id")[["lesion_id", "dx"]]

    if alpha <= 0:
        ids = lesions["lesion_id"].values.copy()
        rng.shuffle(ids)
        parts = [list(p) for p in np.array_split(ids, n_clients)]
    else:
        for _ in range(200):
            parts = [[] for _ in range(n_clients)]
            for _, g in lesions.groupby("dx"):
                ids = g["lesion_id"].values.copy()
                rng.shuffle(ids)
                props = rng.dirichlet(alpha * np.ones(n_clients))
                cuts = (np.cumsum(props) * len(ids)).astype(int)[:-1]
                for k, chunk in enumerate(np.split(ids, cuts)):
                    parts[k].extend(chunk.tolist())
            sizes = [pool_df["lesion_id"].isin(p).sum() for p in parts]
            if min(sizes) >= min_size:
                break
        else:
            raise RuntimeError("Could not build a partition with every client >= min_client_size. "
                               "Increase --alpha, lower --clients, or lower --min_client_size.")

    return [pool_df[pool_df["lesion_id"].isin(set(p))] for p in parts]


# --------------------------------------------------------------------------- #
# Client side
# --------------------------------------------------------------------------- #
def local_train(model, global_params, ds, df, classes, args, device, lr):
    """Train `model` (already loaded with the global weights) on one client's dataset."""
    use_cuda = device.type == "cuda"
    drop_last = len(ds) >= args.batch_size
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                    pin_memory=use_cuda, drop_last=drop_last)

    # Class weights from this client's own labels only (smoothed: absent classes -> count 1)
    counts = df["dx"].value_counts().reindex(classes).fillna(0).values.astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(classes) * counts)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))

    # Fresh optimizer every round (optimizer state is not communicated in FL)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    model.train()
    total_loss, seen = 0.0, 0
    for _ in range(args.local_epochs):
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_cuda):
                loss = criterion(model(x), y)
            if args.mu > 0:  # FedProx: keep local weights close to the global model
                prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_params))
                loss = loss + (args.mu / 2.0) * prox
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * x.size(0)
            seen += x.size(0)
    return total_loss / max(seen, 1)


# --------------------------------------------------------------------------- #
# Server side
# --------------------------------------------------------------------------- #
def new_accumulator(state):
    return {k: torch.zeros_like(v, dtype=torch.float32) for k, v in state.items()
            if v.is_floating_point()}


def accumulate(acc, int_buf, model, n_k):
    """Add n_k * (client weights) into the running sum (avoids holding all client models)."""
    for k, v in model.state_dict().items():
        if k in acc:
            acc[k] += n_k * v.detach().cpu().float()
        else:  # integer buffers (BatchNorm num_batches_tracked): just keep the latest value
            int_buf[k] = v.detach().cpu().clone()


def finalize(acc, int_buf, total_n, template):
    """FedAvg: weighted average  w = sum_k (n_k / n) * w_k."""
    new_state = {}
    for k, v in template.items():
        new_state[k] = (acc[k] / total_n).to(v.dtype) if k in acc else int_buf[k]
    return new_state


@torch.no_grad()
def evaluate(model, ds, args, device):
    use_cuda = device.type == "cuda"
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=use_cuda)
    model.eval()
    preds, labels = [], []
    with torch.autocast(device.type, dtype=torch.float16, enabled=use_cuda):
        for x, y in dl:
            out = model(x.to(device, non_blocking=True))
            preds.append(out.argmax(1).cpu())
            labels.append(y)
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    return {
        "acc": float((preds == labels).mean()),
        "bal_acc": float(balanced_accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="macro")),
        "preds": preds,
        "labels": labels,
    }


def build_model(n_classes, device):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model.to(device)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)

    df = pd.read_csv(args.csv)
    classes = sorted(df["dx"].unique())
    print("Classes:", classes)

    # ---- Server-side splits and client partition ----
    pool_df, val_df, test_df = split_server_and_pool(df, args.seed)
    client_dfs = partition_clients(pool_df, args.clients, args.alpha, args.seed, args.min_client_size)

    print(f"\nTrain pool {len(pool_df)} | global val {len(val_df)} | global test {len(test_df)}")
    dist = pd.DataFrame({f"client{k}": d["dx"].value_counts().reindex(classes).fillna(0).astype(int)
                         for k, d in enumerate(client_dfs)})
    dist.loc["TOTAL"] = dist.sum()
    print(f"\nPer-client class distribution (alpha={args.alpha}):")
    print(dist.T.to_string())
    dist.T.to_csv(f"{args.out_prefix}_client_distribution.csv")

    train_tf, val_tf = get_transforms()
    client_ds = [HAM10000Dataset(d, args.img_dir, train_tf) for d in client_dfs]
    val_ds = HAM10000Dataset(val_df, args.img_dir, val_tf)
    test_ds = HAM10000Dataset(test_df, args.img_dir, val_tf)
    for ds in client_ds + [val_ds, test_ds]:
        assert ds.classes == classes, "class mapping mismatch"

    # ---- Global model ----
    model = build_model(len(classes), device)
    global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model_mb = sum(v.numel() * v.element_size() for v in global_state.values()) / 1e6
    print(f"\nModel size: {model_mb:.1f} MB  (per-client upload per round)")

    algo = f"FedProx(mu={args.mu})" if args.mu > 0 else "FedAvg"
    print(f"Algorithm: {algo} | clients {args.clients} | frac {args.frac} | "
          f"rounds {args.rounds} | local epochs {args.local_epochs}")

    rng = np.random.default_rng(args.seed)
    n_selected = max(1, int(round(args.frac * args.clients)))
    history, best = [], {"bal_acc": -1.0}
    comm_mb = 0.0
    ckpt_path = f"{args.out_prefix}_best.pt"

    for rnd in range(1, args.rounds + 1):
        set_seed(args.seed + rnd)
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * (rnd - 1) / args.rounds))  # cosine over rounds
        selected = sorted(rng.choice(args.clients, n_selected, replace=False).tolist())

        acc, int_buf, total_n = new_accumulator(global_state), {}, 0
        client_losses = []
        for k in selected:
            # Client downloads the global model
            model.load_state_dict(global_state)
            global_params = ([p.detach().clone() for p in model.parameters()]
                             if args.mu > 0 else None)
            # Client trains locally on its private data
            loss = local_train(model, global_params, client_ds[k], client_dfs[k], classes, args, device, lr)
            n_k = len(client_ds[k])
            # Client uploads weights; server folds them into the running average
            accumulate(acc, int_buf, model, n_k)
            total_n += n_k
            client_losses.append(loss)
            print(f"  [Round {rnd:02d}] client {k} (n={n_k}) local loss {loss:.4f}")

        global_state = finalize(acc, int_buf, total_n, global_state)
        comm_mb += 2 * len(selected) * model_mb  # download + upload

        # Server evaluates the aggregated model on the global validation set
        model.load_state_dict(global_state)
        m = evaluate(model, val_ds, args, device)
        history.append({"round": rnd, "clients": len(selected), "avg_local_loss": float(np.mean(client_losses)),
                        "val_acc": m["acc"], "val_bal_acc": m["bal_acc"], "val_macro_f1": m["f1"],
                        "cum_comm_MB": comm_mb})
        print(f"[Round {rnd:02d}/{args.rounds}] clients {selected} | loss {np.mean(client_losses):.4f} | "
              f"val acc {m['acc']:.3f} | bal-acc {m['bal_acc']:.3f} | macro-F1 {m['f1']:.3f} | "
              f"comm {comm_mb / 1000:.2f} GB")

        if m["bal_acc"] > best["bal_acc"]:
            best = {"round": rnd, "bal_acc": m["bal_acc"]}
            torch.save({"model": global_state, "classes": classes, "round": rnd}, ckpt_path)

        pd.DataFrame(history).to_csv(f"{args.out_prefix}_history.csv", index=False)

    # ---- Final evaluation of best global model on the untouched test set ----
    print(f"\nBest round by validation balanced accuracy: {best['round']} ({best['bal_acc']:.3f})")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    t = evaluate(model, test_ds, args, device)
    print(f"TEST | acc {t['acc']:.3f} | bal-acc {t['bal_acc']:.3f} | macro-F1 {t['f1']:.3f}")

    pd.DataFrame({
        "image_id": test_df["image_id"].values,
        "true": [classes[i] for i in t["labels"]],
        "pred": [classes[i] for i in t["preds"]],
    }).to_csv(f"{args.out_prefix}_test_predictions.csv", index=False)

    print("\nPer-class report (global test set):")
    print(classification_report([classes[i] for i in t["labels"]], [classes[i] for i in t["preds"]],
                                labels=classes, digits=3, zero_division=0))


if __name__ == "__main__":  # required on Windows for DataLoader workers
    main()