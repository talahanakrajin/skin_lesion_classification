import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu, wilcoxon


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cam_dir", required=True)
    p.add_argument("--mask_dir", required=True)
    p.add_argument("--oof_csv", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_mask(mask_path, target_size=(224, 224)):
    mask = Image.open(mask_path).convert("L").resize(target_size, Image.NEAREST)
    return np.array(mask) > 127


def make_center_baseline(H, W):
    yy, xx = np.mgrid[:H, :W]
    return np.exp(-(((yy - H / 2) ** 2 + (xx - W / 2) ** 2) / (2 * (H / 4) ** 2)))


def norm(c):
    return (c - c.min()) / (c.max() - c.min() + 1e-8)


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def alignment(cam, mask):
    """cam: (H,W) float, mask: (H,W) bool."""
    n = int(mask.sum())
    if n == 0 or n == mask.size:
        return None
    cam = norm(cam)
    flat = cam.ravel()
    top = np.zeros(flat.size, dtype=bool)
    top[np.argpartition(flat, -n)[-n:]] = True # exactly N pixels, ties broken arbitrarily
    top = top.reshape(cam.shape)
    return {
        "iou_area_matched": iou(top, mask),
        "iou_at_0.5": iou(cam >= 0.5, mask),
        "pointing_game": int(mask[np.unravel_index(cam.argmax(), cam.shape)]),
        "energy_in_mask": cam[mask].sum() / (cam.sum() + 1e-8),
        "auroc": roc_auc_score(mask.ravel(), cam.ravel()),
    }


METRICS = ["iou_area_matched", "iou_at_0.5", "pointing_game", "energy_in_mask", "auroc"]


def main():
    args = get_args()
    rng = np.random.default_rng(args.seed)
    oof = pd.read_csv(args.oof_csv)
    cam_files_all = os.listdir(args.cam_dir)

    rows = []
    for _, row in oof.iterrows():
        image_id = row["image_id"]

        cam_files = [f for f in cam_files_all if f.startswith(image_id + "_fold")]
        if not cam_files:
            continue
        cam = np.load(os.path.join(args.cam_dir, cam_files[0]))

        mask_path = os.path.join(args.mask_dir, f"{image_id}_segmentation.png")
        if not os.path.exists(mask_path):
            continue
        H, W = cam.shape
        mask = load_mask(mask_path, target_size=(W, H))

        sources = {
            "real": cam,
            "center": make_center_baseline(H, W),
            "random": rng.random((H, W)),
        }
        for name, m in sources.items():
            res = alignment(m, mask)
            if res is None:       # empty or full mask -> skip image entirely
                break
            rows.append({
                "image_id": image_id,
                "true": row["true"],
                "pred": row["pred"],
                "correct": row["true"] == row["pred"],
                "source": name,
                **res,
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"Computed alignment for {df['image_id'].nunique()} images")

    print("\n--- Real vs. baselines ---")
    print(df.groupby("source")[METRICS].mean())

    real = df[df.source == "real"]
    print("\n--- Real CAM: correct vs incorrect ---")
    print(real.groupby("correct")[METRICS].mean())

    print("\n--- Real CAM: by true class ---")
    print(real.groupby("true")[METRICS].mean())

    # Paired difference: real minus each baseline, per image
    wide = df.pivot(index="image_id", columns="source", values=METRICS)
    print("\n--- Mean paired gain of real CAM over baseline ---")
    for base in ["center", "random"]:
        diff = {m: (wide[m]["real"] - wide[m][base]).mean() for m in METRICS}
        print(base, {k: round(v, 4) for k, v in diff.items()})
        
    print("\n--- Correct vs incorrect (Mann-Whitney U, real CAM) ---")
    for m in METRICS:
        a, b = real[real.correct][m], real[~real.correct][m]
        if len(a) and len(b):
            print(f"{m}: p = {mannwhitneyu(a, b).pvalue:.4g}")

    print("\n--- Real vs center baseline (paired Wilcoxon) ---")
    for m in METRICS:
        print(f"{m}: p = {wilcoxon(wide[m]['real'], wide[m]['center']).pvalue:.4g}")    


if __name__ == "__main__":
    main()