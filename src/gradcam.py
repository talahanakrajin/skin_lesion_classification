"""
Generate Grad-CAM heatmaps for HAM10000 using the fold checkpoints from train.py.

Runs each fold's model on its own held-out validation split (out-of-fold),
so every image gets a heatmap from a model that never saw it during training.

Usage:
    python gradcam.py --csv HAM10000_metadata.csv --img_dir path/to/all_images \
        --ckpt_dir . --out_prefix resnet50_ham10000

    # Only run on a specific subset, e.g. melanoma misclassified as nevus:
    python gradcam.py --csv ... --img_dir ... --ckpt_dir . \
        --oof_csv resnet50_ham10000_oof_predictions.csv \
        --filter_true mel --filter_pred nv
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import HAM10000Dataset
from sklearn.model_selection import StratifiedGroupKFold

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--img_dir", required=True)
    p.add_argument("--ckpt_dir", required=True, help="folder containing fold checkpoints")
    p.add_argument("--out_prefix", default="resnet50_ham10000")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42, help="must match the seed used in train.py")
    p.add_argument("--oof_csv", default=None, help="optional: filter to specific true/pred pairs")
    p.add_argument("--filter_true", default=None)
    p.add_argument("--filter_pred", default=None)
    p.add_argument("--max_per_fold", type=int, default=None, help="cap images per fold, for quick tests")
    p.add_argument("--cam_dir", default="../outputs/cams")
    p.add_argument("--fig_dir", default="../outputs/figures/gradcam")
    return p.parse_args()

def get_eval_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def load_fold_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return model, classes

def unnormalize(tensor_img):
    """Convert a normalized tensor back to a 0-1 numpy image (H, W, 3) for overlay."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor_img.cpu().numpy().transpose(1, 2, 0)
    img = img * std + mean
    return np.clip(img, 0, 1)

def run_gradcam_on_fold(fold, val_df, classes, args, device):
    ckpt_path = os.path.join(args.ckpt_dir, f"{args.out_prefix}_fold{fold}.pt")
    model, ckpt_classes = load_fold_model(ckpt_path, device)
    assert ckpt_classes == classes, f"class order mismatch in fold {fold} checkpoint"
    
    val_tf = get_eval_transforms()
    val_ds = HAM10000Dataset(val_df, args.img_dir, val_tf)
    
    # Standard target layer for ResNet50 Grad-CAM: the last block of layer4
    target_layer = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layer)
    
    os.makedirs(args.cam_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)
    
    n = len(val_ds) if args.max_per_fold is None else min(args.max_per_fold, len(val_ds))
    for i in range(n):
        img_tensor, label_idx = val_ds[i]
        row = val_df.iloc[i]
        image_id = row["image_id"]
        
        input_tensor = img_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_idx = model(input_tensor).argmax(dim=1).item()
            
        # Explain the model's PREDICTED class (what it actually decided)
        # not necessarily the true label - this part is to test whether the model is shortcut learning or no
        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0] # (H, W), values 0-1
        
        # Save the raw heatmap array for later IoU/pointing-game computation (Week 7-8)
        np.save(os.path.join(args.cam_dir, f"{image_id}_fold{fold}.npy"), grayscale_cam)
        
        # Save a visual overlay for a sanity-check subset
        if i < 20 or (args.filter_true is not None):
            rgb_img = unnormalize(img_tensor)
            overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            Image.fromarray(overlay).save(
                os.path.join(args.fig_dir, f"{image_id}_true-{classes[label_idx]}_pred-{classes[pred_idx]}.png")
            )
            
    del model, cam
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[Fold {fold}] Grad-CAM done for {n} images")
    
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
        
    df = pd.read_csv(args.csv)
    classes = sorted(df["dx"].unique())
        
     # Optional: narrow down to a specific true/pred subset (e.g. mel misclassified as nv)
    if args.oof_csv is not None:
        oof = pd.read_csv(args.oof_csv)
        mask = pd.Series(True, index=oof.index)
        if args.filter_true:
            mask &= oof["true"] == args.filter_true
        if args.filter_pred:
            mask &= oof["pred"] == args.filter_pred
        keep_ids = set(oof.loc[mask, "image_id"])
        print(f"Filter matched {len(keep_ids)} images")
    else:
        keep_ids = None
        
    # Recreate the Exact same folds as train.py (same seed, same splitter)
    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(skf.split(df, df["dx"], groups=df["lesion_id"]))
    
    for fold, (tr_idx, va_idx) in enumerate(splits):
        val_df = df.iloc[va_idx].reset_index(drop=True)
        if keep_ids is not None:
            val_df = val_df[val_df["image_id"].isin(keep_ids)].reset_index(drop=True)
            if len(val_df) == 0:
                continue
        run_gradcam_on_fold(fold, val_df, classes, args, device)
            
if __name__ == "__main__":
    main()