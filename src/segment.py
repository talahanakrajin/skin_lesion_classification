import argparse
import numpy as np, pandas as pd, torch
import segmentation_models_pytorch as smp
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

class SegDataset(Dataset):
    def __init__(self, df, img_dir, mask_dir, train=False, size=256):
        self.df, self.img_dir, self.mask_dir = df.reset_index(drop=True), img_dir, mask_dir
        self.train, self.size = train, size
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        iid = self.df.iloc[i]["image_id"]
        img = Image.open(f"{self.img_dir}/{iid}.jpg").convert("RGB").resize((self.size,)*2, Image.BILINEAR)
        msk = Image.open(f"{self.mask_dir}/{iid}_segmentation.png").convert("L").resize((self.size,)*2, Image.NEAREST)
        x = (torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255 - MEAN) / STD
        y = torch.from_numpy((np.array(msk) > 127).astype(np.float32))[None]
        if self.train:
            if torch.rand(1) < 0.5: x, y = x.flip(-1), y.flip(-1)
            if torch.rand(1) < 0.5: x, y = x.flip(-2), y.flip(-2)
        return x, y
    
def split(df, seed=42):
    g = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, tmp = next(g.split(df, groups=df["lesion_id"]))
    tmp_df = df.iloc[tmp]
    g2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    va, te = next(g2.split(tmp_df, groups=tmp_df["lesion_id"]))
    return df.iloc[tr], tmp_df.iloc[va], tmp_df.iloc[te]
    
@torch.no_grad()
def evaluate(model, dl, device):
    model.eval(); ious, dices = [], []
    for x, y in dl:
        p = (model(x.to(device)).sigmoid() > 0.5).float().cpu()
        inter = (p * y).sum((1, 2, 3)); union = ((p + y) > 0).sum((1, 2, 3))
        ious += (inter / union.clamp(min=1)).tolist()
        dices += (2 * inter / (p.sum((1, 2, 3)) + y.sum((1, 2, 3))).clamp(min=1)).tolist()
    return np.mean(ious), np.mean(dices)
    
def main():
    a = argparse.ArgumentParser()
    a.add_argument("--csv"); a.add_argument("--img_dir"); a.add_argument("--mask_dir")
    a.add_argument("--epochs", type=int, default=20); a.add_argument("--out", default="../outputs/models/unet_resnet50.pt")
    a = a.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(a.csv)
    tr, va, te = split(df)
    mk = lambda d, t: DataLoader(SegDataset(d, a.img_dir, a.mask_dir, train=t), batch_size=16, shuffle=t, num_workers=2)
    tr_dl, va_dl, te_dl = mk(tr, True), mk(va, False), mk(te, False)
        
    model = smp.Unet("resnet50", encoder_weights="imagenet", in_channels=3, classes=1).to(device)
    dice, bce = smp.losses.DiceLoss("binary"), torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    best = 0
    for ep in range(1, a.epochs + 1):
        model.train()
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            out = model(x); loss = dice(out, y) + bce(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
        iou, dsc = evaluate(model, va_dl, device)
        print(f"Epoch {ep}: val IoU={iou:.4f}, Dice={dsc:.4f}")
        if iou > best: best = iou; torch.save(model.state_dict(), a.out)
    model.load_state_dict(torch.load(a.out)) 
    iou, dsc = evaluate(model, te_dl, device)
    print(f"Test IoU={iou:.4f}, Dice={dsc:.4f}")
    
if __name__ == "__main__":
    main()