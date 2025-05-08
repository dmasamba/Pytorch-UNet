import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff

from unet import UNet

def load_npy_file(base_dir: Path, fid: str) -> np.ndarray:
    candidates = [base_dir / fid, base_dir / f"{fid}.npy"]
    for p in candidates:
        if p.exists():
            if p.stat().st_size == 0:
                raise EOFError(f"Empty file: {p}")
            try:
                return np.load(p)
            except EOFError:
                raise EOFError(f"Corrupted file: {p}")
    raise FileNotFoundError(f"No such file: {candidates[0]} or {candidates[1]}")

def compute_confusion(gt: np.ndarray, pred: np.ndarray):
    """Return TP, FP, TN, FN counts for binary masks."""
    tp = int(np.count_nonzero((pred == 1) & (gt == 1)))
    fp = int(np.count_nonzero((pred == 1) & (gt == 0)))
    tn = int(np.count_nonzero((pred == 0) & (gt == 0)))
    fn = int(np.count_nonzero((pred == 0) & (gt == 1)))
    return tp, fp, tn, fn

def hausdorff(gt: np.ndarray, pred: np.ndarray):
    """Return symmetric Hausdorff distance between two binary masks."""
    gt_pts   = np.argwhere(gt == 1)
    pred_pts = np.argwhere(pred == 1)
    if len(gt_pts) == 0 or len(pred_pts) == 0:
        return np.nan
    d1 = directed_hausdorff(gt_pts, pred_pts)[0]
    d2 = directed_hausdorff(pred_pts, gt_pts)[0]
    return max(d1, d2)

class CovidTestDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_dir, masks_dir, id_list, scale=1.0,
                 global_mean=None, global_std=None):
        self.ids = id_list
        self.imgs = Path(imgs_dir)
        self.msks = Path(masks_dir)
        self.scale = scale
        self.mean  = global_mean
        self.std   = global_std

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fid = self.ids[idx]
        img = load_npy_file(self.imgs, fid).astype(np.float32)
        msk = load_npy_file(self.msks, fid).astype(np.float32)

        # window & scale
        img = np.clip(img, -1000, 400)
        img = (img + 1000) / 1400.0
        if self.mean is not None:
            img = (img - self.mean) / (self.std + 1e-8)

        # resize
        if self.scale != 1.0:
            h, w = img.shape
            img = np.array(
                Image.fromarray(img).resize(
                    (int(w*self.scale), int(h*self.scale)),
                    Image.BICUBIC
                )
            )
            msk = np.array(
                Image.fromarray(msk).resize(
                    (int(w*self.scale), int(h*self.scale)),
                    Image.NEAREST
                )
            )

        img_tensor = torch.from_numpy(img[None, ...]).float()
        return img_tensor, msk.astype(np.uint8), fid

def main(args):
    # Load & filter test IDs
    with open(args.test_list, 'r') as f:
        raw_ids = [l.strip() for l in f if l.strip()]
    test_ids = []
    for fid in raw_ids:
        try:
            _ = load_npy_file(Path(args.imgs_dir), fid)
            _ = load_npy_file(Path(args.masks_dir), fid)
            test_ids.append(fid)
        except Exception as e:
            print(f"Skipping {fid}: {e}")
    print(f"Using {len(test_ids)}/{len(raw_ids)} valid test samples.\n")

    # Compute global mean/std if needed
    if args.global_mean is None or args.global_std is None:
        print("Computing global mean/std from training folds…")
        train_ids = []
        for fold in range(args.num_folds):
            with open(f"./data/train_new{fold}.txt") as f:
                train_ids += [l.strip() for l in f if l.strip()]
        s1 = s2 = n = 0
        for fid in train_ids:
            try:
                arr = load_npy_file(Path(args.imgs_dir), fid).astype(np.float32)
                arr = np.clip(arr, -1000, 400)
                arr = (arr + 1000)/1400.0
                s1 += arr.sum()
                s2 += (arr**2).sum()
                n += arr.size
            except Exception:
                continue
        args.global_mean = s1/n
        var = s2/n - (args.global_mean ** 2)
        args.global_std = np.sqrt(max(var, 1e-6))
        print(f"Global mean={args.global_mean:.4f}, std={args.global_std:.4f}\n")

    # Build DataLoader
    ds = CovidTestDataset(
        args.imgs_dir, args.masks_dir, test_ids,
        scale=args.scale,
        global_mean=args.global_mean,
        global_std=args.global_std
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    metrics = {k: [] for k in ['Dice','IoU','Sensitivity','Specificity','Hausdorff']}

    # Evaluate each fold
    for fold in range(args.num_folds):
        print(f"Evaluating Fold {fold}…")
        model = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear)
        ckpt_path = Path(args.ckpt_dir)/f"fold{fold}.pth"
        ckpt = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt)
        model.to(args.device).eval()

        fold_m = {k: [] for k in metrics}
        with torch.no_grad():
            for img_t, gt_mask, _ in loader:
                img_t = img_t.to(args.device)
                logits = model(img_t)
                prob = torch.sigmoid(logits)[0,0].cpu().numpy()
                pred = (prob > 0.5).astype(np.uint8)
                gt   = gt_mask.squeeze(0).cpu().numpy().astype(np.uint8)

                # compute metrics
                tp, fp, tn, fn = compute_confusion(gt, pred)
                dice = (2*tp)/(2*tp+fp+fn+1e-8)
                iou  = tp/(tp+fp+fn+1e-8)
                sens = tp/(tp+fn+1e-8)
                spec = tn/(tn+fp+1e-8)
                hd   = hausdorff(gt, pred)

                fold_m['Dice'].append(dice)
                fold_m['IoU'].append(iou)
                fold_m['Sensitivity'].append(sens)
                fold_m['Specificity'].append(spec)
                fold_m['Hausdorff'].append(hd)

        for k in metrics:
            m = np.nanmean(fold_m[k])
            print(f" Fold{fold} {k}: {m:.4f}")
            metrics[k] += fold_m[k]
        print()

    # Overall stats
    print("Overall test-set performance:")
    for k,v in metrics.items():
        print(f" {k}: {np.nanmean(v):.4f} ± {np.nanstd(v):.4f}")

    os.makedirs(args.plot_dir, exist_ok=True)
    for k, vals in metrics.items():
        clean = [x for x in vals if not np.isnan(x)]
        plt.figure(figsize=(6,4))
        plt.boxplot(clean, vert=True, patch_artist=True)
        plt.title(f"Test-set {k}")
        plt.ylabel(k)
        plt.grid(True, linestyle='--', alpha=0.5)
        out_path = Path(args.plot_dir)/f"{k.lower()}_boxplot.png"
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
    print(f"Saved plots under `{args.plot_dir}`")
    
if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--imgs-dir",   type=str, required=True)
    p.add_argument("--masks-dir",  type=str, required=True)
    p.add_argument("--test-list",  type=str, required=True)
    p.add_argument("--ckpt-dir",   type=str, default="checkpoints")
    p.add_argument("--plot-dir",   type=str, default="plots")
    p.add_argument("--num-folds",  type=int,   default=5)
    p.add_argument("--scale",      type=float, default=0.5)
    p.add_argument("--bilinear",   action="store_true")
    p.add_argument("--device",     type=str,   default="cpu")
    p.add_argument("--global-mean", type=float, default=None)
    p.add_argument("--global-std",  type=float, default=None)
    args = p.parse_args()
    main(args)
