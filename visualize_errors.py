import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

from unet import UNet

def load_npy_file(base_dir: Path, fid: str) -> np.ndarray:
    for p in (base_dir / fid, base_dir / f"{fid}.npy"):
        if p.exists():
            return np.load(p)
    raise FileNotFoundError(f"No file for ID {fid}")

class CovidTestDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_dir, masks_dir, ids, scale=1.0,
                 global_mean=None, global_std=None):
        self.ids = ids
        self.imgs = Path(imgs_dir)
        self.msks = Path(masks_dir)
        self.scale = scale
        self.mean = global_mean
        self.std = global_std

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        fid = self.ids[i]
        img = load_npy_file(self.imgs, fid).astype(np.float32)
        msk = load_npy_file(self.msks, fid).astype(np.uint8)

        # 1) window & scale to [0,1]
        img = np.clip(img, -1000, 400)
        img = (img + 1000) / 1400.0
        if self.mean is not None:
            img = (img - self.mean) / (self.std + 1e-8)

        # 2) resize if needed
        if self.scale != 1.0:
            h, w = img.shape
            img = np.array(
                Image.fromarray(img).resize(
                    (int(w * self.scale), int(h * self.scale)),
                    Image.BICUBIC
                )
            )
            msk = np.array(
                Image.fromarray(msk).resize(
                    (int(w * self.scale), int(h * self.scale)),
                    Image.NEAREST
                )
            )

        return img, msk, fid

def overlay_mask_on_rgb(rgb, mask, color, alpha=0.5):
    """
    Overlay a binary mask on an RGB image.
    rgb: HxWx3 uint8, mask: HxW boolean, color: (r,g,b) 0–255
    """
    overlay = rgb.copy().astype(np.float32)
    for c in range(3):
        overlay[..., c] = np.where(
            mask,
            overlay[..., c] * (1 - alpha) + alpha * color[c],
            overlay[..., c]
        )
    return overlay.astype(np.uint8)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--imgs-dir",     required=True)
    p.add_argument("--masks-dir",    required=True)
    p.add_argument("--test-list",    required=True)
    p.add_argument("--ckpt-dir",     default="checkpoints")
    p.add_argument("--output-dir",   default="error_vis")
    p.add_argument("--num-folds",    type=int, default=5)
    p.add_argument("--threshold",    type=float, default=0.5)
    p.add_argument("--scale",        type=float, default=0.5)
    p.add_argument("--global-mean",  type=float, default=None)
    p.add_argument("--global-std",   type=float, default=None)
    p.add_argument("--device",       default="cpu")
    p.add_argument("--max-samples",  type=int, default=None,
                   help="Max number of test samples to visualize")
    args = p.parse_args()

    # load test IDs
    with open(args.test_list) as f:
        ids = [l.strip() for l in f if l.strip()]
    if args.max_samples:
        ids = ids[:args.max_samples]

    # dataset + loader
    ds = CovidTestDataset(
        args.imgs_dir, args.masks_dir, ids,
        scale=args.scale,
        global_mean=args.global_mean,
        global_std=args.global_std
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    device = torch.device(args.device)
    Path(args.output_dir).mkdir(exist_ok=True)

    # load all fold-models
    models = []
    for fold in range(args.num_folds):
        m = UNet(n_channels=1, n_classes=1, bilinear=False)
        ckpt = torch.load(f"{args.ckpt_dir}/fold{fold}.pth",
                          map_location=device)
        m.load_state_dict(ckpt)
        m.to(device).eval()
        models.append(m)

    # process each sample
    for img_np, gt_mask, fid in loader:
        # img_np: Tensor[1,H,W], convert to numpy HxW
        img_np = img_np[0].cpu().numpy()
        gt = gt_mask[0].cpu().numpy()

        # prepare grayscale RGB for overlay
        gray = (img_np * 255).astype(np.uint8)
        rgb = np.stack([gray] * 3, axis=-1)

        # ensemble soft predictions
        probs = np.zeros_like(img_np, dtype=np.float32)
        with torch.no_grad():
            for m in models:
                inp = torch.from_numpy(img_np[None, None,...]).to(device)
                logits = m(inp)
                prob = torch.sigmoid(logits)[0,0].cpu().numpy()
                probs += prob
        probs /= len(models)
        pred = (probs > args.threshold).astype(np.uint8)

        # generate overlays
        gt_overlay   = overlay_mask_on_rgb(rgb, gt==1,    color=(0,255,0))
        pred_overlay = overlay_mask_on_rgb(rgb, pred==1,  color=(255,0,0))
        fp = (pred==1) & (gt==0)
        fn = (pred==0) & (gt==1)
        err_rgb = overlay_mask_on_rgb(rgb, fp, color=(255,0,0), alpha=0.6)
        err_rgb = overlay_mask_on_rgb(err_rgb, fn, color=(0,0,255), alpha=0.6)

        # plot side-by-side
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(gray, cmap='gray');              axes[0].set_title("CT slice")
        axes[1].imshow(gt_overlay);                     axes[1].set_title("GT mask (green)")
        axes[2].imshow(pred_overlay);                   axes[2].set_title("Pred mask (red)")
        axes[3].imshow(err_rgb);                        axes[3].set_title("Errors: FP red, FN blue")
        for ax in axes:
            ax.axis('off')

        fig.tight_layout()
        fig.savefig(f"{args.output_dir}/{fid}.png", bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    main()
