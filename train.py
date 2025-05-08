import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

import wandb

from unet import UNet
from utils.dice_score import dice_loss
from evaluate import evaluate

# --- Paths ---
DIR_IMG        = Path('./data/imgs/')
DIR_MASK       = Path('./data/masks/')
DIR_CHECKPOINT = Path('./checkpoints/')

def read_split_file(split_file):
    ids = []
    with open(split_file, 'r') as f:
        for line in f:
            name = line.strip()
            if name and not name.startswith('.'):
                ids.append(os.path.splitext(name)[0])
    return ids

class CovidDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, img_ids,
                 scale=1.0, global_mean=None, global_std=None, flip=False):
        self.images_dir  = Path(images_dir)
        self.masks_dir   = Path(masks_dir)
        self.img_ids     = img_ids
        self.scale       = scale
        self.global_mean = global_mean
        self.global_std  = global_std
        self.flip        = flip

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        fid = self.img_ids[idx]
        img = np.load(self.images_dir / f"{fid}.npy").astype(np.float32)
        msk = np.load(self.masks_dir  / f"{fid}.npy").astype(np.float32)

        # 1) Lung-window clip & scale → [0,1]
        img = np.clip(img, -1000, 400)
        img = (img + 1000) / 1400.0

        # 2) Global normalization
        if self.global_mean is not None:
            img = (img - self.global_mean) / (self.global_std + 1e-8)

        # 3) Resize
        if self.scale != 1.0:
            h, w = img.shape
            new_w = int(w * self.scale)
            new_h = int(h * self.scale)
            img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BICUBIC))
            msk = np.array(Image.fromarray(msk).resize((new_w, new_h), Image.NEAREST))

        # 4) Channels & types
        img = img[None, ...]           # (1,H,W)
        msk = (msk > 0.5).astype(np.uint8)

        # 5) Random horizontal flip
        if self.flip and random.random() > 0.5:
            img = np.flip(img, axis=2).copy()
            msk = np.flip(msk, axis=1).copy()

        img = torch.from_numpy(img).float()
        msk = torch.from_numpy(msk).long()
        return {'image': img, 'mask': msk}

def save_predictions(model, loader, device, out_dir, epoch, num_batches=3):
    """
    Save a few input slices, GT masks, and predicted masks to disk.
    """
    model.eval()
    inputs_dir = Path(out_dir) / "inputs"
    gts_dir    = Path(out_dir) / "ground_truths"
    preds_dir  = Path(out_dir) / "predictions"
    for d in (inputs_dir, gts_dir, preds_dir):
        d.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for b, batch in enumerate(loader):
            if b >= num_batches:
                break
            imgs = batch['image'].to(device)      # (B,1,H,W)
            msks = batch['mask'].to(device)       # (B,H,W)
            logits = model(imgs)                  # (B,1,H,W)
            probs  = torch.sigmoid(logits)        # (B,1,H,W)
            preds  = (probs > 0.5).float()        # (B,1,H,W)

            B = imgs.size(0)
            for i in range(B):
                inp = imgs[i,0].cpu().numpy()      # normalized float
                gt  = msks[i].cpu().numpy()        # {0,1}
                pd  = preds[i,0].cpu().numpy()     # {0,1}

                # rescale input back to [0,255] for saving
                inp = ((inp - inp.min()) / (inp.max() - inp.min() + 1e-8) * 255).astype(np.uint8)
                gt  = (gt * 255).astype(np.uint8)
                pd  = (pd * 255).astype(np.uint8)

                fname = f"epoch{epoch}_batch{b}_sample{i}.png"
                Image.fromarray(inp).save(inputs_dir / fname)
                Image.fromarray(gt).save(gts_dir    / fname)
                Image.fromarray(pd).save(preds_dir  / fname)

def train_model_cv(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # --- compute global mean/std over all train folds ---
    all_ids = set()
    for f in range(args.num_folds):
        all_ids |= set(read_split_file(f"./data/train_new{f}.txt"))

    s1 = s2 = n = 0
    for fid in all_ids:
        arr = np.load(DIR_IMG / f"{fid}.npy").astype(np.float32)
        arr = np.clip(arr, -1000, 400)
        arr = (arr + 1000) / 1400.0
        s1 += arr.sum()
        s2 += (arr**2).sum()
        n  += arr.size

    global_mean = s1 / n
    var = s2 / n - global_mean**2
    var = max(var, 1e-6)
    global_std = np.sqrt(var)
    logging.info(f"Global norm → mean: {global_mean:.4f}, std: {global_std:.4f}")

    for fold in range(args.num_folds):
        train_ids = read_split_file(f"./data/train_new{fold}.txt")
        valid_ids = read_split_file(f"./data/valid_new{fold}.txt")

        train_ds = CovidDataset(DIR_IMG, DIR_MASK, train_ids,
                                scale=args.scale,
                                global_mean=global_mean,
                                global_std=global_std,
                                flip=True)
        valid_ds = CovidDataset(DIR_IMG, DIR_MASK, valid_ids,
                                scale=args.scale,
                                global_mean=global_mean,
                                global_std=global_std,
                                flip=False)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=os.cpu_count(),
                                  pin_memory=True)
        val_loader   = DataLoader(valid_ds, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False,
                                  num_workers=os.cpu_count(),
                                  pin_memory=True)

        # compute pos_weight for BCE
        pos = neg = 0
        for sample in train_ds:
            m = sample['mask']
            pos += m.sum().item()
            neg += (m.numel() - m.sum().item())
        pos_weight = torch.tensor(neg / (pos + 1e-8), device=device)
        logging.info(f"Fold {fold} pos_weight = {pos_weight:.4f}")

        model = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear)
        model.to(device=device, memory_format=torch.channels_last)

        optimizer = optim.Adam(model.parameters(),
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5)

        bce_fn  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        dice_fn = dice_loss

        if wandb.run: wandb.finish()
        run = wandb.init(project="U-Net-improved",
                         name=f"fold_{fold}", reinit=True)
        run.config.update(vars(args))

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0
            pbar = tqdm(train_loader,
                        desc=f"[Fold {fold}] Epoch {epoch}/{args.epochs}",
                        unit="batch")
            for batch in pbar:
                imgs = batch['image'].to(device)
                msks = batch['mask'].float().to(device)

                preds = model(imgs)
                l1 = bce_fn( preds.squeeze(1), msks)
                l2 = dice_fn(torch.sigmoid(preds.squeeze(1)), msks)
                loss = 0.4 * l1 + 0.6 * l2

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.grad_clip)
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(train_loader)

            # validation
            model.eval()
            val_dice = evaluate(model, val_loader, device, args.amp)
            scheduler.step(val_dice)

            logging.info(f"Fold {fold} • Epoch {epoch}: "
                         f"Train Loss={avg_loss:.4f}, "
                         f"Val Dice={val_dice:.4f}, "
                         f"LR={optimizer.param_groups[0]['lr']:.2e}")

            wandb.log({"train/loss": avg_loss,
                       "val/dice": val_dice,
                       "epoch": epoch,
                       "lr": optimizer.param_groups[0]['lr']})

            # save a few example predictions
            out_dir = Path(args.output_dir) / f"fold{fold}"
            save_predictions(model, val_loader, device, out_dir, epoch,
                             num_batches=args.vis_batches)

            # optional checkpoint
            if args.save_ckpt:
                DIR_CHECKPOINT.mkdir(exist_ok=True)
                ckpt = DIR_CHECKPOINT / f"fold{fold}_ep{epoch}.pth"
                torch.save(model.state_dict(), ckpt)

        run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs",        type=int,   default=20)
    parser.add_argument("-b", "--batch-size",    type=int,   default=4)
    parser.add_argument("-l", "--learning-rate", type=float, default=1e-4)
    parser.add_argument("-s", "--scale",         type=float, default=0.5)
    parser.add_argument("--num-folds",           type=int,   default=5)
    parser.add_argument("--bilinear",            action="store_true")
    parser.add_argument("--weight-decay",        type=float, default=1e-8)
    parser.add_argument("--grad-clip",           type=float, default=1.0)
    parser.add_argument("--save-ckpt",           action="store_true")
    parser.add_argument("--amp",                 action="store_true",
                                                help="Use torch.cuda.amp in eval")
    parser.add_argument("--output-dir",          type=str,   default="outputs",
                                                help="Where to save visualization folders")
    parser.add_argument("--vis-batches",         type=int,   default=3,
                                                help="How many val batches to visualize each epoch")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    train_model_cv(args)
