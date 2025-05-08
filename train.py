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
import torchvision.transforms as T
from tqdm import tqdm

import wandb

from unet import UNet
from utils.dice_score import dice_loss
from evaluate import evaluate

# Needed paths
DIR_IMG        = Path('./data/imgs/')
DIR_MASK       = Path('./data/masks/')
DIR_CHECKPOINT = Path('./checkpoints/')

# important utilities
def read_split_file(split_file: str):
    ids = []
    with open(split_file, 'r') as f:
        for line in f:
            name = line.strip()
            if name and not name.startswith('.'):
                ids.append(os.path.splitext(name)[0])
    return ids

class CovidDataset(torch.utils.data.Dataset):
    """
    CT slices + masks dataset with optional global normalization,
    random flips, and on-the-fly augmentations.
    """
    def __init__(self,
                 images_dir, masks_dir, img_ids,
                 scale=1.0,
                 global_mean=None, global_std=None,
                 flip=False,
                 augment=False):
        self.images_dir  = Path(images_dir)
        self.masks_dir   = Path(masks_dir)
        self.img_ids     = img_ids
        self.scale       = scale
        self.global_mean = global_mean
        self.global_std  = global_std
        self.flip        = flip
        self.augment     = augment

        # augmentations
        self.max_rot = 10.0    # ±10° rotations
        self.hflip   = True    # allow horizontal flips only
        # color_jitter removed for now
        # self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        fid = self.img_ids[idx]
        img = np.load(self.images_dir / f"{fid}.npy").astype(np.float32)
        msk = np.load(self.masks_dir  / f"{fid}.npy").astype(np.float32)

        # Lung‐window clip & scale to [0,1]
        img = np.clip(img, -1000, 400)
        img = (img + 1000) / 1400.0

        # Global normalization if provided
        if self.global_mean is not None and self.global_std is not None:
            img = (img - self.global_mean) / (self.global_std + 1e-8)

        # Resize
        if self.scale != 1.0:
            h, w = img.shape
            new_w = int(w * self.scale)
            new_h = int(h * self.scale)
            img = np.array(
                Image.fromarray(img).resize((new_w, new_h), Image.BICUBIC)
            )
            msk = np.array(
                Image.fromarray(msk).resize((new_w, new_h), Image.NEAREST)
            )

        # Random flip
        if self.flip and random.random() > 0.5:
            img = np.flip(img, axis=1).copy()
            msk = np.flip(msk, axis=0).copy()

        # augmentation on-the-fly 
        if self.augment:
            img_pil  = Image.fromarray((img * 255).astype(np.uint8))
            mask_pil = Image.fromarray((msk * 255).astype(np.uint8))

            # unified random rotation
            angle = random.uniform(-self.max_rot, self.max_rot)
            img_pil  = img_pil.rotate(angle,  resample=Image.BILINEAR)
            mask_pil = mask_pil.rotate(angle, resample=Image.NEAREST)

            # unified random horizontal flip only
            if self.hflip and random.random() > 0.5:
                img_pil  = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
                mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)

            # back to numpy
            img = np.array(img_pil).astype(np.float32) / 255.0
            msk = (np.array(mask_pil) > 127).astype(np.uint8)

        # Add channel dim and convert to tensors
        img = torch.from_numpy(img[None, ...]).float()
        msk = torch.from_numpy(msk).long()

        return {'image': img, 'mask': msk}

def save_predictions(model, loader, device, out_dir, epoch, num_batches=3):
    """
    Save `num_batches` worth of inputs, ground truths, and predictions
    under out_dir/inputs, /ground_truths, /predictions, with filenames
    prefixed by the epoch number.
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
            imgs   = batch['image'].to(device)      # (B,1,H,W)
            msks   = batch['mask'].to(device)       # (B,H,W)
            logits = model(imgs)                    # (B,1,H,W)
            probs  = torch.sigmoid(logits)          # (B,1,H,W)
            preds  = (probs > 0.5).float()          # (B,1,H,W)

            B = imgs.size(0)
            for i in range(B):
                inp = imgs[i,0].cpu().numpy()
                gt  = msks[i].cpu().numpy()
                pd  = preds[i,0].cpu().numpy()

                # scale input [0..1] → [0..255]
                inp = ((inp - inp.min()) / (inp.max() - inp.min() + 1e-8) * 255).astype(np.uint8)
                gt  = (gt * 255).astype(np.uint8)
                pd  = (pd * 255).astype(np.uint8)

                fname = f"epoch{epoch}_batch{b}_sample{i}.png"
                Image.fromarray(inp).save( inputs_dir / fname )
                Image.fromarray(gt ).save( gts_dir    / fname )
                Image.fromarray(pd ).save( preds_dir  / fname )

def train_model_cv(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # compute global mean/std across all folds —
    all_ids = set()
    for f in range(args.num_folds):
        all_ids |= set(read_split_file(f"./data/train_new{f}.txt"))

    s1 = s2 = count = 0
    for fid in all_ids:
        arr = np.load(DIR_IMG / f"{fid}.npy").astype(np.float32)
        arr = np.clip(arr, -1000, 400)
        arr = (arr + 1000) / 1400.0
        s1    += arr.sum()
        s2    += (arr**2).sum()
        count += arr.size

    global_mean = s1 / count
    var         = s2 / count - (global_mean ** 2)
    var         = max(var, 1e-6)
    global_std  = np.sqrt(var)
    logging.info(f"Global norm → mean: {global_mean:.4f}, std: {global_std:.4f}")

    for fold in range(args.num_folds):
        train_ids = read_split_file(f"./data/train_new{fold}.txt")
        valid_ids = read_split_file(f"./data/valid_new{fold}.txt")

        train_ds = CovidDataset(
            DIR_IMG, DIR_MASK, train_ids,
            scale=args.scale,
            global_mean=global_mean,
            global_std=global_std,
            flip=True,
            augment=args.augment
        )
        valid_ds = CovidDataset(
            DIR_IMG, DIR_MASK, valid_ids,
            scale=args.scale,
            global_mean=global_mean,
            global_std=global_std,
            flip=False,
            augment=False
        )

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=os.cpu_count(), pin_memory=True
        )
        val_loader = DataLoader(
            valid_ds, batch_size=args.batch_size, shuffle=False,
            drop_last=False, num_workers=os.cpu_count(), pin_memory=True
        )

        # compute class imbalance weight
        pos = neg = 0
        for sample in train_ds:
            m = sample['mask']
            pos += m.sum().item()
            neg += (m.numel() - m.sum().item())
        pos_weight = torch.tensor(neg / (pos + 1e-8), device=device)
        logging.info(f"Fold {fold} pos_weight = {pos_weight:.4f}")

        # model, optimizer, scheduler, losses
        model = UNet(n_channels=1, n_classes=1, bilinear=args.bilinear)
        model.to(device=device, memory_format=torch.channels_last)

        optimizer = optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )

        bce_fn  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        dice_fn = dice_loss

        # WandB
        wandb.finish()
        run = wandb.init(
            project="U-Net-improved",
            name=f"fold_{fold}",
            reinit=True
        )
        run.config.update(vars(args))

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss = 0.0
            pbar = tqdm(
                train_loader,
                desc=f"[Fold {fold}] Epoch {epoch}/{args.epochs}",
                unit="batch"
            )
            for batch in pbar:
                imgs = batch['image'].to(device)
                msks = batch['mask'].float().to(device)

                preds     = model(imgs)
                loss_bce  = bce_fn(preds.squeeze(1), msks)
                loss_dice = dice_fn(torch.sigmoid(preds.squeeze(1)), msks)
                loss      = 0.4 * loss_bce + 0.6 * loss_dice

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

            avg_train_loss = epoch_loss / len(train_loader)

            # validation
            model.eval()
            val_dice = evaluate(model, val_loader, device, args.amp)
            scheduler.step(val_dice)

            logging.info(
                f"Fold {fold} • Epoch {epoch}: "
                f"Train Loss={avg_train_loss:.4f}, "
                f"Val Dice={val_dice:.4f}, "
                f"LR={optimizer.param_groups[0]['lr']:.2e}"
            )
            wandb.log({
                "train/avg_loss": avg_train_loss,
                "val/dice":       val_dice,
                "epoch":          epoch,
                "lr":             optimizer.param_groups[0]['lr']
            })

            # save sample predictions every K epochs
            out_dir = Path(args.output_dir) / f"fold{fold}"
            if epoch % args.save_interval == 0:
                save_predictions(model, val_loader, device, out_dir,
                             epoch, num_batches=args.vis_batches)

            # checkpointing
            if args.save_ckpt and (
               epoch % args.checkpoint_interval == 0 or epoch == args.epochs
            ):
                DIR_CHECKPOINT.mkdir(exist_ok=True)
                if epoch == args.epochs:
                    fname = f"fold{fold}.pth"
                else:
                    fname = f"fold{fold}_ep{epoch}.pth"
                torch.save(model.state_dict(), DIR_CHECKPOINT / fname)

        run.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs",              type=int,   default=20)
    parser.add_argument("-b", "--batch-size",          type=int,   default=4)
    parser.add_argument("-l", "--learning-rate",       type=float, default=1e-4)
    parser.add_argument("-s", "--scale",               type=float, default=0.5)
    parser.add_argument("--num-folds",                 type=int,   default=5)
    parser.add_argument("--bilinear",                  action="store_true")
    parser.add_argument("--weight-decay",              type=float, default=1e-8)
    parser.add_argument("--grad-clip",                 type=float, default=1.0)

    parser.add_argument("--save-ckpt",                 action="store_true",
                        help="Save model checkpoints")
    parser.add_argument("--checkpoint-interval",       type=int,   default=10,
                        help="Interval (in epochs) for intermediate checkpoints")

    parser.add_argument("--amp",                       action="store_true",
                        help="Use torch.cuda.amp in evaluation")

    parser.add_argument("--output-dir",                type=str,   default="outputs",
                        help="Where to save predicted images")
    parser.add_argument("--vis-batches",               type=int,   default=3,
                        help="How many validation batches to dump each epoch")
    parser.add_argument("--save-interval", type=int, default=1,
                        help="only dump images every K epochs")

    parser.add_argument("--augment",                   action="store_true",
                        help="Apply on-the-fly data augmentation on train set")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    train_model_cv(args)

if __name__ == "__main__":
    main()
