import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.utils as vutils
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
import numpy as np
from PIL import Image
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss, dice_loss as dice_loss_fn

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')


def read_split_file(split_file):
    files = []
    for line in open(split_file, 'r'):
        name = line.strip()
        if not name:
            continue
        # Remove extension if present (e.g., .npy, .png, .jpg)
        name = os.path.splitext(name)[0]
        # Skip hidden/system files or files like .keep
        if name.startswith('.'):
            continue
        files.append(name)
    return files

def random_flip(img: np.ndarray, mask: np.ndarray):
    # img: (C, H, W), mask: (H, W)
    if random.random() > 0.5:
        img  = np.flip(img, axis=2).copy()  # horizontal flip
        mask = np.flip(mask, axis=1).copy()
    if random.random() > 0.5:
        img  = np.flip(img, axis=1).copy()  # vertical flip
        mask = np.flip(mask, axis=0).copy()
    return img, mask


class CovidDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, masks_dir, img_ids, scale=1.0, transform=None, use_random_flip=False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_ids = img_ids
        self.scale = scale
        self.transform = transform
        self.use_random_flip = use_random_flip

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        fname = self.img_ids[idx]
        image_file = str(self.images_dir / (fname + '.npy'))
        mask_file = str(self.masks_dir / (fname + '.npy'))
        image = np.load(image_file).astype(np.float32)
        mask = np.load(mask_file).astype(np.float32)
        # Windowing and normalization for CT (lung window)
        wmin, wmax = -1000, 400
        image = np.clip(image, wmin, wmax)
        image = (image - wmin) / (wmax - wmin)
        # Normalize to zero mean and unit variance
        image = (image - np.mean(image)) / (np.std(image) + 1e-8)
        # Resize if needed
        if self.scale != 1.0:
            new_shape = (int(image.shape[1] * self.scale), int(image.shape[0] * self.scale))
            image = np.array(Image.fromarray(image).resize(new_shape, resample=Image.BICUBIC))
            mask = np.array(Image.fromarray(mask).resize(new_shape, resample=Image.NEAREST))
        # Add channel dimension if needed
        if image.ndim == 2:
            image = image[None, ...]
        # Convert mask to integer class labels if necessary
        mask = (mask > 0.5).astype(np.uint8)  # For binary masks
        # Apply random_flip if enabled
        if self.use_random_flip:
            image, mask = random_flip(image, mask)
        # Optionally apply transforms
        if self.transform:
            image, mask = self.transform(image, mask)
        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()
        return {'image': image.contiguous(), 'mask': mask.contiguous()}


def get_dataset_from_split(img_dir, mask_dir, file_list, img_scale, use_random_flip=False):
    return CovidDataset(img_dir, mask_dir, file_list, img_scale, use_random_flip=use_random_flip)


def save_visualizations(images, true_masks, pred_masks, fold, epoch, step, max_samples=4):
    import os
    from torchvision.utils import save_image

    base_dir = Path('visualizations') / f'fold_{fold}'
    input_dir = base_dir / 'inputs'
    gt_dir = base_dir / 'ground_truths'
    pred_dir = base_dir / 'predictions'
    input_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(max_samples, images.shape[0])):
        # Save input image (assume single channel)
        img = images[i]
        if img.shape[0] == 1:
            img = img[0]
        # Normalize image for visualization (scale to [0,1])
        img_min, img_max = img.min(), img.max()
        img_vis = (img - img_min) / (img_max - img_min + 1e-8)
        img_path = input_dir / f'epoch{epoch}_step{step}_img{i}.png'
        save_image(img_vis.unsqueeze(0), img_path)

        # Save ground truth mask
        mask = true_masks[i].float()
        mask_path = gt_dir / f'epoch{epoch}_step{step}_gt{i}.png'
        save_image(mask.unsqueeze(0), mask_path)

        # Save predicted mask
        pred = pred_masks[i].float()
        pred_path = pred_dir / f'epoch{epoch}_step{step}_pred{i}.png'
        save_image(pred.unsqueeze(0), pred_path)


def train_model_cv(
        model_fn,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        num_folds: int = 5
):
    for fold in range(num_folds):
        train_split = f'./data/train_new{fold}.txt'
        valid_split = f'./data/valid_new{fold}.txt'
        train_ids = read_split_file(train_split)
        valid_ids = read_split_file(valid_split)

        # Enable random_flip for training set only
        train_dataset = get_dataset_from_split(dir_img, dir_mask, train_ids, img_scale, use_random_flip=True)
        valid_dataset = get_dataset_from_split(dir_img, dir_mask, valid_ids, img_scale, use_random_flip=False)

        loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
        val_loader = DataLoader(valid_dataset, shuffle=False, drop_last=True, **loader_args)

        # Ensure a new wandb run for each fold
        if wandb.run is not None:
            wandb.finish()
        experiment = wandb.init(project='U-Net', resume='allow', anonymous='must', name=f'fold_{fold}')
        experiment.config.update(
            dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                 save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp, fold=fold),
            allow_val_change=True
        )

        logging.info(f'''Starting training on COVID CT scan dataset, fold {fold}:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {len(train_dataset)}
            Validation size: {len(valid_dataset)}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Images scaling:  {img_scale}
            Mixed Precision: {amp}
        ''')

        model = model_fn()
        model.to(memory_format=torch.channels_last)
        model.to(device=device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
        if device.type == 'cuda':
            grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        else:
            grad_scaler = torch.amp.GradScaler(enabled=amp)
        criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
        global_step = 0

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0
            with tqdm(total=len(train_dataset), desc=f'Fold {fold} Epoch {epoch}/{epochs}', unit='img') as pbar:
                for batch in train_loader:
                    images, true_masks = batch['image'], batch['mask']

                    assert images.shape[1] == model.n_channels, \
                        f'Network has been defined with {model.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    true_masks = true_masks.to(device=device, dtype=torch.long)

                    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                        masks_pred = model(images)
                        if not torch.isfinite(masks_pred).all():
                            raise ValueError("NaN in model output. Check model weights and input data.")

                        if model.n_classes > 1:
                            ce_loss = criterion(masks_pred, true_masks)
                            dice = dice_loss_fn(
                                F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )
                            loss = 0.4 * ce_loss + 0.6 * dice
                        else:
                            ce_loss = criterion(masks_pred.squeeze(1), true_masks.float())
                            dice = dice_loss_fn(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                            loss = 0.4 * ce_loss + 0.6 * dice

                    if not torch.isfinite(loss):
                        raise ValueError("NaN loss encountered. Check your data and model outputs.")

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pbar.update(images.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()
                    experiment.log({
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch,
                        'fold': fold
                    })
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    division_step = (len(train_dataset) // (5 * batch_size))
                    if division_step > 0:
                        if global_step % division_step == 0:
                            histograms = {}
                            for tag, value in model.named_parameters():
                                tag = tag.replace('/', '.')
                                if not (torch.isinf(value) | torch.isnan(value)).any():
                                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                            val_score = evaluate(model, val_loader, device, amp)
                            scheduler.step(val_score)

                            # Save visualizations for the first batch of validation set
                            model.eval()
                            with torch.no_grad():
                                val_batch = next(iter(val_loader))
                                val_images = val_batch['image'].to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                                val_true_masks = val_batch['mask'].to(device=device, dtype=torch.long)
                                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                                    val_pred = model(val_images)
                                    if model.n_classes == 1:
                                        val_pred_mask = (F.sigmoid(val_pred.squeeze(1)) > 0.5).long()
                                    else:
                                        val_pred_mask = val_pred.argmax(dim=1)
                                save_visualizations(
                                    val_images.cpu(),
                                    val_true_masks.cpu(),
                                    val_pred_mask.cpu(),
                                    fold=fold,
                                    epoch=epoch,
                                    step=global_step,
                                    max_samples=4
                                )
                            model.train()

                            logging.info(f'Validation Dice score (fold {fold}): {val_score}')
                            try:
                                experiment.log({
                                    'learning rate': optimizer.param_groups[0]['lr'],
                                    'validation Dice': val_score,
                                    'images': wandb.Image(images[0].cpu()),
                                    'masks': {
                                        'true': wandb.Image(true_masks[0].float().cpu()),
                                        'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                    },
                                    'step': global_step,
                                    'epoch': epoch,
                                    'fold': fold,
                                    **histograms
                                })
                            except:
                                pass

                # After each epoch, evaluate on the validation set and log the Dice score
                val_score = evaluate(model, val_loader, device, amp)
                logging.info(f'Validation Dice score (fold {fold}, epoch {epoch}): {val_score}')
                experiment.log({
                    'validation Dice (epoch end)': val_score,
                    'epoch': epoch,
                    'fold': fold
                })

            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                torch.save(state_dict, str(dir_checkpoint / f'checkpoint_fold{fold}_epoch{epoch}.pth'))
                logging.info(f'Checkpoint fold {fold} epoch {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--num-folds', type=int, default=5, help='Number of folds for cross-validation')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    def model_fn():
        return UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)

    try:
        train_model_cv(
            model_fn=model_fn,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            amp=args.amp,
            num_folds=args.num_folds
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()

        def model_fn_ckpt():
            m = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
            m.use_checkpointing()
            return m

        train_model_cv(
            model_fn=model_fn_ckpt,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            amp=args.amp,
            num_folds=args.num_folds
        )
