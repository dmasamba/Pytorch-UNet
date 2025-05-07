# U-Net: Semantic segmentation with PyTorch
<a href="#"><img src="https://img.shields.io/github/actions/workflow/status/milesial/PyTorch-UNet/main.yml?logo=github&style=for-the-badge" /></a>
<a href="https://hub.docker.com/r/milesial/unet"><img src="https://img.shields.io/badge/docker%20image-available-blue?logo=Docker&style=for-the-badge" /></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v1.13+-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/python-v3.6+-blue.svg?logo=python&style=for-the-badge" /></a>

![input and output for a random image in the test dataset](https://i.imgur.com/GD8FcB7.png)


Customized implementation of the [U-Net](https://arxiv.org/abs/1505.04597) in PyTorch for Kaggle's [Carvana Image Masking Challenge](https://www.kaggle.com/c/carvana-image-masking-challenge) from high definition images.

- [Quick start](#quick-start)
  - [Without Docker](#without-docker)
  - [With Docker](#with-docker)
- [Description](#description)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Prediction](#prediction)
  - [Docker](#docker)
- [Weights & Biases](#weights--biases)
- [Pretrained model](#pretrained-model)
- [Data](#data)

## Quick start

### Without Docker

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch 1.13 or later](https://pytorch.org/get-started/locally/)

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download the data and run training:
```bash
bash scripts/download_data.sh
python train.py --amp
```

### With Docker

1. [Install Docker 19.03 or later:](https://docs.docker.com/get-docker/)
```bash
curl https://get.docker.com | sh && sudo systemctl --now enable docker
```
2. [Install the NVIDIA container toolkit:](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```
3. [Download and run the image:](https://hub.docker.com/repository/docker/milesial/unet)
```bash
sudo docker run --rm --shm-size=8g --ulimit memlock=-1 --gpus all -it milesial/unet
```

4. Download the data and run training:
```bash
bash scripts/download_data.sh
python train.py --amp
```

## Description
This model was trained from scratch with 5k images and scored a [Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) of 0.988423 on over 100k test images.

It can be easily used for multiclass segmentation, portrait segmentation, medical segmentation, ...


## Usage
**Note : Use Python 3.6 or newer**

### Data Preparation

1. **Download the data**  
   Download your dataset and place the images and masks in `data/imgs` and `data/masks` respectively.  
   For Carvana, use the helper script:
   ```bash
   bash scripts/download_data.sh
   ```

2. **Clean missing images**  
   Ensure that every image has a corresponding mask and vice versa.  
   You can use a script like:
   ```python
   # clean_missing.py
   import os
   imgs = set(os.path.splitext(f)[0] for f in os.listdir('data/imgs'))
   masks = set(os.path.splitext(f)[0] for f in os.listdir('data/masks'))
   missing_imgs = masks - imgs
   missing_masks = imgs - masks
   for f in missing_imgs:
       print(f"Missing image for mask: {f}")
   for f in missing_masks:
       print(f"Missing mask for image: {f}")
   ```
   Remove or fix any missing pairs before proceeding.

3. **Generate k-fold splits**  
   To use k-fold cross-validation, generate split files for each fold:
   ```python
   # generate_kfold.py
   import os, random
   from sklearn.model_selection import KFold
   img_dir = 'data/imgs'
   all_ids = [os.path.splitext(f)[0] for f in os.listdir(img_dir) if not f.startswith('.')]
   k = 5
   kf = KFold(n_splits=k, shuffle=True, random_state=42)
   for fold, (train_idx, val_idx) in enumerate(kf.split(all_ids)):
       with open(f'data/train_new{fold}.txt', 'w') as f:
           for idx in train_idx:
               f.write(all_ids[idx] + '\n')
       with open(f'data/valid_new{fold}.txt', 'w') as f:
           for idx in val_idx:
               f.write(all_ids[idx] + '\n')
   ```
   This will create `train_new{fold}.txt` and `valid_new{fold}.txt` for each fold.

### Training

Train the model using k-fold cross-validation:
```bash
python train.py --num-folds 5 --epochs 5 --batch-size 4 --amp
```
- `--num-folds`: Number of folds for cross-validation (default: 5)
- `--amp`: Use automatic mixed precision for faster and memory-efficient training
- Other options: `--learning-rate`, `--scale`, `--classes`, etc.

**Notes:**
- The code will automatically use the k-fold split files (`train_new{fold}.txt`, `valid_new{fold}.txt`).
- Images are normalized to zero mean and unit variance per slice.
- Training and validation progress is logged to Weights & Biases (wandb).

### Prediction

After training, you can predict masks for new images:
```bash
python predict.py -m MODEL.pth -i image1.png image2.png --viz
```
See `python predict.py -h` for all options.

### Docker

A docker image containing the code and dependencies is available:
```bash
docker run -it --rm --shm-size=8g --ulimit memlock=-1 --gpus all milesial/unet
```

---

## Weights & Biases

The training progress can be visualized in real-time using [Weights & Biases](https://wandb.ai/).  Loss curves, validation curves, weights and gradient histograms, as well as predicted masks are logged to the platform.

When launching a training, a link will be printed in the console. Click on it to go to your dashboard. If you have an existing W&B account, you can link it
 by setting the `WANDB_API_KEY` environment variable. If not, it will create an anonymous run which is automatically deleted after 7 days.


## Pretrained model
A [pretrained model](https://github.com/milesial/Pytorch-UNet/releases/tag/v3.0) is available for the Carvana dataset. It can also be loaded from torch.hub:

```python
net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)
```
Available scales are 0.5 and 1.0.

## Data
The Carvana data is available on the [Kaggle website](https://www.kaggle.com/c/carvana-image-masking-challenge/data).

You can also download it using the helper script:

```
bash scripts/download_data.sh
```

The input images and target masks should be in the `data/imgs` and `data/masks` folders respectively (note that the `imgs` and `masks` folder should not contain any sub-folder or any other files, due to the greedy data-loader). For Carvana, images are RGB and masks are black and white.

You can use your own dataset as long as you make sure it is loaded properly in `utils/data_loading.py`.


---

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox:

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
