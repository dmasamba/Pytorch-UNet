# Pytorch-UNet for COVID-19 CT Scan Segmentation

A PyTorch implementation of the U-Net architecture for semantic segmentation of lung lesion masks on COVID-19 CT slices.  
Supports data cleaning, 5-fold cross-validation, on-the-fly augmentation, mixed-precision, and per-fold checkpointing.

---

## 🚀 Features

- **Data preparation**  
  - `check_dataset_consistency.py` – Detect & remove missing or mismatched `.npy` files.  
  - `generate_kfold.py` – Split your dataset into 5 train/validation folds (`train_new{fold}.txt`, `valid_new{fold}.txt`).

- **Visualization**  
  - `visualize_covid_ct.py` – Browse sample CT images, binary masks, and overlayed masks.

- **Training**  
  - `train.py` – 5-fold CV pipeline with:  
    - Global intensity normalization  
    - Class-balanced BCE + Dice loss  
    - On-the-fly geometric & color augmentation (`--augment`)  
    - ReduceLROnPlateau scheduler  
    - Mixed-precision inference (`--amp`)  
    - Per-fold checkpointing every *K* epochs & final model  
    - Sample-prediction dumps for visual inspection  

- **Inference**  
  - `predict.py` – Run your trained model on new CT slices, output binary masks.

---

## 📁 Repository Layout

```
Pytorch-UNet/
├─ .github/                    
├─ data/                       
│   ├─ imgs/        Raw CT slices (.npy)
│   └─ masks/       Corresponding binary masks (.npy)
├─ unet/           U-Net model definition
├─ utils/          Dice score/loss and helpers
├─ check_dataset_consistency.py
├─ generate_kfold.py
├─ visualize_covid_ct.py
├─ train.py
├─ predict.py
├─ evaluate.py
├─ requirements.txt
└─ README.md
```

---

## ⚙️ Installation

1. **Clone & enter**  
   ```bash
   git clone https://github.com/dmasamba/Pytorch-UNet.git
   cd Pytorch-UNet
   ```

2. **Create virtual environment & install**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **(Optional) GPU support**  
   ```bash
   # example for CUDA 11.7
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
   ```

---

## 🗂️ Data Preparation

1. **Place your data**  
   - CT slices: `data/imgs/<ID>.npy`  
   - Masks:       `data/masks/<ID>.npy`

2. **Sanity check**  
   ```bash
   python check_dataset_consistency.py      --imgs-dir data/imgs      --masks-dir data/masks
   ```

3. **Generate 5-fold splits**  
   ```bash
   python generate_kfold.py      --imgs-dir data/imgs      --n-folds 5
   ```
   Outputs `data/train_new{0..4}.txt` and `data/valid_new{0..4}.txt`.

---

## 👁️ Visualization

Preview samples of your dataset:

```bash
python visualize_covid_ct.py   --imgs-dir   data/imgs   --masks-dir  data/masks   --n-samples  8   --output-dir figures/
```

---

## 🏋️ Training

```bash
python train.py   --epochs 100   --batch-size 16   --learning-rate 1e-3   --scale 0.5   --num-folds 5   --augment   --amp   --save-ckpt   --checkpoint-interval 10   --vis-batches 1   --output-dir outputs/
```

**Key flags**  
- `--augment` : apply default geometric & color jitter on train set  
- `--amp` : use mixed-precision during evaluation  
- `--save-ckpt` : enable checkpointing  
- `--checkpoint-interval K` : save every K epochs + final model  
- `--vis-batches M` : dump M val batches (imgs/gt/preds) per save  
- `--output-dir DIR` : root folder for visualization outputs  

Outputs:
- **Checkpoints/**  
  ```
  checkpoints/
  ├─ fold0.pth
  ├─ fold1.pth
  └─ … 
  ```
- **Predictions/**  
  ```
  outputs/
  ├─ fold0/
  │   ├─ inputs/
  │   ├─ ground_truths/
  │   └─ predictions/
  └─ fold1/ …
  ```
  Filenames `epoch{N}_batch{B}_sample{S}.png`

---

## 🔍 Inference

```bash
python predict.py   --model       checkpoints/fold0.pth   --input       new_slice.npy   --output      pred_mask.npy   --viz         # also save PNG overlay
```

---

## 📈 Logging

Integrates with [Weights & Biases](https://wandb.ai).  
Set your `WANDB_API_KEY` to log training metrics, images, and curves.

---

## 📜 License

Released under the **GPL-3.0** license.

_U-Net paper_: “U-Net: Convolutional Networks for Biomedical Image Segmentation” by Ronneberger _et al._
