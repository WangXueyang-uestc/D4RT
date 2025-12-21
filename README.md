# D4RT: 4D Reconstruction Transformer

Implementation of D4RT (4D Reconstruction Transformer) for 4D reconstruction from video sequences.

## Overview

D4RT is a transformer-based model for 4D reconstruction that uses:
- **Query-based decoding mechanism**: Independent queries that attend to encoder features
- **Decoupled dimensions**: Spatial (u, v), temporal (t_src, t_tgt, t_cam) dimensions in query vectors
- **Encoder-Decoder architecture**: ViT encoder with alternating local/global attention, lightweight cross-attention decoder

## Key Features

### Encoder
- Vision Transformer (ViT) with alternating intra-frame local attention and global self-attention
- Additional token for encoding original video aspect ratio
- Fixed square resolution (256x256)

### Query Formulation
- Normalized 2D coordinates (u, v) with Fourier feature embedding
- Three time dimensions as learned discrete embeddings: t_src (source frame), t_tgt (target frame), t_cam (camera reference frame)
- Local RGB patch embedding (9x9 patch centered at query location)

### Decoder
- Lightweight Cross-Attention Transformer (6-8 layers)
- Independent Querying: queries do not interact with each other, only attend to encoder features
- Output: 3D coordinates via linear projection

### Loss Functions
- **Main loss (L_3D)**: L1 loss with preprocessing (normalize by mean depth) and transformation (sign(x) * log(1+|x|))
- **Auxiliary losses**:
  - 2D projection loss
  - Surface normal cosine similarity loss
  - Visibility prediction (Binary Cross-Entropy)
  - Motion displacement loss
  - Confidence penalty (-log(c))

### Training Strategy
- Sample N=2048 random queries per batch
- 30% queries on depth discontinuities or motion boundaries (Sobel operator)
- 40% samples with t_tgt = t_cam
- Optimizer: AdamW (weight decay 0.03)
- Scheduler: Cosine Annealing (LR: 1e-4 → 1e-6)

## Installation

```bash
pip install torch torchvision lightning
pip install opencv-python numpy
```

## Project Structure

```
D4RT/
├── d4rt/
│   ├── models/
│   │   ├── encoder.py          # ViT encoder with alternating attention
│   │   ├── decoder.py          # Cross-attention decoder
│   │   ├── query.py            # Query builder with Fourier features
│   │   └── d4rt_model.py       # Complete D4RT model
│   ├── data/
│   │   ├── dataset.py          # Dataset with query sampling strategy
│   │   └── datamodule.py       # Lightning DataModule
│   ├── utils/
│   │   ├── losses.py           # Loss functions
│   │   └── geometry.py         # Geometry utilities (3D-2D projection, normals)
│   ├── train.py                # Lightning training module
│   └── test.py                 # Lightning testing module
├── scripts/
│   ├── train_d4rt.py           # Training script
│   └── test_d4rt.py            # Testing script (Lightning-based)
└── README.md
```

## Usage

### Training

```bash
python scripts/train_d4rt.py \
    --train_data_path /path/to/train/data \
    --val_data_path /path/to/val/data \
    --num_queries 2048 \
    --img_size 256 \
    --batch_size 1 \
    --lr 1e-4 \
    --weight_decay 0.03 \
    --max_epochs 100 \
    --devices 1
```

### Testing

```bash
python scripts/test_d4rt.py \
    --test_data_path /path/to/test/data \
    --ckpt /path/to/checkpoint.ckpt \
    --num_queries 2048 \
    --img_size 256
```

## Dataset Format

The dataset should implement the `__getitem__` method returning a dictionary with:
- `video`: (T, C, H, W) video frames tensor
- `depth_maps`: (T, H, W) depth maps tensor
- `coords_uv`: (N, 2) normalized query coordinates [0, 1]
- `t_src`, `t_tgt`, `t_cam`: (N,) time indices
- `gt_3d`: (N, 3) ground truth 3D coordinates (optional)
- `intrinsics`: (3, 3) camera intrinsics matrix
- `aspect_ratio`: scalar aspect ratio (width/height)

Note: The current `D4RTDataset` class provides the query sampling logic but requires you to implement the actual data loading based on your data format.

## Model Configuration

Default model configuration:
- Encoder: ViT-L (768 dim, 12 layers, 12 heads)
- Decoder: 6 layers, 512 dim, 8 heads
- Query: 512 dim with Fourier features (128 dim), time embeddings (64 dim each), patch embedding (128 dim)
- Patch size: 16x16 for encoder, 9x9 for query patches

## Notes

- The local attention implementation uses a sliding window approach that may be slower for large images. Consider optimizing if needed.
- The dataset class includes boundary detection using Sobel operators but requires actual video/depth/flow data to be loaded.
- All components are designed to work with PyTorch Lightning for easy distributed training and checkpointing.

## Citation

If you use this code, please cite the D4RT paper.

