"""
Training script for D4RT
"""

import argparse
import os
import sys
import torch
import lightning as L

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from d4rt.train import D4RTTrainLit
from d4rt.data.datamodule import PointOdysseyDataModule


def main():
    parser = argparse.ArgumentParser(description="Train D4RT model")
    
    # Data arguments
    parser.add_argument("--dataset_location", type=str, default='/nas2/home/xueyangwang/PointOdyssey', help="Path to PointOdyssey dataset root")
    parser.add_argument("--train_dset", type=str, default='train', help="Training dataset split name")
    parser.add_argument("--val_dset", type=str, default='val', help="Validation dataset split name")
    parser.add_argument("--use_val", action='store_true', help="Use validation dataset (default: False)")
    parser.add_argument("--use_augs", action='store_true', help="Use data augmentations for training")
    parser.add_argument("--S", type=int, default=8, help="Number of frames per clip (48 as per D4RT paper)")
    parser.add_argument("--N", type=int, default=32, help="Number of trajectories")
    parser.add_argument("--strides", type=int, nargs='+', default=[1, 2, 4], help="Stride values for sampling")
    parser.add_argument("--clip_step", type=int, default=2, help="Step size for clip sampling")
    parser.add_argument("--quick", action='store_true', help="Quick mode (use only first sequence)")
    parser.add_argument("--verbose", action='store_true', help="Verbose output")
    parser.add_argument("--num_queries", type=int, default=2048, help="Number of queries per sample")
    parser.add_argument("--img_size", type=int, default=256, help="Input image size")
    parser.add_argument("--boundary_ratio", type=float, default=0.3, help="Ratio of queries on boundaries")
    parser.add_argument("--t_tgt_eq_t_cam_ratio", type=float, default=0.4, help="Ratio of samples with t_tgt = t_cam")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # Model arguments
    parser.add_argument("--encoder_embed_dim", type=int, default=1408, help="Encoder embedding dimension (1408 for ViT-g as per D4RT paper)")
    parser.add_argument("--encoder_depth", type=int, default=40, help="Encoder depth (40 layers for ViT-g as per D4RT paper)")
    parser.add_argument("--encoder_num_heads", type=int, default=16, help="Encoder number of heads (16 for ViT-g as per D4RT paper)")
    parser.add_argument("--decoder_dim", type=int, default=512, help="Decoder dimension")
    parser.add_argument("--decoder_num_heads", type=int, default=8, help="Decoder number of heads")
    parser.add_argument("--decoder_num_layers", type=int, default=8, help="Decoder layers (8 as per D4RT paper)")
    parser.add_argument("--max_frames", type=int, default=100, help="Maximum number of frames")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size (spatial)")
    parser.add_argument("--warmup_steps", type=int, default=2500, help="Warmup steps (2500 as per D4RT paper)")
    parser.add_argument("--max_steps", type=int, default=500000, help="Total training steps (500k as per D4RT paper)")
    
    # Training arguments
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.03, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator type")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Precision")
    parser.add_argument("--gradient_clip_val", type=float, default=10.0, help="Gradient clipping L2 norm (10.0 as per D4RT paper)")
    
    # Loss weights
    parser.add_argument("--lambda_3d", type=float, default=1.0, help="3D loss weight")
    parser.add_argument("--lambda_2d", type=float, default=0.1, help="2D projection loss weight")
    parser.add_argument("--lambda_normal", type=float, default=0.5, help="Normal loss weight (0.5 as per D4RT paper)")
    parser.add_argument("--lambda_visibility", type=float, default=0.1, help="Visibility loss weight")
    parser.add_argument("--lambda_motion", type=float, default=0.1, help="Motion loss weight")
    parser.add_argument("--lambda_confidence", type=float, default=0.2, help="Confidence loss weight (0.2 as per D4RT paper)")
    
    # Other arguments
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--log_dir", type=str, default="lightning_logs", help="Log directory")
    
    args = parser.parse_args()
    
    # Create data module using PointOdysseyDataset
    datamodule = PointOdysseyDataModule(
        dataset_location=args.dataset_location,
        train_dset=args.train_dset,
        val_dset=args.val_dset,
        use_augs=args.use_augs,
        use_val=args.use_val,
        S=args.S,
        N=args.N,
        strides=args.strides,
        clip_step=args.clip_step,
        quick=args.quick,
        verbose=args.verbose,
        img_size=args.img_size,
        num_queries=args.num_queries,
        boundary_ratio=args.boundary_ratio,
        t_tgt_eq_t_cam_ratio=args.t_tgt_eq_t_cam_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = D4RTTrainLit(
        img_size=args.img_size,
        encoder_embed_dim=args.encoder_embed_dim,
        encoder_depth=args.encoder_depth,
        encoder_num_heads=args.encoder_num_heads,
        decoder_dim=args.decoder_dim,
        decoder_num_heads=args.decoder_num_heads,
        decoder_num_layers=args.decoder_num_layers,
        max_frames=args.max_frames,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        lambda_3d=args.lambda_3d,
        lambda_2d=args.lambda_2d,
        lambda_normal=args.lambda_normal,
        lambda_visibility=args.lambda_visibility,
        lambda_motion=args.lambda_motion,
        lambda_confidence=args.lambda_confidence,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs
    )
    
    # Create trainer
    trainer_kwargs = {
        'accelerator': args.accelerator,
        'devices': args.devices,
        'max_epochs': args.max_epochs,
        'precision': args.precision,
        'gradient_clip_val': args.gradient_clip_val,
        'default_root_dir': args.log_dir,
        'log_every_n_steps': 10,
        'enable_checkpointing': True,
        'enable_progress_bar': True,
        'enable_model_summary': True,
    }
    
    # Configure validation only if use_val is True
    if args.use_val:
        trainer_kwargs['val_check_interval'] = 0.5
        trainer_kwargs['limit_val_batches'] = 1.0  # Use all validation batches
    else:
        trainer_kwargs['limit_val_batches'] = 0  # Disable validation
        trainer_kwargs['num_sanity_val_steps'] = 0  # Disable validation sanity check
    
    trainer = L.Trainer(**trainer_kwargs)
    
    # Train
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()

