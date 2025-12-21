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
from d4rt.data.datamodule import D4RTDataModule


def main():
    parser = argparse.ArgumentParser(description="Train D4RT model")
    
    # Data arguments
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_data_path", type=str, default=None, help="Path to validation data")
    parser.add_argument("--num_queries", type=int, default=2048, help="Number of queries per sample")
    parser.add_argument("--img_size", type=int, default=256, help="Input image size")
    parser.add_argument("--max_frames", type=int, default=100, help="Maximum number of frames")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # Model arguments
    parser.add_argument("--encoder_embed_dim", type=int, default=768, help="Encoder embedding dimension")
    parser.add_argument("--encoder_depth", type=int, default=12, help="Encoder depth")
    parser.add_argument("--encoder_num_heads", type=int, default=12, help="Encoder number of heads")
    parser.add_argument("--decoder_dim", type=int, default=512, help="Decoder dimension")
    parser.add_argument("--decoder_num_heads", type=int, default=8, help="Decoder number of heads")
    parser.add_argument("--decoder_num_layers", type=int, default=6, help="Decoder number of layers")
    
    # Training arguments
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.03, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator type")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Precision")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    
    # Loss weights
    parser.add_argument("--lambda_3d", type=float, default=1.0, help="3D loss weight")
    parser.add_argument("--lambda_2d", type=float, default=0.1, help="2D projection loss weight")
    parser.add_argument("--lambda_normal", type=float, default=0.1, help="Normal loss weight")
    parser.add_argument("--lambda_visibility", type=float, default=0.1, help="Visibility loss weight")
    parser.add_argument("--lambda_motion", type=float, default=0.1, help="Motion loss weight")
    parser.add_argument("--lambda_confidence", type=float, default=0.01, help="Confidence loss weight")
    
    # Other arguments
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--log_dir", type=str, default="lightning_logs", help="Log directory")
    
    args = parser.parse_args()
    
    # Create data module
    # Note: You'll need to provide actual video paths list
    # This is a placeholder - adjust based on your data structure
    train_video_paths = [args.train_data_path]  # Replace with actual list of paths
    val_video_paths = [args.val_data_path] if args.val_data_path else None
    
    datamodule = D4RTDataModule(
        train_video_paths=train_video_paths,
        val_video_paths=val_video_paths,
        num_queries=args.num_queries,
        img_size=args.img_size,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers
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
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.max_epochs,
        precision=args.precision,
        gradient_clip_val=args.gradient_clip_val,
        default_root_dir=args.log_dir,
        log_every_n_steps=10,
        val_check_interval=0.5,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Train
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()

