"""
Testing script for D4RT using Lightning architecture
"""

import argparse
import os
import sys
import torch
import lightning as L

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from d4rt.test import D4RTTestLit
from d4rt.data.datamodule import D4RTDataModule


def main():
    parser = argparse.ArgumentParser(description="Test D4RT model")
    
    # Data arguments
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to test data")
    parser.add_argument("--num_queries", type=int, default=2048, help="Number of queries per sample")
    parser.add_argument("--img_size", type=int, default=256, help="Input image size")
    parser.add_argument("--max_frames", type=int, default=100, help="Maximum number of frames")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # Model arguments (should match training config)
    parser.add_argument("--encoder_embed_dim", type=int, default=768, help="Encoder embedding dimension")
    parser.add_argument("--encoder_depth", type=int, default=12, help="Encoder depth")
    parser.add_argument("--encoder_num_heads", type=int, default=12, help="Encoder number of heads")
    parser.add_argument("--decoder_dim", type=int, default=512, help="Decoder dimension")
    parser.add_argument("--decoder_num_heads", type=int, default=8, help="Decoder number of heads")
    parser.add_argument("--decoder_num_layers", type=int, default=6, help="Decoder number of layers")
    
    # Checkpoint
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    
    # Testing arguments
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator type")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Precision")
    
    # Loss weights (for evaluation metrics)
    parser.add_argument("--lambda_3d", type=float, default=1.0, help="3D loss weight")
    parser.add_argument("--lambda_2d", type=float, default=0.1, help="2D projection loss weight")
    parser.add_argument("--lambda_normal", type=float, default=0.1, help="Normal loss weight")
    parser.add_argument("--lambda_visibility", type=float, default=0.1, help="Visibility loss weight")
    parser.add_argument("--lambda_motion", type=float, default=0.1, help="Motion loss weight")
    parser.add_argument("--lambda_confidence", type=float, default=0.01, help="Confidence loss weight")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="test_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create data module
    test_video_paths = [args.test_data_path]  # Replace with actual list of paths
    
    datamodule = D4RTDataModule(
        train_video_paths=test_video_paths,  # Reuse for test
        val_video_paths=None,
        num_queries=args.num_queries,
        img_size=args.img_size,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    model = D4RTTestLit(
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
        checkpoint_path=None  # Will load from ckpt via trainer
    )
    
    # Create trainer
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False  # Disable logging for test
    )
    
    # Setup data
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()
    
    # Test
    results = trainer.test(model, dataloaders=test_loader, ckpt_path=args.ckpt)
    
    # Print results
    if results and len(results) > 0:
        print("\n=== Test Results ===")
        for key, value in results[0].items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.6f}")
        print("==================\n")
    else:
        print("No test results returned.")
    
    # Save results
    import json
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(results[0] if results else {}, f, indent=2)


if __name__ == "__main__":
    main()

