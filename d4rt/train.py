"""
PyTorch Lightning Module for D4RT Training
"""

import torch
import torch.nn as nn
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from typing import Dict, Any, Optional

from .models.d4rt_model import D4RTModel
from .utils.losses import D4RTLoss


class D4RTTrainLit(L.LightningModule):
    """PyTorch Lightning module for training D4RT"""
    
    def __init__(
        self,
        # Model config
        img_size=256,
        patch_size=16,
        encoder_embed_dim=1408,  # ViT-g: 1408 embed_dim as per paper
        encoder_depth=40,  # ViT-g: 40 layers as per paper
        encoder_num_heads=16,  # ViT-g: 16 heads as per paper
        decoder_dim=512,
        decoder_num_heads=8,
        decoder_num_layers=8,  # 8 layers as per paper
        max_frames=100,
        # Loss config (weights from paper)
        lambda_3d=1.0,
        lambda_2d=0.1,
        lambda_normal=0.5,  # Updated from 0.1 to 0.5 as per paper
        lambda_visibility=0.1,
        lambda_motion=0.1,
        lambda_confidence=0.2,  # Updated from 0.01 to 0.2 as per paper
        # Training config
        lr=1e-4,
        weight_decay=0.03,
        max_epochs=100,
        warmup_steps=2500,  # Warmup steps as per paper
        max_steps=500000,  # Total training steps as per paper
        # Other
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = D4RTModel(
            img_size=img_size,
            patch_size=patch_size,
            encoder_embed_dim=encoder_embed_dim,
            encoder_depth=encoder_depth,
            encoder_num_heads=encoder_num_heads,
            decoder_dim=decoder_dim,
            decoder_num_heads=decoder_num_heads,
            decoder_num_layers=decoder_num_layers,
            max_frames=max_frames
        )
        
        # Loss
        self.criterion = D4RTLoss(
            lambda_3d=lambda_3d,
            lambda_2d=lambda_2d,
            lambda_normal=lambda_normal,
            lambda_visibility=lambda_visibility,
            lambda_motion=lambda_motion,
            lambda_confidence=lambda_confidence,
            query_dim=decoder_dim
        )
        
        # Training config
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
    
    def forward(self, batch):
        """Forward pass"""
        video = batch['video']
        coords_uv = batch['coords_uv']
        t_src = batch['t_src']
        t_tgt = batch['t_tgt']
        t_cam = batch['t_cam']
        aspect_ratio = batch.get('aspect_ratio', None)
        video_orig = batch.get('video_orig', None)  # Original resolution video for patch extraction
        
        outputs = self.model(
            video=video,
            coords_uv=coords_uv,
            t_src=t_src,
            t_tgt=t_tgt,
            t_cam=t_cam,
            aspect_ratio=aspect_ratio,
            video_orig=video_orig
        )
        
        return outputs
    
    def extract_gt_data(self, batch):
        """
        Extract ground truth data from PointOdysseyDataset batch format
        
        Since queries are randomly sampled coordinates (not from trajectories),
        we extract GT by unprojecting depth maps at query locations.
        
        Args:
            batch: batch dictionary from PointOdysseyDataset
            
        Returns:
            Dictionary with gt_3d, gt_2d, intrinsics, gt_visibility, mask
        """
        # Get data from batch
        depths = batch.get('depths')        # (B, S, 1, H, W) or (B, S, H, W)
        pix_T_cams = batch.get('pix_T_cams')  # (B, S, 3, 3)
        coords_uv = batch['coords_uv']      # (B, num_queries, 2) normalized [0,1]
        t_tgt = batch['t_tgt']              # (B, num_queries)
        
        B, S = depths.shape[:2]
        if len(depths.shape) == 5:
            # (B, S, 1, H, W) -> (B, S, H, W)
            depths = depths.squeeze(2)
        H, W = depths.shape[2:4]
        num_queries = coords_uv.shape[1]
        
        # Convert normalized coordinates to pixel coordinates
        coords_pix = coords_uv * torch.tensor([W, H], device=coords_uv.device, dtype=coords_uv.dtype)  # (B, num_queries, 2)
        u_pix = coords_pix[:, :, 0].long()  # (B, num_queries)
        v_pix = coords_pix[:, :, 1].long()  # (B, num_queries)
        u_pix = torch.clamp(u_pix, 0, W - 1)
        v_pix = torch.clamp(v_pix, 0, H - 1)
        
        # Clamp t_tgt to valid range
        t_tgt_clamped = torch.clamp(t_tgt, 0, S - 1)  # (B, num_queries)
        batch_indices = torch.arange(B, device=depths.device).unsqueeze(1).expand(B, num_queries)  # (B, num_queries)
        
        # Extract depth values at query locations
        depth_values = depths[batch_indices, t_tgt_clamped, v_pix, u_pix]  # (B, num_queries)
        
        # Extract intrinsics for target frames
        intrinsics_per_query = pix_T_cams[batch_indices, t_tgt_clamped]  # (B, num_queries, 3, 3)
        # Use intrinsics from first query (assuming same intrinsics within a frame)
        intrinsics = intrinsics_per_query[:, 0]  # (B, 3, 3)
        
        # Unproject to 3D: convert pixel coordinates + depth to 3D camera coordinates
        fx = intrinsics[:, 0, 0].unsqueeze(1)  # (B, 1)
        fy = intrinsics[:, 1, 1].unsqueeze(1)  # (B, 1)
        cx = intrinsics[:, 0, 2].unsqueeze(1)  # (B, 1)
        cy = intrinsics[:, 1, 2].unsqueeze(1)  # (B, 1)
        
        # Convert to float for computation
        u_float = coords_pix[:, :, 0]  # (B, num_queries)
        v_float = coords_pix[:, :, 1]  # (B, num_queries)
        
        # Unproject: x = (u - cx) * z / fx, y = (v - cy) * z / fy, z = depth
        z = depth_values  # (B, num_queries)
        x = (u_float - cx) * z / fx  # (B, num_queries)
        y = (v_float - cy) * z / fy  # (B, num_queries)
        
        gt_3d = torch.stack([x, y, z], dim=-1)  # (B, num_queries, 3)
        gt_2d = coords_pix  # (B, num_queries, 2) - use pixel coordinates as GT 2D
        
        # Create mask based on valid depth values (non-zero, reasonable range)
        mask = (depth_values > 1e-3) & (depth_values < 1000.0)  # (B, num_queries)
        
        # Visibility: assume visible if depth is valid (can be improved with actual visibility data)
        gt_visibility = mask.float()  # (B, num_queries)
        
        return {
            'gt_3d': gt_3d,
            'gt_2d': gt_2d,
            'intrinsics': intrinsics,
            'gt_visibility': gt_visibility,
            'mask': mask
        }
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        # Forward pass
        outputs = self.forward(batch)
        pred_3d = outputs['coords_3d']
        pred_2d = outputs.get('coords_2d')
        pred_visibility_logits = outputs.get('visibility_logits')  # Use logits for loss computation
        pred_motion = outputs.get('motion')
        pred_normal = outputs.get('normal')
        pred_confidence = outputs.get('confidence')
        
        # Extract ground truth data from PointOdysseyDataset format
        gt_data = self.extract_gt_data(batch)
        gt_3d = gt_data['gt_3d']
        gt_2d = gt_data['gt_2d']
        gt_visibility = gt_data['gt_visibility']
        mask = gt_data['mask']
        
        # Optional: gt_normal, gt_motion (not available in PointOdysseyDataset)
        gt_normal = batch.get('gt_normal')
        gt_motion = batch.get('gt_motion')
        
        losses = self.criterion(
            pred_3d=pred_3d,
            pred_2d=pred_2d,
            pred_visibility_logits=pred_visibility_logits,  # Use logits for loss computation
            pred_motion=pred_motion,
            pred_normal=pred_normal,
            pred_confidence=pred_confidence,
            gt_3d=gt_3d,
            gt_2d=gt_2d,
            gt_visibility=gt_visibility,
            gt_motion=gt_motion,
            gt_normal=gt_normal,
            mask=mask
        )
        
        # Log losses
        for key, value in losses.items():
            self.log(f'train/{key}', value, on_step=True, on_epoch=True, prog_bar=True)
        
        return losses['loss']
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        outputs = self.forward(batch)
        pred_3d = outputs['coords_3d']
        pred_2d = outputs.get('coords_2d')
        pred_visibility_logits = outputs.get('visibility_logits')  # Use logits for loss computation
        pred_motion = outputs.get('motion')
        pred_normal = outputs.get('normal')
        pred_confidence = outputs.get('confidence')
        
        # Extract ground truth data from PointOdysseyDataset format
        gt_data = self.extract_gt_data(batch)
        gt_3d = gt_data['gt_3d']
        gt_2d = gt_data['gt_2d']
        gt_visibility = gt_data['gt_visibility']
        mask = gt_data['mask']
        
        # Optional: gt_normal, gt_motion (not available in PointOdysseyDataset)
        gt_normal = batch.get('gt_normal')
        gt_motion = batch.get('gt_motion')
        
        losses = self.criterion(
            pred_3d=pred_3d,
            pred_2d=pred_2d,
            pred_visibility_logits=pred_visibility_logits,  # Use logits for loss computation
            pred_motion=pred_motion,
            pred_normal=pred_normal,
            pred_confidence=pred_confidence,
            gt_3d=gt_3d,
            gt_2d=gt_2d,
            gt_visibility=gt_visibility,
            gt_motion=gt_motion,
            gt_normal=gt_normal,
            mask=mask
        )
        
        # Log losses
        for key, value in losses.items():
            self.log(f'val/{key}', value, on_step=False, on_epoch=True, prog_bar=True)
        
        return losses['loss']
    
    def configure_optimizers(self):
        """
        Configure optimizer and scheduler
        Following D4RT paper:
        - Warmup: 2500 steps linear warmup
        - Peak LR: 1e-4
        - Schedule: Cosine Annealing
        - Final LR: 1e-6
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Use step-based scheduler with warmup + cosine annealing as per paper
        # Paper uses 500k steps total, with 2.5k warmup
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
        
        # Warmup scheduler: linear from 0 to peak LR
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-6 / self.lr,  # Start from very small LR
            end_factor=1.0,
            total_iters=self.warmup_steps
        )
        
        # Cosine annealing scheduler: from peak LR to final LR (1e-6)
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_steps - self.warmup_steps,
            eta_min=1e-6
        )
        
        # Sequential: warmup then cosine annealing
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps]
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Step-based, not epoch-based
                'frequency': 1
            }
        }

