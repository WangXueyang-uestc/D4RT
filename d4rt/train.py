"""
PyTorch Lightning Module for D4RT Training
"""

import torch
import torch.nn as nn
import lightning as L
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_dim=512,
        decoder_num_heads=8,
        decoder_num_layers=6,
        max_frames=100,
        # Loss config
        lambda_3d=1.0,
        lambda_2d=0.1,
        lambda_normal=0.1,
        lambda_visibility=0.1,
        lambda_motion=0.1,
        lambda_confidence=0.01,
        # Training config
        lr=1e-4,
        weight_decay=0.03,
        max_epochs=100,
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
    
    def forward(self, batch):
        """Forward pass"""
        video = batch['video']
        coords_uv = batch['coords_uv']
        t_src = batch['t_src']
        t_tgt = batch['t_tgt']
        t_cam = batch['t_cam']
        aspect_ratio = batch.get('aspect_ratio', None)
        
        outputs = self.model(
            video=video,
            coords_uv=coords_uv,
            t_src=t_src,
            t_tgt=t_tgt,
            t_cam=t_cam,
            aspect_ratio=aspect_ratio
        )
        
        return outputs
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        # Forward pass
        outputs = self.forward(batch)
        pred_3d = outputs['coords_3d']
        query_features = outputs['query_features']
        
        # Compute losses
        gt_3d = batch.get('gt_3d')
        gt_2d = batch.get('gt_2d')
        intrinsics = batch.get('intrinsics')
        gt_normal = batch.get('gt_normal')
        gt_visibility = batch.get('gt_visibility')
        gt_motion = batch.get('gt_motion')
        mask = batch.get('mask')
        
        losses = self.criterion(
            pred_3d=pred_3d,
            query_features=query_features,
            gt_3d=gt_3d,
            gt_2d=gt_2d,
            intrinsics=intrinsics,
            gt_normal=gt_normal,
            gt_visibility=gt_visibility,
            gt_motion=gt_motion,
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
        query_features = outputs['query_features']
        
        gt_3d = batch.get('gt_3d')
        gt_2d = batch.get('gt_2d')
        intrinsics = batch.get('intrinsics')
        gt_normal = batch.get('gt_normal')
        gt_visibility = batch.get('gt_visibility')
        gt_motion = batch.get('gt_motion')
        mask = batch.get('mask')
        
        losses = self.criterion(
            pred_3d=pred_3d,
            query_features=query_features,
            gt_3d=gt_3d,
            gt_2d=gt_2d,
            intrinsics=intrinsics,
            gt_normal=gt_normal,
            gt_visibility=gt_visibility,
            gt_motion=gt_motion,
            mask=mask
        )
        
        # Log losses
        for key, value in losses.items():
            self.log(f'val/{key}', value, on_step=False, on_epoch=True, prog_bar=True)
        
        return losses['loss']
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=1e-6  # Minimum LR as specified
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

