"""
Loss functions for D4RT training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class D4RTLoss(nn.Module):
    """
    Composite loss function for D4RT:
    - L_3D: Main 3D coordinate loss with preprocessing and transformation
    - Auxiliary losses: 2D projection, normal, visibility, motion, confidence
    """
    
    def __init__(
        self,
        lambda_3d=1.0,
        lambda_2d=0.1,
        lambda_normal=0.1,
        lambda_visibility=0.1,
        lambda_motion=0.1,
        lambda_confidence=0.01,
        depth_normalize=True,
        use_log_transform=True,
        query_dim=512
    ):
        super().__init__()
        self.lambda_3d = lambda_3d
        self.lambda_2d = lambda_2d
        self.lambda_normal = lambda_normal
        self.lambda_visibility = lambda_visibility
        self.lambda_motion = lambda_motion
        self.lambda_confidence = lambda_confidence
        self.depth_normalize = depth_normalize
        self.use_log_transform = use_log_transform
        
        # Visibility prediction head (if needed)
        self.visibility_head = nn.Sequential(
            nn.Linear(query_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Confidence prediction head
        self.confidence_head = nn.Sequential(
            nn.Linear(query_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def log_transform(self, x):
        """Apply sign(x) * log(1 + |x|) transformation"""
        sign = torch.sign(x)
        abs_x = torch.abs(x)
        return sign * torch.log(1.0 + abs_x)
    
    def compute_l3d_loss(self, pred_3d, gt_3d, mask=None):
        """
        Main 3D coordinate loss with preprocessing and transformation
        
        Args:
            pred_3d: (B, N, 3) predicted 3D coordinates
            gt_3d: (B, N, 3) ground truth 3D coordinates
            mask: (B, N) optional mask for valid points
        """
        # Normalize by mean depth if enabled
        if self.depth_normalize:
            # Compute mean depth (z-coordinate)
            pred_depth_mean = pred_3d[:, :, 2].mean(dim=-1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
            gt_depth_mean = gt_3d[:, :, 2].mean(dim=-1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
            
            # Normalize
            pred_3d_normalized = pred_3d / (pred_depth_mean + 1e-6)
            gt_3d_normalized = gt_3d / (gt_depth_mean + 1e-6)
        else:
            pred_3d_normalized = pred_3d
            gt_3d_normalized = gt_3d
        
        # Apply log transformation if enabled
        if self.use_log_transform:
            pred_3d_transformed = self.log_transform(pred_3d_normalized)
            gt_3d_transformed = self.log_transform(gt_3d_normalized)
        else:
            pred_3d_transformed = pred_3d_normalized
            gt_3d_transformed = gt_3d_normalized
        
        # L1 loss
        loss = F.l1_loss(pred_3d_transformed, gt_3d_transformed, reduction='none')  # (B, N, 3)
        loss = loss.mean(dim=-1)  # (B, N)
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
        
        return loss
    
    def compute_2d_projection_loss(self, pred_3d, gt_2d, intrinsics, mask=None):
        """
        2D projection loss
        
        Args:
            pred_3d: (B, N, 3) predicted 3D coordinates
            gt_2d: (B, N, 2) ground truth 2D coordinates
            intrinsics: (B, 3, 3) camera intrinsics
            mask: (B, N) optional mask
        """
        from .geometry import project_3d_to_2d
        
        pred_2d, _ = project_3d_to_2d(pred_3d, intrinsics)
        loss = F.l1_loss(pred_2d, gt_2d, reduction='none').mean(dim=-1)  # (B, N)
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
        
        return loss
    
    def compute_normal_loss(self, pred_3d, gt_normal, mask=None):
        """
        Surface normal cosine similarity loss
        
        Args:
            pred_3d: (B, N, 3) predicted 3D points
            gt_normal: (B, N, 3) ground truth surface normals
            mask: (B, N) optional mask
        """
        from .geometry import compute_surface_normal
        
        pred_normal = compute_surface_normal(pred_3d)
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(pred_normal, gt_normal, dim=-1)  # (B, N)
        loss = 1.0 - cos_sim  # Convert similarity to loss
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
        
        return loss
    
    def compute_visibility_loss(self, query_features, gt_visibility):
        """
        Visibility prediction loss (Binary Cross-Entropy)
        
        Args:
            query_features: (B, N, D) query features
            gt_visibility: (B, N) ground truth visibility (0 or 1)
        """
        visibility_pred = self.visibility_head(query_features).squeeze(-1)  # (B, N)
        loss = F.binary_cross_entropy(visibility_pred, gt_visibility.float(), reduction='mean')
        return loss
    
    def compute_motion_loss(self, pred_3d, gt_motion, mask=None):
        """
        Motion displacement loss
        
        Args:
            pred_3d: (B, N, 3) predicted 3D coordinates
            gt_motion: (B, N, 3) ground truth motion vectors
            mask: (B, N) optional mask
        """
        # Compute motion from predicted points (difference between consecutive queries)
        # This is simplified - in practice you'd compute motion across frames
        pred_motion = torch.diff(pred_3d, dim=1)
        
        # Pad to match gt_motion shape
        if pred_motion.shape[1] < gt_motion.shape[1]:
            padding = torch.zeros(pred_motion.shape[0], 1, 3, device=pred_motion.device)
            pred_motion = torch.cat([pred_motion, padding], dim=1)
        elif pred_motion.shape[1] > gt_motion.shape[1]:
            pred_motion = pred_motion[:, :gt_motion.shape[1], :]
        
        loss = F.l1_loss(pred_motion, gt_motion, reduction='none').mean(dim=-1)  # (B, N)
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
        
        return loss
    
    def compute_confidence_loss(self, query_features):
        """
        Confidence penalty: -log(c)
        
        Args:
            query_features: (B, N, D) query features
        """
        confidence = self.confidence_head(query_features).squeeze(-1)  # (B, N)
        # Penalty: -log(c)
        loss = -torch.log(confidence + 1e-6).mean()
        return loss
    
    def forward(
        self,
        pred_3d: torch.Tensor,
        query_features: torch.Tensor,
        gt_3d: torch.Tensor,
        gt_2d: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        gt_normal: Optional[torch.Tensor] = None,
        gt_visibility: Optional[torch.Tensor] = None,
        gt_motion: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses
        
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        
        # Main 3D loss
        loss_3d = self.compute_l3d_loss(pred_3d, gt_3d, mask)
        losses['loss_3d'] = loss_3d
        
        total_loss = self.lambda_3d * loss_3d
        
        # Auxiliary losses
        if gt_2d is not None and intrinsics is not None:
            loss_2d = self.compute_2d_projection_loss(pred_3d, gt_2d, intrinsics, mask)
            losses['loss_2d'] = loss_2d
            total_loss = total_loss + self.lambda_2d * loss_2d
        
        if gt_normal is not None:
            loss_normal = self.compute_normal_loss(pred_3d, gt_normal, mask)
            losses['loss_normal'] = loss_normal
            total_loss = total_loss + self.lambda_normal * loss_normal
        
        if gt_visibility is not None:
            loss_visibility = self.compute_visibility_loss(query_features, gt_visibility)
            losses['loss_visibility'] = loss_visibility
            total_loss = total_loss + self.lambda_visibility * loss_visibility
        
        if gt_motion is not None:
            loss_motion = self.compute_motion_loss(pred_3d, gt_motion, mask)
            losses['loss_motion'] = loss_motion
            total_loss = total_loss + self.lambda_motion * loss_motion
        
        # Confidence penalty
        loss_confidence = self.compute_confidence_loss(query_features)
        losses['loss_confidence'] = loss_confidence
        total_loss = total_loss + self.lambda_confidence * loss_confidence
        
        losses['loss'] = total_loss
        
        return losses

