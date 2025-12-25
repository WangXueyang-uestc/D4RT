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
        
        # Note: Visibility and confidence are now directly output from decoder (13-dim output)
        # These heads are kept for backward compatibility but should not be used
    
    def log_transform(self, x):
        """Apply sign(x) * log(1 + |x|) transformation"""
        sign = torch.sign(x)
        abs_x = torch.abs(x)
        return sign * torch.log(1.0 + abs_x)
    
    def compute_l3d_loss(self, pred_3d, gt_3d, confidence=None, mask=None):
        """
        Main 3D coordinate loss with preprocessing and transformation
        According to D4RT paper: c*λ_3D*L_3D (multiplied by confidence)
        
        Args:
            pred_3d: (B, N, 3) predicted 3D coordinates
            gt_3d: (B, N, 3) ground truth 3D coordinates
            confidence: (B, N) or (B, N, 1) predicted confidence values (used to weight the loss)
            mask: (B, N) optional mask for valid points
        Returns:
            loss: scalar if mask provided or confidence weighted, otherwise per-query loss for proper weighting
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
        
        # L1 loss per query
        loss_per_query = F.l1_loss(pred_3d_transformed, gt_3d_transformed, reduction='none')  # (B, N, 3)
        loss_per_query = loss_per_query.mean(dim=-1)  # (B, N) - per-query loss
        
        # Weight by confidence if provided (as per paper formula: c*λ_3D*L_3D)
        if confidence is not None:
            if confidence.dim() == 3:
                confidence = confidence.squeeze(-1)  # (B, N, 1) -> (B, N)
            loss_per_query = loss_per_query * confidence  # (B, N) - multiply per-query loss by confidence
        
        # Apply mask and average
        if mask is not None:
            loss_per_query = loss_per_query * mask
            loss = loss_per_query.sum() / (mask.sum() + 1e-6)
        else:
            loss = loss_per_query.mean()
        
        return loss
    
    def compute_2d_projection_loss(self, pred_2d, gt_2d, mask=None):
        """
        2D projection loss (L1 loss on predicted 2D coordinates)
        
        Args:
            pred_2d: (B, N, 2) predicted 2D coordinates (directly from decoder)
            gt_2d: (B, N, 2) ground truth 2D coordinates
            mask: (B, N) optional mask
        """
        loss = F.l1_loss(pred_2d, gt_2d, reduction='none').mean(dim=-1)  # (B, N)
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
        
        return loss
    
    def compute_2d_projection_loss_from_3d(self, pred_3d, gt_2d, intrinsics, mask=None):
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
    
    def compute_normal_loss(self, pred_normal, gt_normal, mask=None):
        """
        Surface normal cosine similarity loss
        
        Args:
            pred_normal: (B, N, 3) predicted surface normals (directly from decoder)
            gt_normal: (B, N, 3) ground truth surface normals
            mask: (B, N) optional mask
        """
        # Normalize normals
        pred_normal = F.normalize(pred_normal, p=2, dim=-1)
        gt_normal = F.normalize(gt_normal, p=2, dim=-1)
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(pred_normal, gt_normal, dim=-1)  # (B, N)
        loss = 1.0 - cos_sim  # Convert similarity to loss (range: 0 to 2)
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
        
        return loss
    
    def compute_visibility_loss(self, pred_visibility_logits, gt_visibility, mask=None):
        """
        Visibility prediction loss (Binary Cross-Entropy with Logits, safe for autocast)
        
        Args:
            pred_visibility_logits: (B, N) or (B, N, 1) predicted visibility logits (before sigmoid)
            gt_visibility: (B, N) ground truth visibility (0 or 1)
            mask: (B, N) optional mask
        """
        if pred_visibility_logits.dim() == 3:
            pred_visibility_logits = pred_visibility_logits.squeeze(-1)  # (B, N, 1) -> (B, N)
        
        # Use binary_cross_entropy_with_logits for autocast safety
        loss = F.binary_cross_entropy_with_logits(
            pred_visibility_logits, gt_visibility.float(), reduction='none'
        )  # (B, N)
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
        
        return loss
    
    def compute_motion_loss(self, pred_motion, gt_motion, mask=None):
        """
        Motion displacement loss (L1 loss)
        
        Args:
            pred_motion: (B, N, 3) predicted motion/displacement vectors (directly from decoder)
            gt_motion: (B, N, 3) ground truth motion vectors
            mask: (B, N) optional mask
        """
        loss = F.l1_loss(pred_motion, gt_motion, reduction='none').mean(dim=-1)  # (B, N)
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
        
        return loss
    
    def compute_confidence_loss(self, confidence, mask=None):
        """
        Confidence penalty: -log(c) (as per paper: -λ_conf*log(c))
        This prevents the model from cheating by predicting very low confidence
        
        Args:
            confidence: (B, N) or (B, N, 1) predicted confidence values (directly from decoder)
            mask: (B, N) optional mask
        """
        if confidence.dim() == 3:
            confidence = confidence.squeeze(-1)  # (B, N, 1) -> (B, N)
        
        # Penalty: -log(c)
        loss = -torch.log(confidence + 1e-6)  # (B, N)
        
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-6)
        else:
            loss = loss.mean()
        
        return loss
    
    def forward(
        self,
        pred_3d: torch.Tensor,
        pred_2d: Optional[torch.Tensor] = None,
        pred_visibility_logits: Optional[torch.Tensor] = None,
        pred_motion: Optional[torch.Tensor] = None,
        pred_normal: Optional[torch.Tensor] = None,
        pred_confidence: Optional[torch.Tensor] = None,
        gt_3d: torch.Tensor = None,
        gt_2d: Optional[torch.Tensor] = None,
        gt_visibility: Optional[torch.Tensor] = None,
        gt_motion: Optional[torch.Tensor] = None,
        gt_normal: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses according to D4RT paper formula:
        L = 1/N * sum(c*λ_3D*L_3D - λ_conf*log(c) + λ_2D*L_2D + λ_vis*L_vis + λ_disp*L_disp + λ_normal*L_normal)
        
        Args:
            pred_3d: (B, N, 3) predicted 3D coordinates
            pred_2d: (B, N, 2) predicted 2D coordinates (from decoder)
            pred_visibility_logits: (B, N) or (B, N, 1) predicted visibility logits (before sigmoid)
            pred_motion: (B, N, 3) predicted motion/displacement (from decoder)
            pred_normal: (B, N, 3) predicted surface normal (from decoder)
            pred_confidence: (B, N) or (B, N, 1) predicted confidence probabilities (after sigmoid)
            gt_3d: (B, N, 3) ground truth 3D coordinates
            gt_2d: (B, N, 2) ground truth 2D coordinates
            gt_visibility: (B, N) ground truth visibility
            gt_motion: (B, N, 3) ground truth motion
            gt_normal: (B, N, 3) ground truth surface normal
            mask: (B, N) optional mask for valid points
            
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        B, N = pred_3d.shape[:2]
        
        # Main 3D loss: c*λ_3D*L_3D (confidence-weighted, as per paper formula)
        if gt_3d is not None:
            loss_3d = self.compute_l3d_loss(pred_3d, gt_3d, pred_confidence, mask)
            losses['loss_3d'] = loss_3d
            total_loss = self.lambda_3d * loss_3d
        else:
            total_loss = torch.tensor(0.0, device=pred_3d.device)
        
        # Confidence penalty: -λ_conf*log(c) (as per paper: prevents cheating by low confidence)
        if pred_confidence is not None:
            loss_confidence = self.compute_confidence_loss(pred_confidence, mask)
            losses['loss_confidence'] = loss_confidence
            total_loss = total_loss + self.lambda_confidence * loss_confidence
        
        # 2D projection loss: λ_2D*L_2D
        if gt_2d is not None and pred_2d is not None:
            loss_2d = self.compute_2d_projection_loss(pred_2d, gt_2d, mask)
            losses['loss_2d'] = loss_2d
            total_loss = total_loss + self.lambda_2d * loss_2d
        
        # Visibility loss: λ_vis*L_vis
        if gt_visibility is not None and pred_visibility_logits is not None:
            loss_visibility = self.compute_visibility_loss(pred_visibility_logits, gt_visibility, mask)
            losses['loss_visibility'] = loss_visibility
            total_loss = total_loss + self.lambda_visibility * loss_visibility
        
        # Motion/displacement loss: λ_disp*L_disp
        if gt_motion is not None and pred_motion is not None:
            loss_motion = self.compute_motion_loss(pred_motion, gt_motion, mask)
            losses['loss_motion'] = loss_motion
            total_loss = total_loss + self.lambda_motion * loss_motion
        
        # Normal loss: λ_normal*L_normal
        if gt_normal is not None and pred_normal is not None:
            loss_normal = self.compute_normal_loss(pred_normal, gt_normal, mask)
            losses['loss_normal'] = loss_normal
            total_loss = total_loss + self.lambda_normal * loss_normal
        
        losses['loss'] = total_loss
        
        return losses

