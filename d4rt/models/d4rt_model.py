"""
D4RT Model: Complete model combining Encoder, QueryBuilder, and Decoder
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from .encoder import D4RTEncoder
from .decoder import D4RTDecoder
from .query import QueryBuilder


class D4RTModel(nn.Module):
    """
    D4RT Model for 4D Reconstruction from Video
    """
    
    def __init__(
        self,
        # Encoder config
        img_size=256,
        temporal_patch_size=2,
        spatial_patch_size=16,
        patch_size=None,  # Backward compatibility: if provided, use as spatial_patch_size
        encoder_embed_dim=1408,  # ViT-g: 1408 embed_dim
        encoder_depth=40,  # ViT-g: 40 layers
        encoder_num_heads=16,  # ViT-g: 16 heads
        # Decoder config
        decoder_dim=512,
        decoder_num_heads=8,
        decoder_num_layers=6,
        # Query config
        max_frames=100,
        fourier_dim=128,
        time_embed_dim=64,
        patch_embed_dim=128,
        query_dim=512,
        patch_size_query=9,
        # Other
        dropout=0.1
    ):
        super().__init__()
        
        # Handle backward compatibility
        if patch_size is not None:
            spatial_patch_size = patch_size
        
        # Encoder
        # For standard ViT-g: use pure global self-attention (use_local_global_alternate=False)
        # This matches the official ViT-g architecture
        # MLP ratio for ViT-g: 6144 / 1408 ≈ 4.3636
        mlp_ratio = 6144 / encoder_embed_dim if encoder_embed_dim == 1408 else 4.0
        self.encoder = D4RTEncoder(
            img_size=img_size,
            temporal_patch_size=temporal_patch_size,
            spatial_patch_size=spatial_patch_size,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,  # ViT-g uses 6144 hidden dim (4.3636 ratio)
            dropout=dropout,
            use_local_global_alternate=False  # Use pure global attention for standard ViT-g
        )
        
        # Query builder
        self.query_builder = QueryBuilder(
            max_frames=max_frames,
            fourier_dim=fourier_dim,
            time_embed_dim=time_embed_dim,
            patch_embed_dim=patch_embed_dim,
            query_dim=query_dim,
            patch_size=patch_size_query,
            num_fourier_freqs=10
        )
        
        # Project encoder features to query dimension if needed
        if encoder_embed_dim != decoder_dim:
            self.encoder_proj = nn.Linear(encoder_embed_dim, decoder_dim)
        else:
            self.encoder_proj = nn.Identity()
        
        # Decoder
        self.decoder = D4RTDecoder(
            d_model=decoder_dim,
            nhead=decoder_num_heads,
            num_layers=decoder_num_layers,
            dropout=dropout
        )
    
    def forward(
        self,
        video: torch.Tensor,
        coords_uv: torch.Tensor,
        t_src: torch.Tensor,
        t_tgt: torch.Tensor,
        t_cam: torch.Tensor,
        aspect_ratio: Optional[torch.Tensor] = None,
        video_orig: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            video: (B, T, C, H, W) input video (resized to 256x256 for encoder)
            coords_uv: (B, N, 2) normalized 2D coordinates in [0, 1] (relative to resized video)
            t_src: (B, N) source frame indices
            t_tgt: (B, N) target frame indices
            t_cam: (B, N) camera reference frame indices
            aspect_ratio: optional (B,) aspect ratios
            video_orig: optional (B, T, C, H_orig, W_orig) original resolution video for patch extraction
            
        Returns:
            Dictionary with:
                - coords_3d: (B, N, 3) predicted 3D coordinates (XYZ)
                - coords_2d: (B, N, 2) predicted 2D coordinates (UV)
                - visibility: (B, N, 1) predicted visibility
                - motion: (B, N, 3) predicted motion/displacement
                - normal: (B, N, 3) predicted surface normal
                - confidence: (B, N, 1) predicted confidence
                - query_features: (B, N, decoder_dim) query features
                - encoder_features: (B, M, decoder_dim) encoder features
                - outputs_raw: (B, N, 13) raw 13-dimensional decoder output
        """
        # Encode video (always use resized video)
        encoder_features = self.encoder(video, aspect_ratio)  # (B, M, encoder_embed_dim)
        
        # Project to decoder dimension
        encoder_features = self.encoder_proj(encoder_features)  # (B, M, decoder_dim)
        
        # Build queries (use original resolution video if provided, otherwise use resized)
        video_for_patches = video_orig if video_orig is not None else video
        queries = self.query_builder(video_for_patches, coords_uv, t_src, t_tgt, t_cam, video_resized=video)  # (B, N, query_dim)
        
        # Decode to 13-dimensional 4D predictions
        outputs, query_features = self.decoder(queries, encoder_features)  # (B, N, 13), (B, N, decoder_dim)
        
        # Parse 13-dimensional output according to D4RT paper:
        #   dims 0-2: XYZ position (3)
        #   dims 3-4: UV position (2)
        #   dim 5: visibility (1)
        #   dims 6-8: displacement/motion (3)
        #   dims 9-11: surface normal (3)
        #   dim 12: confidence (1)
        coords_3d = outputs[:, :, 0:3]  # (B, N, 3) XYZ position
        coords_2d = outputs[:, :, 3:5]  # (B, N, 2) UV position
        visibility_logits = outputs[:, :, 5:6]  # (B, N, 1) visibility logits (before sigmoid)
        motion = outputs[:, :, 6:9]  # (B, N, 3) displacement/motion
        normal = outputs[:, :, 9:12]  # (B, N, 3) surface normal
        confidence_logits = outputs[:, :, 12:13]  # (B, N, 1) confidence logits (before sigmoid)
        
        # Apply sigmoid for outputs that need probabilities (for 3D loss weighting and other uses)
        # But keep logits for loss computation (binary_cross_entropy_with_logits)
        visibility = torch.sigmoid(visibility_logits)  # (B, N, 1) visibility probabilities
        confidence = torch.sigmoid(confidence_logits)  # (B, N, 1) confidence probabilities
        
        return {
            'coords_3d': coords_3d,
            'coords_2d': coords_2d,
            'visibility': visibility,  # Probabilities (after sigmoid)
            'visibility_logits': visibility_logits,  # Logits (for loss computation)
            'motion': motion,
            'normal': normal,
            'confidence': confidence,  # Probabilities (after sigmoid)
            'confidence_logits': confidence_logits,  # Logits (for loss computation)
            'query_features': query_features,
            'encoder_features': encoder_features,
            'outputs_raw': outputs  # Raw 13-dim output for debugging
        }

