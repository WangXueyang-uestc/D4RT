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
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
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
        
        # Encoder
        self.encoder = D4RTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            dropout=dropout
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
        aspect_ratio: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            video: (B, T, C, H, W) input video
            coords_uv: (B, N, 2) normalized 2D coordinates in [0, 1]
            t_src: (B, N) source frame indices
            t_tgt: (B, N) target frame indices
            t_cam: (B, N) camera reference frame indices
            aspect_ratio: optional (B,) aspect ratios
            
        Returns:
            Dictionary with:
                - coords_3d: (B, N, 3) predicted 3D coordinates
                - query_features: (B, N, decoder_dim) query features (for auxiliary losses)
                - encoder_features: (B, M, decoder_dim) encoder features
        """
        # Encode video
        encoder_features = self.encoder(video, aspect_ratio)  # (B, M, encoder_embed_dim)
        
        # Project to decoder dimension
        encoder_features = self.encoder_proj(encoder_features)  # (B, M, decoder_dim)
        
        # Build queries
        queries = self.query_builder(video, coords_uv, t_src, t_tgt, t_cam)  # (B, N, query_dim)
        
        # Decode to 3D coordinates
        coords_3d, query_features = self.decoder(queries, encoder_features)  # (B, N, 3), (B, N, decoder_dim)
        
        return {
            'coords_3d': coords_3d,
            'query_features': query_features,
            'encoder_features': encoder_features
        }

