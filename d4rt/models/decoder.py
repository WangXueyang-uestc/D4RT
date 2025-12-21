"""
D4RT Decoder: Lightweight Cross-Attention Transformer with Independent Querying
Queries do not interact with each other, only attend to encoder features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for Independent Querying"""
    
    def __init__(
        self,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.GELU
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # Multi-head cross-attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Self-attention (can be removed for true independence, but helps with convergence)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Cross-attention: queries attend to encoder features
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, queries, encoder_features, query_mask=None, encoder_mask=None):
        """
        Args:
            queries: (B, N, d_model) independent query vectors
            encoder_features: (B, M, d_model) encoder output features
            query_mask: optional mask for queries
            encoder_mask: optional mask for encoder features
        Returns:
            out: (B, N, d_model) updated query vectors
        """
        # Self-attention (optional, for stability)
        # Note: For true independent querying, we could remove this
        q_norm = self.norm1(queries)
        self_attn_out, _ = self.self_attn(q_norm, q_norm, q_norm, key_padding_mask=query_mask)
        queries = queries + self.dropout1(self_attn_out)
        
        # Cross-attention: queries attend to encoder features
        q_norm = self.norm2(queries)
        cross_attn_out, attn_weights = self.cross_attn(
            q_norm, encoder_features, encoder_features,
            key_padding_mask=encoder_mask
        )
        queries = queries + self.dropout2(cross_attn_out)
        
        # Feed-forward
        queries = queries + self.dropout3(self.ffn(self.norm2(queries)))
        
        return queries


class D4RTDecoder(nn.Module):
    """
    D4RT Decoder: Cross-attention Transformer with Independent Querying
    """
    
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        output_dim=3
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Decoder layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Final output projection to 3D coordinates
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_dim)
    
    def forward(self, queries, encoder_features, query_mask=None, encoder_mask=None):
        """
        Decode queries to 3D coordinates
        
        Args:
            queries: (B, N, d_model) query vectors from QueryBuilder
            encoder_features: (B, M, d_model) encoder output features
            query_mask: optional (B, N) mask for queries
            encoder_mask: optional (B, M) mask for encoder features
        Returns:
            coords_3d: (B, N, 3) predicted 3D coordinates
            queries: (B, N, d_model) final query representations (for auxiliary losses)
        """
        # Apply decoder layers
        for layer in self.layers:
            queries = layer(queries, encoder_features, query_mask, encoder_mask)
        
        # Final normalization
        queries = self.norm(queries)
        
        # Project to 3D coordinates
        coords_3d = self.output_proj(queries)  # (B, N, 3)
        
        return coords_3d, queries

