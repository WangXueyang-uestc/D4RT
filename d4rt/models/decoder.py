"""
D4RT Decoder: Lightweight Cross-Attention Transformer with Independent Querying
Queries do not interact with each other, only attend to encoder features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for Independent Querying
    
    According to D4RT paper: Queries are independent and only attend to encoder features.
    No self-attention between queries to ensure independence.
    """
    
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
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)  # For cross-attention
        self.norm2 = nn.LayerNorm(d_model)  # For feed-forward
        
        # Cross-attention: queries attend to encoder features (Key and Value from encoder)
        # Query from queries, Key and Value from encoder_features
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
    
    def forward(self, queries, encoder_features, query_mask=None, encoder_mask=None):
        """
        Args:
            queries: (B, N, d_model) independent query vectors
            encoder_features: (B, M, d_model) encoder output features (used as Key and Value)
            query_mask: optional mask for queries (not used for independent queries)
            encoder_mask: optional mask for encoder features
        Returns:
            out: (B, N, d_model) updated query vectors
        """
        # Cross-attention: queries attend to encoder features
        # Query from queries, Key and Value from encoder_features
        # This ensures queries are independent (no interaction between queries)
        q_norm = self.norm1(queries)
        cross_attn_out, attn_weights = self.cross_attn(
            q_norm,  # Query: (B, N, d_model)
            encoder_features,  # Key: (B, M, d_model)
            encoder_features,  # Value: (B, M, d_model)
            key_padding_mask=encoder_mask
        )
        queries = queries + self.dropout1(cross_attn_out)
        
        # Feed-forward
        queries = queries + self.dropout2(self.ffn(self.norm2(queries)))
        
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
        output_dim=13  # 13 dims: XYZ(3) + UV(2) + vis(1) + disp(3) + normal(3) + conf(1)
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Decoder layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Final output projection to 13 dimensions as per D4RT paper
        # Output breakdown:
        #   dims 0-2: XYZ position (3)
        #   dims 3-4: UV position (2)
        #   dim 5: visibility (1)
        #   dims 6-8: displacement/motion (3)
        #   dims 9-11: surface normal (3)
        #   dim 12: confidence (1)
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_dim)
    
    def forward(self, queries, encoder_features, query_mask=None, encoder_mask=None):
        """
        Decode queries to 4D predictions (13 dimensions)
        
        Args:
            queries: (B, N, d_model) query vectors from QueryBuilder
            encoder_features: (B, M, d_model) encoder output features
            query_mask: optional (B, N) mask for queries
            encoder_mask: optional (B, M) mask for encoder features
        Returns:
            outputs: (B, N, 13) 4D predictions:
                - dims 0-2: XYZ position (3)
                - dims 3-4: UV position (2)
                - dim 5: visibility (1)
                - dims 6-8: displacement/motion (3)
                - dims 9-11: surface normal (3)
                - dim 12: confidence (1)
            queries: (B, N, d_model) final query representations
        """
        # Apply decoder layers
        for layer in self.layers:
            queries = layer(queries, encoder_features, query_mask, encoder_mask)
        
        # Final normalization
        queries = self.norm(queries)
        
        # Project to 13-dimensional output
        outputs = self.output_proj(queries)  # (B, N, 13)
        
        return outputs, queries

