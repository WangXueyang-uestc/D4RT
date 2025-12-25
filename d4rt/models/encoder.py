"""
D4RT Encoder: Vision Transformer with alternating intra-frame local attention
and global self-attention, plus aspect ratio token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LocalAttention(nn.Module):
    """Local attention within a frame (spatial neighborhood)"""
    
    def __init__(self, dim, num_heads=8, local_window_size=7, qkv_bias=False, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_window_size = local_window_size
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
    
    def forward(self, x, H, W):
        """
        Args:
            x: (B, N, C) where N = T * H * W
            H, W: height and width of spatial dimensions
        Returns:
            out: (B, N, C)
        """
        B, N, C = x.shape
        T = N // (H * W)
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Reshape to (B, T, num_heads, H, W, head_dim)
        q = q.reshape(B, self.num_heads, T, H, W, self.head_dim)
        k = k.reshape(B, self.num_heads, T, H, W, self.head_dim)
        v = v.reshape(B, self.num_heads, T, H, W, self.head_dim)
        
        # Apply local window attention
        half_window = self.local_window_size // 2
        out_list = []
        
        for t in range(T):
            q_frame = q[:, :, t, :, :, :]  # (B, num_heads, H, W, head_dim)
            k_frame = k[:, :, t, :, :, :]
            v_frame = v[:, :, t, :, :, :]
            
            # For each spatial position, attend to local window
            q_padded = F.pad(q_frame, (0, 0, half_window, half_window, half_window, half_window), mode='constant')
            k_padded = F.pad(k_frame, (0, 0, half_window, half_window, half_window, half_window), mode='constant')
            v_padded = F.pad(v_frame, (0, 0, half_window, half_window, half_window, half_window), mode='constant')
            
            attn_out = []
            for h in range(H):
                for w in range(W):
                    q_patch = q_padded[:, :, h:h+self.local_window_size, w:w+self.local_window_size, :]
                    k_patch = k_padded[:, :, h:h+self.local_window_size, w:w+self.local_window_size, :]
                    v_patch = v_padded[:, :, h:h+self.local_window_size, w:w+self.local_window_size, :]
                    
                    # Flatten spatial window
                    q_flat = q_patch[:, :, half_window, half_window, :].unsqueeze(-2)  # (B, num_heads, 1, head_dim)
                    k_flat = k_patch.reshape(B, self.num_heads, -1, self.head_dim)
                    v_flat = v_patch.reshape(B, self.num_heads, -1, self.head_dim)
                    
                    # Compute attention
                    attn = (q_flat @ k_flat.transpose(-2, -1)) * self.scale
                    attn = F.softmax(attn, dim=-1)
                    attn = self.dropout(attn)
                    
                    out_patch = (attn @ v_flat).squeeze(-2)  # (B, num_heads, head_dim)
                    attn_out.append(out_patch)
            
            frame_out = torch.stack(attn_out, dim=2).reshape(B, self.num_heads, H, W, self.head_dim)
            out_list.append(frame_out)
        
        # Stack frames and reshape
        out = torch.stack(out_list, dim=2)  # (B, num_heads, T, H, W, head_dim)
        out = out.permute(0, 2, 3, 4, 1, 5).reshape(B, T * H * W, C)
        
        out = self.proj(out)
        out = self.dropout(out)
        return out


class GlobalAttention(nn.Module):
    """Global self-attention across all tokens (standard ViT attention)"""
    
    def __init__(self, dim, num_heads=8, dropout=0.0, batch_first=True):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first
        )
    
    def forward(self, x, x_kv=None, x_v=None):
        """
        Standard self-attention: Q, K, V all from x
        
        Args:
            x: (B, N, C) input tensor
            x_kv: optional, if None uses x
            x_v: optional, if None uses x
        """
        if x_kv is None:
            x_kv = x
        if x_v is None:
            x_v = x_kv
        out, _ = self.attn(x, x_kv, x_v)
        return out


class TransformerBlock(nn.Module):
    """Transformer block with alternating local/global attention"""
    
    def __init__(
        self,
        dim,
        num_heads=8,
        mlp_ratio=4.0,
        local_window_size=7,
        use_local_attn=True,
        dropout=0.1,
        activation=nn.GELU
    ):
        super().__init__()
        self.use_local_attn = use_local_attn
        
        # Attention
        if use_local_attn:
            self.attn = LocalAttention(dim, num_heads, local_window_size, dropout=dropout)
        else:
            self.attn = GlobalAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, H=None, W=None):
        """
        Args:
            x: (B, N, C) where N = T * H * W (+ 1 for aspect ratio token)
            H, W: spatial dimensions (required for local attention)
        """
        if self.use_local_attn and H is not None and W is not None:
            # Separate aspect ratio token if present
            if x.shape[1] == H * W + 1:
                x_tokens = x[:, :-1, :]
                ar_token = x[:, -1:, :]
                x_tokens = x_tokens + self.attn(self.norm1(x_tokens), H, W)
                x = torch.cat([x_tokens, ar_token], dim=1)
            else:
                x = x + self.attn(self.norm1(x), H, W)
        else:
            # Global self-attention: Q, K, V all from normalized x
            x_norm = self.norm1(x)
            x = x + self.attn(x_norm, x_norm, x_norm)
        
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbedding3D(nn.Module):
    """3D patch embedding for video frames (temporal patch of 2 frames, spatial patch of 16x16)"""
    
    def __init__(self, img_size=256, temporal_patch_size=2, spatial_patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_patch_size = spatial_patch_size
        self.n_patches_per_frame = (img_size // spatial_patch_size) ** 2
        self.embed_dim = embed_dim
        
        # 3D convolution: (temporal, height, width) = (2, 16, 16)
        self.proj = nn.Conv3d(
            in_chans, 
            embed_dim, 
            kernel_size=(temporal_patch_size, spatial_patch_size, spatial_patch_size),
            stride=(temporal_patch_size, spatial_patch_size, spatial_patch_size)
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) video tensor
        Returns:
            patches: (B, (T//temporal_patch_size) * n_patches, embed_dim)
        """
        B, T, C, H, W = x.shape
        
        # Rearrange to (B, C, T, H, W) for Conv3d
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
        
        # Apply 3D convolution
        patches = self.proj(x)  # (B, embed_dim, T_p, H_p, W_p)
        
        # Flatten spatial and temporal dimensions
        B, C, T_p, H_p, W_p = patches.shape
        patches = patches.permute(0, 2, 3, 4, 1)  # (B, T_p, H_p, W_p, C)
        patches = patches.reshape(B, T_p * H_p * W_p, C)  # (B, T_p * H_p * W_p, embed_dim)
        
        patches = self.norm(patches)
        return patches


class D4RTEncoder(nn.Module):
    """
    D4RT Encoder: ViT with alternating local/global attention
    """
    
    def __init__(
        self,
        img_size=256,
        temporal_patch_size=2,
        spatial_patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.3636,  # ViT-g: 6144 / 1408 ≈ 4.3636
        local_window_size=7,
        dropout=0.1,
        use_local_global_alternate=True
    ):
        super().__init__()
        self.img_size = img_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_patch_size = spatial_patch_size
        self.n_patches_per_frame = (img_size // spatial_patch_size) ** 2
        self.embed_dim = embed_dim
        
        # 3D Patch embedding
        self.patch_embed = PatchEmbedding3D(
            img_size, temporal_patch_size, spatial_patch_size, in_chans, embed_dim
        )
        
        # Aspect ratio token embedding (learnable)
        self.aspect_ratio_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional embedding for 3D patches
        # We need to handle both temporal and spatial positions
        # For 3D patches: (T//temporal_patch_size) * n_patches_per_frame patches
        max_temporal_patches = 50  # Can be adjusted (max_frames // temporal_patch_size)
        max_patches = max_temporal_patches * self.n_patches_per_frame + 1  # +1 for aspect ratio token
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, embed_dim))
        
        # Separate temporal and spatial positional embeddings
        self.temporal_pos_embed = nn.Parameter(torch.randn(1, max_temporal_patches, embed_dim))
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, self.n_patches_per_frame, embed_dim))
        
        # Transformer blocks
        # For standard ViT-g: use pure global self-attention (use_local_global_alternate=False)
        # For D4RT variant: can use alternating local/global attention (use_local_global_alternate=True)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                local_window_size=local_window_size,
                use_local_attn=(use_local_global_alternate and (i % 2 == 0)),
                dropout=dropout
            )
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, aspect_ratio=None):
        """
        Args:
            x: (B, T, C, H, W) video tensor
            aspect_ratio: (B,) or scalar, original aspect ratio (width/height)
        Returns:
            features: (B, (T//temporal_patch_size) * n_patches + 1, embed_dim) with aspect ratio token
        """
        B, T, C, H, W = x.shape
        assert H == W == self.img_size, f"Input size {H}x{W} must match img_size {self.img_size}"
        
        # 3D Patch embedding
        x = self.patch_embed(x)  # (B, (T//temporal_patch_size) * n_patches, embed_dim)
        
        # Add aspect ratio token
        ar_token = self.aspect_ratio_token.expand(B, -1, -1)
        
        # Encode aspect ratio if provided
        if aspect_ratio is not None:
            if isinstance(aspect_ratio, (int, float)):
                ar_value = torch.tensor([aspect_ratio], device=x.device, dtype=x.dtype).expand(B, 1, 1)
            else:
                ar_value = aspect_ratio.unsqueeze(-1).unsqueeze(-1)
            # Simple encoding: multiply token by aspect ratio
            ar_token = ar_token * (1.0 + ar_value * 0.1)
        
        x = torch.cat([x, ar_token], dim=1)  # (B, (T//temporal_patch_size) * n_patches + 1, embed_dim)
        
        # Add positional embedding
        # For 3D patches, we combine temporal and spatial positional embeddings
        T_patches = T // self.temporal_patch_size
        seq_len = x.shape[1] - 1  # Exclude aspect ratio token
        
        # Create combined positional embedding
        # Reshape patches to (B, T_patches, n_patches_per_frame, embed_dim)
        n_patches_per_frame = self.n_patches_per_frame
        if seq_len == T_patches * n_patches_per_frame:
            patches_reshaped = x[:, :-1, :].reshape(B, T_patches, n_patches_per_frame, self.embed_dim)
            
            # Add temporal and spatial positional embeddings
            temporal_pos = self.temporal_pos_embed[:, :T_patches, :].unsqueeze(2)  # (1, T_patches, 1, embed_dim)
            spatial_pos = self.spatial_pos_embed.unsqueeze(0).unsqueeze(1)  # (1, 1, n_patches_per_frame, embed_dim)
            
            patches_reshaped = patches_reshaped + temporal_pos + spatial_pos
            x_patches = patches_reshaped.reshape(B, seq_len, self.embed_dim)
            
            # Add aspect ratio token back
            x = torch.cat([x_patches, ar_token], dim=1)
        else:
            # Fallback: use simple positional embedding
            pos_embed = self.pos_embed[:, :seq_len+1, :]
            x = x + pos_embed
        
        x = self.dropout(x)
        
        # Apply transformer blocks
        H_spatial = W_spatial = self.img_size // self.spatial_patch_size
        for block in self.blocks:
            if hasattr(block, 'use_local_attn') and block.use_local_attn:
                x = block(x, H=H_spatial, W=W_spatial)
            else:
                x = block(x)
        
        x = self.norm(x)
        return x

