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


class GlobalAttention(nn.MultiheadAttention):
    """Global self-attention across all tokens"""
    pass


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
            x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbedding(nn.Module):
    """Patch embedding for video frames"""
    
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches_per_frame = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) video tensor
        Returns:
            patches: (B, T * n_patches, embed_dim)
        """
        B, T, C, H, W = x.shape
        # Process each frame
        patches_list = []
        for t in range(T):
            frame = x[:, t, :, :, :]  # (B, C, H, W)
            patch = self.proj(frame)  # (B, embed_dim, H_p, W_p)
            patch = patch.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
            patches_list.append(patch)
        
        patches = torch.cat(patches_list, dim=1)  # (B, T * n_patches, embed_dim)
        patches = self.norm(patches)
        return patches


class D4RTEncoder(nn.Module):
    """
    D4RT Encoder: ViT with alternating local/global attention
    """
    
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        local_window_size=7,
        dropout=0.1,
        use_local_global_alternate=True
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches_per_frame = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        
        # Aspect ratio token embedding (learnable)
        self.aspect_ratio_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional embedding (learnable, separate for each frame position)
        # We'll use a learnable positional embedding that can handle variable T
        max_frames = 100  # Can be adjusted
        self.pos_embed = nn.Parameter(torch.randn(1, max_frames * self.n_patches_per_frame + 1, embed_dim))
        
        # Transformer blocks with alternating attention
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
            features: (B, T * n_patches + 1, embed_dim) with aspect ratio token
        """
        B, T, C, H, W = x.shape
        assert H == W == self.img_size, f"Input size {H}x{W} must match img_size {self.img_size}"
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, T * n_patches, embed_dim)
        
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
        
        x = torch.cat([x, ar_token], dim=1)  # (B, T * n_patches + 1, embed_dim)
        
        # Add positional embedding (truncate to actual sequence length)
        seq_len = x.shape[1]
        pos_embed = self.pos_embed[:, :seq_len, :]
        x = x + pos_embed
        x = self.dropout(x)
        
        # Apply transformer blocks
        H_spatial = W_spatial = self.img_size // self.patch_size
        for block in self.blocks:
            if hasattr(block, 'use_local_attn') and block.use_local_attn:
                x = block(x, H=H_spatial, W=W_spatial)
            else:
                x = block(x)
        
        x = self.norm(x)
        return x

