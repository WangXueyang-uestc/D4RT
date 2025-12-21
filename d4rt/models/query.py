"""
Query Builder for D4RT
Constructs queries q = (u, v, t_src, t_tgt, t_cam) with:
- Fourier features for normalized 2D coordinates (u, v)
- Learned discrete embeddings for time dimensions
- Local RGB patch embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FourierFeatureEmbedding(nn.Module):
    """Fourier feature embedding for 2D coordinates (u, v)"""
    
    def __init__(self, num_frequencies=10, embedding_dim=128):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.embedding_dim = embedding_dim
        
        # Generate frequency bands
        self.register_buffer('freq_bands', torch.linspace(0.5, num_frequencies, num_frequencies))
        
        # Projection layer
        self.proj = nn.Linear(num_frequencies * 4, embedding_dim)  # 4 = 2 coords * 2 (sin, cos)
    
    def forward(self, coords):
        """
        Args:
            coords: (B, N, 2) normalized coordinates in [0, 1]
        Returns:
            embedded: (B, N, embedding_dim)
        """
        # coords: (B, N, 2)
        B, N, _ = coords.shape
        
        # Expand frequencies: (num_freq,)
        frequencies = self.freq_bands.unsqueeze(0).unsqueeze(0)  # (1, 1, num_freq)
        frequencies = frequencies.expand(B, N, -1)  # (B, N, num_freq)
        
        # Compute sin and cos for each coordinate and frequency
        coords_expanded = coords.unsqueeze(-1)  # (B, N, 2, 1)
        frequencies_expanded = frequencies.unsqueeze(2)  # (B, N, 1, num_freq)
        
        # (B, N, 2, num_freq)
        angles = 2 * np.pi * coords_expanded * frequencies_expanded
        
        # Compute sin and cos
        sin_features = torch.sin(angles)  # (B, N, 2, num_freq)
        cos_features = torch.cos(angles)  # (B, N, 2, num_freq)
        
        # Concatenate and flatten: (B, N, 2 * num_freq * 2)
        fourier_features = torch.cat([sin_features, cos_features], dim=2)
        fourier_features = fourier_features.reshape(B, N, -1)
        
        # Project to embedding dimension
        embedded = self.proj(fourier_features)
        return embedded


class LocalRGBPatchEmbedding(nn.Module):
    """Extract and embed local RGB patches centered at (u, v)"""
    
    def __init__(self, patch_size=9, embedding_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.half_patch = patch_size // 2
        
        # Convolutional patch encoder
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, kernel_size=patch_size, stride=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=2)
        )
    
    def extract_patch(self, images, coords):
        """
        Extract patches centered at normalized coordinates (u, v)
        
        Args:
            images: (B, T, C, H, W) video frames
            coords: (B, N, 2) normalized coordinates in [0, 1]
        Returns:
            patches: (B, N, C, patch_size, patch_size)
        """
        B, T, C, H, W = images.shape
        N = coords.shape[1]
        
        # Convert normalized coords to pixel coordinates
        pixel_coords = coords * torch.tensor([W, H], device=coords.device, dtype=coords.dtype)
        pixel_coords = pixel_coords.round().long()
        
        # Clamp to valid range
        pixel_coords[:, :, 0] = torch.clamp(pixel_coords[:, :, 0], self.half_patch, W - self.half_patch - 1)
        pixel_coords[:, :, 1] = torch.clamp(pixel_coords[:, :, 1], self.half_patch, H - self.half_patch - 1)
        
        patches_list = []
        for b in range(B):
            for n in range(N):
                u, v = pixel_coords[b, n, 0].item(), pixel_coords[b, n, 1].item()
                
                # Extract patch for each frame and average (or select source frame)
                # For simplicity, we'll extract from first frame. In practice, should extract from t_src
                patch = images[b, 0, :, 
                              v - self.half_patch : v + self.half_patch + 1,
                              u - self.half_patch : u + self.half_patch + 1]
                patches_list.append(patch)
        
        patches = torch.stack(patches_list, dim=0)  # (B*N, C, patch_size, patch_size)
        patches = patches.reshape(B, N, C, self.patch_size, self.patch_size)
        return patches
    
    def forward(self, images, coords, t_src):
        """
        Args:
            images: (B, T, C, H, W) video frames
            coords: (B, N, 2) normalized coordinates
            t_src: (B, N) source frame indices
        Returns:
            embedded: (B, N, embedding_dim)
        """
        B, T, C, H, W = images.shape
        N = coords.shape[1]
        
        # Extract patches (using t_src to select source frame)
        patches_list = []
        pixel_coords = coords * torch.tensor([W, H], device=coords.device, dtype=coords.dtype)
        pixel_coords = pixel_coords.round().long()
        pixel_coords[:, :, 0] = torch.clamp(pixel_coords[:, :, 0], self.half_patch, W - self.half_patch - 1)
        pixel_coords[:, :, 1] = torch.clamp(pixel_coords[:, :, 1], self.half_patch, H - self.half_patch - 1)
        
        for b in range(B):
            for n in range(N):
                u, v = pixel_coords[b, n, 0].item(), pixel_coords[b, n, 1].item()
                t_idx = t_src[b, n].item() if t_src.dim() > 0 else t_src[b, n].item()
                t_idx = int(torch.clamp(torch.tensor(t_idx), 0, T - 1).item())
                patch = images[b, t_idx, :, 
                              v - self.half_patch : v + self.half_patch + 1,
                              u - self.half_patch : u + self.half_patch + 1]
                patches_list.append(patch)
        
        patches = torch.stack(patches_list, dim=0).reshape(B, N, C, self.patch_size, self.patch_size)
        
        # Embed patches
        embedded_patches = []
        for n in range(N):
            patch = patches[:, n, :, :, :]  # (B, C, patch_size, patch_size)
            embedded = self.conv(patch)  # (B, embedding_dim, 1, 1) -> (B, embedding_dim)
            embedded_patches.append(embedded.squeeze(-1))
        
        embedded = torch.stack(embedded_patches, dim=1)  # (B, N, embedding_dim)
        return embedded


class QueryBuilder(nn.Module):
    """
    Builds query vectors q = (u, v, t_src, t_tgt, t_cam) for D4RT decoder
    """
    
    def __init__(
        self,
        max_frames=100,
        fourier_dim=128,
        time_embed_dim=64,
        patch_embed_dim=128,
        query_dim=512,
        patch_size=9,
        num_fourier_freqs=10
    ):
        super().__init__()
        self.max_frames = max_frames
        self.query_dim = query_dim
        
        # Fourier feature embedding for (u, v)
        self.fourier_embed = FourierFeatureEmbedding(
            num_frequencies=num_fourier_freqs,
            embedding_dim=fourier_dim
        )
        
        # Time embeddings (discrete learned embeddings)
        self.t_src_embed = nn.Embedding(max_frames, time_embed_dim)
        self.t_tgt_embed = nn.Embedding(max_frames, time_embed_dim)
        self.t_cam_embed = nn.Embedding(max_frames, time_embed_dim)
        
        # Local RGB patch embedding
        self.patch_embed = LocalRGBPatchEmbedding(
            patch_size=patch_size,
            embedding_dim=patch_embed_dim
        )
        
        # Combine all features
        total_dim = fourier_dim + 3 * time_embed_dim + patch_embed_dim
        self.query_proj = nn.Linear(total_dim, query_dim)
    
    def forward(self, images, coords_uv, t_src, t_tgt, t_cam):
        """
        Build query vectors from components
        
        Args:
            images: (B, T, C, H, W) video frames
            coords_uv: (B, N, 2) normalized 2D coordinates in [0, 1]
            t_src: (B, N) source frame indices (long tensor)
            t_tgt: (B, N) target frame indices (long tensor)
            t_cam: (B, N) camera reference frame indices (long tensor)
        
        Returns:
            queries: (B, N, query_dim) query vectors
        """
        B, N = coords_uv.shape[:2]
        
        # Fourier features for (u, v)
        fourier_feat = self.fourier_embed(coords_uv)  # (B, N, fourier_dim)
        
        # Time embeddings
        t_src_emb = self.t_src_embed(t_src)  # (B, N, time_embed_dim)
        t_tgt_emb = self.t_tgt_embed(t_tgt)  # (B, N, time_embed_dim)
        t_cam_emb = self.t_cam_embed(t_cam)  # (B, N, time_embed_dim)
        
        # Local RGB patch embedding
        patch_feat = self.patch_embed(images, coords_uv, t_src)  # (B, N, patch_embed_dim)
        
        # Concatenate all features
        query_features = torch.cat([
            fourier_feat,
            t_src_emb,
            t_tgt_emb,
            t_cam_emb,
            patch_feat
        ], dim=-1)  # (B, N, total_dim)
        
        # Project to query dimension
        queries = self.query_proj(query_features)  # (B, N, query_dim)
        
        return queries

