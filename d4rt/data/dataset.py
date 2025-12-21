"""
Dataset for D4RT training with query sampling strategy
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple
import cv2


class D4RTDataset(Dataset):
    """
    Dataset for D4RT with query sampling strategy:
    - 30% queries on depth discontinuities or motion boundaries (Sobel operator)
    - 40% samples with t_tgt = t_cam
    - N=2048 queries per batch
    """
    
    def __init__(
        self,
        video_paths: list,
        num_queries: int = 2048,
        img_size: int = 256,
        max_frames: int = 100,
        boundary_ratio: float = 0.3,
        t_tgt_eq_t_cam_ratio: float = 0.4,
        cache_boundaries: bool = True
    ):
        """
        Args:
            video_paths: List of paths to video data files
            num_queries: Number of queries per sample
            img_size: Input image size (assumed square)
            max_frames: Maximum number of frames
            boundary_ratio: Ratio of queries on boundaries (0.3 = 30%)
            t_tgt_eq_t_cam_ratio: Ratio of samples with t_tgt = t_cam (0.4 = 40%)
            cache_boundaries: Whether to cache precomputed boundaries
        """
        self.video_paths = video_paths
        self.num_queries = num_queries
        self.img_size = img_size
        self.max_frames = max_frames
        self.boundary_ratio = boundary_ratio
        self.t_tgt_eq_t_cam_ratio = t_tgt_eq_t_cam_ratio
        self.cache_boundaries = cache_boundaries
        
        # Cache for boundaries (depth edges and motion boundaries)
        self.boundary_cache = {} if cache_boundaries else None
    
    def compute_boundaries(self, depth_map, flow_map=None):
        """
        Compute depth discontinuities and motion boundaries using Sobel operator
        
        Args:
            depth_map: (H, W) depth map
            flow_map: (H, W, 2) optional optical flow map
        
        Returns:
            boundary_mask: (H, W) binary mask of boundary pixels
        """
        # Depth discontinuities
        sobel_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
        depth_gradient = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Threshold to get boundaries (top 20% of gradients)
        depth_threshold = np.percentile(depth_gradient, 80)
        depth_boundary = depth_gradient > depth_threshold
        
        # Motion boundaries (if flow provided)
        if flow_map is not None:
            flow_magnitude = np.sqrt(flow_map[:, :, 0]**2 + flow_map[:, :, 1]**2)
            sobel_x_flow = cv2.Sobel(flow_magnitude, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y_flow = cv2.Sobel(flow_magnitude, cv2.CV_64F, 0, 1, ksize=3)
            flow_gradient = np.sqrt(sobel_x_flow**2 + sobel_y_flow**2)
            
            flow_threshold = np.percentile(flow_gradient, 80)
            flow_boundary = flow_gradient > flow_threshold
            
            boundary_mask = depth_boundary | flow_boundary
        else:
            boundary_mask = depth_boundary
        
        return boundary_mask.astype(np.float32)
    
    def sample_queries(
        self,
        video: np.ndarray,
        depth_maps: np.ndarray,
        flow_maps: Optional[np.ndarray] = None,
        T: int = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample queries according to the strategy
        
        Args:
            video: (T, H, W, 3) video frames
            depth_maps: (T, H, W) depth maps
            flow_maps: (T, H, W, 2) optional optical flow
            T: number of frames
        
        Returns:
            coords_uv: (N, 2) normalized 2D coordinates
            t_src: (N,) source frame indices
            t_tgt: (N,) target frame indices
            t_cam: (N,) camera reference frame indices
        """
        if T is None:
            T = video.shape[0]
        H, W = video.shape[1:3]
        
        # Compute boundaries for middle frame (or average across frames)
        mid_frame = T // 2
        boundary_mask = self.compute_boundaries(
            depth_maps[mid_frame],
            flow_maps[mid_frame] if flow_maps is not None else None
        )
        
        # Number of queries on boundaries
        num_boundary_queries = int(self.num_queries * self.boundary_ratio)
        num_random_queries = self.num_queries - num_boundary_queries
        
        # Sample boundary queries
        boundary_pixels = np.where(boundary_mask > 0.5)
        if len(boundary_pixels[0]) > 0:
            boundary_indices = np.random.choice(
                len(boundary_pixels[0]),
                size=min(num_boundary_queries, len(boundary_pixels[0])),
                replace=False
            )
            boundary_coords = np.stack([
                boundary_pixels[1][boundary_indices],  # u (width)
                boundary_pixels[0][boundary_indices]   # v (height)
            ], axis=1)
        else:
            boundary_coords = np.zeros((0, 2))
        
        # Sample random queries
        num_random_queries_actual = num_random_queries + (num_boundary_queries - len(boundary_coords))
        random_u = np.random.randint(0, W, size=num_random_queries_actual)
        random_v = np.random.randint(0, H, size=num_random_queries_actual)
        random_coords = np.stack([random_u, random_v], axis=1)
        
        # Combine coordinates
        all_coords = np.concatenate([boundary_coords, random_coords], axis=0)
        
        # Normalize to [0, 1]
        coords_uv = all_coords.astype(np.float32)
        coords_uv[:, 0] /= W  # u
        coords_uv[:, 1] /= H  # v
        
        # Sample time indices
        # 40% with t_tgt = t_cam
        num_t_tgt_eq_t_cam = int(self.num_queries * self.t_tgt_eq_t_cam_ratio)
        
        # For t_tgt = t_cam samples
        t_cam_eq = np.random.randint(0, T, size=num_t_tgt_eq_t_cam)
        t_src_eq = np.random.randint(0, T, size=num_t_tgt_eq_t_cam)
        t_tgt_eq = t_cam_eq.copy()
        
        # For remaining samples
        num_remaining = self.num_queries - num_t_tgt_eq_t_cam
        t_src_remaining = np.random.randint(0, T, size=num_remaining)
        t_tgt_remaining = np.random.randint(0, T, size=num_remaining)
        t_cam_remaining = np.random.randint(0, T, size=num_remaining)
        
        # Combine
        t_src = np.concatenate([t_src_eq, t_src_remaining])
        t_tgt = np.concatenate([t_tgt_eq, t_tgt_remaining])
        t_cam = np.concatenate([t_cam_eq, t_cam_remaining])
        
        return coords_uv, t_src, t_tgt, t_cam
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Returns a dictionary with:
            - video: (T, C, H, W) video frames
            - depth_maps: (T, H, W) depth maps
            - coords_uv: (N, 2) query coordinates
            - t_src, t_tgt, t_cam: (N,) time indices
            - gt_3d: (N, 3) ground truth 3D coordinates (if available)
            - intrinsics: (3, 3) camera intrinsics
            - aspect_ratio: scalar aspect ratio
        """
        # This is a placeholder - you'll need to implement actual data loading
        # based on your data format
        raise NotImplementedError(
            "Please implement __getitem__ based on your data format. "
            "Expected return format:\n"
            "  - video: (T, C, H, W) torch.Tensor\n"
            "  - depth_maps: (T, H, W) torch.Tensor\n"
            "  - coords_uv: (N, 2) torch.Tensor (normalized [0, 1])\n"
            "  - t_src, t_tgt, t_cam: (N,) torch.LongTensor\n"
            "  - gt_3d: (N, 3) torch.Tensor (optional)\n"
            "  - intrinsics: (3, 3) torch.Tensor\n"
            "  - aspect_ratio: float or torch.Tensor"
        )

