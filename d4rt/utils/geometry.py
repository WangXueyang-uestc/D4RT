"""
Geometry utility functions for 3D-2D projections and surface normal computation
"""

import torch
import torch.nn.functional as F


def project_3d_to_2d(points_3d, intrinsics):
    """
    Project 3D points to 2D image coordinates
    
    Args:
        points_3d: (B, N, 3) 3D points in camera coordinates
        intrinsics: (B, 3, 3) camera intrinsic matrix
        
    Returns:
        points_2d: (B, N, 2) 2D image coordinates
        depth: (B, N) depth values
    """
    # Extract intrinsic parameters
    fx = intrinsics[:, 0, 0].unsqueeze(-1)  # (B, 1)
    fy = intrinsics[:, 1, 1].unsqueeze(-1)
    cx = intrinsics[:, 0, 2].unsqueeze(-1)
    cy = intrinsics[:, 1, 2].unsqueeze(-1)
    
    x, y, z = points_3d[..., 0], points_3d[..., 1], points_3d[..., 2]
    
    # Avoid division by zero
    z = torch.clamp(z, min=1e-6)
    
    # Project to 2D
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    
    points_2d = torch.stack([u, v], dim=-1)
    depth = z
    
    return points_2d, depth


def compute_surface_normal(points_3d, neighbors=3):
    """
    Compute surface normals from 3D points using local neighborhood
    
    Args:
        points_3d: (B, N, 3) 3D points
        neighbors: number of neighbors to use for normal estimation
        
    Returns:
        normals: (B, N, 3) surface normals (normalized)
    """
    B, N, _ = points_3d.shape
    
    # For simplicity, compute normals using cross products of nearby points
    # In practice, you might want to use a more sophisticated method
    normals = torch.zeros_like(points_3d)
    
    # Compute normals by taking cross products of edges
    # This is a simplified version - you might want to use PCA or similar
    if N > 2:
        # Shift points to compute differences
        p0 = points_3d
        p1 = torch.roll(points_3d, shifts=1, dims=1)
        p2 = torch.roll(points_3d, shifts=2, dims=1)
        
        # Compute edge vectors
        v1 = p1 - p0
        v2 = p2 - p1
        
        # Cross product to get normal
        normals = torch.cross(v1, v2, dim=-1)
        
        # Normalize
        norm = torch.norm(normals, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-6)
        normals = normals / norm
    
    return normals

