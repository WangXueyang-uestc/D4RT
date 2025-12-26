"""
Dataset for D4RT training with query sampling strategy
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple
import cv2
import os
import glob
import sys
from PIL import Image
from torchvision import transforms
import random
from scipy.ndimage import gaussian_filter

# Import utility functions
from ..utils.geometry import GeomUtils
from ..utils.misc import farthest_point_sample_py

# Create utils namespace for compatibility
class MiscNamespace:
    """Namespace for misc functions"""
    @staticmethod
    def farthest_point_sample_py(points, n_samples):
        return farthest_point_sample_py(points, n_samples)

class UtilsNamespace:
    def __init__(self):
        self.geom = GeomUtils()
        self.misc = MiscNamespace()

utils = UtilsNamespace()


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

class PointOdysseyDataset(Dataset):
    def __init__(self,
                 dataset_location='/orion/group/point_odyssey',
                 dset='train',
                 use_augs=False,
                 S=8,
                 N=32,
                 strides=[1,2,4],
                 clip_step=2,
                 quick=False,
                 verbose=False,
                 img_size=256,
                 num_queries=2048,
                 boundary_ratio=0.3,
                 t_tgt_eq_t_cam_ratio=0.4,
                 cache_boundaries=True,
    ):
        print('loading pointodyssey dataset...')

        self.S = S
        self.N = N
        self.verbose = verbose
        self.img_size = img_size
        self.num_queries = num_queries
        self.boundary_ratio = boundary_ratio
        self.t_tgt_eq_t_cam_ratio = t_tgt_eq_t_cam_ratio
        self.cache_boundaries = cache_boundaries

        self.use_augs = use_augs
        self.dset = dset
        
        # Cache for boundaries
        self.boundary_cache = {} if cache_boundaries else None

        self.rgb_paths = []
        self.depth_paths = []
        self.normal_paths = []
        self.traj_paths = []
        self.annotation_paths = []
        self.full_idxs = []

        self.subdirs = []
        self.sequences = []
        self.subdirs.append(os.path.join(dataset_location, dset))

        for subdir in self.subdirs:
            for seq in glob.glob(os.path.join(subdir, "*/")):
                seq_name = seq.split('/')[-1]
                self.sequences.append(seq)

        self.sequences = sorted(self.sequences)
        if self.verbose:
            print(self.sequences)
        print('found %d unique videos in %s (dset=%s)' % (len(self.sequences), dataset_location, dset))
        
        ## load trajectories
        print('loading trajectories...')

        if quick:
           self.sequences = self.sequences[:1] 
        
        for seq in self.sequences:
            if self.verbose: 
                print('seq', seq)

            rgb_path = os.path.join(seq, 'rgbs')
            info_path = os.path.join(seq, 'info.npz')
            annotations_path = os.path.join(seq, 'anno.npz')
            
            if os.path.isfile(info_path) and os.path.isfile(annotations_path):

                info = np.load(info_path, allow_pickle=True)
                trajs_3d_shape = info['trajs_3d'].astype(np.float32)

                # Relax requirement: we only need a minimum number of trajectory points
                # Queries are sampled independently from the image, not from trajectories
                if len(trajs_3d_shape) and trajs_3d_shape[1] > 10:
                
                    for stride in strides:
                        for ii in range(0,len(os.listdir(rgb_path))-self.S*stride+1, clip_step):
                            full_idx = ii + np.arange(self.S)*stride
                            self.rgb_paths.append([os.path.join(seq, 'rgbs', 'rgb_%05d.jpg' % idx) for idx in full_idx])
                            self.depth_paths.append([os.path.join(seq, 'depths', 'depth_%05d.png' % idx) for idx in full_idx])
                            self.normal_paths.append([os.path.join(seq, 'normals', 'normal_%05d.jpg' % idx) for idx in full_idx])
                            self.annotation_paths.append(os.path.join(seq, 'anno.npz'))
                            self.full_idxs.append(full_idx)
                        if self.verbose:
                            sys.stdout.write('.')
                            sys.stdout.flush()
                elif self.verbose:
                    print('rejecting seq for missing 3d')
            elif self.verbose:
                print('rejecting seq for missing info or anno')

        print('collected %d clips of length %d in %s (dset=%s)' % (
            len(self.rgb_paths), self.S, dataset_location, dset))
    
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
    
    def sample_queries(self, depth_maps, trajs_2d, visibs=None, valids=None, T=None):
        """
        Sample queries according to the strategy
        Ensures that sampled points are visible and valid at t_src frames
        
        Args:
            depth_maps: (T, H, W) depth maps
            trajs_2d: (T, N, 2) existing trajectories (for reference)
            visibs: (T, N) visibility flags for trajectories (optional)
            valids: (T, N) validity flags for trajectories (optional)
            T: number of frames
        
        Returns:
            coords_uv: (num_queries, 2) normalized 2D coordinates
            t_src: (num_queries,) source frame indices
            t_tgt: (num_queries,) target frame indices
            t_cam: (num_queries,) camera reference frame indices
        """
        if T is None:
            T = depth_maps.shape[0]
        H, W = depth_maps.shape[1:3]
        
        # Create validity masks for each frame based on depth maps
        # A pixel is valid if it has valid depth (> 0) and is within image bounds
        validity_masks = []
        for t in range(T):
            depth_mask = (depth_maps[t] > 0).astype(np.float32)
            # Also avoid 1px edge for consistency with trajectory filtering
            edge_mask = np.ones((H, W), dtype=np.float32)
            edge_mask[0, :] = 0
            edge_mask[-1, :] = 0
            edge_mask[:, 0] = 0
            edge_mask[:, -1] = 0
            validity_mask = depth_mask * edge_mask
            validity_masks.append(validity_mask)
        
        # Sample time indices first
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
        
        # Sample coordinates ensuring validity at t_src frames
        all_coords = []
        num_boundary_queries = int(self.num_queries * self.boundary_ratio)
        
        for i in range(self.num_queries):
            t_src_i = t_src[i]
            validity_mask = validity_masks[t_src_i]  # Use validity mask for t_src frame
            
            # Compute boundaries for t_src frame (if boundary sampling)
            if i < num_boundary_queries:
                boundary_mask = self.compute_boundaries(depth_maps[t_src_i])
                # Combine with validity: boundary AND valid
                valid_boundary_mask = boundary_mask * validity_mask
                boundary_pixels = np.where(valid_boundary_mask > 0.5)
                
                if len(boundary_pixels[0]) > 0:
                    # Sample from valid boundary pixels
                    idx = np.random.randint(len(boundary_pixels[0]))
                    u = boundary_pixels[1][idx]
                    v = boundary_pixels[0][idx]
                    all_coords.append([u, v])
                else:
                    # Fallback to valid random sampling
                    valid_pixels = np.where(validity_mask > 0.5)
                    if len(valid_pixels[0]) > 0:
                        idx = np.random.randint(len(valid_pixels[0]))
                        u = valid_pixels[1][idx]
                        v = valid_pixels[0][idx]
                        all_coords.append([u, v])
                    else:
                        # Last resort: sample anywhere (should be rare)
                        u = np.random.randint(1, W-1)
                        v = np.random.randint(1, H-1)
                        all_coords.append([u, v])
            else:
                # Random sampling from valid pixels only
                valid_pixels = np.where(validity_mask > 0.5)
                if len(valid_pixels[0]) > 0:
                    idx = np.random.randint(len(valid_pixels[0]))
                    u = valid_pixels[1][idx]
                    v = valid_pixels[0][idx]
                    all_coords.append([u, v])
                else:
                    # Fallback: sample anywhere (should be rare)
                    u = np.random.randint(1, W-1)
                    v = np.random.randint(1, H-1)
                    all_coords.append([u, v])
        
        # Convert to numpy array and normalize to [0, 1]
        coords_uv = np.array(all_coords, dtype=np.float32)
        coords_uv[:, 0] /= W  # u
        coords_uv[:, 1] /= H  # v
        
        return coords_uv, t_src, t_tgt, t_cam
    
    def apply_photometric_augmentation(self, rgbs):
        """
        Apply time-consistent photometric augmentations
        
        Args:
            rgbs: (S, H, W, 3) numpy array
        
        Returns:
            rgbs: (S, H, W, 3) augmented numpy array
        """
        S, H, W, C = rgbs.shape
        
        # Time-consistent Color Jittering (brightness, saturation, contrast, hue)
        if random.random() < 0.8:  # Apply with 80% probability
            brightness_factor = random.uniform(0.8, 1.2)
            saturation_factor = random.uniform(0.8, 1.2)
            contrast_factor = random.uniform(0.8, 1.2)
            hue_factor = random.uniform(-0.1, 0.1)
            
            for s in range(S):
                # Convert to float
                rgb = rgbs[s].astype(np.float32) / 255.0
                
                # Brightness
                rgb = rgb * brightness_factor
                
                # Convert to HSV for saturation and hue
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] *= saturation_factor  # Saturation
                hsv[:, :, 0] = (hsv[:, :, 0] + hue_factor * 180) % 180  # Hue
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                
                # Contrast
                rgb = (rgb - 0.5) * contrast_factor + 0.5
                
                # Clamp and convert back
                rgb = np.clip(rgb, 0, 1)
                rgbs[s] = (rgb * 255.0).astype(np.uint8)
        
        # Random Color Drop (probability 0.2)
        if random.random() < 0.2:
            channel_to_drop = random.randint(0, 2)
            for s in range(S):
                rgbs[s][:, :, channel_to_drop] = 0
        
        # Gaussian Blur (probability 0.4)
        if random.random() < 0.4:
            kernel_size = random.choice([3, 5])
            sigma = random.uniform(0.5, 2.0)
            for s in range(S):
                rgbs[s] = cv2.GaussianBlur(rgbs[s], (kernel_size, kernel_size), sigma)
        
        return rgbs
    
    def apply_geometric_augmentation(self, rgbs, depths, normals, trajs_2d, pix_T_cams):
        """
        Apply geometric augmentations with corresponding GT transformations
        
        Args:
            rgbs: (S, H, W, 3) numpy array
            depths: (S, H, W) numpy array
            normals: (S, H, W, 3) numpy array
            trajs_2d: (S, N, 2) numpy array
            pix_T_cams: (S, 3, 3) camera intrinsics
        
        Returns:
            rgbs, depths, normals, trajs_2d, pix_T_cams: transformed arrays
            aspect_ratio: original aspect ratio (width/height)
        """
        S, H_orig, W_orig, C = rgbs.shape
        aspect_ratio = W_orig / H_orig
        
        # Random Crop (ratio 0.3 to 1.0)
        if random.random() < 0.8:
            crop_ratio = random.uniform(0.3, 1.0)
            crop_h = int(H_orig * crop_ratio)
            crop_w = int(W_orig * crop_ratio)
            
            top = random.randint(0, H_orig - crop_h)
            left = random.randint(0, W_orig - crop_w)
            
            rgbs = rgbs[:, top:top+crop_h, left:left+crop_w, :]
            depths = depths[:, top:top+crop_h, left:left+crop_w]
            normals = normals[:, top:top+crop_h, left:left+crop_w, :]
            
            # Update trajectories
            trajs_2d[:, :, 0] -= left
            trajs_2d[:, :, 1] -= top
            
            # Update intrinsics
            for s in range(S):
                pix_T_cams[s, 0, 2] -= left  # cx
                pix_T_cams[s, 1, 2] -= top    # cy
            
            H_orig, W_orig = crop_h, crop_w
        
        # Random Aspect Ratio (log-uniform sampling, then resize to square)
        if random.random() < 0.7:
            log_min_ar = np.log(0.5)
            log_max_ar = np.log(2.0)
            log_ar = random.uniform(log_min_ar, log_max_ar)
            target_ar = np.exp(log_ar)
            
            if target_ar > 1.0:
                new_h = int(self.img_size / target_ar)
                new_w = self.img_size
            else:
                new_h = self.img_size
                new_w = int(self.img_size * target_ar)
        else:
            new_h = self.img_size
            new_w = self.img_size
        
        # Resize to target size
        rgbs_resized = []
        depths_resized = []
        normals_resized = []
        
        for s in range(S):
            rgb_resized = cv2.resize(rgbs[s], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            depth_resized = cv2.resize(depths[s], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            normal_resized = cv2.resize(normals[s], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            rgbs_resized.append(rgb_resized)
            depths_resized.append(depth_resized)
            normals_resized.append(normal_resized)
        
        rgbs = np.stack(rgbs_resized, axis=0)
        depths = np.stack(depths_resized, axis=0)
        normals = np.stack(normals_resized, axis=0)
        
        # Update trajectories
        scale_x = new_w / W_orig
        scale_y = new_h / H_orig
        trajs_2d[:, :, 0] *= scale_x
        trajs_2d[:, :, 1] *= scale_y
        
        # Update intrinsics
        for s in range(S):
            pix_T_cams[s, 0, 0] *= scale_x  # fx
            pix_T_cams[s, 0, 2] *= scale_x  # cx
            pix_T_cams[s, 1, 1] *= scale_y  # fy
            pix_T_cams[s, 1, 2] *= scale_y  # cy
        
        # Pad or crop to square
        if new_h != self.img_size or new_w != self.img_size:
            rgbs_square = []
            depths_square = []
            normals_square = []
            
            pad_h = max(0, self.img_size - new_h)
            pad_w = max(0, self.img_size - new_w)
            top_pad = pad_h // 2
            left_pad = pad_w // 2
            
            for s in range(S):
                rgb_square = np.pad(rgbs[s], ((top_pad, pad_h-top_pad), (left_pad, pad_w-left_pad), (0, 0)), mode='constant')
                depth_square = np.pad(depths[s], ((top_pad, pad_h-top_pad), (left_pad, pad_w-left_pad)), mode='constant')
                normal_square = np.pad(normals[s], ((top_pad, pad_h-top_pad), (left_pad, pad_w-left_pad), (0, 0)), mode='constant')
                
                # Crop if needed
                if rgb_square.shape[0] > self.img_size:
                    rgb_square = rgb_square[:self.img_size, :, :]
                if rgb_square.shape[1] > self.img_size:
                    rgb_square = rgb_square[:, :self.img_size, :]
                if depth_square.shape[0] > self.img_size:
                    depth_square = depth_square[:self.img_size, :]
                if depth_square.shape[1] > self.img_size:
                    depth_square = depth_square[:, :self.img_size]
                if normal_square.shape[0] > self.img_size:
                    normal_square = normal_square[:self.img_size, :, :]
                if normal_square.shape[1] > self.img_size:
                    normal_square = normal_square[:, :self.img_size, :]
                
                rgbs_square.append(rgb_square)
                depths_square.append(depth_square)
                normals_square.append(normal_square)
            
            rgbs = np.stack(rgbs_square, axis=0)
            depths = np.stack(depths_square, axis=0)
            normals = np.stack(normals_square, axis=0)
            
            # Update trajectories
            trajs_2d[:, :, 0] += left_pad
            trajs_2d[:, :, 1] += top_pad
            
            # Update intrinsics
            for s in range(S):
                pix_T_cams[s, 0, 2] += left_pad  # cx
                pix_T_cams[s, 1, 2] += top_pad   # cy
        
        # Random Zoom In (probability 0.05)
        if random.random() < 0.05:
            zoom_factor = random.uniform(1.2, 2.0)
            center_h, center_w = self.img_size // 2, self.img_size // 2
            
            new_size = int(self.img_size / zoom_factor)
            top = center_h - new_size // 2
            left = center_w - new_size // 2
            
            rgbs_zoom = []
            depths_zoom = []
            normals_zoom = []
            
            for s in range(S):
                rgb_crop = rgbs[s, top:top+new_size, left:left+new_size, :]
                depth_crop = depths[s, top:top+new_size, left:left+new_size]
                normal_crop = normals[s, top:top+new_size, left:left+new_size, :]
                
                rgb_zoom = cv2.resize(rgb_crop, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                depth_zoom = cv2.resize(depth_crop, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                normal_zoom = cv2.resize(normal_crop, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                
                rgbs_zoom.append(rgb_zoom)
                depths_zoom.append(depth_zoom)
                normals_zoom.append(normal_zoom)
            
            rgbs = np.stack(rgbs_zoom, axis=0)
            depths = np.stack(depths_zoom, axis=0)
            normals = np.stack(normals_zoom, axis=0)
            
            # Update trajectories
            trajs_2d[:, :, 0] = (trajs_2d[:, :, 0] - left) * zoom_factor
            trajs_2d[:, :, 1] = (trajs_2d[:, :, 1] - top) * zoom_factor
            
            # Update intrinsics
            for s in range(S):
                pix_T_cams[s, 0, 0] *= zoom_factor  # fx
                pix_T_cams[s, 0, 2] = (pix_T_cams[s, 0, 2] - left) * zoom_factor  # cx
                pix_T_cams[s, 1, 1] *= zoom_factor  # fy
                pix_T_cams[s, 1, 2] = (pix_T_cams[s, 1, 2] - top) * zoom_factor  # cy
        
        return rgbs, depths, normals, trajs_2d, pix_T_cams, aspect_ratio
    
    def apply_temporal_subsampling(self, rgbs, depths, normals, trajs_2d, pix_T_cams, cams_T_world):
        """
        Apply temporal subsampling with random stride
        
        Args:
            All inputs with temporal dimension S
        
        Returns:
            Subsampled arrays, and indices used (for updating visibs/valids)
        """
        indices = None
        if random.random() < 0.5:  # Apply with 50% probability
            stride = random.choice([2, 3, 4])
            S_current = rgbs.shape[0]
            if stride < S_current:
                indices = np.arange(0, S_current, stride)
                if len(indices) < 2:
                    return rgbs, depths, normals, trajs_2d, pix_T_cams, cams_T_world, None
                
                rgbs = rgbs[indices]
                depths = depths[indices]
                normals = normals[indices]
                trajs_2d = trajs_2d[indices]
                pix_T_cams = pix_T_cams[indices]
                cams_T_world = cams_T_world[indices]
        
        return rgbs, depths, normals, trajs_2d, pix_T_cams, cams_T_world, indices

    def getitem_helper(self, index):
        sample = None
        gotit = False

        rgb_paths = self.rgb_paths[index]
        depth_paths = self.depth_paths[index]
        normal_paths = self.normal_paths[index]
        full_idx = self.full_idxs[index]
        annotations_path = self.annotation_paths[index]
        annotations = np.load(annotations_path, allow_pickle=True)
        trajs_2d = annotations['trajs_2d'][full_idx].astype(np.float32)
        visibs = annotations['visibs'][full_idx].astype(np.float32)
        valids = annotations['valids'][full_idx].astype(np.float32)
        trajs_world = annotations['trajs_3d'][full_idx].astype(np.float32)
        pix_T_cams = annotations['intrinsics'][full_idx].astype(np.float32)
        cams_T_world = annotations['extrinsics'][full_idx].astype(np.float32)

        # ensure no weird/huge values 
        trajs_world_sum = np.sum(np.abs(trajs_world - trajs_world[0:1]), axis=(0,2))
        not_huge = trajs_world_sum < 100
        trajs_world = trajs_world[:,not_huge]
        trajs_2d = trajs_2d[:,not_huge]
        valids = valids[:,not_huge]
        visibs = visibs[:,not_huge]
        
        # ensure that the point is good in frame0
        vis_and_val = valids * visibs
        vis0 = vis_and_val[0] > 0
        trajs_2d = trajs_2d[:,vis0]
        trajs_world = trajs_world[:,vis0]
        visibs = visibs[:,vis0]
        valids = valids[:,vis0]

        S,N,D = trajs_2d.shape
        assert(D==2)
        assert(S==self.S)
        
        # Relax the requirement: we just need enough points for query sampling
        # Queries are sampled independently from the image, not from trajectories
        if N < 10:  # Just need a minimum number for validation/filtering
            print('returning before cropping: N=%d; need at least 10 points' % N)
            return None, False
        
        trajs_cam = utils.geom.apply_4x4_py(cams_T_world, trajs_world)
        trajs_pix = utils.geom.apply_pix_T_cam_py(pix_T_cams, trajs_cam)

        # get rid of infs and nans in 2d
        valids_xy = np.ones_like(trajs_2d)
        inf_idx = np.where(np.isinf(trajs_2d))
        trajs_world[inf_idx] = 0
        trajs_cam[inf_idx] = 0
        trajs_2d[inf_idx] = 0
        valids_xy[inf_idx] = 0
        nan_idx = np.where(np.isnan(trajs_2d))
        trajs_world[nan_idx] = 0
        trajs_cam[nan_idx] = 0
        trajs_2d[nan_idx] = 0
        valids_xy[nan_idx] = 0
        inv_idx = np.where(np.sum(valids_xy, axis=2)<2) # S,N
        visibs[inv_idx] = 0
        valids[inv_idx] = 0

        rgbs = []
        for rgb_path in rgb_paths:
            with Image.open(rgb_path) as im:
                rgbs.append(np.array(im)[:, :, :3])

        depths = []
        for depth_path in depth_paths:
            depth16 = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            depth = depth16.astype(np.float32) / 65535.0 * 1000.0
            depths.append(depth)

        normals = []
        for normal_path in normal_paths:
            with Image.open(normal_path) as im:
                normals.append(np.array(im)[:, :, :3])

        H,W,C = rgbs[0].shape
        assert(C==3)
        
        # Store original aspect ratio
        original_aspect_ratio = W / H
        
        # Save original resolution images BEFORE any resizing/augmentation
        # These will be used for high-resolution patch extraction
        # Note: For augmentation case, we'll save after photometric aug but before geometric aug
        rgbs_orig = None  # Will be set later
        
        # Apply augmentations if enabled
        if self.use_augs:
            # Photometric augmentation (applied to original resolution too)
            rgbs = self.apply_photometric_augmentation(rgbs)
            # Save original resolution after photometric aug (before geometric aug)
            # After geometric aug, coordinates will be transformed, so we use the geometrically augmented original
            rgbs_orig_temp = np.stack(rgbs, axis=0).copy()  # (S, H_orig, W_orig, 3)
            H_orig, W_orig = H, W
            
            # Geometric augmentation
            # NOTE: After geometric augmentation (crop/resize), the relationship between 
            # resized coordinates and original image becomes complex. For simplicity, 
            # we save the original resolution images before geometric augmentation.
            # When using data augmentation, high-res patch extraction may not work correctly
            # for cropped regions. Consider using original resolution patch extraction only 
            # when use_augs=False, or implement coordinate mapping for augmented case.
            rgbs, depths, normals, trajs_2d, pix_T_cams, aspect_ratio = self.apply_geometric_augmentation(
                rgbs, depths, normals, trajs_2d, pix_T_cams
            )
            # Save original resolution (before geometric augmentation)
            rgbs_orig = rgbs_orig_temp
            
            # Temporal subsampling
            rgbs, depths, normals, trajs_2d, pix_T_cams, cams_T_world, subsample_indices = self.apply_temporal_subsampling(
                rgbs, depths, normals, trajs_2d, pix_T_cams, cams_T_world
            )
            
            # Apply same temporal subsampling to original resolution images
            if subsample_indices is not None:
                rgbs_orig = rgbs_orig[subsample_indices]
            
            # Update S and related arrays after temporal subsampling
            S = rgbs.shape[0]
            # Update visibs and valids to match new S (use indices if subsampling occurred)
            if subsample_indices is not None:
                visibs = visibs[subsample_indices]
                valids = valids[subsample_indices]
                trajs_cam = trajs_cam[subsample_indices]
                trajs_world = trajs_world[subsample_indices]
                trajs_pix = trajs_pix[subsample_indices]
            else:
                # No subsampling, but still ensure dimensions match
                visibs = visibs[:S]
                valids = valids[:S]
                trajs_cam = trajs_cam[:S]
                trajs_world = trajs_world[:S]
                trajs_pix = trajs_pix[:S]
        else:
            # Still resize to square for consistency
            # Save original resolution BEFORE resize
            rgbs_orig = np.stack(rgbs, axis=0).copy()  # (S, H_orig, W_orig, 3)
            H_orig, W_orig = H, W
            
            rgbs_resized = []
            depths_resized = []
            normals_resized = []
            
            for s in range(S):
                rgb_resized = cv2.resize(rgbs[s], (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                depth_resized = cv2.resize(depths[s], (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
                normal_resized = cv2.resize(normals[s], (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                
                rgbs_resized.append(rgb_resized)
                depths_resized.append(depth_resized)
                normals_resized.append(normal_resized)
            
            rgbs = np.stack(rgbs_resized, axis=0)
            depths = np.stack(depths_resized, axis=0)
            normals = np.stack(normals_resized, axis=0)
            
            # Update trajectories
            scale_x = self.img_size / W
            scale_y = self.img_size / H
            trajs_2d[:, :, 0] *= scale_x
            trajs_2d[:, :, 1] *= scale_y
            
            # Update intrinsics
            for s in range(S):
                pix_T_cams[s, 0, 0] *= scale_x  # fx
                pix_T_cams[s, 0, 2] *= scale_x  # cx
                pix_T_cams[s, 1, 1] *= scale_y  # fy
                pix_T_cams[s, 1, 2] *= scale_y  # cy
            
            # Convert aspect_ratio to tensor
            aspect_ratio = torch.tensor(original_aspect_ratio, dtype=torch.float32)
        
        H, W = rgbs.shape[1:3]
        
        # update visibility annotations
        for si in range(S):
            # avoid 1px edge
            oob_inds = np.logical_or(
                np.logical_or(trajs_2d[si,:,0] < 1, trajs_2d[si,:,0] > W-2),
                np.logical_or(trajs_2d[si,:,1] < 1, trajs_2d[si,:,1] > H-2))
            visibs[si,oob_inds] = 0

            # when a point moves far oob, don't supervise with it
            very_oob_inds = np.logical_or(
                np.logical_or(trajs_2d[si,:,0] < -64, trajs_2d[si,:,0] > W+64),
                np.logical_or(trajs_2d[si,:,1] < -64, trajs_2d[si,:,1] > H+64))
            valids[si,very_oob_inds] = 0

        # ensure that the point is good in frame0
        vis_and_val = valids * visibs
        vis0 = vis_and_val[0] > 0
        trajs_2d = trajs_2d[:,vis0]
        trajs_cam = trajs_cam[:,vis0]
        trajs_world = trajs_world[:,vis0]
        trajs_pix = trajs_pix[:,vis0]
        visibs = visibs[:,vis0]
        valids = valids[:,vis0]

        # ensure that the point is good in frame1 (only if S >= 2)
        if S >= 2:
            vis_and_val = valids * visibs
            vis1 = vis_and_val[1] > 0
            trajs_2d = trajs_2d[:,vis1]
            trajs_cam = trajs_cam[:,vis1]
            trajs_world = trajs_world[:,vis1]
            trajs_pix = trajs_pix[:,vis1]
            visibs = visibs[:,vis1]
            valids = valids[:,vis1]

        # ensure that the point is good in at least sqrt(S) frames
        val_ok = np.sum(valids, axis=0) >= max(np.sqrt(S),2)
        trajs_2d = trajs_2d[:,val_ok]
        trajs_cam = trajs_cam[:,val_ok]
        trajs_world = trajs_world[:,val_ok]
        trajs_pix = trajs_pix[:,val_ok]
        visibs = visibs[:,val_ok]
        valids = valids[:,val_ok]
        
        N = trajs_2d.shape[1]
        
        # Since queries are sampled independently from the image (not from trajectories),
        # we don't need to strictly limit N. We can keep more trajectory points if available,
        # but still use FPS to ensure good distribution if we have too many.
        max_traj_points = max(self.N, 100)  # Keep more points if available, but cap for efficiency
        
        if N < 10:  # Minimum threshold for valid trajectories
            return None, False
        
        # Use FPS to sample trajectory points (for efficiency, but not strictly required for queries)
        if N > max_traj_points:
            # even out the distribution, across initial positions and velocities
            # fps based on xy0 and mean motion
            xym = np.concatenate([trajs_2d[0], np.mean(trajs_2d[1:] - trajs_2d[:-1], axis=0)], axis=-1)
            inds = utils.misc.farthest_point_sample_py(xym, max_traj_points)
            trajs_2d = trajs_2d[:,inds]
            trajs_cam = trajs_cam[:,inds]
            trajs_world = trajs_world[:,inds]
            trajs_pix = trajs_pix[:,inds]
            visibs = visibs[:,inds]
            valids = valids[:,inds]
            N = max_traj_points

        # we won't supervise with the extremes, but let's clamp anyway just to be safe
        trajs_2d = np.minimum(np.maximum(trajs_2d, np.array([-64,-64])), np.array([W+64, H+64])) # S,N,2
        trajs_pix = np.minimum(np.maximum(trajs_pix, np.array([-64,-64])), np.array([W+64, H+64])) # S,N,2
            
        N = trajs_2d.shape[1]
        # Use all available trajectory points (up to a reasonable limit for batching)
        # Since queries are sampled independently, we don't need to strictly pad to self.N
        N_used = min(N, self.N)  # Use up to self.N for consistency, but can be less
        if N > self.N:
            # Randomly sample if we have more than self.N
            inds = np.random.choice(N, N_used, replace=False)
        else:
            inds = np.arange(N)
            N_used = N

        # prep for batching, pad to self.N for consistent batch shapes
        trajs_2d_full = np.zeros((S, self.N, 2)).astype(np.float32)
        trajs_cam_full = np.zeros((S, self.N, 3)).astype(np.float32)
        trajs_world_full = np.zeros((S, self.N, 3)).astype(np.float32)
        trajs_pix_full = np.zeros((S, self.N, 2)).astype(np.float32)
        visibs_full = np.zeros((S, self.N)).astype(np.float32)
        valids_full = np.zeros((S, self.N)).astype(np.float32)
        trajs_2d_full[:,:N_used] = trajs_2d[:,inds]
        trajs_cam_full[:,:N_used] = trajs_cam[:,inds]
        trajs_world_full[:,:N_used] = trajs_world[:,inds]
        trajs_pix_full[:,:N_used] = trajs_pix[:,inds]
        visibs_full[:,:N_used] = visibs[:,inds]
        valids_full[:,:N_used] = valids[:,inds]

        # Sample queries BEFORE converting to tensors (need numpy arrays for query sampling)
        # Ensure depth shape is (S, H, W) for query sampling
        depths_for_sampling = np.array(depths)
        if len(depths_for_sampling.shape) == 4 and depths_for_sampling.shape[1] == 1:
            depths_for_sampling = depths_for_sampling.squeeze(1)  # (S, 1, H, W) -> (S, H, W)
        elif len(depths_for_sampling.shape) == 3:
            pass  # Already (S, H, W)
        else:
            raise ValueError(f"Unexpected depth shape: {depths_for_sampling.shape}")
        
        # Generate queries: coords_uv are in normalized coordinates [0,1] relative to resized video
        # The sample_queries method ensures sampled points have valid depth (>0) at their t_src frames
        # This guarantees visibility and validity at the source frame
        coords_uv, t_src, t_tgt, t_cam = self.sample_queries(
            depths_for_sampling,
            trajs_2d_full,  # (S, N, 2) - Only used for reference, not for sampling
            visibs=None,  # Not used for coordinate sampling, validity determined by depth maps
            valids=None,  # Not used for coordinate sampling, validity determined by depth maps
            T=S
        )
        
        # Now convert everything to tensors
        # Convert to float32 and normalize to [0, 1] for model input
        rgbs = torch.from_numpy(rgbs).float() / 255.0  # Convert uint8 [0,255] to float32 [0,1]
        rgbs = rgbs.permute(0,3,1,2)  # (S, H, W, 3) -> (S, 3, H, W)
        # Convert original resolution images to tensor
        rgbs_orig = torch.from_numpy(rgbs_orig).float() / 255.0  # Convert uint8 to float32 [0,1]
        rgbs_orig = rgbs_orig.permute(0,3,1,2)  # (S, H_orig, W_orig, 3) -> (S, 3, H_orig, W_orig)
        
        # Handle depths: ensure it's (S, 1, H, W) and float32
        if len(depths.shape) == 3:
            depths = torch.from_numpy(depths).float().unsqueeze(1)  # (S, H, W) -> (S, 1, H, W)
        else:
            depths = torch.from_numpy(depths).float()  # Already (S, 1, H, W)
        # Convert normals to float32 (if not already)
        normals = torch.from_numpy(normals).float().permute(0,3,1,2)  # (S, H, W, 3) -> (S, 3, H, W)
        # Convert all arrays to float32 tensors
        trajs_2d = torch.from_numpy(trajs_2d_full).float()  # S,N,2
        trajs_cam = torch.from_numpy(trajs_cam_full).float()  # S,N,3
        trajs_world = torch.from_numpy(trajs_world_full).float()  # S,N,3
        trajs_pix = torch.from_numpy(trajs_pix_full).float()  # S,N,2
        visibs = torch.from_numpy(visibs_full).float()  # S,N
        valids = torch.from_numpy(valids_full).float()  # S,N
        pix_T_cams = torch.from_numpy(pix_T_cams).float()  # S,3,3
        cams_T_world = torch.from_numpy(cams_T_world).float()  # S,4,4
        
        # Convert to tensors - queries are ready for network input
        coords_uv = torch.from_numpy(coords_uv).float()  # (num_queries, 2) normalized [0,1]
        t_src = torch.from_numpy(t_src).long()  # (num_queries,)
        t_tgt = torch.from_numpy(t_tgt).long()  # (num_queries,)
        t_cam = torch.from_numpy(t_cam).long()  # (num_queries,)
        
        # Ensure we have exactly num_queries queries
        assert coords_uv.shape[0] == self.num_queries, \
            f"Expected {self.num_queries} queries, got {coords_uv.shape[0]}"

        # Convert aspect_ratio to tensor if it's a scalar
        if isinstance(aspect_ratio, (int, float)):
            aspect_ratio = torch.tensor(aspect_ratio, dtype=torch.float32)
        
        sample = {
            'video': rgbs,  # Resized video (S, 3, H, W) for encoder
            'video_orig': rgbs_orig,  # Original resolution video (S, 3, H_orig, W_orig) for patch extraction
            'depths': depths,
            'normals': normals,
            'trajs_2d': trajs_2d,
            'trajs_cam': trajs_cam,
            'trajs_world': trajs_world,
            'trajs_pix': trajs_pix,
            'pix_T_cams': pix_T_cams,
            'cams_T_world': cams_T_world,
            'visibs': visibs,
            'valids': valids,
            'coords_uv': coords_uv,
            't_src': t_src,
            't_tgt': t_tgt,
            't_cam': t_cam,
            'aspect_ratio': aspect_ratio,
            'annotations_path': annotations_path,
        }
        
        return sample, True

    
    def __getitem__(self, index):
        gotit = False
        sample, gotit = self.getitem_helper(index)
        if not gotit:
            print('warning: sampling failed')  
            # return a fake sample, so we can still collate
            sample = {
                'video': torch.zeros((self.S, 3, self.img_size, self.img_size), dtype=torch.float32),  # Resized video
                'video_orig': torch.zeros((self.S, 3, 540, 960), dtype=torch.float32),  # Original resolution video (dummy size)
                'depths': torch.zeros((self.S, 1, self.img_size, self.img_size), dtype=torch.float32),
                'normals': torch.zeros((self.S, 3, self.img_size, self.img_size), dtype=torch.float32),
                'trajs_2d': torch.zeros((self.S, self.N, 2), dtype=torch.float32),
                'trajs_cam': torch.zeros((self.S, self.N, 3), dtype=torch.float32),
                'trajs_world': torch.zeros((self.S, self.N, 3), dtype=torch.float32),
                'trajs_pix': torch.zeros((self.S, self.N, 2), dtype=torch.float32),
                'pix_T_cams': torch.zeros((self.S, 3, 3), dtype=torch.float32),
                'cams_T_world': torch.zeros((self.S, 4, 4), dtype=torch.float32),
                'visibs': torch.zeros((self.S, self.N), dtype=torch.float32),
                'valids': torch.zeros((self.S, self.N), dtype=torch.float32),
                'coords_uv': torch.zeros((self.num_queries, 2), dtype=torch.float32),  # num_queries (2048) query coordinates
                't_src': torch.zeros((self.num_queries,), dtype=torch.long),
                't_tgt': torch.zeros((self.num_queries,), dtype=torch.long),
                't_cam': torch.zeros((self.num_queries,), dtype=torch.long),
                'aspect_ratio': torch.tensor(1.0, dtype=torch.float32),
                'annotations_path': '',
            }
        return sample, gotit

    def __len__(self):
        return len(self.rgb_paths)