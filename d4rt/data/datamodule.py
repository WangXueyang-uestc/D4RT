"""
PyTorch Lightning DataModule for D4RT
"""

import torch
from torch.utils.data import DataLoader
import lightning as L
from typing import Optional

from .dataset import D4RTDataset, PointOdysseyDataset


class D4RTDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for D4RT"""
    
    def __init__(
        self,
        train_video_paths: list,
        val_video_paths: Optional[list] = None,
        num_queries: int = 2048,
        img_size: int = 256,
        max_frames: int = 100,
        batch_size: int = 1,
        num_workers: int = 4,
        pin_memory: bool = True,
        **dataset_kwargs
    ):
        super().__init__()
        self.train_video_paths = train_video_paths
        self.val_video_paths = val_video_paths
        self.num_queries = num_queries
        self.img_size = img_size
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_kwargs = dataset_kwargs
    
    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            self.train_dataset = D4RTDataset(
                self.train_video_paths,
                num_queries=self.num_queries,
                img_size=self.img_size,
                max_frames=self.max_frames,
                **self.dataset_kwargs
            )
            
            if self.val_video_paths is not None:
                self.val_dataset = D4RTDataset(
                    self.val_video_paths,
                    num_queries=self.num_queries,
                    img_size=self.img_size,
                    max_frames=self.max_frames,
                    **self.dataset_kwargs
                )
            else:
                # Split training data
                total_size = len(self.train_dataset)
                val_size = int(total_size * 0.1)
                train_size = total_size - val_size
                self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                    self.train_dataset, [train_size, val_size]
                )
        
        if stage == "test" or stage is None:
            if hasattr(self, 'val_dataset'):
                self.test_dataset = self.val_dataset
            else:
                self.test_dataset = D4RTDataset(
                    self.val_video_paths if self.val_video_paths else self.train_video_paths,
                    num_queries=self.num_queries,
                    img_size=self.img_size,
                    max_frames=self.max_frames,
                    **self.dataset_kwargs
                )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function for batching"""
        # Assuming batch is a list of dictionaries
        # You may need to adjust this based on your dataset's __getitem__ return format
        collated = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                collated[key] = torch.stack([item[key] for item in batch])
            else:
                collated[key] = [item[key] for item in batch]
        return collated


class PointOdysseyDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for PointOdysseyDataset"""
    
    def __init__(
        self,
        dataset_location='/orion/group/point_odyssey',
        train_dset='train',
        val_dset='val',
        use_augs=True,
        use_val=False,  # Whether to use validation dataset (default: False)
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
        batch_size=1,
        num_workers=4,
        pin_memory=True,
    ):
        super().__init__()
        self.dataset_location = dataset_location
        self.train_dset = train_dset
        self.val_dset = val_dset
        self.use_augs = use_augs
        self.use_val = use_val
        self.S = S
        self.N = N
        self.strides = strides
        self.clip_step = clip_step
        self.quick = quick
        self.verbose = verbose
        self.img_size = img_size
        self.num_queries = num_queries
        self.boundary_ratio = boundary_ratio
        self.t_tgt_eq_t_cam_ratio = t_tgt_eq_t_cam_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            self.train_dataset = PointOdysseyDataset(
                dataset_location=self.dataset_location,
                dset=self.train_dset,
                use_augs=self.use_augs,
                S=self.S,
                N=self.N,
                strides=self.strides,
                clip_step=self.clip_step,
                quick=self.quick,
                verbose=self.verbose,
                img_size=self.img_size,
                num_queries=self.num_queries,
                boundary_ratio=self.boundary_ratio,
                t_tgt_eq_t_cam_ratio=self.t_tgt_eq_t_cam_ratio,
            )
            
            # Create validation dataset only if use_val is True
            if self.use_val:
                self.val_dataset = PointOdysseyDataset(
                    dataset_location=self.dataset_location,
                    dset=self.val_dset,
                    use_augs=False,  # No augmentation for validation
                    S=self.S,
                    N=self.N,
                    strides=self.strides,
                    clip_step=self.clip_step,
                    quick=self.quick,
                    verbose=self.verbose,
                    img_size=self.img_size,
                    num_queries=self.num_queries,
                    boundary_ratio=self.boundary_ratio,
                    t_tgt_eq_t_cam_ratio=self.t_tgt_eq_t_cam_ratio,
                )
            else:
                self.val_dataset = None
        
        if stage == "test" or stage is None:
            self.test_dataset = PointOdysseyDataset(
                dataset_location=self.dataset_location,
                dset=self.val_dset,
                use_augs=False,
                S=self.S,
                N=self.N,
                strides=self.strides,
                clip_step=self.clip_step,
                quick=self.quick,
                verbose=self.verbose,
                img_size=self.img_size,
                num_queries=self.num_queries,
                boundary_ratio=self.boundary_ratio,
                t_tgt_eq_t_cam_ratio=self.t_tgt_eq_t_cam_ratio,
            )
    
    def train_dataloader(self):
        # Check if dataset is empty
        if len(self.train_dataset) == 0:
            raise ValueError(
                f"Training dataset is empty! "
                f"Dataset location: {self.dataset_location}, "
                f"Dataset split: {self.train_dset}. "
                f"Please check if the dataset path is correct and contains data."
            )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn_pointodyssey
        )
    
    def val_dataloader(self):
        # Return validation dataloader if available
        # Note: If use_val=False, limit_val_batches=0 should be set in trainer to skip validation
        if not hasattr(self, 'val_dataset') or self.val_dataset is None:
            # Create a dummy dataset to avoid errors, but limit_val_batches=0 will skip it
            from torch.utils.data import TensorDataset
            dummy_data = torch.zeros((1, 1))
            dummy_dataset = TensorDataset(dummy_data)
            return DataLoader(
                dummy_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                pin_memory=False
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn_pointodyssey
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn_pointodyssey
        )
    
    def _collate_fn_pointodyssey(self, batch):
        """Custom collate function for PointOdysseyDataset batching"""
        # Filter out failed samples
        valid_samples = [item for item, gotit in batch if gotit]
        
        if len(valid_samples) == 0:
            # Return a dummy batch if all samples failed
            sample = batch[0][0] if len(batch) > 0 else {}
            return {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v] for k, v in sample.items()}
        
        collated = {}
        for key in valid_samples[0].keys():
            if isinstance(valid_samples[0][key], torch.Tensor):
                try:
                    collated[key] = torch.stack([item[key] for item in valid_samples])
                except RuntimeError:
                    # Handle variable-length sequences
                    collated[key] = [item[key] for item in valid_samples]
            else:
                collated[key] = [item[key] for item in valid_samples]
        
        return collated

