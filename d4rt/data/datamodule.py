"""
PyTorch Lightning DataModule for D4RT
"""

import torch
from torch.utils.data import DataLoader
import lightning as L
from typing import Optional

from .dataset import D4RTDataset


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

