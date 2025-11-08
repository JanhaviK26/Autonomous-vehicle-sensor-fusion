"""
Dataset Classes for KITTI Sensor Fusion

This module provides PyTorch dataset classes for loading and preprocessing
KITTI data for depth prediction and segmentation tasks.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional, Any
import yaml
from pathlib import Path

from .processing import DataFusion


class KITTIDepthDataset(Dataset):
    """Dataset for depth prediction task"""
    
    def __init__(self, data_path: str, config: Dict[str, Any], 
                 split: str = 'train', transform=None):
        self.data_path = Path(data_path)
        self.config = config
        self.split = split
        self.transform = transform
        
        # Initialize data fusion processor
        self.fusion = DataFusion(config)
        
        # Load data split
        self.samples = self._load_split()
        
    def _load_split(self) -> List[Dict[str, str]]:
        """Load data split from directory structure"""
        split_path = self.data_path / self.split
        
        samples = []
        for sample_dir in sorted(split_path.iterdir()):
            if sample_dir.is_dir():
                # Look for image and LiDAR files
                image_path = sample_dir / 'image.png'
                lidar_path = sample_dir / 'lidar.bin'
                calib_path = sample_dir / 'calib.txt'
                
                if all(p.exists() for p in [image_path, lidar_path, calib_path]):
                    samples.append({
                        'image': str(image_path),
                        'lidar': str(lidar_path),
                        'calib': str(calib_path),
                        'sample_id': sample_dir.name
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        sample = self.samples[idx]
        
        try:
            # Fuse data
            fused_data = self.fusion.fuse_data(
                sample['image'], 
                sample['lidar'], 
                sample['calib']
            )
            
            # Convert to tensors
            image = torch.from_numpy(fused_data['image']).permute(2, 0, 1)  # HWC -> CHW
            depth = torch.from_numpy(fused_data['depth']).unsqueeze(0)  # Add channel dim
            
            # Apply transforms if provided
            if self.transform:
                # Stack image and depth for joint transformation
                stacked = torch.cat([image, depth], dim=0)
                stacked = self.transform(stacked)
                image = stacked[:3]  # RGB channels
                depth = stacked[3:4]  # Depth channel
            
            return {
                'image': image,
                'depth': depth,
                'sample_id': sample['sample_id']
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a dummy sample
            dummy_image = torch.zeros(3, 256, 256)
            dummy_depth = torch.zeros(1, 256, 256)
            return {
                'image': dummy_image,
                'depth': dummy_depth,
                'sample_id': f"error_{idx}"
            }


class KITTISegmentationDataset(Dataset):
    """Dataset for drivable area segmentation task"""
    
    def __init__(self, data_path: str, config: Dict[str, Any], 
                 split: str = 'train', transform=None):
        self.data_path = Path(data_path)
        self.config = config
        self.split = split
        self.transform = transform
        
        # Initialize data fusion processor
        self.fusion = DataFusion(config)
        
        # Load data split
        self.samples = self._load_split()
        
    def _load_split(self) -> List[Dict[str, str]]:
        """Load data split from directory structure"""
        split_path = self.data_path / self.split
        
        samples = []
        for sample_dir in sorted(split_path.iterdir()):
            if sample_dir.is_dir():
                # Look for image, LiDAR, and segmentation mask files
                image_path = sample_dir / 'image.png'
                lidar_path = sample_dir / 'lidar.bin'
                calib_path = sample_dir / 'calib.txt'
                mask_path = sample_dir / 'mask.png'
                
                if all(p.exists() for p in [image_path, lidar_path, calib_path, mask_path]):
                    samples.append({
                        'image': str(image_path),
                        'lidar': str(lidar_path),
                        'calib': str(calib_path),
                        'mask': str(mask_path),
                        'sample_id': sample_dir.name
                    })
        
        return samples
    
    def __load_mask(self, mask_path: str) -> np.ndarray:
        """Load segmentation mask"""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        # Resize to target size
        mask = cv2.resize(mask, self.config['data']['image_size'], 
                         interpolation=cv2.INTER_NEAREST)
        
        # Convert to binary mask (0: background, 1: drivable)
        mask = (mask > 128).astype(np.uint8)
        
        return mask
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        sample = self.samples[idx]
        
        try:
            # Fuse data
            fused_data = self.fusion.fuse_data(
                sample['image'], 
                sample['lidar'], 
                sample['calib']
            )
            
            # Load segmentation mask
            mask = self.__load_mask(sample['mask'])
            
            # Convert to tensors
            image = torch.from_numpy(fused_data['image']).permute(2, 0, 1)  # HWC -> CHW
            depth = torch.from_numpy(fused_data['depth']).unsqueeze(0)  # Add channel dim
            mask = torch.from_numpy(mask).long()  # Segmentation mask
            
            # Apply transforms if provided
            if self.transform:
                # Stack image and depth for joint transformation
                stacked = torch.cat([image, depth], dim=0)
                stacked = self.transform(stacked)
                image = stacked[:3]  # RGB channels
                depth = stacked[3:4]  # Depth channel
                
                # Apply same transform to mask
                mask = self.transform(mask.unsqueeze(0)).squeeze(0).long()
            
            return {
                'image': image,
                'depth': depth,
                'mask': mask,
                'sample_id': sample['sample_id']
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a dummy sample
            dummy_image = torch.zeros(3, 256, 256)
            dummy_depth = torch.zeros(1, 256, 256)
            dummy_mask = torch.zeros(256, 256, dtype=torch.long)
            return {
                'image': dummy_image,
                'depth': dummy_depth,
                'mask': dummy_mask,
                'sample_id': f"error_{idx}"
            }


class DataAugmentation:
    """Data augmentation transforms"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.aug_config = config.get('augmentation', {})
    
    def get_transforms(self, split: str = 'train'):
        """Get transforms for specific split"""
        if split == 'train':
            return self._get_train_transforms()
        else:
            return self._get_val_transforms()
    
    def _get_train_transforms(self):
        """Get training transforms"""
        transforms = []
        
        # Horizontal flip
        if self.aug_config.get('horizontal_flip', False):
            transforms.append(self._horizontal_flip)
        
        # Random rotation
        if 'rotation_range' in self.aug_config:
            transforms.append(self._random_rotation)
        
        # Color jitter
        if 'brightness_range' in self.aug_config or 'contrast_range' in self.aug_config:
            transforms.append(self._color_jitter)
        
        return transforms
    
    def _get_val_transforms(self):
        """Get validation transforms (no augmentation)"""
        return []
    
    def _horizontal_flip(self, data: torch.Tensor) -> torch.Tensor:
        """Random horizontal flip"""
        if torch.rand(1) > 0.5:
            return torch.flip(data, dims=[2])  # Flip width dimension
        return data
    
    def _random_rotation(self, data: torch.Tensor) -> torch.Tensor:
        """Random rotation"""
        angle = torch.rand(1) * self.aug_config['rotation_range'] * 2 - self.aug_config['rotation_range']
        
        # Convert to PIL for rotation
        if data.dim() == 3:  # CHW format
            data_pil = torchvision.transforms.functional.to_pil_image(data)
            rotated = torchvision.transforms.functional.rotate(data_pil, angle.item())
            return torchvision.transforms.functional.to_tensor(rotated)
        
        return data
    
    def _color_jitter(self, data: torch.Tensor) -> torch.Tensor:
        """Color jittering"""
        if data.shape[0] >= 3:  # Has RGB channels
            brightness = self.aug_config.get('brightness_range', 0)
            contrast = self.aug_config.get('contrast_range', 0)
            
            # Apply brightness
            if brightness > 0:
                factor = 1 + torch.rand(1) * brightness * 2 - brightness
                data[:3] = data[:3] * factor
            
            # Apply contrast
            if contrast > 0:
                factor = 1 + torch.rand(1) * contrast * 2 - contrast
                mean = data[:3].mean()
                data[:3] = (data[:3] - mean) * factor + mean
        
        return data


def create_dataloaders(config: Dict[str, Any], data_path: str) -> Dict[str, DataLoader]:
    """Create data loaders for all splits"""
    dataloaders = {}
    
    # Get augmentation transforms
    aug = DataAugmentation(config)
    
    for split in ['train', 'val', 'test']:
        # Create dataset
        if 'segmentation' in config.get('model', {}).get('name', ''):
            dataset = KITTISegmentationDataset(
                data_path, config, split=split, 
                transform=aug.get_transforms(split)
            )
        else:
            dataset = KITTIDepthDataset(
                data_path, config, split=split,
                transform=aug.get_transforms(split)
            )
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=(split == 'train'),
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory'],
            drop_last=(split == 'train')
        )
        
        dataloaders[split] = dataloader
    
    return dataloaders


def main():
    """Example usage"""
    # Load configuration
    with open('configs/depth_model.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data loaders
    dataloaders = create_dataloaders(config, 'data/processed')
    
    # Test data loading
    for split, dataloader in dataloaders.items():
        print(f"\n{split.upper()} DataLoader:")
        print(f"  Number of samples: {len(dataloader.dataset)}")
        print(f"  Number of batches: {len(dataloader)}")
        
        # Test first batch
        try:
            batch = next(iter(dataloader))
            print(f"  Batch keys: {list(batch.keys())}")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key} shape: {value.shape}")
        except Exception as e:
            print(f"  Error loading batch: {e}")


if __name__ == "__main__":
    main()
