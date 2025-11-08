"""
LiDAR and Camera Data Processing for KITTI Dataset

This module handles:
- LiDAR point cloud to depth image conversion
- Camera-LiDAR calibration and alignment
- Multi-modal data fusion
"""

import numpy as np
import cv2
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import yaml
from dataclasses import dataclass


@dataclass
class CalibrationData:
    """Camera-LiDAR calibration parameters"""
    P2: np.ndarray  # 3x4 projection matrix
    R0_rect: np.ndarray  # 3x3 rectification matrix
    Tr_velo_to_cam: np.ndarray  # 3x4 transformation matrix
    Tr_imu_to_velo: np.ndarray  # 3x4 transformation matrix


class LidarProcessor:
    """Processes LiDAR point clouds and converts to depth images"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.depth_max = config['data']['depth_max']
        self.image_size = tuple(config['data']['image_size'])
        
    def load_lidar_points(self, bin_path: str) -> np.ndarray:
        """Load LiDAR points from .bin file"""
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        return points
    
    def filter_points(self, points: np.ndarray) -> np.ndarray:
        """Filter points based on range and height"""
        # Remove points too far or too close
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
        mask = (distances > 1.0) & (distances < self.depth_max)
        
        # Remove ground points (optional)
        height_threshold = -1.5  # meters
        mask &= points[:, 2] > height_threshold
        
        return points[mask]
    
    def project_to_image(self, points: np.ndarray, calib: CalibrationData) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3D points to 2D image coordinates"""
        # Transform points to camera coordinates
        points_cam = self.transform_to_camera(points, calib)
        
        # Add homogeneous coordinate for projection
        points_cam_homo = np.column_stack([points_cam, np.ones(len(points_cam))])
        
        # Project to image plane using P2 matrix (3x4)
        points_img = calib.P2 @ points_cam_homo.T  # 3x4 @ 4xN = 3xN
        points_img = points_img.T
        
        # Normalize homogeneous coordinates
        u = points_img[:, 0] / points_img[:, 2]
        v = points_img[:, 1] / points_img[:, 2]
        depth = points_img[:, 2]
        
        # Filter points within image bounds
        valid = (u >= 0) & (u < 1242) & (v >= 0) & (v < 375) & (depth > 0)
        
        return np.column_stack([u[valid], v[valid], depth[valid]]), valid
    
    def transform_to_camera(self, points: np.ndarray, calib: CalibrationData) -> np.ndarray:
        """Transform LiDAR points to camera coordinate system"""
        # Add homogeneous coordinate
        points_homo = np.column_stack([points[:, :3], np.ones(len(points))])
        
        # Apply transformations: Tr_velo_to_cam is 3x4, so we need to handle it properly
        # First transform to camera coordinates
        points_cam = calib.Tr_velo_to_cam @ points_homo.T  # 3x4 @ 4xN = 3xN
        
        # Then apply rectification
        points_cam = calib.R0_rect @ points_cam  # 3x3 @ 3xN = 3xN
        
        return points_cam.T
    
    def create_depth_image(self, projected_points: np.ndarray, 
                          image_shape: Tuple[int, int]) -> np.ndarray:
        """Create depth image from projected points"""
        depth_image = np.zeros(image_shape, dtype=np.float32)
        
        if len(projected_points) == 0:
            return depth_image
        
        u, v, depth = projected_points[:, 0], projected_points[:, 1], projected_points[:, 2]
        
        # Convert to integer coordinates
        u_int = np.round(u).astype(int)
        v_int = np.round(v).astype(int)
        
        # Filter valid coordinates
        valid = (u_int >= 0) & (u_int < image_shape[1]) & (v_int >= 0) & (v_int < image_shape[0])
        
        if np.any(valid):
            # Use minimum depth for overlapping points
            for i in range(len(u_int)):
                if valid[i]:
                    if depth_image[v_int[i], u_int[i]] == 0 or depth[i] < depth_image[v_int[i], u_int[i]]:
                        depth_image[v_int[i], u_int[i]] = depth[i]
        
        return depth_image
    
    def resize_depth_image(self, depth_image: np.ndarray) -> np.ndarray:
        """Resize depth image to target size"""
        return cv2.resize(depth_image, self.image_size, interpolation=cv2.INTER_NEAREST)


class CameraProcessor:
    """Processes camera images"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_size = tuple(config['data']['image_size'])
        self.camera_id = config['data']['camera_id']
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load camera image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size"""
        return cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        return image.astype(np.float32) / 255.0


class DataFusion:
    """Fuses camera and LiDAR data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lidar_processor = LidarProcessor(config)
        self.camera_processor = CameraProcessor(config)
    
    def load_calibration(self, calib_path: str) -> CalibrationData:
        """Load calibration data from file"""
        calib_data = {}
        
        with open(calib_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    calib_data[key.strip()] = value.strip()
        
        # Parse matrices
        P2 = np.array(calib_data['P2'].split(), dtype=np.float32).reshape(3, 4)
        R0_rect = np.array(calib_data['R0_rect'].split(), dtype=np.float32).reshape(3, 3)
        Tr_velo_to_cam = np.array(calib_data['Tr_velo_to_cam'].split(), dtype=np.float32).reshape(3, 4)
        Tr_imu_to_velo = np.array(calib_data['Tr_imu_to_velo'].split(), dtype=np.float32).reshape(3, 4)
        
        return CalibrationData(P2, R0_rect, Tr_velo_to_cam, Tr_imu_to_velo)
    
    def fuse_data(self, image_path: str, lidar_path: str, calib_path: str) -> Dict[str, np.ndarray]:
        """Fuse camera image and LiDAR data"""
        # Load calibration
        calib = self.load_calibration(calib_path)
        
        # Load and process camera image
        image = self.camera_processor.load_image(image_path)
        image = self.camera_processor.resize_image(image)
        image = self.camera_processor.normalize_image(image)
        
        # Load and process LiDAR points
        points = self.lidar_processor.load_lidar_points(lidar_path)
        points = self.lidar_processor.filter_points(points)
        
        # Project to image and create depth map
        projected_points, valid = self.lidar_processor.project_to_image(points, calib)
        depth_image = self.lidar_processor.create_depth_image(
            projected_points, (375, 1242)  # Original KITTI image size
        )
        depth_image = self.lidar_processor.resize_depth_image(depth_image)
        
        # Normalize depth
        if self.config['preprocessing']['normalize_depth']:
            depth_image = depth_image / self.config['data']['depth_max']
        
        # Create fused input (RGB + Depth)
        fused_input = np.concatenate([image, depth_image[..., np.newaxis]], axis=2)
        
        return {
            'image': image,
            'depth': depth_image,
            'fused_input': fused_input,
            'points': points[valid]
        }


def main():
    """Example usage"""
    # Load configuration
    with open('configs/preprocessing.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize fusion processor
    fusion = DataFusion(config)
    
    # Example paths (you'll need to provide actual KITTI paths)
    image_path = "data/raw/kitti/training/image_2/000000.png"
    lidar_path = "data/raw/kitti/training/velodyne/000000.bin"
    calib_path = "data/raw/kitti/training/calib/000000.txt"
    
    try:
        # Fuse data
        fused_data = fusion.fuse_data(image_path, lidar_path, calib_path)
        
        print("Data fusion successful!")
        print(f"Image shape: {fused_data['image'].shape}")
        print(f"Depth shape: {fused_data['depth'].shape}")
        print(f"Fused input shape: {fused_data['fused_input'].shape}")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please ensure KITTI dataset is properly downloaded and paths are correct.")


if __name__ == "__main__":
    main()
