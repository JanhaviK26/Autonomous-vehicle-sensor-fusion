"""
KITTI Unsynced/Unrectified Data Processor

This module handles the raw KITTI data format (unsynced and unrectified)
and converts it to the format needed for sensor fusion training.
"""

import numpy as np
import cv2
import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import yaml
from dataclasses import dataclass
import struct


@dataclass
class RawKITTISequence:
    """Raw KITTI sequence data structure"""
    sequence_id: str
    date: str
    drive_id: str
    timestamps: List[float]
    images: List[str]
    lidar: List[str]
    calibration: str


class RawKITTIProcessor:
    """Processor for raw KITTI unsynced/unrectified data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_size = tuple(config['data']['image_size'])
        self.depth_max = config['data']['depth_max']
        
    def load_raw_sequence(self, sequence_path: str) -> RawKITTISequence:
        """Load a raw KITTI sequence"""
        seq_path = Path(sequence_path)
        
        # Extract sequence info from path
        parts = seq_path.name.split('_')
        date = f"{parts[0]}_{parts[1]}_{parts[2]}"
        drive_id = parts[3]
        sequence_id = seq_path.name
        
        # Load timestamps
        timestamps_file = seq_path / "timestamps.txt"
        timestamps = []
        if timestamps_file.exists():
            with open(timestamps_file, 'r') as f:
                timestamps = [float(line.strip()) for line in f.readlines()]
        
        # Find image files
        image_dir = seq_path / "image_02" / "data"  # Left color camera
        images = []
        if image_dir.exists():
            images = sorted([str(f) for f in image_dir.glob("*.png")])
        
        # Find LiDAR files
        lidar_dir = seq_path / "velodyne_points" / "data"
        lidar = []
        if lidar_dir.exists():
            lidar = sorted([str(f) for f in lidar_dir.glob("*.bin")])
        
        # Find calibration file
        calib_file = seq_path / "calib_cam_to_cam.txt"
        calibration = str(calib_file) if calib_file.exists() else ""
        
        return RawKITTISequence(
            sequence_id=sequence_id,
            date=date,
            drive_id=drive_id,
            timestamps=timestamps,
            images=images,
            lidar=lidar,
            calibration=calibration
        )
    
    def process_raw_image(self, image_path: str) -> np.ndarray:
        """Process raw unrectified image"""
        # Load raw image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def process_raw_lidar(self, lidar_path: str) -> np.ndarray:
        """Process raw LiDAR point cloud"""
        # Load raw LiDAR data
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        
        # Filter points by range
        distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2)
        mask = (distances > 1.0) & (distances < self.depth_max)
        
        # Filter by height (remove ground points)
        height_threshold = -1.5
        mask &= points[:, 2] > height_threshold
        
        return points[mask]
    
    def load_raw_calibration(self, calib_path: str) -> Dict[str, np.ndarray]:
        """Load raw calibration data"""
        calib_data = {}
        
        with open(calib_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    calib_data[key.strip()] = value.strip()
        
        # Parse matrices for raw data
        matrices = {}
        
        # Camera matrices (raw/unrectified)
        for i in range(4):
            key = f'K_{i:02d}'
            if key in calib_data:
                matrices[key] = np.array(calib_data[key].split(), dtype=np.float32).reshape(3, 3)
        
        # Distortion coefficients
        for i in range(4):
            key = f'D_{i:02d}'
            if key in calib_data:
                matrices[key] = np.array(calib_data[key].split(), dtype=np.float32)
        
        # Rectification matrices
        for i in range(4):
            key = f'R_{i:02d}'
            if key in calib_data:
                matrices[key] = np.array(calib_data[key].split(), dtype=np.float32).reshape(3, 3)
        
        # Projection matrices
        for i in range(4):
            key = f'P_{i:02d}'
            if key in calib_data:
                matrices[key] = np.array(calib_data[key].split(), dtype=np.float32).reshape(3, 4)
        
        # LiDAR calibration
        if 'Tr_velo_to_cam' in calib_data:
            matrices['Tr_velo_to_cam'] = np.array(calib_data['Tr_velo_to_cam'].split(), dtype=np.float32).reshape(3, 4)
        
        return matrices
    
    def rectify_image(self, image: np.ndarray, K: np.ndarray, D: np.ndarray, R: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Rectify raw image using calibration parameters"""
        # Convert to uint8 for OpenCV
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Rectify image
        map1, map2 = cv2.initUndistortRectifyMap(K, D, R, P, self.image_size, cv2.CV_32FC1)
        rectified = cv2.remap(image_uint8, map1, map2, cv2.INTER_LINEAR)
        
        # Convert back to float
        rectified = rectified.astype(np.float32) / 255.0
        
        return rectified
    
    def project_lidar_to_image(self, points: np.ndarray, calib_matrices: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Project LiDAR points to rectified image coordinates"""
        # Use camera 2 (left color camera) calibration
        K = calib_matrices['K_02']  # Raw camera matrix
        D = calib_matrices['D_02']  # Distortion coefficients
        R = calib_matrices['R_02']  # Rectification matrix
        P = calib_matrices['P_02']  # Projection matrix
        Tr_velo_to_cam = calib_matrices['Tr_velo_to_cam']
        
        # Transform LiDAR points to camera coordinates
        points_homo = np.column_stack([points[:, :3], np.ones(len(points))])
        points_cam = Tr_velo_to_cam @ points_homo.T
        points_cam = points_cam.T
        
        # Apply rectification
        points_cam_rect = R @ points_cam.T
        points_cam_rect = points_cam_rect.T
        
        # Add homogeneous coordinate for projection
        points_cam_homo = np.column_stack([points_cam_rect, np.ones(len(points_cam_rect))])
        
        # Project to image plane
        points_img = P @ points_cam_homo.T
        points_img = points_img.T
        
        # Normalize homogeneous coordinates
        u = points_img[:, 0] / points_img[:, 2]
        v = points_img[:, 1] / points_img[:, 2]
        depth = points_img[:, 2]
        
        # Filter points within image bounds
        valid = (u >= 0) & (u < self.image_size[1]) & (v >= 0) & (v < self.image_size[0]) & (depth > 0)
        
        return np.column_stack([u[valid], v[valid], depth[valid]]), valid
    
    def create_depth_image(self, projected_points: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
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
    
    def process_sequence(self, sequence_path: str, output_path: str, max_frames: Optional[int] = None):
        """Process entire raw KITTI sequence"""
        print(f"🔄 Processing sequence: {sequence_path}")
        
        # Load sequence
        sequence = self.load_raw_sequence(sequence_path)
        
        # Load calibration
        calib_matrices = self.load_raw_calibration(sequence.calibration)
        
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process frames
        num_frames = min(len(sequence.images), len(sequence.lidar))
        if max_frames:
            num_frames = min(num_frames, max_frames)
        
        print(f"📊 Processing {num_frames} frames...")
        
        for i in range(num_frames):
            try:
                # Process image
                image = self.process_raw_image(sequence.images[i])
                
                # Rectify image
                K = calib_matrices['K_02']
                D = calib_matrices['D_02']
                R = calib_matrices['R_02']
                P = calib_matrices['P_02']
                
                rectified_image = self.rectify_image(image, K, D, R, P)
                
                # Process LiDAR
                points = self.process_raw_lidar(sequence.lidar[i])
                
                # Project to image
                projected_points, valid = self.project_lidar_to_image(points, calib_matrices)
                
                # Create depth image
                depth_image = self.create_depth_image(projected_points, self.image_size)
                
                # Normalize depth
                if self.config['preprocessing']['normalize_depth']:
                    depth_image = depth_image / self.depth_max
                
                # Create fused input
                fused_input = np.concatenate([rectified_image, depth_image[..., np.newaxis]], axis=2)
                
                # Save processed data
                frame_id = f"{i:06d}"
                
                # Save rectified image
                cv2.imwrite(str(output_dir / f"{frame_id}_image.png"), 
                           (rectified_image * 255).astype(np.uint8))
                
                # Save depth image
                np.save(str(output_dir / f"{frame_id}_depth.npy"), depth_image)
                
                # Save fused input
                np.save(str(output_dir / f"{frame_id}_fused.npy"), fused_input)
                
                if i % 10 == 0:
                    print(f"  ✅ Processed frame {i+1}/{num_frames}")
                    
            except Exception as e:
                print(f"  ❌ Error processing frame {i}: {e}")
                continue
        
        print(f"✅ Sequence processing complete! {num_frames} frames processed.")
        return num_frames


def main():
    """Example usage"""
    # Load configuration
    with open('configs/preprocessing.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize processor
    processor = RawKITTIProcessor(config)
    
    # Example: Process a raw sequence
    sequence_path = "data/raw/kitti/downloads/2011_09_26_drive_0001"
    output_path = "data/processed/2011_09_26_drive_0001"
    
    try:
        frames_processed = processor.process_sequence(sequence_path, output_path, max_frames=50)
        print(f"🎉 Successfully processed {frames_processed} frames!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have raw KITTI data downloaded in the correct format.")


if __name__ == "__main__":
    main()
