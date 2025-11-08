#!/usr/bin/env python3
"""
KITTI Dataset Organization Helper

This script helps organize downloaded KITTI dataset files into the correct structure
for the sensor fusion project.
"""

import os
import shutil
import zipfile
from pathlib import Path
import argparse


def create_kitti_structure(base_path: str = "data/raw/kitti"):
    """Create the correct KITTI directory structure"""
    directories = [
        "training/image_2",
        "training/velodyne", 
        "training/calib",
        "testing/image_2",
        "testing/velodyne",
        "testing/calib"
    ]
    
    print("📁 Creating KITTI directory structure...")
    for directory in directories:
        Path(base_path, directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")
    
    print("✅ Directory structure created!")


def organize_kitti_data(extracted_path: str, target_path: str = "data/raw/kitti"):
    """Organize extracted KITTI data into correct structure"""
    
    print("🔄 Organizing KITTI data...")
    
    # Find all image files
    image_files = list(Path(extracted_path).rglob("*.png"))
    print(f"  📸 Found {len(image_files)} image files")
    
    # Find all LiDAR files  
    lidar_files = list(Path(extracted_path).rglob("*.bin"))
    print(f"  🔍 Found {len(lidar_files)} LiDAR files")
    
    # Find all calibration files
    calib_files = list(Path(extracted_path).rglob("*.txt"))
    print(f"  📐 Found {len(calib_files)} calibration files")
    
    # Copy files to correct locations
    for img_file in image_files:
        if "image_2" in str(img_file):
            target_file = Path(target_path) / "training" / "image_2" / img_file.name
            shutil.copy2(img_file, target_file)
    
    for lidar_file in lidar_files:
        if "velodyne" in str(lidar_file):
            target_file = Path(target_path) / "training" / "velodyne" / lidar_file.name
            shutil.copy2(lidar_file, target_file)
    
    for calib_file in calib_files:
        if "calib" in str(calib_file):
            target_file = Path(target_path) / "training" / "calib" / calib_file.name
            shutil.copy2(calib_file, target_file)
    
    print("✅ Data organization complete!")


def verify_kitti_structure(base_path: str = "data/raw/kitti"):
    """Verify that KITTI data is properly organized"""
    
    print("🔍 Verifying KITTI data structure...")
    
    required_dirs = [
        "training/image_2",
        "training/velodyne", 
        "training/calib"
    ]
    
    all_good = True
    
    for directory in required_dirs:
        dir_path = Path(base_path) / directory
        if dir_path.exists():
            file_count = len(list(dir_path.glob("*")))
            print(f"  ✅ {directory}: {file_count} files")
            if file_count == 0:
                all_good = False
                print(f"    ⚠️  Warning: {directory} is empty!")
        else:
            print(f"  ❌ {directory}: Directory not found!")
            all_good = False
    
    if all_good:
        print("✅ KITTI data structure is correct!")
    else:
        print("❌ KITTI data structure needs attention.")
        print("\n📋 Required files:")
        print("  - training/image_2/*.png (camera images)")
        print("  - training/velodyne/*.bin (LiDAR point clouds)")
        print("  - training/calib/*.txt (calibration files)")
    
    return all_good


def create_sample_data_info():
    """Create information about expected KITTI data format"""
    
    info = {
        "kitti_dataset_structure": {
            "description": "Expected KITTI dataset structure for sensor fusion",
            "required_files": {
                "images": {
                    "path": "training/image_2/",
                    "format": "PNG",
                    "description": "Left color camera images",
                    "example": "000000.png, 000001.png, ..."
                },
                "lidar": {
                    "path": "training/velodyne/", 
                    "format": "BIN",
                    "description": "Velodyne LiDAR point clouds",
                    "example": "000000.bin, 000001.bin, ..."
                },
                "calibration": {
                    "path": "training/calib/",
                    "format": "TXT", 
                    "description": "Camera-LiDAR calibration parameters",
                    "example": "000000.txt, 000001.txt, ..."
                }
            },
            "file_correspondence": {
                "description": "Files with same number correspond to same scene",
                "example": "000000.png + 000000.bin + 000000.txt = one scene"
            }
        },
        "download_instructions": {
            "website": "http://www.cvlibs.net/datasets/kitti/",
            "required_downloads": [
                "data_object_image_2.zip (training images)",
                "data_object_velodyne.zip (training LiDAR)", 
                "data_object_calib.zip (training calibration)"
            ],
            "extraction": "Extract all zip files and run organize_kitti_data()"
        }
    }
    
    import json
    with open('data/kitti_data_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print("📋 Created KITTI data information file: data/kitti_data_info.json")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Organize KITTI dataset')
    parser.add_argument('--action', choices=['create', 'organize', 'verify', 'info'], 
                       default='verify', help='Action to perform')
    parser.add_argument('--extracted_path', type=str, 
                       help='Path to extracted KITTI data')
    parser.add_argument('--target_path', type=str, default='data/raw/kitti',
                       help='Target path for organized data')
    
    args = parser.parse_args()
    
    if args.action == 'create':
        create_kitti_structure(args.target_path)
    elif args.action == 'organize':
        if not args.extracted_path:
            print("❌ Error: --extracted_path required for organize action")
            return
        organize_kitti_data(args.extracted_path, args.target_path)
    elif args.action == 'verify':
        verify_kitti_structure(args.target_path)
    elif args.action == 'info':
        create_sample_data_info()


if __name__ == "__main__":
    main()
