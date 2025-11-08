"""
Utility Functions for Sensor Fusion Project

This module contains various utility functions for data processing,
visualization, and project management.
"""

import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import torch
import cv2
from datetime import datetime
import pandas as pd


def create_directory_structure(base_path: str = "."):
    """Create the complete directory structure for the project"""
    directories = [
        "data/raw/kitti",
        "data/processed/train",
        "data/processed/val", 
        "data/processed/test",
        "data/calibration",
        "models/depth_prediction",
        "models/segmentation",
        "notebooks",
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/utils",
        "configs",
        "experiments",
        "metrics",
        "plots",
        "reports",
        "dashboard",
        "tests"
    ]
    
    for directory in directories:
        Path(base_path, directory).mkdir(parents=True, exist_ok=True)
    
    print("Directory structure created successfully!")


def save_config_template():
    """Save configuration templates"""
    # Depth model config template
    depth_config = {
        'model': {
            'name': 'unet_depth',
            'architecture': {
                'encoder': 'resnet34',
                'decoder_channels': [256, 128, 64, 32],
                'input_channels': 4,
                'output_channels': 1
            },
            'loss': {
                'type': 'berhu',
                'weight': 1.0
            },
            'dropout': 0.1,
            'weight_decay': 1e-4
        },
        'training': {
            'batch_size': 16,
            'epochs': 100,
            'learning_rate': 1e-4,
            'scheduler': 'cosine',
            'patience': 15,
            'min_delta': 0.001,
            'use_amp': True,
            'save_best': True,
            'save_last': True,
            'checkpoint_dir': 'models/depth_prediction'
        },
        'data': {
            'train_path': 'data/processed/train',
            'val_path': 'data/processed/val',
            'test_path': 'data/processed/test',
            'num_workers': 4,
            'pin_memory': True
        },
        'evaluation': {
            'metrics': ['rmse', 'mae', 'delta1', 'delta2', 'delta3'],
            'save_predictions': True,
            'num_samples': 50
        }
    }
    
    with open('configs/depth_model_template.yaml', 'w') as f:
        yaml.dump(depth_config, f, default_flow_style=False)
    
    # Segmentation model config template
    seg_config = {
        'model': {
            'name': 'deeplabv3plus',
            'architecture': {
                'backbone': 'resnet50',
                'num_classes': 2,
                'input_channels': 4
            },
            'loss': {
                'type': 'focal',
                'alpha': 0.25,
                'gamma': 2.0,
                'weight': 1.0
            },
            'dropout': 0.1,
            'weight_decay': 1e-4
        },
        'training': {
            'batch_size': 8,
            'epochs': 150,
            'learning_rate': 1e-4,
            'scheduler': 'poly',
            'patience': 20,
            'min_delta': 0.001,
            'use_amp': True,
            'save_best': True,
            'save_last': True,
            'checkpoint_dir': 'models/segmentation'
        },
        'data': {
            'train_path': 'data/processed/train',
            'val_path': 'data/processed/val',
            'test_path': 'data/processed/test',
            'num_workers': 4,
            'pin_memory': True
        },
        'evaluation': {
            'metrics': ['iou', 'f1', 'pixel_accuracy', 'dice'],
            'save_predictions': True,
            'num_samples': 50,
            'class_names': ['Background', 'Drivable']
        }
    }
    
    with open('configs/segmentation_model_template.yaml', 'w') as f:
        yaml.dump(seg_config, f, default_flow_style=False)
    
    print("Configuration templates saved!")


def generate_project_summary():
    """Generate a comprehensive project summary"""
    summary = {
        "project_name": "Mini-KITTI Sensor Fusion",
        "description": "Sensor fusion for autonomous vehicles using LiDAR and camera data",
        "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "components": {
            "data_processing": {
                "lidar_processing": "Point cloud to depth image conversion",
                "camera_processing": "RGB image preprocessing",
                "calibration": "Camera-LiDAR alignment",
                "fusion": "Multi-modal data combination"
            },
            "models": {
                "depth_prediction": "U-Net architecture for depth estimation",
                "segmentation": "DeepLabV3+ for drivable area detection",
                "fusion_net": "Custom RGB+LiDAR fusion network"
            },
            "training": {
                "framework": "PyTorch",
                "loss_functions": ["BerHu", "Focal", "Dice", "Combined"],
                "optimizers": ["Adam"],
                "schedulers": ["Cosine", "Poly", "Step"]
            },
            "evaluation": {
                "depth_metrics": ["RMSE", "MAE", "δ1", "δ2", "δ3"],
                "segmentation_metrics": ["IoU", "F1", "Pixel Accuracy", "Dice"]
            },
            "mlops": {
                "experiment_tracking": "MLflow",
                "data_versioning": "DVC",
                "model_registry": "MLflow",
                "visualization": "Streamlit"
            }
        },
        "file_structure": {
            "data": "Raw and processed KITTI dataset",
            "models": "Model definitions and trained weights",
            "src": "Source code modules",
            "configs": "Configuration files",
            "experiments": "MLflow experiment tracking",
            "dashboard": "Streamlit web application"
        }
    }
    
    with open('project_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Project summary generated!")
    return summary


def create_sample_data_info():
    """Create sample data information file"""
    sample_info = {
        "kitti_dataset": {
            "description": "KITTI autonomous driving dataset",
            "components": {
                "images": "Left color camera images (PNG format)",
                "lidar": "Velodyne LiDAR point clouds (BIN format)",
                "calibration": "Camera-LiDAR calibration files (TXT format)"
            },
            "structure": {
                "training": "Training data split",
                "testing": "Testing data split"
            },
            "download": "http://www.cvlibs.net/datasets/kitti/",
            "license": "Creative Commons Attribution-NonCommercial-ShareAlike 3.0"
        },
        "preprocessing": {
            "lidar_to_depth": "Convert point clouds to depth images",
            "image_resize": "Resize images to 256x256",
            "calibration": "Align camera and LiDAR coordinate systems",
            "normalization": "Normalize depth and RGB values"
        },
        "sample_files": {
            "image": "000000.png (example camera image)",
            "lidar": "000000.bin (example LiDAR data)",
            "calib": "000000.txt (example calibration file)"
        }
    }
    
    with open('data/sample_data_info.json', 'w') as f:
        json.dump(sample_info, f, indent=2)
    
    print("Sample data information created!")


def create_test_scripts():
    """Create test scripts for the project"""
    
    # Test data processing
    test_data_script = '''
"""
Test script for data processing functionality
"""
import sys
sys.path.append('src')
from data.processing import DataFusion
import yaml

def test_data_fusion():
    """Test data fusion functionality"""
    try:
        # Load config
        with open('configs/preprocessing.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize fusion
        fusion = DataFusion(config)
        print("✅ DataFusion initialized successfully")
        
        # Test with dummy data (in real scenario, use actual KITTI files)
        print("ℹ️  To test with real data, provide KITTI file paths")
        
    except Exception as e:
        print(f"❌ Error testing data fusion: {e}")

if __name__ == "__main__":
    test_data_fusion()
'''
    
    with open('tests/test_data_processing.py', 'w') as f:
        f.write(test_data_script)
    
    # Test model creation
    test_model_script = '''
"""
Test script for model creation
"""
import sys
sys.path.append('src')
from models.architectures import create_model
import yaml
import torch

def test_model_creation():
    """Test model creation functionality"""
    try:
        # Test depth model
        with open('configs/depth_model.yaml', 'r') as f:
            depth_config = yaml.safe_load(f)
        
        depth_model = create_model(depth_config)
        print("✅ Depth model created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(1, 4, 256, 256)
        with torch.no_grad():
            output = depth_model(dummy_input)
        print(f"✅ Depth model forward pass successful, output shape: {output.shape}")
        
        # Test segmentation model
        with open('configs/segmentation_model.yaml', 'r') as f:
            seg_config = yaml.safe_load(f)
        
        seg_model = create_model(seg_config)
        print("✅ Segmentation model created successfully")
        
        # Test forward pass
        with torch.no_grad():
            output = seg_model(dummy_input)
        print(f"✅ Segmentation model forward pass successful, output shape: {output.shape}")
        
    except Exception as e:
        print(f"❌ Error testing model creation: {e}")

if __name__ == "__main__":
    test_model_creation()
'''
    
    with open('tests/test_model_creation.py', 'w') as f:
        f.write(test_model_script)
    
    print("Test scripts created!")


def create_notebook_templates():
    """Create Jupyter notebook templates"""
    
    # Data exploration notebook
    exploration_notebook = '''
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# KITTI Dataset Exploration\\n",
    "\\n",
    "This notebook explores the KITTI dataset structure and visualizes sensor data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\\n",
    "sys.path.append('src')\\n",
    "\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import cv2\\n",
    "from data.processing import DataFusion\\n",
    "import yaml\\n",
    "\\n",
    "# Load configuration\\n",
    "with open('configs/preprocessing.yaml', 'r') as f:\\n",
    "    config = yaml.safe_load(f)\\n",
    "\\n",
    "# Initialize data fusion\\n",
    "fusion = DataFusion(config)\\n",
    "print('Data fusion initialized successfully!')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Load and Visualize Sample Data\\n",
    "\\n",
    "Load a sample from the KITTI dataset and visualize the sensor data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Example paths (replace with actual KITTI paths)\\n",
    "image_path = 'data/raw/kitti/training/image_2/000000.png'\\n",
    "lidar_path = 'data/raw/kitti/training/velodyne/000000.bin'\\n",
    "calib_path = 'data/raw/kitti/training/calib/000000.txt'\\n",
    "\\n",
    "# Fuse data\\n",
    "try:\\n",
    "    fused_data = fusion.fuse_data(image_path, lidar_path, calib_path)\\n",
    "    \\n",
    "    # Visualize results\\n",
    "    fig, axes = plt.subplots(1, 3, figsize=(15, 5))\\n",
    "    \\n",
    "    axes[0].imshow(fused_data['image'])\\n",
    "    axes[0].set_title('RGB Image')\\n",
    "    axes[0].axis('off')\\n",
    "    \\n",
    "    im = axes[1].imshow(fused_data['depth'], cmap='viridis')\\n",
    "    axes[1].set_title('Depth Map')\\n",
    "    axes[1].axis('off')\\n",
    "    plt.colorbar(im, ax=axes[1])\\n",
    "    \\n",
    "    axes[2].imshow(fused_data['fused_input'][:, :, 3], cmap='viridis')\\n",
    "    axes[2].set_title('Fused Input (Depth Channel)')\\n",
    "    axes[2].axis('off')\\n",
    "    \\n",
    "    plt.tight_layout()\\n",
    "    plt.show()\\n",
    "    \\n",
    "except FileNotFoundError:\\n",
    "    print('KITTI files not found. Please download the dataset and update paths.')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
'''
    
    with open('notebooks/data_exploration.ipynb', 'w') as f:
        f.write(exploration_notebook)
    
    print("Notebook templates created!")


def main():
    """Main function to set up the project"""
    print("🚀 Setting up Mini-KITTI Sensor Fusion Project...")
    
    # Create directory structure
    create_directory_structure()
    
    # Save configuration templates
    save_config_template()
    
    # Generate project summary
    generate_project_summary()
    
    # Create sample data info
    create_sample_data_info()
    
    # Create test scripts
    create_test_scripts()
    
    # Create notebook templates
    create_notebook_templates()
    
    print("\\n✅ Project setup complete!")
    print("\\nNext steps:")
    print("1. Download KITTI dataset")
    print("2. Update configuration files")
    print("3. Run preprocessing: python src/data/preprocess.py")
    print("4. Train models: python src/training/train.py")
    print("5. Launch dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
