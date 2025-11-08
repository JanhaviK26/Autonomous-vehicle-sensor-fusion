#!/usr/bin/env python3
"""
Project Setup Script for Mini-KITTI Sensor Fusion

This script initializes the complete project structure and creates
all necessary configuration files and directories.
"""

import os
import sys
from pathlib import Path
import subprocess
import yaml
import json


def create_directories():
    """Create all necessary directories"""
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
        "tests",
        "docs"
    ]
    
    print("📁 Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("✅ Directory structure created!")


def create_init_files():
    """Create __init__.py files for Python packages"""
    init_files = [
        "src/__init__.py",
        "src/data/__init__.py",
        "src/models/__init__.py",
        "src/training/__init__.py",
        "src/evaluation/__init__.py",
        "src/utils/__init__.py"
    ]
    
    print("🐍 Creating Python package files...")
    for init_file in init_files:
        Path(init_file).touch()
    print("✅ Python packages initialized!")


def create_gitignore():
    """Create .gitignore file"""
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyTorch
*.pth
*.pt

# Jupyter Notebook
.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Data
data/raw/
*.bin
*.png
*.jpg
*.jpeg

# Models
models/*.pth
models/*.pt

# MLflow
mlruns/
mlflow.db

# DVC
.dvc/
.dvcignore

# Logs
*.log
logs/

# Temporary files
*.tmp
*.temp
"""
    
    print("📝 Creating .gitignore...")
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    print("✅ .gitignore created!")


def create_setup_script():
    """Create setup script for easy project initialization"""
    setup_script = '''#!/bin/bash
# Setup script for Mini-KITTI Sensor Fusion Project

echo "🚀 Setting up Mini-KITTI Sensor Fusion Project..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Initialize MLflow
echo "🔬 Setting up MLflow..."
python src/utils/mlflow_utils.py

# Initialize DVC (optional)
echo "📊 Setting up DVC..."
dvc init

# Create sample data info
echo "📋 Creating sample data information..."
python -c "from src.utils.setup import create_sample_data_info; create_sample_data_info()"

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download KITTI dataset and place in data/raw/kitti/"
echo "2. Update configuration files in configs/"
echo "3. Run preprocessing: python src/data/preprocess.py"
echo "4. Train models: python src/training/train.py"
echo "5. Launch dashboard: streamlit run dashboard/app.py"
'''
    
    print("🔧 Creating setup script...")
    with open('setup.sh', 'w') as f:
        f.write(setup_script)
    
    # Make executable
    os.chmod('setup.sh', 0o755)
    print("✅ Setup script created!")


def create_quick_start_guide():
    """Create quick start guide"""
    guide_content = """# Quick Start Guide

## 1. Environment Setup
```bash
# Run the setup script
./setup.sh

# Or manually:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Download KITTI Dataset
1. Go to http://www.cvlibs.net/datasets/kitti/
2. Download the following:
   - Left color images (PNG format)
   - Velodyne point clouds (BIN format)
   - Calibration files (TXT format)
3. Place files in `data/raw/kitti/` following KITTI structure

## 3. Configure Project
Edit configuration files in `configs/`:
- `preprocessing.yaml` - Data preprocessing settings
- `depth_model.yaml` - Depth prediction model settings
- `segmentation_model.yaml` - Segmentation model settings

## 4. Run Preprocessing
```bash
python src/data/preprocess.py --config configs/preprocessing.yaml
```

## 5. Train Models
```bash
# Train depth prediction model
python src/training/train.py --config configs/depth_model.yaml --model_type depth

# Train segmentation model
python src/training/train.py --config configs/segmentation_model.yaml --model_type segmentation
```

## 6. Evaluate Models
```bash
python src/evaluation/evaluate.py --config configs/depth_model.yaml --model models/depth_prediction/best_checkpoint.pth --model_type depth
```

## 7. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

## 8. View Experiments
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Troubleshooting

### Common Issues:
1. **CUDA out of memory**: Reduce batch size in config files
2. **File not found**: Check KITTI dataset paths
3. **Import errors**: Ensure virtual environment is activated
4. **MLflow errors**: Check if MLflow server is running

### Getting Help:
- Check the technical documentation in `docs/`
- Look at example notebooks in `notebooks/`
- Run test scripts in `tests/`
- Open an issue on GitHub
"""
    
    print("📖 Creating quick start guide...")
    with open('QUICKSTART.md', 'w') as f:
        f.write(guide_content)
    print("✅ Quick start guide created!")


def create_project_info():
    """Create project information file"""
    project_info = {
        "name": "Mini-KITTI Sensor Fusion",
        "version": "1.0.0",
        "description": "Sensor fusion for autonomous vehicles using LiDAR and camera data",
        "author": "Your Name",
        "email": "your.email@example.com",
        "license": "MIT",
        "created": "2024-01-01",
        "python_version": ">=3.8",
        "dependencies": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "numpy>=1.21.0",
            "opencv-python>=4.5.0",
            "pillow>=8.3.0",
            "scikit-learn>=1.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "pandas>=1.3.0",
            "h5py>=3.1.0",
            "tqdm>=4.62.0",
            "mlflow>=2.0.0",
            "dvc>=2.0.0",
            "streamlit>=1.20.0",
            "plotly>=5.0.0",
            "jupyter>=1.0.0"
        ],
        "features": [
            "LiDAR point cloud processing",
            "Camera image processing",
            "Multi-modal data fusion",
            "Depth prediction models",
            "Semantic segmentation models",
            "MLOps pipeline with DVC and MLflow",
            "Interactive Streamlit dashboard",
            "Comprehensive evaluation metrics"
        ],
        "structure": {
            "data": "Dataset storage and preprocessing",
            "models": "Model definitions and trained weights",
            "src": "Source code modules",
            "configs": "Configuration files",
            "experiments": "MLflow experiment tracking",
            "dashboard": "Streamlit web application",
            "notebooks": "Jupyter notebooks for exploration",
            "tests": "Unit tests",
            "docs": "Documentation"
        }
    }
    
    print("ℹ️ Creating project information...")
    with open('project_info.json', 'w') as f:
        json.dump(project_info, f, indent=2)
    print("✅ Project information created!")


def main():
    """Main setup function"""
    print("🚀 Initializing Mini-KITTI Sensor Fusion Project...")
    print("=" * 60)
    
    # Create directory structure
    create_directories()
    
    # Create Python package files
    create_init_files()
    
    # Create .gitignore
    create_gitignore()
    
    # Create setup script
    create_setup_script()
    
    # Create quick start guide
    create_quick_start_guide()
    
    # Create project info
    create_project_info()
    
    print("=" * 60)
    print("✅ Project initialization complete!")
    print("")
    print("Next steps:")
    print("1. Run: ./setup.sh")
    print("2. Download KITTI dataset")
    print("3. Update configuration files")
    print("4. Start training!")
    print("")
    print("For more information, see:")
    print("- README.md")
    print("- QUICKSTART.md")
    print("- docs/TECHNICAL_DOCS.md")


if __name__ == "__main__":
    main()
