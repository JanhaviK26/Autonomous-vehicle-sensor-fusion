#!/usr/bin/env python3
"""
Simple Project Setup Script for Mini-KITTI Sensor Fusion
"""

import os
from pathlib import Path


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
    gitignore_content = """# Python
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
    
    print("=" * 60)
    print("✅ Project initialization complete!")
    print("")
    print("Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download KITTI dataset")
    print("3. Update configuration files")
    print("4. Start training!")
    print("")
    print("For more information, see:")
    print("- README.md")
    print("- docs/TECHNICAL_DOCS.md")


if __name__ == "__main__":
    main()
