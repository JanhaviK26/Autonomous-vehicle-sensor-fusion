# Mini-KITTI Sensor Fusion Project

A comprehensive implementation of sensor fusion for autonomous vehicles, combining LiDAR point clouds and camera images to predict depth maps and drivable areas.

## 🎯 Project Overview

This project demonstrates how autonomous vehicles fuse multi-modal sensor data (LiDAR + Camera) to enhance perception capabilities. We implement:

- **Depth Prediction**: Converting LiDAR point clouds to depth images and training CNN models
- **Drivable Area Segmentation**: Semantic segmentation of road surfaces using fused sensor data
- **MLOps Pipeline**: Complete data versioning, model tracking, and deployment pipeline

## 🏗️ Project Structure

```
├── data/                   # Dataset storage and preprocessing
│   ├── raw/               # Original KITTI data
│   ├── processed/         # Preprocessed images and depth maps
│   └── calibration/       # Camera-LiDAR calibration files
├── models/                # Model definitions and weights
│   ├── depth_prediction/  # Depth estimation models
│   └── segmentation/      # Drivable area segmentation models
├── notebooks/             # Jupyter notebooks for exploration
├── src/                   # Source code
│   ├── data/             # Data loading and preprocessing
│   ├── models/           # Model architectures
│   ├── training/         # Training scripts
│   ├── evaluation/       # Evaluation metrics and visualization
│   └── utils/            # Utility functions
├── configs/              # Configuration files
├── experiments/          # MLflow experiment tracking
├── dashboard/           # Streamlit dashboard
└── tests/               # Unit tests
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd autonomous-vehicle-sensor-fusion

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Download KITTI dataset (you'll need to provide the path)
python src/data/download_kitti.py --data_path /path/to/kitti

# Preprocess LiDAR and camera data
python src/data/preprocess.py --config configs/preprocessing.yaml
```

### 3. Training

```bash
# Train depth prediction model
python src/training/train.py --config configs/depth_model.yaml --model_type depth

# Train segmentation model
python src/training/train.py --config configs/segmentation_model.yaml --model_type segmentation
```

### 4. Evaluation

```bash
# Evaluate models
python src/evaluation/evaluate.py --config configs/depth_model.yaml --model models/depth_prediction/best_checkpoint.pth --model_type depth
```

### 5. Dashboard

```bash
# Launch Streamlit dashboard
streamlit run dashboard/app.py
```

## 📊 Key Features

### Data Processing
- LiDAR point cloud to depth image conversion
- Camera-LiDAR calibration and alignment
- Multi-modal data fusion techniques

### Model Architectures
- **U-Net**: Encoder-decoder for depth prediction
- **DeepLabV3+**: Advanced segmentation architecture
- **Fusion Networks**: RGB + LiDAR depth channel fusion

### MLOps Pipeline
- **DVC**: Data versioning and pipeline management
- **MLflow**: Experiment tracking and model registry
- **Automated Training**: Reproducible training pipelines

### Evaluation Metrics
- **Depth**: RMSE, MAE, δ accuracy
- **Segmentation**: IoU, F1-score, Pixel Accuracy
- **Visualization**: Side-by-side comparisons, error heatmaps

## 🔬 Experiments

### Depth Prediction Results
| Model | RMSE ↓ | MAE ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ |
|-------|--------|-------|------|------|------|
| RGB Only | 4.23 | 2.89 | 0.78 | 0.92 | 0.97 |
| RGB + LiDAR | 3.45 | 2.12 | 0.85 | 0.95 | 0.98 |

### Segmentation Results
| Model | IoU ↑ | F1 ↑ | Pixel Acc ↑ |
|-------|-------|------|-------------|
| RGB Only | 0.72 | 0.84 | 0.89 |
| RGB + LiDAR | 0.81 | 0.90 | 0.94 |


