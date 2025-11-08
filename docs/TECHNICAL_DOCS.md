# Mini-KITTI Sensor Fusion Project - Technical Documentation

## Table of Contents
1. [Project Architecture](#project-architecture)
2. [Data Processing Pipeline](#data-processing-pipeline)
3. [Model Architectures](#model-architectures)
4. [Training Pipeline](#training-pipeline)
5. [Evaluation Framework](#evaluation-framework)
6. [MLOps Infrastructure](#mlops-infrastructure)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)

## Project Architecture

### Overview
The project implements a complete sensor fusion pipeline for autonomous vehicles, combining LiDAR point clouds and camera images to enhance perception capabilities.

### Core Components

#### 1. Data Processing (`src/data/`)
- **`processing.py`**: Core LiDAR and camera data processing
- **`dataset.py`**: PyTorch dataset classes for training
- **`preprocessing.py`**: Data preprocessing utilities

#### 2. Model Architectures (`src/models/`)
- **`architectures.py`**: Neural network architectures
- **`losses.py`**: Custom loss functions and metrics

#### 3. Training (`src/training/`)
- **`train.py`**: Training pipeline with MLflow integration

#### 4. Evaluation (`src/evaluation/`)
- **`evaluate.py`**: Model evaluation and visualization

#### 5. Utilities (`src/utils/`)
- **`mlflow_utils.py`**: MLflow experiment management
- **`setup.py`**: Project setup utilities

## Data Processing Pipeline

### LiDAR Processing
```python
class LidarProcessor:
    def load_lidar_points(self, bin_path: str) -> np.ndarray
    def filter_points(self, points: np.ndarray) -> np.ndarray
    def project_to_image(self, points: np.ndarray, calib: CalibrationData) -> Tuple[np.ndarray, np.ndarray]
    def create_depth_image(self, projected_points: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray
```

### Camera Processing
```python
class CameraProcessor:
    def load_image(self, image_path: str) -> np.ndarray
    def resize_image(self, image: np.ndarray) -> np.ndarray
    def normalize_image(self, image: np.ndarray) -> np.ndarray
```

### Data Fusion
```python
class DataFusion:
    def load_calibration(self, calib_path: str) -> CalibrationData
    def fuse_data(self, image_path: str, lidar_path: str, calib_path: str) -> Dict[str, np.ndarray]
```

## Model Architectures

### U-Net for Depth Prediction
```python
class UNetDepth(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        # Encoder blocks
        self.enc1 = self._make_encoder_block(4, 64)  # RGB + Depth input
        self.enc2 = self._make_encoder_block(64, 128, stride=2)
        self.enc3 = self._make_encoder_block(128, 256, stride=2)
        self.enc4 = self._make_encoder_block(256, 512, stride=2)
        
        # Bottleneck
        self.bottleneck = self._make_encoder_block(512, 1024, stride=2)
        
        # Decoder with skip connections
        self.dec4 = self._make_decoder_block(1024, 512)
        self.dec3 = self._make_decoder_block(512, 256)
        self.dec2 = self._make_decoder_block(256, 128)
        self.dec1 = self._make_decoder_block(128, 64)
        
        # Final prediction
        self.final_conv = nn.Conv2d(64, 1, 1)
```

### DeepLabV3+ for Segmentation
```python
class DeepLabV3Plus(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        # Modified ResNet backbone for 4-channel input
        self.backbone = self._get_backbone()
        self._modify_first_layer()
        
        # ASPP module
        self.aspp = ASPP(2048, 256)
        
        # Decoder
        self.decoder = Decoder(256, 48, num_classes)
```

## Training Pipeline

### Trainer Base Class
```python
class Trainer:
    def __init__(self, config: Dict[str, Any], model_name: str):
        self.setup()
    
    def setup(self):
        # Initialize model, optimizer, scheduler, data loaders
        pass
    
    def train_epoch(self) -> float:
        # Training loop with mixed precision
        pass
    
    def validate_epoch(self) -> tuple:
        # Validation loop
        pass
    
    def train(self):
        # Main training loop with MLflow logging
        pass
```

### Specialized Trainers
- **`DepthTrainer`**: For depth prediction models
- **`SegmentationTrainer`**: For segmentation models

## Evaluation Framework

### Metrics
#### Depth Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **δ Accuracy**: Percentage of pixels with max(pred/target, target/pred) < threshold

#### Segmentation Metrics
- **IoU**: Intersection over Union
- **F1-Score**: Harmonic mean of precision and recall
- **Pixel Accuracy**: Percentage of correctly classified pixels

### Visualization
- Side-by-side comparisons
- Error heatmaps
- Class distribution plots
- Model comparison charts

## MLOps Infrastructure

### DVC Pipeline
```yaml
stages:
  preprocess:
    cmd: python src/data/preprocess.py --config configs/preprocessing.yaml
    deps: [src/data/preprocess.py, configs/preprocessing.yaml]
    outs: [data/processed/train, data/processed/val, data/processed/test]
    metrics: [metrics/preprocessing.json]
  
  train_depth:
    cmd: python src/training/train.py --config configs/depth_model.yaml --model_type depth
    deps: [src/training/train.py, configs/depth_model.yaml]
    outs: [models/depth_prediction/best_checkpoint.pth]
    metrics: [metrics/depth_training.json]
```

### MLflow Integration
```python
class MLflowManager:
    def setup_experiments(self):
        # Create MLflow experiments
    
    def log_training_metrics(self, metrics: Dict[str, Any], config: Dict[str, Any]):
        # Log training metrics and model
    
    def log_evaluation_metrics(self, metrics: Dict[str, Any]):
        # Log evaluation results
```

## API Reference

### Data Processing Functions

#### `load_lidar_points(bin_path: str) -> np.ndarray`
Load LiDAR points from .bin file.
- **Parameters**: `bin_path` - Path to .bin file
- **Returns**: Array of shape (N, 4) with [x, y, z, intensity]

#### `project_to_image(points: np.ndarray, calib: CalibrationData) -> Tuple[np.ndarray, np.ndarray]`
Project 3D points to 2D image coordinates.
- **Parameters**: 
  - `points` - 3D points array
  - `calib` - Calibration data
- **Returns**: Tuple of (projected_points, valid_mask)

### Model Functions

#### `create_model(config: Dict[str, Any]) -> nn.Module`
Create model based on configuration.
- **Parameters**: `config` - Model configuration dictionary
- **Returns**: PyTorch model instance

#### `create_loss_function(config: Dict[str, Any]) -> nn.Module`
Create loss function based on configuration.
- **Parameters**: `config` - Loss configuration dictionary
- **Returns**: PyTorch loss function

### Training Functions

#### `train_depth_model(config_path: str)`
Train depth prediction model.
- **Parameters**: `config_path` - Path to configuration file

#### `train_segmentation_model(config_path: str)`
Train segmentation model.
- **Parameters**: `config_path` - Path to configuration file

### Evaluation Functions

#### `evaluate_model(config_path: str, model_path: str, model_type: str) -> Dict[str, float]`
Evaluate a trained model.
- **Parameters**:
  - `config_path` - Path to configuration file
  - `model_path` - Path to model checkpoint
  - `model_type` - Type of model ('depth' or 'segmentation')
- **Returns**: Dictionary of evaluation metrics

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size in config
batch_size: 8  # Instead of 16

# Use gradient accumulation
gradient_accumulation_steps: 2
```

#### 2. Data Loading Errors
```bash
# Check file paths in configuration
# Ensure KITTI dataset is properly downloaded
# Verify calibration files exist
```

#### 3. Model Convergence Issues
```bash
# Adjust learning rate
learning_rate: 1e-5  # Lower learning rate

# Use different scheduler
scheduler: "cosine"  # Instead of "step"

# Increase training epochs
epochs: 200  # More training time
```

#### 4. MLflow Connection Issues
```bash
# Check MLflow server status
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Verify experiment exists
mlflow experiments list
```

### Performance Optimization

#### 1. Mixed Precision Training
```python
# Enable in config
use_amp: true

# Use appropriate loss scaling
scaler = torch.cuda.amp.GradScaler()
```

#### 2. Data Loading Optimization
```python
# Increase number of workers
num_workers: 8

# Enable pin memory
pin_memory: true

# Use persistent workers
persistent_workers: true
```

#### 3. Model Optimization
```python
# Use efficient architectures
backbone: "resnet50"  # Instead of "resnet101"

# Reduce model complexity
decoder_channels: [128, 64, 32]  # Smaller decoder
```

### Debugging Tips

#### 1. Enable Debug Mode
```python
# Add to config
debug: true
save_debug_images: true
```

#### 2. Monitor Training
```python
# Use MLflow UI
mlflow ui

# Check tensorboard logs
tensorboard --logdir experiments/
```

#### 3. Validate Data Pipeline
```python
# Test data loading
python tests/test_data_processing.py

# Test model creation
python tests/test_model_creation.py
```

## Contributing Guidelines

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Add docstrings to all functions
- Use meaningful variable names

### Testing
- Write unit tests for new functions
- Test with different configurations
- Validate on small datasets first

### Documentation
- Update README for new features
- Add examples for new functions
- Document configuration options

### Pull Request Process
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Submit pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
