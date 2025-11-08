"""
Training Pipeline with MLflow Integration

This module provides training scripts for depth prediction and segmentation models
with comprehensive logging, checkpointing, and experiment tracking.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import time
from tqdm import tqdm
import numpy as np

from ..data.dataset import create_dataloaders
from ..models.architectures import create_model
from ..models.losses import create_loss_function, DepthMetrics, SegmentationMetrics, EarlyStopping, LearningRateScheduler


class Trainer:
    """Base trainer class with MLflow integration"""
    
    def __init__(self, config: Dict[str, Any], model_name: str):
        self.config = config
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize MLflow
        self.experiment_name = f"{model_name}_{config['model']['name']}"
        mlflow.set_experiment(self.experiment_name)
        
        # Training components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.dataloaders = None
        self.early_stopping = None
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        
    def setup(self):
        """Setup training components"""
        # Create model
        self.model = create_model(self.config).to(self.device)
        
        # Create loss function
        self.criterion = create_loss_function(self.config)
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['model']['weight_decay']
        )
        
        # Create scheduler
        self.scheduler = LearningRateScheduler.get_scheduler(self.optimizer, self.config['training'])
        
        # Create data loaders
        self.dataloaders = create_dataloaders(self.config, self.config['data']['train_path'])
        
        # Create early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['training']['patience'],
            min_delta=self.config['training']['min_delta']
        )
        
        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.config['training']['use_amp'] else None
        
        print(f"Training setup complete!")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training samples: {len(self.dataloaders['train'].dataset)}")
        print(f"Validation samples: {len(self.dataloaders['val'].dataset)}")
    
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.dataloaders['train'])
        
        pbar = tqdm(self.dataloaders['train'], desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            targets = batch[self.get_target_key()].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def validate_epoch(self) -> tuple:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(self.dataloaders['val'], desc="Validation"):
                # Move data to device
                images = batch['image'].to(self.device)
                targets = batch[self.get_target_key()].to(self.device)
                
                # Forward pass
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                
                # Compute metrics
                metrics = self.compute_metrics(outputs, targets)
                all_metrics.append(metrics)
        
        # Average metrics across batches
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return total_loss / len(self.dataloaders['val']), avg_metrics
    
    def compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute evaluation metrics - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_target_key(self) -> str:
        """Get target key from batch - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_primary_metric(self, metrics: Dict[str, float]) -> float:
        """Get primary metric for model selection - to be implemented by subclasses"""
        raise NotImplementedError
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        if self.config['training']['save_last']:
            torch.save(checkpoint, checkpoint_dir / 'last_checkpoint.pth')
        
        # Save best checkpoint
        if is_best and self.config['training']['save_best']:
            torch.save(checkpoint, checkpoint_dir / 'best_checkpoint.pth')
            print(f"New best model saved! Primary metric: {self.get_primary_metric(metrics):.4f}")
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics to MLflow"""
        for key, value in metrics.items():
            mlflow.log_metric(f"{prefix}{key}", value, step=self.current_epoch)
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config['training']['epochs']} epochs...")
        
        with mlflow.start_run():
            # Log hyperparameters
            mlflow.log_params({
                'model_name': self.config['model']['name'],
                'batch_size': self.config['training']['batch_size'],
                'learning_rate': self.config['training']['learning_rate'],
                'epochs': self.config['training']['epochs'],
                'loss_type': self.config['model']['loss']['type'],
                'optimizer': 'Adam',
                'scheduler': self.config['training']['scheduler']
            })
            
            # Log model
            mlflow.pytorch.log_model(self.model, "model")
            
            for epoch in range(self.config['training']['epochs']):
                self.current_epoch = epoch
                
                # Train
                train_loss = self.train_epoch()
                self.train_losses.append(train_loss)
                
                # Validate
                val_loss, val_metrics = self.validate_epoch()
                self.val_losses.append(val_loss)
                self.val_metrics.append(val_metrics)
                
                # Update scheduler
                if self.scheduler:
                    self.scheduler.step()
                
                # Get primary metric
                primary_metric = self.get_primary_metric(val_metrics)
                
                # Check if best model
                is_best = primary_metric < self.best_metric
                if is_best:
                    self.best_metric = primary_metric
                
                # Save checkpoint
                self.save_checkpoint(epoch, val_metrics, is_best)
                
                # Log metrics
                self.log_metrics({'train_loss': train_loss}, '')
                self.log_metrics({'val_loss': val_loss}, '')
                self.log_metrics(val_metrics, 'val_')
                
                # Print progress
                print(f"Epoch {epoch+1}/{self.config['training']['epochs']}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Primary Metric: {primary_metric:.4f}")
                print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # Early stopping
                if self.early_stopping(primary_metric, self.model):
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            print("Training completed!")
            print(f"Best primary metric: {self.best_metric:.4f}")


class DepthTrainer(Trainer):
    """Trainer for depth prediction models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "depth_prediction")
    
    def compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute depth prediction metrics"""
        return DepthMetrics.compute_all_metrics(outputs, targets)
    
    def get_target_key(self) -> str:
        return 'depth'
    
    def get_primary_metric(self, metrics: Dict[str, float]) -> float:
        return metrics['rmse']  # Lower is better


class SegmentationTrainer(Trainer):
    """Trainer for segmentation models"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config, "segmentation")
    
    def compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute segmentation metrics"""
        num_classes = self.config['model']['architecture']['num_classes']
        return SegmentationMetrics.compute_all_metrics(outputs, targets, num_classes)
    
    def get_target_key(self) -> str:
        return 'mask'
    
    def get_primary_metric(self, metrics: Dict[str, float]) -> float:
        return -metrics['mean_iou']  # Negative because lower is better for early stopping


def train_depth_model(config_path: str):
    """Train depth prediction model"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    trainer = DepthTrainer(config)
    trainer.setup()
    trainer.train()


def train_segmentation_model(config_path: str):
    """Train segmentation model"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    trainer = SegmentationTrainer(config)
    trainer.setup()
    trainer.train()


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train sensor fusion models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_type', type=str, choices=['depth', 'segmentation'], 
                       required=True, help='Type of model to train')
    
    args = parser.parse_args()
    
    if args.model_type == 'depth':
        train_depth_model(args.config)
    elif args.model_type == 'segmentation':
        train_segmentation_model(args.config)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


if __name__ == "__main__":
    main()
