"""
Evaluation and Visualization Module

This module provides comprehensive evaluation tools for depth prediction and
segmentation models, including metrics computation and visualization utilities.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
from PIL import Image
import yaml
from tqdm import tqdm

from ..models.architectures import create_model
from ..models.losses import DepthMetrics, SegmentationMetrics
from ..data.dataset import create_dataloaders


class ModelEvaluator:
    """Base class for model evaluation"""
    
    def __init__(self, config: Dict[str, Any], model_path: str):
        self.config = config
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Create data loader
        self.dataloader = self._create_dataloader()
        
    def _load_model(self) -> nn.Module:
        """Load trained model"""
        model = create_model(self.config)
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model.to(self.device)
    
    def _create_dataloader(self):
        """Create test data loader"""
        dataloaders = create_dataloaders(self.config, self.config['data']['test_path'])
        return dataloaders['test']
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on test set"""
        all_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Evaluating"):
                # Move data to device
                images = batch['image'].to(self.device)
                targets = batch[self.get_target_key()].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute metrics
                metrics = self.compute_metrics(outputs, targets)
                all_metrics.append(metrics)
        
        # Average metrics across all batches
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        return avg_metrics
    
    def compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute evaluation metrics - to be implemented by subclasses"""
        raise NotImplementedError
    
    def get_target_key(self) -> str:
        """Get target key from batch - to be implemented by subclasses"""
        raise NotImplementedError
    
    def visualize_predictions(self, num_samples: int = 5, save_path: Optional[str] = None):
        """Visualize model predictions - to be implemented by subclasses"""
        raise NotImplementedError


class DepthEvaluator(ModelEvaluator):
    """Evaluator for depth prediction models"""
    
    def compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute depth prediction metrics"""
        return DepthMetrics.compute_all_metrics(outputs, targets)
    
    def get_target_key(self) -> str:
        return 'depth'
    
    def visualize_predictions(self, num_samples: int = 5, save_path: Optional[str] = None):
        """Visualize depth predictions"""
        self.model.eval()
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if i >= num_samples:
                    break
                
                # Move data to device
                images = batch['image'].to(self.device)
                targets = batch['depth'].to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                
                # Convert to numpy for visualization
                rgb = images[0, :3].cpu().permute(1, 2, 0).numpy()
                depth_gt = targets[0, 0].cpu().numpy()
                depth_pred = predictions[0, 0].cpu().numpy()
                
                # Denormalize if needed
                if self.config['preprocessing']['normalize_depth']:
                    depth_max = self.config['data']['depth_max']
                    depth_gt *= depth_max
                    depth_pred *= depth_max
                
                # Plot RGB image
                axes[i, 0].imshow(rgb)
                axes[i, 0].set_title('RGB Image')
                axes[i, 0].axis('off')
                
                # Plot ground truth depth
                im1 = axes[i, 1].imshow(depth_gt, cmap='viridis', vmin=0, vmax=80)
                axes[i, 1].set_title('Ground Truth Depth')
                axes[i, 1].axis('off')
                plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
                
                # Plot predicted depth
                im2 = axes[i, 2].imshow(depth_pred, cmap='viridis', vmin=0, vmax=80)
                axes[i, 2].set_title('Predicted Depth')
                axes[i, 2].axis('off')
                plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
                
                # Plot error map
                error = np.abs(depth_pred - depth_gt)
                im3 = axes[i, 3].imshow(error, cmap='hot', vmin=0, vmax=10)
                axes[i, 3].set_title('Absolute Error')
                axes[i, 3].axis('off')
                plt.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()


class SegmentationEvaluator(ModelEvaluator):
    """Evaluator for segmentation models"""
    
    def compute_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Compute segmentation metrics"""
        num_classes = self.config['model']['architecture']['num_classes']
        return SegmentationMetrics.compute_all_metrics(outputs, targets, num_classes)
    
    def get_target_key(self) -> str:
        return 'mask'
    
    def visualize_predictions(self, num_samples: int = 5, save_path: Optional[str] = None):
        """Visualize segmentation predictions"""
        self.model.eval()
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        class_names = self.config['evaluation']['class_names']
        colors = ['black', 'green']  # Background, Drivable
        
        with torch.no_grad():
            for i, batch in enumerate(self.dataloader):
                if i >= num_samples:
                    break
                
                # Move data to device
                images = batch['image'].to(self.device)
                targets = batch['mask'].to(self.device)
                
                # Forward pass
                predictions = self.model(images)
                
                # Convert to numpy for visualization
                rgb = images[0, :3].cpu().permute(1, 2, 0).numpy()
                mask_gt = targets[0].cpu().numpy()
                mask_pred = torch.argmax(predictions[0], dim=0).cpu().numpy()
                
                # Plot RGB image
                axes[i, 0].imshow(rgb)
                axes[i, 0].set_title('RGB Image')
                axes[i, 0].axis('off')
                
                # Plot ground truth mask
                axes[i, 1].imshow(mask_gt, cmap='viridis', vmin=0, vmax=1)
                axes[i, 1].set_title('Ground Truth Mask')
                axes[i, 1].axis('off')
                
                # Plot predicted mask
                axes[i, 2].imshow(mask_pred, cmap='viridis', vmin=0, vmax=1)
                axes[i, 2].set_title('Predicted Mask')
                axes[i, 2].axis('off')
                
                # Plot overlay
                overlay = rgb.copy()
                mask_overlay = mask_pred == 1  # Drivable area
                overlay[mask_overlay] = [0, 1, 0]  # Green for drivable
                axes[i, 3].imshow(overlay)
                axes[i, 3].set_title('Prediction Overlay')
                axes[i, 3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()


class ComparisonAnalyzer:
    """Analyze performance differences between models"""
    
    def __init__(self, configs: List[Dict[str, Any]], model_paths: List[str]):
        self.configs = configs
        self.model_paths = model_paths
        self.results = {}
    
    def compare_models(self) -> Dict[str, Dict[str, float]]:
        """Compare multiple models"""
        for i, (config, model_path) in enumerate(zip(self.configs, self.model_paths)):
            model_name = config['model']['name']
            print(f"Evaluating {model_name}...")
            
            if 'depth' in model_name:
                evaluator = DepthEvaluator(config, model_path)
            else:
                evaluator = SegmentationEvaluator(config, model_path)
            
            metrics = evaluator.evaluate()
            self.results[model_name] = metrics
        
        return self.results
    
    def create_comparison_plot(self, save_path: Optional[str] = None):
        """Create comparison plots"""
        if not self.results:
            self.compare_models()
        
        # Determine if depth or segmentation metrics
        first_model = list(self.results.keys())[0]
        if 'depth' in first_model:
            self._plot_depth_comparison(save_path)
        else:
            self._plot_segmentation_comparison(save_path)
    
    def _plot_depth_comparison(self, save_path: Optional[str] = None):
        """Plot depth prediction comparison"""
        metrics = ['rmse', 'mae', 'delta1', 'delta2', 'delta3']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            model_names = list(self.results.keys())
            values = [self.results[model][metric] for model in model_names]
            
            bars = axes[i].bar(model_names, values)
            axes[i].set_title(f'{metric.upper()}')
            axes[i].set_ylabel('Value')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')
        
        # Remove empty subplot
        axes[-1].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def _plot_segmentation_comparison(self, save_path: Optional[str] = None):
        """Plot segmentation comparison"""
        metrics = ['mean_iou', 'mean_f1', 'pixel_accuracy']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            model_names = list(self.results.keys())
            values = [self.results[model][metric] for model in model_names]
            
            bars = axes[i].bar(model_names, values)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Value')
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()


def evaluate_model(config_path: str, model_path: str, model_type: str = 'depth'):
    """Evaluate a single model"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if model_type == 'depth':
        evaluator = DepthEvaluator(config, model_path)
    else:
        evaluator = SegmentationEvaluator(config, model_path)
    
    # Evaluate
    metrics = evaluator.evaluate()
    
    print(f"\nEvaluation Results for {config['model']['name']}:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric:20}: {value:.4f}")
    
    # Visualize
    evaluator.visualize_predictions(num_samples=5)
    
    return metrics


def compare_models(config_paths: List[str], model_paths: List[str]):
    """Compare multiple models"""
    configs = []
    for config_path in config_paths:
        with open(config_path, 'r') as f:
            configs.append(yaml.safe_load(f))
    
    analyzer = ComparisonAnalyzer(configs, model_paths)
    results = analyzer.compare_models()
    
    print("\nModel Comparison Results:")
    print("=" * 60)
    
    # Print comparison table
    first_model = list(results.keys())[0]
    metrics = list(results[first_model].keys())
    
    # Header
    print(f"{'Model':<20}", end="")
    for metric in metrics:
        print(f"{metric:>12}", end="")
    print()
    print("-" * (20 + 12 * len(metrics)))
    
    # Data rows
    for model_name, model_metrics in results.items():
        print(f"{model_name:<20}", end="")
        for metric in metrics:
            print(f"{model_metrics[metric]:>12.4f}", end="")
        print()
    
    # Create comparison plot
    analyzer.create_comparison_plot()
    
    return results


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate sensor fusion models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['depth', 'segmentation'], 
                       default='depth', help='Type of model to evaluate')
    
    args = parser.parse_args()
    
    evaluate_model(args.config, args.model, args.model_type)


if __name__ == "__main__":
    main()
