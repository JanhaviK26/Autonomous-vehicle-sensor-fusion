"""
Loss Functions and Training Utilities

This module contains custom loss functions for depth prediction and segmentation,
along with training utilities and metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
import math


class BerhuLoss(nn.Module):
    """BerHu loss for depth prediction - combines L1 and L2 losses"""
    
    def __init__(self, threshold: float = 0.2):
        super().__init__()
        self.threshold = threshold
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            pred: Predicted depth map
            target: Ground truth depth map
            mask: Optional mask for valid pixels
        """
        diff = torch.abs(pred - target)
        
        if mask is not None:
            diff = diff * mask
            valid_pixels = mask.sum()
        else:
            valid_pixels = diff.numel()
        
        # Calculate threshold based on maximum error
        delta = self.threshold * torch.max(diff)
        
        # BerHu loss: L1 for small errors, L2 for large errors
        loss = torch.where(
            diff <= delta,
            diff,  # L1 loss
            (diff**2 + delta**2) / (2 * delta)  # L2 loss
        )
        
        return loss.sum() / valid_pixels


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance in segmentation"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            inputs: Logits from model (N, C, H, W)
            targets: Ground truth labels (N, H, W)
        """
        # Convert to probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply focal loss formula
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            inputs: Logits from model (N, C, H, W)
            targets: Ground truth labels (N, H, W)
        """
        # Convert logits to probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        
        # Calculate Dice coefficient for each class
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        
        # Return 1 - dice (loss)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combined loss function for segmentation"""
    
    def __init__(self, focal_weight: float = 1.0, dice_weight: float = 1.0, 
                 alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_loss = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        return self.focal_weight * focal + self.dice_weight * dice


class DepthMetrics:
    """Metrics for depth prediction evaluation"""
    
    @staticmethod
    def rmse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """Root Mean Square Error"""
        if mask is not None:
            diff = (pred - target) * mask
            mse = (diff ** 2).sum() / mask.sum()
        else:
            mse = F.mse_loss(pred, target)
        return math.sqrt(mse.item())
    
    @staticmethod
    def mae(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """Mean Absolute Error"""
        if mask is not None:
            diff = torch.abs(pred - target) * mask
            return diff.sum().item() / mask.sum().item()
        else:
            return F.l1_loss(pred, target).item()
    
    @staticmethod
    def delta_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float, 
                      mask: Optional[torch.Tensor] = None) -> float:
        """δ accuracy: percentage of pixels with max(pred/target, target/pred) < threshold"""
        if mask is not None:
            pred_masked = pred * mask
            target_masked = target * mask
            valid_pixels = mask.sum()
        else:
            pred_masked = pred
            target_masked = target
            valid_pixels = pred.numel()
        
        # Avoid division by zero
        ratio1 = pred_masked / (target_masked + 1e-8)
        ratio2 = target_masked / (pred_masked + 1e-8)
        max_ratio = torch.max(ratio1, ratio2)
        
        if mask is not None:
            accurate_pixels = ((max_ratio < threshold) * mask).sum()
        else:
            accurate_pixels = (max_ratio < threshold).sum()
        
        return (accurate_pixels / valid_pixels).item()
    
    @staticmethod
    def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor, 
                          mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Compute all depth metrics"""
        metrics = {
            'rmse': DepthMetrics.rmse(pred, target, mask),
            'mae': DepthMetrics.mae(pred, target, mask),
            'delta1': DepthMetrics.delta_accuracy(pred, target, 1.25, mask),
            'delta2': DepthMetrics.delta_accuracy(pred, target, 1.25**2, mask),
            'delta3': DepthMetrics.delta_accuracy(pred, target, 1.25**3, mask)
        }
        return metrics


class SegmentationMetrics:
    """Metrics for segmentation evaluation"""
    
    @staticmethod
    def iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, float]:
        """Intersection over Union for each class"""
        pred_labels = torch.argmax(pred, dim=1)
        
        ious = {}
        for cls in range(num_classes):
            pred_cls = (pred_labels == cls)
            target_cls = (target == cls)
            
            intersection = (pred_cls & target_cls).sum().float()
            union = (pred_cls | target_cls).sum().float()
            
            if union > 0:
                ious[f'iou_class_{cls}'] = (intersection / union).item()
            else:
                ious[f'iou_class_{cls}'] = 0.0
        
        # Mean IoU
        ious['mean_iou'] = np.mean(list(ious.values()))
        
        return ious
    
    @staticmethod
    def f1_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, float]:
        """F1 score for each class"""
        pred_labels = torch.argmax(pred, dim=1)
        
        f1_scores = {}
        for cls in range(num_classes):
            pred_cls = (pred_labels == cls)
            target_cls = (target == cls)
            
            tp = (pred_cls & target_cls).sum().float()
            fp = (pred_cls & ~target_cls).sum().float()
            fn = (~pred_cls & target_cls).sum().float()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1_scores[f'f1_class_{cls}'] = f1.item()
        
        # Mean F1
        f1_scores['mean_f1'] = np.mean(list(f1_scores.values()))
        
        return f1_scores
    
    @staticmethod
    def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Pixel-wise accuracy"""
        pred_labels = torch.argmax(pred, dim=1)
        correct = (pred_labels == target).sum().float()
        total = target.numel()
        return (correct / total).item()
    
    @staticmethod
    def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor, 
                          num_classes: int) -> Dict[str, float]:
        """Compute all segmentation metrics"""
        metrics = {}
        
        # IoU metrics
        iou_metrics = SegmentationMetrics.iou(pred, target, num_classes)
        metrics.update(iou_metrics)
        
        # F1 metrics
        f1_metrics = SegmentationMetrics.f1_score(pred, target, num_classes)
        metrics.update(f1_metrics)
        
        # Pixel accuracy
        metrics['pixel_accuracy'] = SegmentationMetrics.pixel_accuracy(pred, target)
        
        return metrics


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 mode: str = 'min', restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """
        Args:
            score: Current validation score
            model: Model to potentially restore weights for
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save model checkpoint"""
        self.best_weights = model.state_dict().copy()


class LearningRateScheduler:
    """Learning rate scheduler utilities"""
    
    @staticmethod
    def get_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
        """Get learning rate scheduler based on config"""
        scheduler_type = config.get('scheduler', 'step')
        
        if scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=config.get('step_size', 30),
                gamma=config.get('gamma', 0.1)
            )
        elif scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.get('T_max', 100)
            )
        elif scheduler_type == 'poly':
            return torch.optim.lr_scheduler.PolynomialLR(
                optimizer,
                total_iters=config.get('total_iters', 100),
                power=config.get('power', 0.9)
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")


def create_loss_function(config: Dict[str, Any]) -> nn.Module:
    """Create loss function based on configuration"""
    loss_config = config['model']['loss']
    loss_type = loss_config['type']
    
    if loss_type == 'berhu':
        return BerhuLoss(threshold=loss_config.get('threshold', 0.2))
    elif loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'l2':
        return nn.MSELoss()
    elif loss_type == 'smooth_l1':
        return nn.SmoothL1Loss()
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=loss_config.get('alpha', 0.25),
            gamma=loss_config.get('gamma', 2.0)
        )
    elif loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'combined':
        return CombinedLoss(
            focal_weight=loss_config.get('focal_weight', 1.0),
            dice_weight=loss_config.get('dice_weight', 1.0),
            alpha=loss_config.get('alpha', 0.25),
            gamma=loss_config.get('gamma', 2.0)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def main():
    """Test loss functions and metrics"""
    # Test depth metrics
    pred_depth = torch.randn(2, 1, 64, 64)
    target_depth = torch.randn(2, 1, 64, 64)
    
    depth_metrics = DepthMetrics.compute_all_metrics(pred_depth, target_depth)
    print("Depth Metrics:")
    for metric, value in depth_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test segmentation metrics
    pred_seg = torch.randn(2, 2, 64, 64)  # 2 classes
    target_seg = torch.randint(0, 2, (2, 64, 64))
    
    seg_metrics = SegmentationMetrics.compute_all_metrics(pred_seg, target_seg, 2)
    print("\nSegmentation Metrics:")
    for metric, value in seg_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test loss functions
    berhu_loss = BerhuLoss()
    focal_loss = FocalLoss()
    
    depth_loss = berhu_loss(pred_depth, target_depth)
    seg_loss = focal_loss(pred_seg, target_seg)
    
    print(f"\nLoss Values:")
    print(f"  BerHu Loss: {depth_loss:.4f}")
    print(f"  Focal Loss: {seg_loss:.4f}")


if __name__ == "__main__":
    main()
