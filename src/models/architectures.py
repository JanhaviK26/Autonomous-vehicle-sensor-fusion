"""
Model Architectures for Sensor Fusion

This module contains neural network architectures for depth prediction
and semantic segmentation using fused LiDAR and camera data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List, Optional, Dict, Any
import math


class ResidualBlock(nn.Module):
    """Residual block for encoder"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out


class UNetDepth(nn.Module):
    """U-Net architecture for depth prediction"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        arch_config = config['model']['architecture']
        self.input_channels = arch_config['input_channels']  # RGB + Depth
        self.output_channels = arch_config['output_channels']  # Depth
        self.dropout = config['model']['dropout']
        
        # Encoder
        self.enc1 = self._make_encoder_block(self.input_channels, 64)
        self.enc2 = self._make_encoder_block(64, 128, stride=2)
        self.enc3 = self._make_encoder_block(128, 256, stride=2)
        self.enc4 = self._make_encoder_block(256, 512, stride=2)
        
        # Bottleneck
        self.bottleneck = self._make_encoder_block(512, 1024, stride=2)
        
        # Decoder
        self.dec4 = self._make_decoder_block(1024, 512)
        self.dec3 = self._make_decoder_block(512, 256)
        self.dec2 = self._make_decoder_block(256, 128)
        self.dec1 = self._make_decoder_block(128, 64)
        
        # Final prediction
        self.final_conv = nn.Conv2d(64, self.output_channels, 1)
        
        # Dropout
        self.dropout_layer = nn.Dropout2d(self.dropout)
        
    def _make_encoder_block(self, in_channels: int, out_channels: int, stride: int = 1):
        """Create encoder block with residual connections"""
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels)
        )
    
    def _make_decoder_block(self, in_channels: int, out_channels: int):
        """Create decoder block with upsampling"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualBlock(out_channels, out_channels)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        # Bottleneck
        bottleneck = self.bottleneck(enc4)
        bottleneck = self.dropout_layer(bottleneck)
        
        # Decoder with skip connections
        dec4 = self.dec4(bottleneck)
        dec4 = dec4 + enc4  # Skip connection
        
        dec3 = self.dec3(dec4)
        dec3 = dec3 + enc3  # Skip connection
        
        dec2 = self.dec2(dec3)
        dec2 = dec2 + enc2  # Skip connection
        
        dec1 = self.dec1(dec2)
        dec1 = dec1 + enc1  # Skip connection
        
        # Final prediction
        depth = self.final_conv(dec1)
        
        return depth


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ architecture for semantic segmentation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        arch_config = config['model']['architecture']
        self.input_channels = arch_config['input_channels']  # RGB + Depth
        self.num_classes = arch_config['num_classes']
        self.backbone_name = arch_config['backbone']
        self.dropout = config['model']['dropout']
        
        # Modify first layer to accept 4 channels
        self.backbone = self._get_backbone()
        self._modify_first_layer()
        
        # ASPP module
        self.aspp = ASPP(2048, 256)
        
        # Decoder
        self.decoder = Decoder(256, 256, self.num_classes)  # Fix: low_level_channels should be 256 for ResNet50
        
        # Dropout
        self.dropout_layer = nn.Dropout2d(self.dropout)
        
    def _get_backbone(self):
        """Get backbone network"""
        if self.backbone_name == 'resnet50':
            return models.resnet50(pretrained=True)
        elif self.backbone_name == 'resnet101':
            return models.resnet101(pretrained=True)
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
    
    def _modify_first_layer(self):
        """Modify first layer to accept 4 channels"""
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            self.input_channels, 
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Initialize new weights
        with torch.no_grad():
            self.backbone.conv1.weight[:, :3] = original_conv.weight
            if self.input_channels == 4:
                # Initialize 4th channel (depth) with small random weights
                nn.init.normal_(self.backbone.conv1.weight[:, 3:], 0, 0.01)
    
    def forward(self, x):
        input_size = x.shape[-2:]
        
        # Extract features
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        low_level_features = x
        
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # ASPP
        x = self.aspp(x)
        x = self.dropout_layer(x)
        
        # Decoder
        x = self.decoder(x, low_level_features)
        
        # Upsample to input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return x


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(inplace=True)
        )
        
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        size = x.shape[2:]
        
        x1 = self.conv1x1(x)
        x2 = self.conv3x3_1(x)
        x3 = self.conv3x3_2(x)
        x4 = self.conv3x3_3(x)
        
        x5 = self.global_pool(x)
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=False)
        
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.conv_out(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x


class Decoder(nn.Module):
    """Decoder for DeepLabV3+"""
    
    def __init__(self, in_channels: int, low_level_channels: int, num_classes: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1)
        self.conv2 = nn.Conv2d(in_channels + 48, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, num_classes, 1)
        
        self.bn1 = nn.BatchNorm2d(48)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, low_level_features):
        # Process low-level features
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.bn1(low_level_features)
        low_level_features = self.relu(low_level_features)
        
        # Upsample high-level features
        x = F.interpolate(x, size=low_level_features.shape[2:], 
                         mode='bilinear', align_corners=False)
        
        # Concatenate features
        x = torch.cat([x, low_level_features], dim=1)
        
        # Process concatenated features
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        
        return x


class FusionNet(nn.Module):
    """Custom fusion network combining RGB and LiDAR features"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.input_channels = config['model']['architecture']['input_channels']
        self.output_channels = config['model']['architecture']['output_channels']
        
        # Separate encoders for RGB and depth
        self.rgb_encoder = self._make_encoder(3, 64)
        self.depth_encoder = self._make_encoder(1, 64)
        
        # Fusion layers
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = self._make_decoder(128, self.output_channels)
        
    def _make_encoder(self, in_channels: int, out_channels: int):
        """Create encoder block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    
    def _make_decoder(self, in_channels: int, out_channels: int):
        """Create decoder block"""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, 64, 2, 2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1)
        )
    
    def forward(self, x):
        # Split input into RGB and depth
        rgb = x[:, :3]
        depth = x[:, 3:4]
        
        # Encode separately
        rgb_features = self.rgb_encoder(rgb)
        depth_features = self.depth_encoder(depth)
        
        # Fuse features
        fused_features = torch.cat([rgb_features, depth_features], dim=1)
        fused_features = self.fusion_conv(fused_features)
        
        # Decode
        output = self.decoder(fused_features)
        
        return output


def create_model(config: Dict[str, Any]) -> nn.Module:
    """Create model based on configuration"""
    model_name = config['model']['name']
    
    if model_name == 'unet_depth':
        return UNetDepth(config)
    elif model_name == 'deeplabv3plus':
        return DeepLabV3Plus(config)
    elif model_name == 'fusion_net':
        return FusionNet(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """Test model creation and forward pass"""
    import yaml
    
    # Test depth prediction model
    with open('configs/depth_model.yaml', 'r') as f:
        depth_config = yaml.safe_load(f)
    
    depth_model = create_model(depth_config)
    print(f"Depth Model: {depth_model.__class__.__name__}")
    print(f"Parameters: {count_parameters(depth_model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 4, 256, 256)  # Batch of 2, 4 channels (RGB+Depth), 256x256
    with torch.no_grad():
        depth_output = depth_model(dummy_input)
    print(f"Depth output shape: {depth_output.shape}")
    
    # Test segmentation model
    with open('configs/segmentation_model.yaml', 'r') as f:
        seg_config = yaml.safe_load(f)
    
    seg_model = create_model(seg_config)
    print(f"\nSegmentation Model: {seg_model.__class__.__name__}")
    print(f"Parameters: {count_parameters(seg_model):,}")
    
    with torch.no_grad():
        seg_output = seg_model(dummy_input)
    print(f"Segmentation output shape: {seg_output.shape}")


if __name__ == "__main__":
    main()
