"""
Loss functions for training
"""

import torch
import torch.nn.functional as F


def l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    L2 loss (MSE)
    
    Args:
        pred: Predicted tensor
        target: Target tensor
    
    Returns:
        L2 loss
    """
    return ((pred - target) ** 2).mean()


def cos_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cosine similarity loss
    
    Args:
        pred: Predicted tensor
        target: Target tensor
    
    Returns:
        Cosine loss (1 - cosine_similarity)
    """
    return 1 - F.cosine_similarity(pred, target, dim=-1).mean()


def tv_loss(image: torch.Tensor) -> torch.Tensor:
    """
    Total variation loss for smoothness
    
    Args:
        image: [B, C, H, W] image tensor
    
    Returns:
        TV loss
    """
    diff_i = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    diff_j = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
    return diff_i.mean() + diff_j.mean()


def ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Structural Similarity Index loss
    
    Args:
        pred: [B, C, H, W] predicted image
        target: [B, C, H, W] target image
        window_size: Window size for SSIM computation
    
    Returns:
        SSIM loss (1 - SSIM)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = F.avg_pool2d(pred, window_size, 1, window_size//2)
    mu2 = F.avg_pool2d(target, window_size, 1, window_size//2)
    
    sigma1_sq = (F.avg_pool2d(pred * pred, window_size, 1, window_size//2) - mu1 * mu1).clamp(min=0)
    sigma2_sq = (F.avg_pool2d(target * target, window_size, 1, window_size//2) - mu2 * mu2).clamp(min=0)
    sigma12 = F.avg_pool2d(pred * target, window_size, 1, window_size//2) - mu1 * mu2
    
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return 1 - ssim_map.mean()
