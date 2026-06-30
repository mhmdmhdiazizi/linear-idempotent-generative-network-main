"""Utility functions for IGN (Image-Guided Network) applications."""

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np


def save_image_grid(images, filename, nrow=None, padding=2, normalize=True):
    """
    Save a grid of images to a file.
    
    Args:
        images: Tensor of shape [B, C, H, W] or list of tensors
        filename: Output filename (supports .png, .jpg, .pdf)
        nrow: Number of images per row (default: auto)
        padding: Padding between images in pixels
        normalize: Whether to normalize images to [0, 1]
    """
    if isinstance(images, list):
        images = torch.cat(images, dim=0)
    
    if normalize and images.min() < 0:
        images = (images + 1) / 2  # Assume [-1, 1] range
    
    images = torch.clamp(images, 0, 1)
    
    if nrow is None:
        nrow = int(np.ceil(np.sqrt(images.shape[0])))
    
    grid = make_grid(images, nrow=nrow, padding=padding, pad_value=1)
    
    # Convert to numpy and save
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    if filename.endswith('.pdf'):
        fig, ax = plt.subplots(figsize=(nrow * 2, nrow * 2))
        ax.imshow(grid_np)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()
    else:
        from PIL import Image
        grid_img = Image.fromarray((grid_np * 255).astype(np.uint8))
        grid_img.save(filename)
    
    print(f"Saved image grid to {filename}")


def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0 if img1.max() <= 1 else 255.0
    psnr = 20 * torch.log10(torch.tensor(max_pixel)) - 10 * torch.log10(mse)
    return psnr.item()


def calculate_ssim(img1, img2, window_size=11):
    """Calculate SSIM between two images."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def prepare_condition(condition_type, image):
    """
    Prepare condition image based on type.
    
    Args:
        condition_type: Type of condition ('edge', 'segmentation', 'low_res', 'inpainting')
        image: Input image tensor [B, C, H, W]
        
    Returns:
        Conditioned image
    """
    if condition_type == 'edge':
        # Simple Sobel edge detection
        from torchvision.transforms import Sobel
        sobel_x = Sobel(1)
        sobel_y = Sobel(0)
        gray = image.mean(dim=1, keepdim=True)  # Convert to grayscale
        edges_x = sobel_x(gray)
        edges_y = sobel_y(gray)
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        return edges
    
    elif condition_type == 'low_res':
        # Downsample and upsample
        h, w = image.shape[2:]
        low_res = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=False)
        low_res = F.interpolate(low_res, size=(h, w), mode='bilinear', align_corners=False)
        return low_res
    
    elif condition_type == 'inpainting':
        # Create mask with center region masked
        h, w = image.shape[2:]
        mask = torch.ones_like(image)
        mask[:, :, h//4:3*h//4, w//4:3*w//4] = 0
        return image * mask
    
    elif condition_type == 'segmentation':
        # Simplified segmentation (just color quantization for demo)
        quantized = (image * 7).round() / 7
        return quantized
    
    else:
        raise ValueError(f"Unknown condition type: {condition_type}")


def get_model_summary(model):
    """Get a summary of model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params
    }
