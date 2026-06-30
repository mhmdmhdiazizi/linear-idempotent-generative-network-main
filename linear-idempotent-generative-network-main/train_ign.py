"""
Training script for Idempotent Generative Network (IGN).

This script demonstrates how to train an IGN model using the losses
described in the paper: reconstruction, sparsity, and isometry.
"""

import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from ign import IGNLinearizer, InverseUnet
from ign.common.song__unet import SongUNet



def create_song_unet(model_channels, in_channels, out_channels, img_resolution,
                     channel_mult=(1, 2, 4), num_blocks=2):
    """Factory function to create SongUNet instances."""
    return SongUNet(
        img_resolution=img_resolution,
        in_channels=in_channels,
        out_channels=out_channels,
        model_channels=model_channels,
        channel_mult=channel_mult,
        num_blocks=num_blocks,
    )


def train_ign(args):
    """Train the IGN model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    transform = transforms.Compose([
        transforms.Resize((args.img_resolution, args.img_resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    if args.dataset == 'cifar10':
        dataset = CIFAR10(root='./data', train=True, download=False, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Create invertible encoder/decoder
    gx = InverseUnet(
        num_of_layers=args.num_layers,
        in_ch=args.channels,
        img_resolution=args.img_resolution,
        creat_song_unet=create_song_unet,
        model_channels=args.model_channels
    ).to(device)
    
    # Create IGN linearizer with idempotent projector
    latent_dim = gx.dim
    print(f"Latent dimension: {latent_dim}")
    
    ign = IGNLinearizer(
        gx=gx,
        latent_dim=latent_dim,
        learnable_projector=True
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(ign.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 2, gamma=0.5)
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_rec_loss = 0.0
        total_sparse_loss = 0.0
        total_isometry_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                # If batch is a list/tuple, take the first element (the tensor)
                if len(batch) > 0:
                    x = batch[0]
                else:
                    continue
            else:
                x = batch
            
            # Move to device and ensure tensor
            if isinstance(x, torch.Tensor):
                x = x.to(device)

            # x : [B, C, H, W])

            optimizer.zero_grad()
            
            # Compute losses
            losses = ign.compute_losses(
                x,
                lambda_rec=args.lambda_rec,
                lambda_sparse=args.lambda_sparse,
                lambda_isometry=args.lambda_isometry
            )
            
            loss = losses['total']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_rec_loss += losses['reconstruction'].item()
            total_sparse_loss += losses['sparsity'].item()
            total_isometry_loss += losses['isometry'].item()

            # Update progress bar description with current loss values
            pbar.set_postfix({
                'loss': loss.item(),
                'rec': losses['reconstruction'].item(),
                'sp': losses['sparsity'].item()
            })
            num_batches += 1
        
        scheduler.step()
        
        # Log progress
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_rec = total_rec_loss / num_batches
            avg_sparse = total_sparse_loss / num_batches
            avg_isometry = total_isometry_loss / num_batches
        else:
            avg_loss = avg_rec = avg_sparse = avg_isometry = 0.0
        
        active_dims = ign.get_active_dimensions()
        
        print(f"Epoch [{epoch+1}/{args.epochs}]: "
              f"Total Loss: {avg_loss:.4f}, "
              f"Rec: {avg_rec:.4f}, "
              f"Sparse: {avg_sparse:.4f}, "
              f"Isometry: {avg_isometry:.4f}, "
              f"Active Dims: {active_dims}/{latent_dim}")
        
        # Verify idempotency periodically
        if (epoch + 1) % args.verify_every == 0:
            with torch.no_grad():
                test_input = torch.randn(1, args.channels, args.img_resolution, 
                                        args.img_resolution, device=device)
                is_idempotent, diff = ign.verify_idempotency(test_input)
                print(f"  Idempotency check: {'PASS' if is_idempotent else 'FAIL'} "
                      f"(max diff: {diff:.2e})")
    
    # Save model
    if args.save_path:
        torch.save({
            'model_state_dict': ign.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': args.epochs,
        }, args.save_path)
        print(f"Model saved to {args.save_path}")
    
    return ign


def main():
    parser = argparse.ArgumentParser(description='Train Idempotent Generative Network')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10'],
                       help='Dataset to use for training')
    parser.add_argument('--img-resolution', type=int, default=32,
                       help='Image resolution')
    parser.add_argument('--channels', type=int, default=3,
                       help='Number of input channels')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num-layers', type=int, default=4,
                       help='Number of invertible layers')
    parser.add_argument('--model-channels', type=int, default=32,
                       help='Base channel count for UNet')
    parser.add_argument('--lambda-rec', type=float, default=1.0,
                       help='Weight for reconstruction loss')
    parser.add_argument('--lambda-sparse', type=float, default=0.1,
                       help='Weight for sparsity loss')
    parser.add_argument('--lambda-isometry', type=float, default=0.001,
                       help='Weight for isometry loss')
    parser.add_argument('--verify-every', type=int, default=10,
                       help='Verify idempotency every N epochs')
    parser.add_argument('--save-path', type=str, default=None,
                       help='Path to save the trained model')
    
    args = parser.parse_args()
    train_ign(args)


if __name__ == '__main__':
    main()