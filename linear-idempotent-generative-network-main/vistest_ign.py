"""
Visual Test Script for Idempotent Generative Network (IGN).

This script is designed to be run AFTER training. It loads a pre-trained
checkpoint and generates visual proofs for the core claims of the paper:
1. Fixed Point Property: f(x) ≈ x for real data (Reconstruction)
2. Global Projection: f(noise) projects ambient noise onto the data manifold
3. Latent Sparsity: Visualizing the binary diagonal projector Λ
"""

import argparse
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch.nn.functional as F

from ign.modules.ign_linearizer import IGNLinearizer
from ign.modules.invertable_network import InverseUnet
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
        num_blocks=num_blocks,  # Fixed: Added num_blocks parameter
    )


def load_pretrained_ign(args, device):
    """Instantiate the architecture and load the pre-trained weights."""
    gx = InverseUnet(
        num_of_layers=args.num_layers,
        in_ch=args.channels,
        img_resolution=args.img_resolution,
        creat_song_unet=create_song_unet,
        model_channels=args.model_channels
    ).to(device)
    
    ign = IGNLinearizer(gx=gx, latent_dim=gx.dim).to(device)
    
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        ign.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Successfully loaded model from: {args.checkpoint}")
        print(f"  Trained for {checkpoint.get('epoch', 'unknown')} epochs.")
    except Exception as e:
        print(f"✗ Failed to load checkpoint. Error: {e}")
        print("  Ensure the --img-resolution, --channels, --num-layers, and --model-channels")
        print("  exactly match the ones used during training.")
        exit(1)
        
    return ign


def denorm(img):
    """Convert images from [-1, 1] back to [0, 1] for matplotlib."""
    return torch.clamp(img * 0.5 + 0.5, 0, 1)


def run_visual_tests(ign, args, device):
    """Run inference and generate visualization plots."""
    ign.eval()
    
    # 1. Load real data (CIFAR-10)
    transform = transforms.Compose([
        transforms.Resize((args.img_resolution, args.img_resolution)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.num_samples, shuffle=True)
    real_images = next(iter(dataloader))[0].to(device)
    
    # 2. Generate random ambient noise
    random_noise = torch.randn_like(real_images)
    
    # 3. Model Inference
    with torch.no_grad():
        reconstructed = ign.project(real_images)
        projected_noise = ign.project(random_noise)
        
        # Extract binary mask from the STE probabilities - with error handling
        try:
            if hasattr(ign, 'projector') and hasattr(ign.projector, 'logits'):
                probs = torch.sigmoid(ign.projector.logits)
                binary_mask = torch.round(probs).cpu()
            elif hasattr(ign, 'get_active_dimensions'):
                # Fallback: use active dimensions
                active_dims = ign.get_active_dimensions()
                total_dims = ign.latent_dim
                binary_mask = torch.zeros(total_dims)
                binary_mask[:active_dims] = 1
                print(f"Using fallback for binary mask: {active_dims}/{total_dims} active")
            else:
                binary_mask = torch.ones(ign.latent_dim)  # Default: all active
                print("Warning: Could not extract projector mask, using all ones")
        except Exception as e:
            print(f"Warning: Could not extract projector mask: {e}")
            binary_mask = torch.ones(ign.latent_dim)
        
    # Compute quantitative metric for reconstruction
    mse_rec = F.mse_loss(reconstructed, real_images).item()
    psnr = 10 * torch.log10(1.0 / torch.tensor(mse_rec)).item()

    # --- PLOT 1: Image Comparisons ---
    fig, axs = plt.subplots(3, args.num_samples, figsize=(12, 9))
    fig.suptitle(f"Linear IGN Visual Evaluation (Reconstruction PSNR: {psnr:.2f} dB)", fontsize=14)

    for i in range(args.num_samples):
        # Top Row: Original Real Images
        axs[0, i].imshow(denorm(real_images[i]).cpu().permute(1, 2, 0).numpy())
        axs[0, i].set_title("Real Data", fontsize=10)
        axs[0, i].axis('off')
        
        # Middle Row: Reconstructed f(x) -> Should equal Real Data
        axs[1, i].imshow(denorm(reconstructed[i]).cpu().permute(1, 2, 0).numpy())
        axs[1, i].set_title("f(x) [Fixed Pt]", fontsize=10)
        axs[1, i].axis('off')
        
        # Bottom Row: Projected Noise f(noise) -> Should look like structured data
        axs[2, i].imshow(denorm(projected_noise[i]).cpu().permute(1, 2, 0).numpy())
        axs[2, i].set_title("f(noise) [Projected]", fontsize=10)
        axs[2, i].axis('off')

    plt.tight_layout()
    plt.savefig('vis_ign_projections.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visual comparison to vis_ign_projections.png")
    plt.close()

    # --- PLOT 2: Latent Binary Mask ---
    active_dims = int(binary_mask.sum().item())
    total_dims = len(binary_mask)
    
    plt.figure(figsize=(12, 2.5))
    plt.bar(range(total_dims), binary_mask, width=1.0, color='black')
    plt.title(f"Binary Projector Λ (Active Dims: {active_dims} / {total_dims} [{active_dims/total_dims*100:.1f}%])")
    plt.xlabel("Latent Dimension Index")
    plt.ylabel("State (0=Off, 1=On)")
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1])
    plt.xlim(0, total_dims)
    plt.tight_layout()
    plt.savefig('vis_ign_latent_mask.png', dpi=150, bbox_inches='tight')
    print("Saved latent mask to vis_ign_latent_mask.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visual Tests for Trained IGN')
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the .pth checkpoint file')
    
    # Architecture hyperparameters (MUST match training config)
    parser.add_argument('--img-resolution', type=int, default=32)
    parser.add_argument('--channels', type=int, default=3)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--model-channels', type=int, default=32)
    
    # Visuals
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of image pairs to visualize')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 60)
    print("IGN Visual Test Suite")
    print("=" * 60)
    
    ign = load_pretrained_ign(args, device)
    run_visual_tests(ign, args, device)
    
    print("\nVisual tests complete.")


if __name__ == '__main__':
    main()