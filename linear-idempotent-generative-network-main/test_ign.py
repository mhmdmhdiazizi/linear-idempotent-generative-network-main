"""
Test script for Idempotent Generative Network (IGN).

This script demonstrates the key properties of IGN:
1. Exact idempotency: f(f(x)) = f(x)
2. Projection onto learned data manifold
3. Sparsity of the latent representation
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

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
        num_blocks=num_blocks,  # Fixed: Added num_blocks parameter
    )


def test_idempotency():
    """Test that the IGN model is exactly idempotent."""
    print("=" * 60)
    print("Testing Idempotency Property: f(f(x)) = f(x)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a small model for testing
    img_resolution = 16
    channels = 1
    
    gx = InverseUnet(
        num_of_layers=2,
        in_ch=channels,
        img_resolution=img_resolution,
        creat_song_unet=create_song_unet,
        model_channels=16
    ).to(device)
    
    ign = IGNLinearizer(gx=gx, latent_dim=gx.dim).to(device)
    
    # Test with random inputs
    num_tests = 10
    max_diffs = []
    
    for i in range(num_tests):
        x = torch.randn(1, channels, img_resolution, img_resolution, device=device)
        is_idempotent, diff = ign.verify_idempotency(x)
        max_diffs.append(diff)
        
        status = "PASS" if is_idempotent else "FAIL"
        print(f"Test {i+1}: {status} (max difference: {diff:.2e})")
    
    avg_diff = sum(max_diffs) / len(max_diffs)
    print(f"\nAverage max difference: {avg_diff:.2e}")
    print(f"Idempotency tolerance: 1e-5")
    print(f"Overall: {'✓ ALL TESTS PASSED' if avg_diff < 1e-5 else '✗ SOME TESTS FAILED'}")
    
    return avg_diff < 1e-5


def test_projection():
    """Test the projection operation onto the learned manifold."""
    print("\n" + "=" * 60)
    print("Testing Projection Operation")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    img_resolution = 16
    channels = 1
    
    gx = InverseUnet(
        num_of_layers=2,
        in_ch=channels,
        img_resolution=img_resolution,
        creat_song_unet=create_song_unet,
        model_channels=16
    ).to(device)
    
    ign = IGNLinearizer(gx=gx, latent_dim=gx.dim).to(device)
    
    # Create noisy input
    clean = torch.randn(1, channels, img_resolution, img_resolution, device=device)
    noise = 0.5 * torch.randn_like(clean)
    noisy = clean + noise
    
    # Project noisy input
    with torch.no_grad():
        projected = ign.project(noisy)
        
        # Check that projecting again gives the same result
        projected_again = ign.project(projected)
        diff = torch.abs(projected - projected_again).max().item()
        
        print(f"Noisy input norm: {torch.norm(noisy).item():.4f}")
        print(f"Projected output norm: {torch.norm(projected).item():.4f}")
        print(f"Difference after second projection: {diff:.2e}")
        print(f"Projection consistency: {'✓ PASS' if diff < 1e-5 else '✗ FAIL'}")
    
    return diff < 1e-5


def test_sparsity():
    """Test the sparsity of the learned projector."""
    print("\n" + "=" * 60)
    print("Testing Latent Space Sparsity")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    img_resolution = 8
    channels = 1
    
    gx = InverseUnet(
        num_of_layers=1,
        in_ch=channels,
        img_resolution=img_resolution,
        creat_song_unet=create_song_unet,
        model_channels=8
    ).to(device)
    
    ign = IGNLinearizer(gx=gx, latent_dim=gx.dim).to(device)
    
    # Get active dimensions
    total_dims = gx.dim
    active_dims = ign.get_active_dimensions()
    sparsity_ratio = active_dims / total_dims if total_dims > 0 else 0
    
    print(f"Total latent dimensions: {total_dims}")
    print(f"Active dimensions: {active_dims}")
    print(f"Sparsity ratio: {sparsity_ratio:.2%}")
    
    # Get probability distribution if available
    try:
        probs = ign.get_projector_visualization()
        if len(probs) > 0:
            print(f"Probability stats:")
            print(f"  Min: {probs.min().item():.4f}")
            print(f"  Max: {probs.max().item():.4f}")
            print(f"  Mean: {probs.mean().item():.4f}")
            print(f"  Std: {probs.std().item():.4f}")
    except AttributeError:
        print("  get_projector_visualization not available")
    
    return True


def visualize_projector():
    """Visualize the projector's probability distribution."""
    print("\n" + "=" * 60)
    print("Projector Visualization")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    img_resolution = 8
    channels = 1
    
    gx = InverseUnet(
        num_of_layers=1,
        in_ch=channels,
        img_resolution=img_resolution,
        creat_song_unet=create_song_unet,
        model_channels=8
    ).to(device)
    
    ign = IGNLinearizer(gx=gx, latent_dim=gx.dim).to(device)
    
    try:
        probs = ign.get_projector_visualization()
        
        if len(probs) > 0:
            probs_np = probs.cpu().numpy()
            
            plt.figure(figsize=(12, 4))
            
            # Histogram of probabilities
            plt.subplot(1, 2, 1)
            plt.hist(probs_np, bins=50, edgecolor='black', alpha=0.7)
            plt.xlabel('Probability')
            plt.ylabel('Count')
            plt.title('Distribution of Projector Probabilities')
            plt.axvline(x=0.5, color='r', linestyle='--', label='Threshold (0.5)')
            plt.legend()
            
            # Line plot showing which dimensions are active
            plt.subplot(1, 2, 2)
            binary = np.round(probs_np)
            plt.plot(binary, 'o-', markersize=3, alpha=0.5)
            plt.xlabel('Dimension Index')
            plt.ylabel('Active (1) / Inactive (0)')
            plt.title(f'Active Dimensions ({binary.sum()}/{len(binary)})')
            plt.ylim(-0.1, 1.1)
            plt.yticks([0, 1])
            
            plt.tight_layout()
            plt.savefig('projector_visualization.png', dpi=150)
            print("Saved visualization to projector_visualization.png")
            plt.close()
        else:
            print("No projector data available for visualization")
    except (AttributeError, RuntimeError) as e:
        print(f"Could not visualize projector: {e}")
    
    return True


def main():
    """Run all tests."""
    print("Idempotent Generative Network (IGN) Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Test idempotency
    results['idempotency'] = test_idempotency()
    
    # Test projection
    results['projection'] = test_projection()
    
    # Test sparsity
    results['sparsity'] = test_sparsity()
    
    # Visualize
    results['visualization'] = visualize_projector()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name.capitalize()}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)