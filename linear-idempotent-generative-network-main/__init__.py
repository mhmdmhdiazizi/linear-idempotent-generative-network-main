"""
Idempotent Generative Network (IGN) Package.

This package implements the Linear IGN architecture from the paper:
"Idempotent Generative Networks" (Shocher et al., 2024)

Key Features:
- Exact idempotency through architectural design (f(f(x)) = f(x))
- Binary projector matrix using Straight-Through Estimator (STE)
- Invertible encoder/decoder networks
- Training losses: reconstruction, sparsity, and isometry

Example usage:
    from ign import IGNLinearizer, IdempotentLinearModule
    from ign.modules import IGNInverseUnet
    
    # Create invertible encoder/decoder
    gx = IGNInverseUnet(num_of_layers=4, in_ch=3, img_resolution=64, 
                        creat_song_unet=create_song_unet)
    
    # Create IGN linearizer with idempotent projector
    ign = IGNLinearizer(gx=gx, latent_dim=gx.dim)
    
    # Training loop
    for x in dataloader:
        losses = ign.compute_losses(x, lambda_rec=1.0, lambda_sparse=0.1, 
                                    lambda_isometry=0.001)
        loss = losses['total']
        loss.backward()
        optimizer.step()
    
    # Inference: project onto learned manifold
    projected = ign.project(noisy_image)
    
    # Verify idempotency
    is_idempotent, diff = ign.verify_idempotency(test_input)
"""

from .modules import (
    IdempotentLinearModule,
    IGNLinearizer,
    InverseUnet,
    InvUnet,
)

__version__ = '1.0.0'
__all__ = [
    'IdempotentLinearModule',
    'IGNLinearizer', 
    'IGNInverseUnet',
    'InvUnet',
    'ConditionalInvUnet',
]