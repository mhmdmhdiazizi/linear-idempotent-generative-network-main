from typing_extensions import override
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from .linearizer import Linearizer, G


class IdempotentLinearModule(nn.Module):
    """
    Idempotent linear module using Straight-Through Estimator (STE).
    
    Implements the binary projector matrix A = Q * Λ * Q^(-1) where Λ is 
    a diagonal matrix with binary entries {0, 1}. The matrices Q, Q^(-1) 
    are absorbed into g, g^(-1), so we train a Linearizer g^(-1)(Λ g(·)).
    
    Uses STE for differentiable binarization:
    - Forward: rounded values (0 or 1)
    - Backward: continuous gradients through underlying probabilities
    """
    
    def __init__(self, dim: int, learnable_probs: bool = True):
        """
        Initialize the idempotent linear module.
        
        Args:
            dim: Dimension of the latent space
            learnable_probs: Whether to learn probabilities or compute from input
        """
        super().__init__()
        self.dim = dim
        self.learnable_probs = learnable_probs
        
        if learnable_probs:
            # Learnable probability parameters for each dimension
            self.logits = nn.Parameter(torch.zeros(dim))
        else:
            # Predict probabilities from input
            self.prob_net = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim),
                nn.Sigmoid()
            )
    
    def _get_binary_projector(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Get binary projector matrix using Straight-Through Estimator.
        
        Args:
            probs: Probability tensor in [0, 1]
            
        Returns:
            Binary diagonal matrix (as vector for efficiency)
        """
        # Round to get binary values for forward pass
        binary = torch.round(probs)
        # STE: forward uses rounded values, backward uses continuous probs
        return binary + probs - probs.detach()
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply idempotent projection: f(x) = g^{-1}(Λ g(x)).
        
        Since Q, Q^{-1} are absorbed into g, g^{-1}, we just apply Λ.
        
        Args:
            x: Input tensor [B, D] or [B, C, H, W] (flattened internally)
            
        Returns:
            Projected output with same shape as input
        """
        original_shape = x.shape
        
        # Flatten to [B, D]
        if x.dim() > 2:
            x_flat = x.view(x.shape[0], -1)
        else:
            x_flat = x
        
        # Ensure dimension matches
        if x_flat.shape[1] != self.dim:
            # Pad or truncate to match
            if x_flat.shape[1] < self.dim:
                x_padded = torch.zeros(x_flat.shape[0], self.dim, 
                                       device=x.device, dtype=x.dtype)
                x_padded[:, :x_flat.shape[1]] = x_flat
                x_flat = x_padded
            else:
                x_flat = x_flat[:, :self.dim]
        
        # Get probabilities
        if self.learnable_probs:
            probs = torch.sigmoid(self.logits).unsqueeze(0).expand(x_flat.shape[0], -1)
        else:
            probs = self.prob_net(x_flat)
        
        # Get binary projector via STE
        Lambda = self._get_binary_projector(probs)
        
        # Apply projection: element-wise multiplication (diagonal matrix)
        y_flat = x_flat * Lambda
        
        # Restore original shape
        if len(original_shape) > 2:
            # Need to reshape back, accounting for potential padding
            if original_shape[1:].numel() <= self.dim:
                y = y_flat[:, :original_shape[1:].numel()].view(original_shape)
            else:
                y = y_flat.view(x.shape[0], *original_shape[1:])
        else:
            y = y_flat[:, :original_shape[1]]
        
        return y
    
    def get_sparsity_loss(self) -> torch.Tensor:
        """
        Compute sparsity loss: L_sparse = (1/N) * Rank(A) = (1/N) * sum(λ_i).
        
        This encourages the model to use the smallest possible latent manifold.
        
        Returns:
            Sparsity loss scalar
        """
        if self.learnable_probs:
            probs = torch.sigmoid(self.logits)
        else:
            # For non-learnable, we need a representative input
            # This is typically called during training with actual inputs
            return torch.tensor(0.0, device=self.logits.device if self.learnable_probs else torch.device('cpu'))
        
        # Expected rank = sum of probabilities (soft relaxation for training)
        return probs.mean()
    
    def get_projector_rank(self) -> float:
        """Get the current rank of the projector (number of active dimensions)."""
        if self.learnable_probs:
            probs = torch.sigmoid(self.logits)
            binary = torch.round(probs)
            return binary.sum().item()
        return 0.0


class IGNLinearizer(Linearizer):
    """
    Idempotent Generative Network (IGN) Linearizer.
    
    Implements the Linear IGN architecture that enforces exact idempotency
    through architectural design rather than approximate optimization.
    
    Key components:
    - Invertible encoder g: maps data to latent space
    - Binary projector Λ: diagonal matrix with {0, 1} entries (via STE)
    - Invertible decoder g^{-1}: maps latent back to data space
    
    The function f(x) = g^{-1}(Λ g(x)) is exactly idempotent because Λ^2 = Λ.
    
    Training losses:
    - L_rec = ||f(x) - x||^2: Reconstruction (data should be fixed points)
    - L_sparse = (1/N) * sum(λ_i): Sparsity (tightest latent manifold)
    - L_isometry = |||g(x) - g(0)||^2 - ||x||^2||_1: Isometry regularization
    """
    
    def __init__(self, gx: G, linear_network: Optional[nn.Module] = None, 
                 gy: G = None, latent_dim: Optional[int] = None,
                 learnable_projector: bool = True):
        """
        Initialize IGN Linearizer.
        
        Args:
            gx: Encoder network g
            linear_network: Optional custom linear network (if None, creates IdempotentLinearModule)
            gy: Decoder network (defaults to gx if None)
            latent_dim: Latent space dimension (required if linear_network is None)
            learnable_projector: Whether to use learnable probability parameters
        """
        # Create idempotent linear module if not provided
        if linear_network is None:
            if latent_dim is None:
                latent_dim = gx.dim
            linear_network = IdempotentLinearModule(latent_dim, learnable_projector)
        
        super().__init__(gx=gx, linear_network=linear_network, gy=gy)
        
        # Store reference to projector for loss computation
        if isinstance(linear_network, IdempotentLinearModule):
            self.projector = linear_network
        else:
            self.projector = None
    
    @override
    def gx(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through g_x (encoding from data space to latent space)."""
        return self.net_gx(x, mode='gx', **kwargs)
    
    @override
    def gy(self, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass through g_y (encoding from target space to latent space)."""
        return self.net_gy(y, mode='gy', **kwargs)
    
    @override
    def gx_inverse(self, g_x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Inverse pass through g_x (decoding from latent space to data space)."""
        return self.net_gx.inverse(g_x, mode='gx', **kwargs)
    
    @override
    def gy_inverse(self, g_y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Inverse pass through g_y (decoding from latent space to target space)."""
        return self.net_gy.inverse(g_y, mode='gy', **kwargs)
    
    def A(self, g_x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply the idempotent projector A (which is Λ in our case).
        
        Args:
            g_x: Encoded latent representation
            **kwargs: Additional arguments
            
        Returns:
            Projected latent representation
        """
        if isinstance(self.linear_network, IdempotentLinearModule):
            return self.linear_network(g_x, **kwargs)
        else:
            return self.linear_network(g_x, **kwargs)
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass: f(x) = g^{-1}(Λ g(x)).
        
        Due to idempotency, f(f(x)) = f(x), meaning outputs are fixed points.
        
        Args:
            x: Input image
            **kwargs: Additional arguments
            
        Returns:
            Projected output (should equal input for training data)
        """
        g_x = self.gx(x, **kwargs)
        g_y = self.A(g_x, **kwargs)
        y_pred = self.gy_inverse(g_y, **kwargs)
        return y_pred
    
    def compute_losses(self, x: torch.Tensor, 
                       lambda_rec: float = 1.0,
                       lambda_sparse: float = 0.1,
                       lambda_isometry: float = 0.001) -> Dict[str, torch.Tensor]:
        """
        Compute all IGN training losses.
        
        Args:
            x: Input batch
            lambda_rec: Weight for reconstruction loss
            lambda_sparse: Weight for sparsity loss
            lambda_isometry: Weight for isometry regularization
            
        Returns:
            Dictionary of losses
        """
        # Encode input
        g_x = self.gx(x)
        
        # Apply projector
        g_y = self.A(g_x)
        
        # Decode
        x_recon = self.gy_inverse(g_y)
        
        # Reconstruction loss: L_rec = ||f(x) - x||^2
        loss_rec = F.mse_loss(x_recon, x)
        
        # Sparsity loss: L_sparse = (1/N) * Rank(A)
        if self.projector is not None:
            loss_sparse = self.projector.get_sparsity_loss()
        else:
            loss_sparse = torch.tensor(0.0, device=x.device)
        
        # Isometry loss: L_isometry = |||g(x) - g(0)||^2 - ||x||^2||_1
        with torch.no_grad():
            zero_input = torch.zeros_like(x[:1])
        g_zero = self.gx(zero_input.expand_as(x))
        
        norm_diff_latent = torch.norm(g_x - g_zero, dim=list(range(1, g_x.dim())), p=2) ** 2
        norm_diff_data = torch.norm(x, dim=list(range(1, x.dim())), p=2) ** 2
        
        loss_isometry = F.l1_loss(norm_diff_latent, norm_diff_data)
        
        # Total loss
        total_loss = (lambda_rec * loss_rec + 
                     lambda_sparse * loss_sparse + 
                     lambda_isometry * loss_isometry)
        
        return {
            'total': total_loss,
            'reconstruction': loss_rec,
            'sparsity': loss_sparse,
            'isometry': loss_isometry
        }
    
    def verify_idempotency(self, x: torch.Tensor, tol: float = 1e-5) -> Tuple[bool, float]:
        """
        Verify that f(f(x)) ≈ f(x) (idempotency property).
        
        Args:
            x: Input tensor
            tol: Tolerance for equality check
            
        Returns:
            Tuple of (is_idempotent, max_difference)
        """
        fx = self.forward(x)
        ffx = self.forward(fx)
        
        diff = torch.abs(ffx - fx).max().item()
        is_idempotent = diff < tol
        
        return is_idempotent, diff
    
    @torch.no_grad()
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input onto the learned data manifold.
        
        This is the main inference operation. Due to idempotency,
        applying this multiple times gives the same result.
        
        Args:
            x: Input image (can be noisy, corrupted, or out-of-distribution)
            
        Returns:
            Projected image on the learned manifold
        """
        self.eval()
        return self.forward(x)
    
    @torch.no_grad()
    def get_active_dimensions(self) -> int:
        """Get the number of active (non-zero) dimensions in the projector."""
        if self.projector is not None:
            return int(self.projector.get_projector_rank())
        return 0
    
    def get_projector_visualization(self) -> torch.Tensor:
        """
        Get visualization of the projector's probability distribution.
        
        Returns:
            Tensor of probabilities for each dimension
        """
        if self.projector is not None and self.projector.learnable_probs:
            return torch.sigmoid(self.projector.logits)
        return torch.tensor([])
