import math

import torch
from torch import nn
from torch.nn import functional as F


class PositionalEmbedding(nn.Module):
    """
    Positional embedding for normalized coordinates using Fourier features.
    
    Args:
        coord_dim (int): Dimension of input coordinates (e.g., 6 for 3D bounding box)
        emb_dim (int): Output embedding dimension
        num_frequencies: Number of frequency bands for Fourier features
    """
    def __init__(self, coord_dim: int = 6, emb_dim: int = 512, num_frequencies: int = 10) -> None:
        super().__init__()
        self.coord_dim = coord_dim
        self.emb_dim = emb_dim
        self.num_frequencies = num_frequencies
        
        # Fourier feature frequencies (learnable or fixed)
        # Fixed frequencies: more stable
        freq_bands = torch.pow(2, torch.linspace(0, num_frequencies - 1, num_frequencies))
        self.register_buffer('freq_bands', freq_bands)
        
        # MLP to project Fourier features to embedding dimension
        fourier_dim = coord_dim * num_frequencies * 2  # *2 for sin and cos
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )
    
    def forward(self, coords):
        """
        Args:
            coords: [B, coord_dim] normalized coordinates in [0, 1]
        
        Returns:
            embeddings: [B, emb_dim]
        """
        B = coords.shape[0]
        
        # Create Fourier features
        # coords: [B, coord_dim] -> [B, coord_dim, 1]
        coords = coords.unsqueeze(-1)
        
        # Multiply by frequencies: [B, coord_dim, num_frequencies]
        scaled = coords * self.freq_bands.view(1, 1, -1) * 2 * math.pi
        
        # Apply sin and cos
        fourier_features = torch.cat([
            torch.sin(scaled),
            torch.cos(scaled)
        ], dim=-1)  # [B, coord_dim, num_frequencies * 2]
        
        # Flatten: [B, coord_dim * num_frequencies * 2]
        fourier_features = fourier_features.view(B, -1)
        
        # Project to embedding dimension
        embeddings = self.mlp(fourier_features)
        
        return embeddings


# Alternative: Simpler MLP-based version
class PositionalEmbeddingSimple(nn.Module):
    """
    Simple MLP-based positional embedding for coordinates.
    
    Args:
        coord_dim: Dimension of input coordinates
        emb_dim: Output embedding dimension
        hidden_dim: Hidden layer dimension
    """
    def __init__(self, coord_dim=6, emb_dim=512):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(coord_dim, emb_dim // 2),
            nn.SiLU(),
            nn.Linear(emb_dim // 2, emb_dim // 2),
            nn.SiLU(),
            nn.Linear(emb_dim // 2, emb_dim)
        )
    
    def forward(self, coords):
        """
        Args:
            coords: [B, coord_dim] normalized coordinates in [0, 1]
        
        Returns:
            embeddings: [B, emb_dim]
        """
        return self.mlp(coords)
