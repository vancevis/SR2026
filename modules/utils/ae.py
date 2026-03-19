"""
Autoencoder for compressing CLIP features
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List


class Autoencoder(nn.Module):
    """
    Autoencoder for compressing 512D CLIP features to 32D
    """
    
    def __init__(
        self,
        input_dim=512,
        latent_dim=3,
        encoder_hidden_dims: List[int] = None,
        decoder_hidden_dims: List[int] = None
    ):
        super().__init__()
        
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [256, 128, 64, 32, 3]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = [32, 64, 128, 256, 512]
        
        # Build encoder
        encoder_layers = []
        for i in range(len(encoder_hidden_dims)):
            if i == 0:
                encoder_layers.append(nn.Linear(512, encoder_hidden_dims[i]))
            else:
                encoder_layers.append(nn.GroupNorm(2, encoder_hidden_dims[i-1]))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Linear(encoder_hidden_dims[i-1], encoder_hidden_dims[i]))
        self.encoder = nn.ModuleList(encoder_layers)
        
        # Build decoder
        decoder_layers = []
        for i in range(len(decoder_hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.Linear(encoder_hidden_dims[-1], decoder_hidden_dims[i]))
            else:
                decoder_layers.append(nn.GroupNorm(2, decoder_hidden_dims[i-1]))
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Linear(decoder_hidden_dims[i-1], decoder_hidden_dims[i]))
        self.decoder = nn.ModuleList(decoder_layers)
        
        self.latent_dim = encoder_hidden_dims[-1]
        
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode features
        
        Args:
            x: [N, 512] CLIP features
        
        Returns:
            z: [N, latent_dim] compressed features
        """
        for layer in self.encoder:
            x = layer(x)
        # Normalize latent vector to unit sphere (matches 3DitScene)
        x = torch.nn.functional.normalize(x, dim=-1)
        return x
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent features
        
        Args:
            z: [N, latent_dim] compressed features
        
        Returns:
            x: [N, 512] reconstructed features
        """
        for layer in self.decoder:
            z = layer(z)
        return z
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass
        
        Args:
            x: [N, 512] input features
        
        Returns:
            z: [N, latent_dim] compressed features
            x_recon: [N, 512] reconstructed features
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return z, x_recon


class AutoencoderDataset(Dataset):
    """Dataset for autoencoder training"""
    
    def __init__(self, data):
        """
        Args:
            data: [N, 512] numpy array of CLIP features
        """
        self.data = data
    
    def __getitem__(self, index: int):
        return torch.tensor(self.data[index], dtype=torch.float32)
    
    def __len__(self):
        return self.data.shape[0]
