"""Positional Embeddings for Neural Operators."""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn 
from torch.types import Device

from typing import List, Optional

class Embedding(nn.Module, ABC): 
    """Abstract class for positional embeddings."""

    def __init__(self, *args, **kwargs):
        """__init__ method for Embedding Abstract Class."""
        super().__init__(*args, **kwargs)
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass for the embedding."""
        pass


class GridEmbedding(Embedding):
    """Grid embedding for Neural Operators."""

    in_features: int 
    """Number of input features of the embedding"""

    dim: int
    """Number of spatial dimensions of the embedding"""

    domain: List[List[float]]
    """The grid boundaries of the embedding"""

    def __init__(self, in_features: int, dim: int, domain: List[List[float]]): 
        """__init__ method for GridEmbedding class."""
        super().__init__()

        self.in_features = in_features
        self.dim = dim
        self.domain = domain
        self._grid = None 
        self._res: Optional[torch.Size] = None
        self.out_channels = self.in_features + self.dim

    def grid(self, n_dim: torch.Size, device: Device, dtype: torch.dtype):
        """Generate an N-dimensional grid for positional encoding.
        
        Arguments:
            n_dim (int): Number of dimensions for the grid.
            device (torch.device): Device to place the grid on.
            dtype (torch.dtype): Data type of the grid tensor.
        
        Returns:
            torch.Tensor: The generated grid tensor.

        """
        if self._grid is None or self._res != n_dim: 
            grids = make_regular_grid(n_dim, domain = self.domain)

            grids = [x.to(device).to(dtype).unsqueeze(0).unsqueeze(0) for x in grids]
            self._grid = grids
            self._res = n_dim
        
        return self._grid
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """Forward pass for the grid embedding.
        
        Arguments:
            x (torch.Tensor): Input tensor of shape (B, C, d_1, d_2, ...).
            
        Returns:
            torch.Tensor: The output tensor of shape (B, C + D, d_1, d_2, ...).

        """
        batch_size = x.shape[0]

        grids = self.grid(n_dim = x.shape[2:], device=x.device, dtype=x.dtype)
        grids = [x.repeat(batch_size, *[1] * (self.dim + 1)) for x in grids]
        out = torch.cat((x, *grids), dim=1)

        return out

def make_regular_grid(resolutions: torch.Size, domain: List[List[float]]):
    """Generate an n-dimensional regular grid.

    Arguments:
        resolutions: torch.Size
            A torch.Size object representing the resolutions for the grid in each dimension.
        domain: List[List[float]]
            A list of boundaries for the grid in each dimension.

    Returns:
        List[torch.Tensor]: A list of tensors representing the grid in each dimension.
    
    """
    if len(resolutions) != len(domain):
        raise ValueError("The number of resolutions must match the number of dimensions in the domain.")
    
    dim = len(resolutions)

    meshgrid_i = [] 

    for res, (start, stop) in zip(resolutions, domain):
        meshgrid_i.append(torch.linspace(start, stop, res+1)[:-1])
    
    grid = torch.meshgrid(*meshgrid_i, indexing='ij')
    grid = tuple([x.repeat([1]*dim) for x in grid])
    
    return grid