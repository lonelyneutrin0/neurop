"""Skip connections for neural operators."""
import torch 
import torch.nn as nn

from torch.types import Tensor
from typing import Literal, Optional, Union

Connection = Union['SoftGatingConnection', 'IdentityConnection', 'ConvConnection']
ConnectionType = Literal['identity', 'soft-gating', 'conv']
        
class SoftGatingConnection(nn.Module): 
    """Soft-gating connection that applies a learnable gating mechanism to the skip connection."""

    in_features: int
    """Number of input features"""
    
    weight: nn.Parameter
    """Weight matrix for the connection"""

    bias: Optional[nn.Parameter]
    """Optional bias to be added to the skip connection layer"""

    def __init__(self, in_features: int, n_dim: int, bias: bool = False):
        """Initialize the soft-gating connection.
        
        Args:
            in_features (int): Number of input features.
            n_dim (int): Number of spatial dimensions.
            bias (bool): Whether to include a bias term in the connection.

        """
        super().__init__() 

        self.in_features = in_features
        self.weight = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, self.in_features, *(1,) * n_dim))
        else:
            self.bias = None
        
    def forward(self, x: Tensor, transformed_x: Tensor) -> Tensor:
        """Forward pass for the soft-gating connection.

        Args:
            x (Tensor): The input tensor [B, C, d_1, d_2, ...]
            transformed_x (Tensor) : The transformed input tensor [B, C, d_1, d_2, ...]

        Returns:
            Tensor [B, C, d_1, d_2, ....]

        """
        if self.bias is not None:
            return self.weight * x + self.bias + transformed_x

        return self.weight * x + transformed_x

class ConvConnection(nn.Module): 
    """Convolution Skip Connection Layer."""

    def __init__(self, in_features: int, out_features: int, n_kernel: int, bias=False):
        """Convolution based skip connection.
        
        Arguments:
        in_features (int): The number of input features 
        out_features (int) : The number of output features 
        n_kernel (int): The size of the kernel 
        bias (bool): Bias of the layer

        """
        super().__init__() 
        self.layer = nn.Conv1d(in_features, out_features, n_kernel, bias=bias)
    
    def forward(self, x: torch.Tensor, transformed_x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the convolutional skip connection.

        Args:
            x (Tensor): Input tensor 
            transformed_x (Tensor): The transformed input tensor

        Returns:
            Tensor: Output tensor

        """
        size = list(x.shape)

        x = x.view(*size[:2], -1)
        x = self.layer(x)   

        x = x.view(size[0], self.layer.out_channels, *size[2:])

        return x + transformed_x

class IdentityConnection(nn.Module):
    """Identity connection that returns the transformed input.

    This maintains consistent interface with other skip connections.
    """
    
    def forward(self, x: Tensor, transformed_x: Tensor) -> Tensor:
        """Forward pass for the identity connection.

        Args:
            x (Tensor): Input tensor (ignored).
            transformed_x (Tensor): Transformed input tensor.

        Returns:
            Tensor: The transformed input tensor.

        """
        return transformed_x

def create_skip_connection(
    in_features: int, 
    out_features: int, 
    n_dim: int = -1, 
    n_kernel: int = -1, 
    bias: bool = False, 
    connection_type: ConnectionType = 'soft-gating',
) -> Connection: 
    """Create a skip connection module.
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features (optional, defaults to in_features)
        n_dim (int): Number of spatial dimensions (for soft-gating)
        n_kernel (int): Kernel size (for convolution)
        bias (bool): Whether to include bias parameters
        connection_type (ConnectionType): Type of connection ('identity', 'soft-gating', 'conv')

    Returns:
        The appropriate skip connection module

    """
    if connection_type == 'identity':
        return IdentityConnection()
    
    if connection_type == 'soft-gating':
        if n_dim == -1:
            raise ValueError('Specify the number of spatial dimensions for the soft-gating connection.')
        return SoftGatingConnection(in_features=in_features, n_dim=n_dim, bias=bias)

    if connection_type == 'conv':
        if n_kernel == -1: 
            raise ValueError('Specify a kernel size for the convolution skip connection.')
        return ConvConnection(in_features=in_features, out_features=out_features, n_kernel=n_kernel, bias=bias)

    raise ValueError(f"Unknown connection type: {connection_type}")
