"""Skip connections for neural operators."""
import torch 
import torch.nn as nn

from typing import Optional

from abc import ABC, abstractmethod

from enum import Enum

class ConnectionType(Enum): 
    SOFT_GATING = 'soft-gating'
    CONV = 'conv'
    IDENTITY = 'identity'
class SkipConnection(nn.Module, ABC): 

    def __init__(self, *args, **kwargs): 
        """__init__ method for Skip Connections."""
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward method for Skip Connections."""
        pass

class SoftGatingConnection(SkipConnection): 
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
        
    def forward(self, x: torch.Tensor, transformed_x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the soft-gating connection.

        Args:
            x (torch.Tensor): The input tensor [B, C, d_1, d_2, ...]
            transformed_x (torch.Tensor) : The transformed input tensor [B, C, d_1, d_2, ...]

        Returns:
            torch.Tensor [B, C, d_1, d_2, ....]

        """
        if self.bias is not None:
            return self.weight * x + self.bias + transformed_x

        return self.weight * x + transformed_x

class ConvConnection(SkipConnection): 
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
            x (torch.Tensor): Input tensor 
            transformed_x (torch.Tensor): The transformed input tensor

        Returns:
            torch.Tensor: Output tensor

        """
        size = list(x.shape)

        x = x.view(*size[:2], -1)
        x = self.layer(x)   

        x = x.view(size[0], self.layer.out_channels, *size[2:])

        return x + transformed_x

class IdentityConnection(SkipConnection):
    """Identity connection that returns the transformed input.

    This maintains consistent interface with other skip connections.
    """

    def forward(self, x: torch.Tensor, transformed_x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the identity connection.

        Args:
            x (torch.Tensor): Input tensor (ignored).
            transformed_x (torch.Tensor): Transformed input tensor.

        Returns:
            torch.Tensor: The transformed input tensor.

        """
        return transformed_x

def create_skip_connection(
    in_features: int, 
    out_features: int, 
    n_dim: int = -1, 
    n_kernel: int = -1, 
    bias: bool = False, 
    connection_type: ConnectionType = ConnectionType.SOFT_GATING,
) -> SkipConnection: 
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
    if connection_type == ConnectionType.IDENTITY:
        return IdentityConnection()

    if connection_type == ConnectionType.SOFT_GATING:
        if n_dim == -1:
            raise ValueError('Specify the number of spatial dimensions for the soft-gating connection.')
        return SoftGatingConnection(in_features=in_features, n_dim=n_dim, bias=bias)

    if connection_type == ConnectionType.CONV:
        if n_kernel == -1: 
            raise ValueError('Specify a kernel size for the convolution skip connection.')
        return ConvConnection(in_features=in_features, out_features=out_features, n_kernel=n_kernel, bias=bias)

    raise ValueError(f"Unknown connection type: {connection_type}")
