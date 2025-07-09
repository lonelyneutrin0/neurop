"""FNO Unit Module."""
import torch
import torch.nn as nn

from torch.types import Tensor
from typing import Type, Union, List

from .spectral_convolution import SpectralConv
from .skip_connections import create_skip_connection, ConnectionType

class FNOUnit(nn.Module):
    """Single Fourier Neural Operator Unit.
    
    Consists of: Spectral Convolution -> Activation -> Skip Connection
    """

    in_features: int
    """Number of input features (channels)."""

    out_features: int
    """Number of output features (channels)."""

    modes: Union[int, List[int]]
    """Number of Fourier modes to consider in each spatial dimension. Can be a single integer or a list of integers."""

    n_dim: int
    """Number of spatial dimensions (2D, 3D, etc.)."""

    activation_function: nn.Module
    """Activation function to apply after spectral convolution."""

    skip_connection_type: ConnectionType
    """Type of skip connection to use (identity, linear, soft-gating, residual).""" 

    conv_module: Type[SpectralConv]
    """Spectral convolution module to use. Defaults to SpectralConv."""

    bias: bool
    """Whether to include bias parameters in the skip connection."""

    init_scale: float
    """Scale for initializing the weights of the spectral convolution layer."""

    dtype: torch.dtype
    """Data type for the spectral convolution layer output, typically complex (torch.cfloat)."""

    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 modes: Union[int, List[int]], 
                 n_dim: int,
                 activation_function: nn.Module = nn.ReLU(),
                 conv_module: Type[SpectralConv] = SpectralConv,
                 skip_connection: ConnectionType = 'soft-gating',
                 bias: bool = True,
                 init_scale: float = 1.0,
                 dtype: torch.dtype = torch.cfloat
                 ):
        """Initialize the FNO unit with the given parameters.
        
        Args:
            in_features (int): Number of input features (channels).
            out_features (int): Number of output features (channels).
            modes (Union[int, List[int]]): Number of Fourier modes to consider in each spatial dimension.
            n_dim (int): Number of spatial dimensions (2D, 3D, etc.).
            activation_function (nn.Module): Activation function to apply after spectral convolution.
            conv_module (Type[SpectralConv]): Spectral convolution module to use.
            skip_connection (ConnectionType): Type of skip connection to use.
            bias (bool): Whether to include bias parameters in the skip connection.
            init_scale (float): Scale for initializing the weights of the spectral convolution layer.
            dtype (torch.dtype): Data type for the spectral convolution layer output, typically complex (torch.cfloat).

        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.modes = modes
        self.n_dim = n_dim
        self.activation_function = activation_function
        self.skip_connection_type = skip_connection

        # Spectral convolution layer
        self.spectral_conv = conv_module(
            in_features=in_features,
            out_features=out_features,
            modes=modes,
            init_scale=init_scale, 
            dtype=dtype
        )
        
        # Skip connection
        self.skip_connection = create_skip_connection(
            in_features=in_features,
            out_features=out_features,
            n_dim=n_dim,
            bias=bias,
            connection_type=skip_connection
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through single FNO unit.
        
        Args:
            x (Tensor): Input tensor of shape (B, in_features, *spatial_dims)
            
        Returns:
            Tensor: Output tensor of shape (B, out_features, *spatial_dims)

        """
        if x.shape[1] != self.in_features:
            raise ValueError(f"Expected {self.in_features} input features, got {x.shape[1]}")

        # Store input for skip connection
        residual = x
        
        # Apply spectral convolution
        x = self.spectral_conv(x)
        
        # Apply activation function
        x = self.activation_function(x)
        
        # Apply skip connection
        x = self.skip_connection(residual, x)
        
        return x


