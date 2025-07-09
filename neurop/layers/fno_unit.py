"""FNO Unit Module."""
import torch
import torch.nn as nn

from torch.types import Tensor
from typing import Type, Union, List, Optional

from .spectral_convolution import SpectralConv
from .feature_mlp import ConvFeatureMLP
from .skip_connections import create_skip_connection, ConnectionType

class FNOUnit(nn.Module):
    """Single Fourier Neural Operator Unit.
    
    Consists of: Spectral Convolution -> Skip Connection -> Activation Function -> Feature MLP -> Skip Connection -> Activation Function.
    """

    in_features: int
    """Number of input features (channels)."""

    out_features: int
    """Number of output features (channels)."""

    modes: Union[int, List[int]]
    """Number of Fourier modes to consider in each spatial dimension."""

    n_dim: int
    """Number of spatial dimensions (2D, 3D, etc.)."""

    activation_function: Optional[nn.Module]
    """Activation function to apply after spectral convolution."""

    skip_connection_type: ConnectionType
    """Type of skip connection to use (e.g., 'soft-gating', 'identity')."""

    feature_mlp: Optional[nn.Module]
    """Feature MLP for additional processing of features."""

    spectral_conv: SpectralConv
    """Spectral convolution layer for global processing of features."""

    skip_connection: nn.Module
    """Skip connection module for combining input and transformed features."""

    feature_mlp_skip_connection: nn.Module
    """Skip connection module for combining input and transformed features after the feature MLP."""

    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 modes: Union[int, List[int]], 
                 n_dim: int,
                 activation_function: Optional[Type[nn.Module]] = None,
                 conv_module: Type[SpectralConv] = SpectralConv,
                 skip_connection: ConnectionType = 'soft-gating',
                 use_feature_mlp: bool = True,
                 feature_mlp_module: Type[nn.Module] = ConvFeatureMLP,
                 feature_mlp_depth: int = 2,
                 feature_mlp_skip_connection: ConnectionType = 'soft-gating',
                 feature_expansion_factor: float = 1.0,
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
            activation_function (Optional[Type[nn.Module]]): Activation function to apply after spectral convolution. Defaults to None.
            conv_module (Type[SpectralConv]): Spectral convolution module to use.
            skip_connection (ConnectionType): Type of skip connection to use.
            use_feature_mlp (bool): Whether to use a feature MLP for additional processing.
            feature_mlp_module (Type[nn.Module]): Feature MLP module to use for additional processing.
            feature_mlp_depth (int): Depth of the feature MLP.
            feature_mlp_skip_connection (ConnectionType): Type of skip connection to use for the feature MLP.
            feature_expansion_factor (float): Factor by which to expand the features in the feature MLP.
            bias (bool): Whether to include bias parameters in the skip connection.
            init_scale (float): Scale for initializing the weights of the spectral convolution layer.
            dtype (torch.dtype): Data type for the spectral convolution layer output, typically complex (torch.cfloat).

        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.modes = modes
        self.n_dim = n_dim
        self.activation_function = activation_function() if activation_function is not None else None
        self.skip_connection_type = skip_connection

        if use_feature_mlp:
            self.feature_mlp = feature_mlp_module(
                in_features=in_features,
                hidden_features=int(feature_expansion_factor * in_features),
                out_features=out_features,
                depth=feature_mlp_depth,
                n_dim=n_dim,
                activation_function=activation_function if activation_function is not None else torch.nn.ReLU,
            )

            self.feature_mlp_skip_connection = create_skip_connection(
                in_features=in_features,
                out_features=out_features,
                n_dim=n_dim,
                bias=bias,
                connection_type=feature_mlp_skip_connection
            )

        else: 
            self.feature_mlp = None
        
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

        Convolution -> Skip Connection -> Activation -> FeatureMLP -> Skip Connection -> Activation -> Output
        
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
        
        # Apply skip connection
        x = self.skip_connection(x = residual, transformed_x = x)

        if self.activation_function is not None:
            x = self.activation_function(x)
        
        new_residual = x

        if self.feature_mlp is not None:
            # FeatureMLP 
            x = self.feature_mlp(x)

            # FeatureMLP Skip Connection
            x = self.feature_mlp_skip_connection(x = new_residual, transformed_x = x)

            if self.activation_function is not None:
                x = self.activation_function(x)
            
        return x


