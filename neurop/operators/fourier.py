"""Fourier Neural Operator (FNO) implementation for learning mappings between functions in the Fourier domain."""
from torch.types import Tensor
from typing import List, Union, Type, Optional
from ..layers.skip_connections import ConnectionType

import torch
import torch.nn as nn

from ..base import NeuralOperator
from ..layers.spectral_convolution import SpectralConv, NormType
from ..layers.fno_unit import FNOUnit
from ..layers.feature_mlp import ConvFeatureMLP, LinearFeatureMLP
class FourierOperator(NeuralOperator):
    """ 
    Fourier Neural Operator (FNO) for learning mappings between functions in the Fourier domain.

    This operator uses spectral convolutions to learn the mapping between input and output functions.
    """

    readin: LinearFeatureMLP
    """Layer to read input features and project them to hidden features."""

    fno_units: nn.ModuleList
    """List of FNO units that apply spectral convolutions and activation functions."""

    readout: LinearFeatureMLP
    """Layer to read output features and project them to the final output features."""

    spectral_normalizer: Optional[nn.Module]
    """Normalizer to apply to the spectral convolution output."""

    feature_normalizer: Optional[nn.Module]
    """Normalizer to apply to the feature MLP output."""

    learnable_normalizers: bool
    """Whether the normalization parameters are learnable."""

    normalizer_eps: float
    """Epsilon value for numerical stability in normalization layers."""

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 modes: Union[int, List[int]],
                 n_dim: int,
                 depth: int = 4,
                 activation_function: Type[nn.Module] = nn.ReLU,
                 conv_module: Type[SpectralConv] = SpectralConv,
                 skip_connections: Union[ConnectionType, List[ConnectionType]] = 'soft-gating',
                 use_feature_mlp: bool = True,
                 feature_mlp_module: Type[nn.Module] = ConvFeatureMLP,
                 feature_mlp_depth: int = 4,
                 feature_mlp_skip_connections: Union[ConnectionType, List[ConnectionType]] = 'soft-gating',
                 feature_expansion_factor: float = 1.0,
                 bias: bool = True,
                 init_scale: float = 1.0,
                 dtype: torch.dtype = torch.cfloat,
                 norm: NormType = 'ortho',
                 spectral_normalizer: Optional[Type[nn.Module]] = None, 
                 feature_normalizer: Optional[Type[nn.Module]] = None,
                 *, 
                 learnable_normalizers: bool = True,
                 normalizer_eps: float = 1e-10):
        """Initialize the FourierOperator with the given parameters.

        Args:
            in_features (int): Number of input features (channels).
            hidden_features (int): Number of hidden features (channels).
            out_features (int): Number of output features (channels).
            modes (Union[int, List[int]]): Number of Fourier modes to consider in each spatial dimension.
            n_dim (int): Number of spatial dimensions (2D, 3D, etc.).
            depth (int): Number of FNO units in the network.
            activation_function (torch.nn.Module): Activation function to apply after spectral convolution.
            conv_module (Type[SpectralConv]): Spectral convolution module to use.
            skip_connections (Union[ConnectionType, List[ConnectionType]]): Type of skip connection to use.
            use_feature_mlp (bool): Whether to use a feature MLP for additional processing.
            feature_mlp_module (Type[nn.Module]): Feature MLP module to use for additional processing.
            feature_mlp_depth (int): Depth of the feature MLP.
            feature_mlp_skip_connections (Union[ConnectionType, List[ConnectionType]]): Type of skip connection to use for the feature MLP.
            feature_expansion_factor (float): Factor by which to expand the features in the feature MLP.
            bias (bool): Whether to include bias parameters in the skip connection.
            init_scale (float): Scale for initializing the weights of the spectral convolution layer.
            dtype (torch.dtype): Data type for the spectral convolution layer output, typically complex (torch.cfloat).
            norm (NormType): Normalization type for the spectral convolution layer, can be 'backward', 'forward', or 'ortho'.
            spectral_normalizer (Optional[Type[nn.Module]]): Normalizer class to apply to the spectral convolution output.
            feature_normalizer (Optional[Type[nn.Module]]): Normalizer class to apply to the feature MLP output.
            learnable_normalizers (bool): Whether the normalization parameters are learnable.
            normalizer_eps (float): Epsilon value for numerical stability in normalization layers.

        """
        super().__init__()

        self.readin = LinearFeatureMLP(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=hidden_features,
            depth=1,
            n_dim=n_dim,
        )

        self.fno_units = torch.nn.ModuleList(
            [
                FNOUnit(
                    in_features=hidden_features, 
                    out_features=hidden_features,  
                    modes=modes,
                    n_dim=n_dim, 
                    activation_function=activation_function if i < depth - 1 else None,
                    conv_module=conv_module,
                    skip_connection=skip_connections[i] if isinstance(skip_connections, list) else skip_connections,
                    use_feature_mlp=use_feature_mlp,
                    feature_mlp_module=feature_mlp_module,
                    feature_mlp_depth=feature_mlp_depth,
                    feature_mlp_skip_connection=feature_mlp_skip_connections[i] if isinstance(feature_mlp_skip_connections, list) else feature_mlp_skip_connections,
                    feature_expansion_factor=feature_expansion_factor,
                    bias=bias,
                    init_scale=init_scale,
                    dtype=dtype,
                    norm=norm,
                    spectral_normalizer=spectral_normalizer,
                    feature_normalizer=feature_normalizer,      
                    learnable_normalizers=learnable_normalizers,
                    normalizer_eps=normalizer_eps
                )
                for i in range(depth)
            ]
        )

        self.readout = LinearFeatureMLP(
            in_features=hidden_features,
            hidden_features=hidden_features,
            out_features=out_features,
            depth=1,
            n_dim=n_dim
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the Fourier Operator.
        
        Args:
            x (Tensor): Input tensor of shape (B, in_features, *spatial_dims).
        
        Returns:
            Tensor: Output tensor of shape (B, out_features, *spatial_dims).

        """
        x = self.readin(x)
        
        for fno_unit in self.fno_units:
            x = fno_unit(x)
        
        x = self.readout(x)
        
        return x