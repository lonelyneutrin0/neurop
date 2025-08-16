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

from dataclasses import dataclass

@dataclass
class CoreParams: 
    """Dataclass for core parameters of FNOs."""

    in_features: int
    hidden_features: int
    out_features: int
    modes: Union[int, List[int]]
    n_dim: int 
    depth: int = 4
    n_kernel: int = -1
    activation_function: Type[nn.Module] = nn.ReLU
    conv_module: Type[SpectralConv] = SpectralConv
    skip_connections: Union[ConnectionType, List[ConnectionType]] = 'soft-gating'
    bias: bool = True
    init_scale: float = 1.0
    dtype: torch.dtype = torch.float

@dataclass
class FeatureMLPParams: 
    """Dataclass for Feature MLP parameters."""

    use_feature_mlp: bool = True
    feature_mlp_module: Type[nn.Module] = ConvFeatureMLP
    feature_mlp_depth: int = 4
    feature_mlp_skip_connections: Union[ConnectionType, List[ConnectionType]] = 'soft-gating'
    feature_expansion_factor: float = 1.0

@dataclass
class NormalizerParams: 
    """Dataclass for Normalizer parameters."""

    norm: NormType = 'ortho'
    spectral_normalizer: Optional[Type[nn.Module]] = None 
    feature_normalizer: Optional[Type[nn.Module]] = None
    learnable_normalizers: bool = True
    normalizer_eps: float = 1e-10

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

    def __init__(self, core: CoreParams, feature: FeatureMLPParams, normalizer: NormalizerParams, *args, **kwargs):
        """Initialize the FourierOperator with the given parameters.

        Args:
            core (CoreParams): Core parameters for the FNO.
            feature (FeatureMLPParams): Feature MLP parameters.
            normalizer (NormalizerParams): Normalizer parameters.
            *args: Additional args
            **kwargs: Additional keyword args

        """
        super().__init__(*args, **kwargs)

        self.readin = LinearFeatureMLP(
            in_features=core.in_features,
            hidden_features=core.hidden_features,
            out_features=core.hidden_features,
            depth=1,
            n_dim=core.n_dim,
        )

        self.fno_units = torch.nn.ModuleList(
            [
                FNOUnit(
                    in_features=core.hidden_features,
                    out_features=core.hidden_features,
                    modes=core.modes,
                    n_dim=core.n_dim,
                    n_kernel=core.n_kernel,
                    activation_function=core.activation_function if i < core.depth - 1 else None,
                    conv_module=core.conv_module,
                    skip_connection=core.skip_connections[i] if isinstance(core.skip_connections, list) else core.skip_connections,
                    use_feature_mlp=feature.use_feature_mlp,
                    feature_mlp_module=feature.feature_mlp_module,
                    feature_mlp_depth=feature.feature_mlp_depth,
                    feature_mlp_skip_connection=feature.feature_mlp_skip_connections[i] if isinstance(feature.feature_mlp_skip_connections, list) else feature.feature_mlp_skip_connections,
                    feature_expansion_factor=feature.feature_expansion_factor,
                    bias=core.bias,
                    init_scale=core.init_scale,
                    dtype=core.dtype,
                    norm=normalizer.norm,
                    spectral_normalizer=normalizer.spectral_normalizer,
                    feature_normalizer=normalizer.feature_normalizer,      
                    learnable_normalizers=normalizer.learnable_normalizers,
                    normalizer_eps=normalizer.normalizer_eps
                )
                for i in range(core.depth)
            ]
        )

        self.readout = LinearFeatureMLP(
            in_features=core.hidden_features,
            hidden_features=core.hidden_features,
            out_features=core.out_features,
            depth=1,
            n_dim=core.n_dim
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