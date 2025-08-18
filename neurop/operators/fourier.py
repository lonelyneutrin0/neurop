"""Fourier Neural Operator (FNO) implementation for learning mappings between functions in the Fourier domain."""
from typing import List, Union, Type, Optional

import torch
import torch.nn as nn

from ..layers.skip_connections import SkipConnection, IdentityConnection
from ..base import NeuralOperator, NeuralOperatorBuilder
from ..layers.spectral_convolution import Conv, SpectralConv, FFTNormType
from ..layers.fno_unit import FNOUnit
from ..layers.feature_mlp import FeatureMLP, ConvFeatureMLP, LinearFeatureMLP
from ..layers.normalizers import Normalizer

from typing import Self

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

    spectral_normalizer: Optional[Normalizer]
    """Normalizer to apply to the spectral convolution output."""

    feature_normalizer: Optional[Normalizer]
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
                 conv_module: Type[Conv] = SpectralConv,
                 skip_connections: Union[Type[SkipConnection], List[Type[SkipConnection]]] = IdentityConnection,
                 n_kernel: int = -1,
                 use_feature_mlp: bool = True,
                 feature_mlp_module: Type[FeatureMLP] = ConvFeatureMLP,
                 feature_mlp_depth: int = 4,
                 feature_mlp_skip_connections: Union[Type[SkipConnection], List[Type[SkipConnection]]] = IdentityConnection,
                 feature_expansion_factor: float = 1.0,
                 bias: bool = True,
                 init_scale: float = 1.0,
                 dtype: torch.dtype = torch.cfloat,
                 norm: FFTNormType = FFTNormType.ORTHO,
                 conv_normalizer: Optional[Type[Normalizer]] = None,
                 feature_mlp_normalizer: Optional[Type[Normalizer]] = None,
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
            activation_function (nn.Module): Activation function to apply after spectral convolution.
            conv_module (Type[Conv]): Spectral convolution module to use.
            skip_connections (Union[Type[SkipConnection], List[Type[SkipConnection]]]): Type of skip connection to use.
            n_kernel (int): Kernel size to use in the convolution skip connection.
            use_feature_mlp (bool): Whether to use a feature MLP for additional processing.
            feature_mlp_module (Type[FeatureMLP]): Feature MLP module to use for additional processing.
            feature_mlp_depth (int): Depth of the feature MLP.
            feature_mlp_skip_connections (Union[Type[SkipConnection], List[Type[SkipConnection]]]): Type of skip connection to use for the feature MLP.
            feature_expansion_factor (float): Factor by which to expand the features in the feature MLP.
            bias (bool): Whether to include bias parameters in the skip connection.
            init_scale (float): Scale for initializing the weights of the spectral convolution layer.
            dtype (torch.dtype): Data type for the spectral convolution layer output, typically complex (torch.cfloat).
            norm (FFTNormType): Normalization type for the spectral convolution layer, can be 'backward', 'forward', or 'ortho'.
            conv_normalizer (Optional[Type[Normalizer]]): Normalizer class to apply to the spectral convolution output.
            feature_mlp_normalizer (Optional[Type[Normalizer]]): Normalizer class to apply to the feature MLP output.
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

        self.fno_units = nn.ModuleList(
            [
                FNOUnit(
                    in_features=hidden_features, 
                    out_features=hidden_features,  
                    modes=modes,
                    n_dim=n_dim, 
                    activation_function=activation_function if i < depth - 1 else None,
                    conv_module=conv_module,
                    skip_connection=skip_connections[i] if isinstance(skip_connections, list) else skip_connections,
                    n_kernel=n_kernel,
                    use_feature_mlp=use_feature_mlp,
                    feature_mlp_module=feature_mlp_module,
                    feature_mlp_depth=feature_mlp_depth,
                    feature_mlp_skip_connection=feature_mlp_skip_connections[i] if isinstance(feature_mlp_skip_connections, list) else feature_mlp_skip_connections,
                    feature_expansion_factor=feature_expansion_factor,
                    bias=bias,
                    init_scale=init_scale,
                    dtype=dtype,
                    norm=norm,
                    conv_normalizer=conv_normalizer,
                    feature_mlp_normalizer=feature_mlp_normalizer,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Fourier Operator.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, in_features, *spatial_dims).
        
        Returns:
            torch.Tensor: Output tensor of shape (B, out_features, *spatial_dims).

        """
        x = self.readin(x)
        
        for fno_unit in self.fno_units:
            x = fno_unit(x)
        
        x = self.readout(x)
        
        return x
    
class FourierOperatorBuilder(NeuralOperatorBuilder):
    """Builder class for constructing Fourier Neural Operators."""

    def __init__(self):
        """__init__ method for FourierOperatorBuilder."""
        super().__init__()
        self._reset()

    def _reset(self): 
        """Reset all parameters to their default values."""
        # Required parameters (no defaults)
        self.in_features = None
        self.out_features = None
        self.modes = None
        self.n_dim = None
        self.depth = 4
        
        # Optional parameters with defaults
        self.activation_function = None
        self.conv_module = SpectralConv
        self.skip_connection = IdentityConnection
        self.n_kernel  = -1
        self.use_feature_mlp = False
        self.feature_mlp_module = ConvFeatureMLP
        self.feature_mlp_depth = 2
        self.feature_mlp_skip_connection = IdentityConnection
        self.feature_expansion_factor = 1.0
        self.bias = True
        self.init_scale = 1.0
        self.dtype = torch.float64
        self.norm = FFTNormType.ORTHO
        self.conv_normalizer = None
        self.feature_mlp_normalizer = None
        self.learnable_normalizers = True
        self.normalizer_eps = 1e-10

    def set_architecture(self, 
                         in_features: int, 
                         hidden_features: int, 
                         out_features: int, 
                         modes: Union[int, List[int]],
                         n_dim: int, 
                         depth: int,
                         ) -> Self:
        """
        Set the architecture for the FourierOperator.
        
        Arguments:
            in_features (int): Number of input features.
            hidden_features (int): Number of hidden features.
            out_features (int): Number of output features.
            modes (Union[int, List[int]]): Number of modes to consider for the convolution layer.
            n_dim (int): Number of spatial dimensions.
            depth (int): Number of FNO Units.
        
        Returns: 
            FourierOperatorBuilder 

        """
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.modes = modes
        self.n_dim = n_dim
        self.depth = depth
        return self
    
    def set_activation_function(self, activation_function: Type[nn.Module]) -> Self:
        """
        Set the activation function for the FourierOperator.
        
        Arguments: 
            activation_function (Type[nn.Module]): Activation function to use in the FourierOperator.
        
        Returns:
            FourierOperatorBuilder
        """
        self.activation_function = activation_function
        return self

    def set_conv_module(self, 
                        conv_module: Type[Conv], 
                        skip_connection: Type[SkipConnection],
                        conv_normalizer: Type[Normalizer],
                        *,
                        norm: FFTNormType = FFTNormType.ORTHO,
                        n_kernel: int = -1
                        ) -> Self:
        """
        Set the convolution module for the FourierOperator.

        Arguments:
            conv_module (Type[SpectralConv]): Convolution module to use in the FourierOperator.
            skip_connection (Type[SkipConnection]): Skip connection to use in the FourierOperator.
            conv_normalizer (Type[Normalizer]): Normalizer to use for the convolution layer.
            norm (FFTNormType, optional): Normalization type to use. Defaults to FFTNormType.ORTHO.
            n_kernel (int, optional): Kernel size for the convolution layer. Defaults to -1.

        Returns:
            FourierOperatorBuilder
        """
        self.conv_module = conv_module
        self.skip_connection = skip_connection
        self.conv_normalizer = conv_normalizer
        self.norm = norm
        self.n_kernel = n_kernel
        return self

    def set_feature_mlp(self, 
                        feature_mlp_module: Type[FeatureMLP], 
                        feature_mlp_depth: int, 
                        feature_mlp_skip_connection: Type[SkipConnection],
                        feature_expansion_factor: float,
                        feature_mlp_normalizer: Type[Normalizer],
                        ) -> Self:
        """
        Set the feature MLP for the FourierOperator.

        Arguments:
            feature_mlp_module (Type[ConvFeatureMLP]): Feature MLP module to use in the FourierOperator.
            feature_mlp_depth (int): Depth of the feature MLP.
            feature_mlp_skip_connection (Type[SkipConnection]): Skip connection to use in the feature MLP.
            feature_expansion_factor (float): Expansion factor for the feature MLP.
            feature_mlp_normalizer (Type[Normalizer]): Normalizer to use for the feature MLP.

        Returns:
            FourierOperatorBuilder
        """
        self.use_feature_mlp = True
        self.feature_mlp_module = feature_mlp_module
        self.feature_mlp_depth = feature_mlp_depth
        self.feature_mlp_skip_connection = feature_mlp_skip_connection
        self.feature_expansion_factor = feature_expansion_factor
        self.feature_mlp_normalizer = feature_mlp_normalizer
        return self

    def use_bias(self, toggle: bool) -> Self: 
        """
        Whether or not to use bias for the FourierOperator.

        Arguments:
            toggle (bool): Whether to use bias or not.
        
        Returns: 
            FourierOperatorBuilder
        """
        self.bias = toggle
        return self
    
    def use_learnable_normalizers(self, learnable_normalizers: bool, normalizer_eps: float) -> Self:
        """
        Whether or not to use learnable normalizers for the FourierOperator.

        Arguments:
            learnable_normalizers (bool): Whether to use learnable normalizers or not.
            normalizer_eps (float): Epsilon value for the normalizers.

        Returns:
            FourierOperatorBuilder
        """
        self.learnable_normalizers = learnable_normalizers
        self.normalizer_eps = normalizer_eps
        return self
    
    def build(self) -> FourierOperator:
        """
        Build the FourierOperator with the specified parameters.

        Returns:
            FourierOperator: The constructed FourierOperator instance.
        """
        if self.in_features is None or self.out_features is None or self.modes is None or self.n_dim is None or self.depth is None or self.activation_function is None:
            raise ValueError("Required parameters (in_features, out_features, modes, n_dim, depth, activation_function) must be set before building.")

        operator =  FourierOperator(
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            out_features=self.out_features,
            modes=self.modes,
            n_dim=self.n_dim,
            depth=self.depth,
            activation_function=self.activation_function,
            conv_module=self.conv_module,
            skip_connections=self.skip_connection,
            n_kernel=self.n_kernel,
            use_feature_mlp=self.use_feature_mlp,
            feature_mlp_module=self.feature_mlp_module,
            feature_mlp_depth=self.feature_mlp_depth,
            feature_mlp_skip_connections=self.feature_mlp_skip_connection,
            feature_expansion_factor=self.feature_expansion_factor,
            bias=self.bias,
            init_scale=self.init_scale,
            dtype=self.dtype,
            norm=self.norm,
            conv_normalizer=self.conv_normalizer,
            feature_mlp_normalizer=self.feature_mlp_normalizer,
            learnable_normalizers=self.learnable_normalizers,
            normalizer_eps=self.normalizer_eps
        )

        self._reset() 

        return operator
