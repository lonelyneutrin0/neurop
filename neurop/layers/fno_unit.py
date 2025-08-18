"""FNO Unit Module."""
import torch
import torch.nn as nn

from typing import Type, Union, List, Optional
from typing_extensions import Self

from .spectral_convolution import Conv, SpectralConv, FFTNormType
from .feature_mlp import FeatureMLP, ConvFeatureMLP
from .skip_connections import SkipConnection, IdentityConnection

from .normalizers import Normalizer

from ..base import NeuralOperatorUnitBuilder, NeuralOperatorUnit

class FNOUnit(NeuralOperatorUnit):
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

    feature_mlp: Optional[FeatureMLP]
    """Feature MLP for additional processing of features."""

    spectral_conv: Conv
    """Spectral convolution layer for global processing of features."""

    skip_connection: SkipConnection
    """Skip connection module for combining input and transformed features."""

    feature_mlp_skip_connection: SkipConnection
    """Skip connection module for combining input and transformed features after the feature MLP."""

    spectral_normalizer: Optional[Normalizer]
    """Normalizer to apply to the spectral convolution output."""

    feature_mlp_normalizer: Optional[Normalizer]
    """Normalizer to apply to the feature MLP output."""

    learnable_normalizers: bool
    """Whether the normalization parameters are learnable."""

    normalizer_eps: float
    """Epsilon value for numerical stability in normalization layers."""
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 modes: Union[int, List[int]], 
                 n_dim: int,
                 activation_function: Optional[Type[nn.Module]] = None,
                 conv_module: Type[Conv] = SpectralConv,
                 skip_connection: Type[SkipConnection] = IdentityConnection,
                 n_kernel: int = -1,
                 use_feature_mlp: bool = False,
                 feature_mlp_module: Type[FeatureMLP] = ConvFeatureMLP,
                 feature_mlp_depth: int = 2,
                 feature_mlp_skip_connection: Type[SkipConnection] = IdentityConnection,
                 feature_expansion_factor: float = 1.0,
                 bias: bool = True,
                 init_scale: float = 1.0,
                 dtype: torch.dtype = torch.float64,
                 norm: FFTNormType = FFTNormType.ORTHO,
                 conv_normalizer: Optional[Type[Normalizer]] = None, 
                 feature_mlp_normalizer: Optional[Type[Normalizer]] = None,
                 *, 
                 learnable_normalizers: bool = True,
                 normalizer_eps: float = 1e-10,
                 ):
        """Initialize the FNO unit with the given parameters.
        
        Args:
            in_features (int): Number of input features (channels).
            out_features (int): Number of output features (channels).
            modes (Union[int, List[int]]): Number of Fourier modes to consider in each spatial dimension.
            n_dim (int): Number of spatial dimensions (2D, 3D, etc.).
            activation_function (Optional[Type[nn.Module]]): Activation function to apply after spectral convolution. Defaults to None.
            conv_module (Type[Conv]): Spectral convolution module to use.
            skip_connection (Type[SkipConnection]): Type of skip connection to use.
            n_kernel (int): Number of kernels to use in the convolution.
            use_feature_mlp (bool): Whether to use a feature MLP for additional processing.
            feature_mlp_module (Type[FeatureMLP]): Feature MLP module to use for additional processing.
            feature_mlp_depth (int): Depth of the feature MLP.
            feature_mlp_skip_connection (Type[SkipConnection]): Type of skip connection to use for the feature MLP.
            feature_expansion_factor (float): Factor by which to expand the features in the feature MLP.
            bias (bool): Whether to include bias parameters in the skip connection.
            init_scale (float): Scale for initializing the weights of the spectral convolution layer.
            dtype (torch.dtype): Data type for the spectral convolution layer output, typically complex (torch.cfloat).
            norm (FFTNormType): Normalization type for the spectral convolution layer, can be BACKWARD, FORWARD, or ORTHO.
            conv_normalizer (Optional[Type[Normalizer]]): Normalizer class to apply to the spectral convolution output.
            feature_mlp_normalizer (Optional[Type[Normalizer]]): Normalizer class to apply to the feature MLP output.
            learnable_normalizers (bool): Whether the normalization parameters are learnable.
            normalizer_eps (float): Epsilon value for numerical stability in normalization layers.

        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if isinstance(modes, int):
            self.modes = [modes] * n_dim
        else:
            if len(modes) != n_dim:
                raise ValueError(f"Expected {n_dim} modes, got {len(modes)}")
            self.modes = modes
        
        self.n_dim = n_dim
        self.activation_function = activation_function() if activation_function is not None else None

        if use_feature_mlp:
            self.feature_mlp = feature_mlp_module(
                in_features=out_features,
                hidden_features=int(feature_expansion_factor * out_features),
                out_features=out_features,
                depth=feature_mlp_depth,
                n_dim=n_dim,
                activation_function=activation_function if activation_function is not None else nn.ReLU,
            )

            self.feature_mlp_skip_connection = feature_mlp_skip_connection(
                in_features=out_features,
                out_features=out_features,
                n_dim=n_dim,
                n_kernel=n_kernel,
                bias=bias,
            )

        else: 
            self.feature_mlp = None
        
        # Spectral convolution layer
        self.spectral_conv = conv_module(
            in_features=in_features,
            out_features=out_features,
            modes=self.modes,
            init_scale=init_scale, 
            dtype=dtype,
            norm=norm
        )
        
        # Skip connection
        self.skip_connection = skip_connection(
            in_features=in_features,
            out_features=out_features,
            n_dim=n_dim,
            bias=bias,
            n_kernel=n_kernel,
        )

        # Initialize normalizers as None
        self.conv_normalizer = None
        self.feature_mlp_normalizer = None

        if conv_normalizer is not None: 
            self.conv_normalizer = conv_normalizer(
                num_features=out_features,
                ndim=n_dim,
                learnable=learnable_normalizers,
                tol=normalizer_eps
            )
        
        if feature_mlp_normalizer is not None: 
            self.feature_mlp_normalizer = feature_mlp_normalizer(
                num_features=out_features,
                ndim=n_dim,
                learnable=learnable_normalizers,
                tol=normalizer_eps
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through single FNO unit.

        Convolution -> Normalization -> Add Skip Connection -> Activation -> FeatureMLP -> Normalization -> Add Skip Connection -> Activation -> Output

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_features, *spatial_dims)
            
        Returns:
            torch.Tensor: Output tensor of shape (B, out_features, *spatial_dims)

        """
        if x.shape[1] != self.in_features:
            raise ValueError(f"Expected {self.in_features} input features, got {x.shape[1]}")

        # Apply spectral convolution
        x_conv = self.spectral_conv(x)
        if self.conv_normalizer is not None:
            x_conv = self.conv_normalizer(x_conv)
        
        # Apply skip connection
        x = self.skip_connection(x = x, transformed_x = x_conv)

        if self.activation_function is not None:
            x = self.activation_function(x)
        
        if self.feature_mlp is not None:
            # FeatureMLP 
            x_mlp = self.feature_mlp(x)

            if self.feature_mlp_normalizer is not None:
                x_mlp = self.feature_mlp_normalizer(x_mlp)

            x = self.feature_mlp_skip_connection(x = x, transformed_x = x_mlp)

            if self.activation_function is not None:
                x = self.activation_function(x)
            
        return x

class FNOUnitBuilder(NeuralOperatorUnitBuilder): 
    """Builder class for FNO Units for custom configurations with YAML/JSON."""

    def __init__(self): 
        """__init__ method for FNOUnitBuilder class."""
        super().__init__() 
        self._reset()

    def _reset(self): 
        """Reset all parameters to their default values."""
        # Required parameters (no defaults)
        self.in_features = None
        self.out_features = None
        self.modes = None
        self.n_dim = None
        
        # Optional parameters with defaults
        self.activation_function = None
        self.conv_module = SpectralConv
        self.skip_connection = IdentityConnection
        self.n_kernel = -1
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
        self.post_activation = True

    def set_architecture(self, 
                         in_features: int, 
                         out_features: int, 
                         modes: Union[int, List[int]], 
                         n_dim: int, *, 
                         dtype: torch.dtype=torch.float64, 
                         init_scale: float=1.0) -> Self:
        """Set the architecture for the FNO Unit.
        
        Arguments:
            in_features (int): Number of input features for the FNO Unit.
            out_features (int): Number of output features for the FNO Unit. 
            modes (int): Number of Fourier modes to consider in each spatial dimension.
            n_dim (int): Number of spatial dimensions (2D, 3D, etc.)
            dtype (torch.dtype): Data type for the spectral convolution layer output, typically float (torch.float64).
            init_scale (float): Initial scale for the spectral convolution layer weights.

        Returns:
            An FNOUnitBuilder object

        """
        self.in_features = in_features
        self.out_features = out_features
        self.modes = modes
        self.n_dim = n_dim
        self.dtype = dtype
        self.init_scale = init_scale

        return self

    def set_activation_function(self, activation_function: Type[nn.Module],) -> Self:
        """Set the activation function for the FNO Unit.
        
        Arguments:
            activation_function (Type[nn.Module]): The activation function to apply.
        
        Returns:
            An FNOUnitBuilder object
        
        """
        self.activation_function = activation_function

        return self

    def set_conv_module(self, 
                        conv_module: Type[Conv], 
                        skip_connection: Type[SkipConnection], 
                        conv_normalizer: Type[Normalizer], 
                        *, 
                        norm: FFTNormType = FFTNormType.ORTHO,
                        conv_kernel: int = -1) -> Self:
        """Set the convolution module for the FNO Unit.

        Arguments:
            conv_module (Type[Conv]): The convolution module to apply.
            skip_connection (Type[SkipConnection]): The type of skip connection to use.
            conv_normalizer (Type[Normalizer]): The normalization module to apply after the convolution.
            norm (FFTNormType): The normalization type to use for FFTs.
            conv_kernel (int): THe kernel size for the convolution skip connection.

        Returns:
            An FNOUnitBuilder object

        """
        self.conv_module = conv_module
        self.skip_connection = skip_connection
        self.conv_normalizer = conv_normalizer
        self.norm = norm
        self.n_kernel = conv_kernel

        return self
    
    def set_feature_mlp(self, 
                        feature_mlp_module: Type[FeatureMLP], 
                        feature_mlp_depth: int, 
                        feature_mlp_skip_connection: Type[SkipConnection], 
                        feature_expansion_factor: float, 
                        feature_mlp_normalizer: Type[Normalizer]) -> Self:
        """Set the feature MLP for the FNO Unit.
        
        Arguments:
            feature_mlp_module (nn.Module): The feature MLP module to apply.
            feature_mlp_depth (int): Depth of the feature MLP.
            feature_mlp_skip_connection (Connection): Type of skip connection to use for the feature MLP.
            feature_expansion_factor (float): Factor by which to expand the features in the feature MLP.
            feature_mlp_normalizer (Type[Normalizer]): Normalization layer to use for the feature MLP.

        Returns:
            An FNOUnitBuilder object

        """
        self.use_feature_mlp = True
        self.feature_mlp_module = feature_mlp_module
        self.feature_mlp_depth = feature_mlp_depth
        self.feature_mlp_skip_connection = feature_mlp_skip_connection
        self.feature_expansion_factor = feature_expansion_factor
        self.feature_mlp_normalizer = feature_mlp_normalizer
        return self

    def use_bias(self, toggle: bool) -> Self:
        """Set whether to use bias parameters in the skip connections.
        
        Arguments:
            toggle (bool): Whether to include bias parameters in the skip connections.
        
        Returns:
            An FNOUnitBuilder object

        """
        self.bias = toggle
        return self

    def use_learnable_normalizers(self, learnable_normalizers: bool, normalizer_eps: float) -> Self:
        """Set whether to use learnable normalizers and their tolerance value.
        
        Arguments:
            learnable_normalizers (bool): Whether or not to use learnable normalizers.
            normalizer_eps (float): The epsilon value for the normalizers.
        
        Returns:
            An FNOUnitBuilder object

        """
        self.learnable_normalizers = learnable_normalizers
        self.normalizer_eps = normalizer_eps
        return self

    def build(self,) -> FNOUnit: 
        """Build the FNO Unit with the specified parameters. Resets the builder to default settings.
        
        Returns:
            An FNOUnit object initialized with the specified parameters.

        """
        if self.in_features is None or self.out_features is None or self.modes is None or self.n_dim is None:
            raise ValueError("Required parameters (in_features, out_features, modes, n_dim) must be set before building.")
        
        unit = FNOUnit(
                in_features = self.in_features, 
                out_features = self.out_features,
                modes = self.modes,
                n_dim = self.n_dim,
                activation_function = self.activation_function,
                conv_module = self.conv_module,
                skip_connection = self.skip_connection,
                n_kernel = self.n_kernel,
                use_feature_mlp = self.use_feature_mlp,
                feature_mlp_module = self.feature_mlp_module,
                feature_mlp_depth = self.feature_mlp_depth,
                feature_mlp_skip_connection = self.feature_mlp_skip_connection,
                feature_expansion_factor = self.feature_expansion_factor,
                bias = self.bias,
                init_scale = self.init_scale,
                dtype = self.dtype,
                norm = self.norm,
                conv_normalizer = self.conv_normalizer,
                feature_mlp_normalizer = self.feature_mlp_normalizer,
                learnable_normalizers = self.learnable_normalizers,
                normalizer_eps = self.normalizer_eps,
        )

        self._reset() 
        return unit