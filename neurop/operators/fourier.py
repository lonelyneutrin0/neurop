from torch.types import Tensor
from typing import List, Union, Literal, Type
from ..layers.skip_connections import Connection, ConnectionType
import torch

from ..base import NeuralOperator
from ..layers.io_layers import ReadinLayer, ReadoutLayer
from ..layers.spectral_convolution import SpectralConv
from ..layers.fno_unit import FNOUnit
class FourierOperator(NeuralOperator):
    """ 
    Fourier Neural Operator (FNO) for learning mappings between functions in the Fourier domain.
    This operator uses spectral convolutions to learn the mapping between input and output functions.
    """

    readin: ReadinLayer
    """Layer to read input features and project them to hidden features."""

    fno_units: torch.nn.ModuleList
    """List of FNO units that apply spectral convolutions and activation functions."""

    readout: ReadoutLayer
    """Layer to read output features and project them to the final output features."""

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 modes: Union[int, List[int]],
                 n_dim: int,
                 depth: int = 4,
                 activation_function: torch.nn.Module = torch.nn.ReLU(),
                 conv_module: Type[SpectralConv] = SpectralConv,
                 skip_connections: Union[ConnectionType, List[ConnectionType]] = 'soft-gating',
                 bias: bool = True,
                 init_scale: float = 1.0,
                 dtype: torch.dtype = torch.cfloat):
        """
        Initializes the FourierOperator with the given parameters.

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
            bias (bool): Whether to include bias parameters in the skip connection.
            init_scale (float): Scale for initializing the weights of the spectral convolution layer.
            dtype (torch.dtype): Data type for the spectral convolution layer output, typically complex (torch.cfloat).
        """
        super().__init__()

        self.readin = ReadinLayer(in_features=in_features, hidden_features=hidden_features)

        self.fno_units = torch.nn.ModuleList(
            [
                FNOUnit(
                    in_features=hidden_features, 
                    out_features=hidden_features,  
                    modes=modes,
                    n_dim=n_dim, 
                    activation_function=activation_function,
                    conv_module=conv_module,
                    skip_connection=skip_connections[i] if isinstance(skip_connections, list) else skip_connections,
                    bias=bias,
                    init_scale=init_scale,
                    dtype=dtype
                )
                for i in range(depth)
            ]
        )

        self.readout = ReadoutLayer(hidden_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Fourier Operator.
        
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