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

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 modes: Union[int, List[int]],
                 n_dim: int,
                 depth: int = 4,
                 activation_function: torch.nn.Module = torch.nn.ReLU(),
                 conv_modules: Type[SpectralConv] = SpectralConv,
                 skip_connections: Union[ConnectionType, List[ConnectionType]] = 'soft-gating',
                 bias: bool = True,
                 init_scale: float = 1.0,
                 dtype: torch.dtype = torch.cfloat):
        """
        Initializes the FourierOperator with the given parameters.
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
                    conv_module=conv_modules,
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