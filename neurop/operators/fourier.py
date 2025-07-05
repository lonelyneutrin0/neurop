from torch.types import Tensor
from typing import List, Union, Callable

import torch

from ..base import NeuralOperator
from ..layers.iolayers import ReadinLayer, ReadoutLayer
from ..layers.spectralconv import SpectralConv1DLayer, SpectralConv2DLayer, SpectralConv3DLayer, SpectralConvNDLayer


class FourierOperator(NeuralOperator):
    """
    Fourier Neural Operator. 
    It learns a spectral kernel that can be used to approximate functions in the frequency domain.
    """
    
    readin: ReadinLayer
    """Layer that reads in input data and projects it to a higher dimensional space."""

    kernel_integral: torch.nn.ModuleList
    """List of spectral convolution layers that apply Fourier transforms to the input data."""

    readout: ReadoutLayer
    """Layer that reads out data to a lower dimensional space."""

    depth: int
    """Depth of the operator, i.e., number of spectral convolution layers."""

    activation_function: Callable[[Tensor], Tensor]
    """Activation function to apply after each layer."""
    def __init__(self, 
                 in_features: int, 
                 hidden_features: int, 
                 out_features: int,
                 modes: Union[int, List], 
                 depth: int = 3,
                 activation_function: Callable[[Tensor], Tensor] = torch.relu
                 ):
        """
        Initializes the Fourier operator with the given parameters.

        Args:
            in_features (int): Number of input features.
            hidden_features (int): Number of hidden features.
            out_features (int): Number of output features.
            modes (Union[int, List]): Number of Fourier modes to consider. 
                                       If a list, it should contain the number of modes for each dimension.
            depth (int): Depth of the operator, i.e., number of spectral convolution layers.
            activation_function (Callable[[Tensor], Tensor]): Activation function to apply after each layer.
        
        Returns:
            None
        """
        super().__init__()
        self.readin = ReadinLayer(in_features, hidden_features)

        if isinstance(modes, int):
            self.kernel_integral = torch.nn.ModuleList([
                SpectralConv1DLayer(hidden_features, hidden_features, modes = modes) for _ in range(depth)
             ])
        elif len(modes) == 2:
            self.kernel_integral = torch.nn.ModuleList([
                SpectralConv2DLayer(hidden_features, hidden_features, mode_h = modes[0], mode_w = modes[1]) for _ in range(depth)
            ])
        elif len(modes) == 3:
            self.kernel_integral = torch.nn.ModuleList([
                SpectralConv3DLayer(hidden_features, hidden_features, mode_d = modes[0], mode_h = modes[1], mode_w = modes[2]) for _ in range(depth)
            ])
        
        else:
            self.kernel_integral = torch.nn.ModuleList([
                SpectralConvNDLayer(hidden_features, hidden_features, modes = modes) for _ in range(depth)
            ])

        self.readout = ReadoutLayer(hidden_features, out_features)
        self.depth = depth
        self.activation_function = activation_function

    def forward(self, x) -> Tensor:
        """
        Forward pass for the Fourier operator.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Tensor after applying the Fourier operator.
        """
        x = self.readin(x)
        x = self.activation_function(x)

        for layer in self.kernel_integral:
            x = layer(x)
            x = self.activation_function(x)

        x = self.readout(x)
        return x