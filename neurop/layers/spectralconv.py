import torch
from ..base import Layer

from torch import Tensor

import warnings

class SpectralConv1DLayer(Layer):
    """
    Spectral Convolution 1 Dimensional Layer.
    This layer applies a convolution operation to 1 dimensional data in the frequency domain.

    Given an input signal $x$ of shape $(B, C, N)$, where:
        - $B$ is the batch size,
        - $C$ is the number of input channels,
        - $N$ is the length of the signal,
    
    the layer first transforms the data to the frequency domain:
        $$\vec{x} = \mathcal{F}\left [ \vec x\right ]$$ 
    
    Then, a convolution is applied (which is multiplication with a weight matrix in the frequency domain).
    The shape of the weight matrix is $(C, O, K)$, where:
        - $C$ is the number of input channels,
        - $O$ is the number of output channels,
        - $K$ is the number of modes considered in the spectral convolution.    
    
    $$\hat{y_{b, o, k} = \hat W_{c, o, k} \hat{x_{b, c, k}} $$
    In other words, it's a tensor contraction over the Fourier coefficients of the input signal and the weights. 
    Finally, the output is transformed back to the time domain using the inverse Fourier transform. 
    $$\vec{y} = \mathcal{F}^{-1}\left [ \hat y \right ]$$
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            modes (int): Number of modes to consider in the spectral convolution.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Complex weights
        self.weight = torch.nn.Parameter(
            torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the spectral convolution layer.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor after applying spectral convolution.
        """
        batchsize, _, n = x.shape

        if self.modes > n:
            warnings.warn(
                f"Number of modes {self.modes} cannot be greater than input length {n}. Clipping modes to {n}..."
            )
            self.modes = n

        x_ft = torch.fft.fft(x, dim=-1)
        out_ft = torch.zeros(
            batchsize, self.out_channels, n, dtype=torch.cfloat, device=x.device
        )
        out_ft[..., :self.modes] = torch.einsum(
            "bix,iox->box", x_ft[..., :self.modes], self.weight
        )

        x_out = torch.fft.ifft(out_ft, dim=-1).real 
        return x_out
        