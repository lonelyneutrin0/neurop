import torch
from ..base import Layer

from torch import Tensor
from typing import Tuple   

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
    $$x = \mathcal{F} [x]$$ 
    
    Then, a convolution is applied (which is multiplication with a weight matrix in the frequency domain).
    The shape of the weight matrix is $(C, O, K)$, where:
        - $C$ is the number of input channels,
        - $O$ is the number of output channels,
        - $K$ is the number of modes considered in the spectral convolution.    
    
    $$y_{b, o, k} =  W_{c, o, k} x_{b, c, k} $$
    In other words, it's a tensor contraction over the Fourier coefficients of the input signal and the weights. 
    Finally, the output is transformed back to the time domain using the inverse Fourier transform. 
    $$y = \mathcal{F}^{-1} [ y  ]$$
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

        modes = min(self.modes, n)  # Clip modes to input length
        if self.modes > n:
            warnings.warn(
                f"Number of modes {self.modes} cannot be greater than input length {n}. Clipping modes to {n}..."
            )

        x_ft = torch.fft.fft(x, dim=-1)
        out_ft = torch.zeros(
            batchsize, self.out_channels, n, dtype=torch.cfloat, device=x.device
        )
        out_ft[..., :modes] = torch.einsum(
            "bix,iox->box", x_ft[..., :modes], self.weight
        )

        x_out = torch.fft.ifft(out_ft, dim=-1).real 
        return x_out


class SpectralConv2DLayer(Layer):
    """
    Spectral Convolution 2 Dimensional Layer.
    This layer applies a convolution operation to 2 dimensional data in the frequency domain.

    Given an input signal $x$ of shape $(B, C, H, W)$, where:
        - $B$ is the batch size,
        - $C$ is the number of input channels,
        - $H$ is the height of the signal,
        - $W$ is the width of the signal,
    
    the layer first transforms the data to the frequency domain:
        $$x = \mathcal{F} [ x ]$$ 
    
    Then, a convolution is applied (which is multiplication with a weight matrix in the frequency domain).
    The shape of the weight matrix is $(C, O, K_H, K_W)$, where:
        - $C$ is the number of input channels,
        - $O$ is the number of output channels,
        - $K_H$ is the number of modes considered in height,
        - $K_W$ is the number of modes considered in width.    
    
    $$y_{b, o, k_h, k_w} = W_{c, o, k_h, k_w} x_{b, o, k_h, k_w} $$
    In other words, it's a tensor contraction over the Fourier coefficients of the input signal and the weights. 
    Finally, the output is transformed back to the time domain using the inverse Fourier transform. 
    $$y = \mathcal{F}^{-1}[  y ]$$
    """

    def __init__(self, in_channels: int, out_channels: int, modes: Tuple[int, int]):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            modes (Tuple[int, int]): Number of modes to consider in height and width.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Complex weights
        self.weight = torch.nn.Parameter(
            torch.randn(in_channels, out_channels, *modes, dtype=torch.cfloat)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the Spectral Convolution 2D layer.
        """
        batchsize, _, h, w = x.shape

        modes_h = min(self.modes[0], h)
        modes_w = min(self.modes[1], w)

        if self.modes[0] > h or self.modes[1] > w:
            warnings.warn(
                f"Number of modes {self.modes} cannot be greater than input dimensions {(h, w)}. Clipping modes..."
            )

        x_ft = torch.fft.fft2(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, h, w, dtype=torch.cfloat, device=x.device
        )
        out_ft[..., :modes_h, :modes_w] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[..., :modes_h, :modes_w], self.weight[..., :modes_h, :modes_w]
        )

        x_out = torch.fft.ifft2(out_ft).real
        return x_out
    
class SpectralConv3DLayer(Layer):
    """
    Spectral Convolution 3 Dimensional Layer.
    This layer applies a convolution operation to 3 dimensional data in the frequency domain.

    Given an input signal $x$ of shape $(B, C, D, H, W)$, where:
        - $B$ is the batch size,
        - $C$ is the number of input channels,
        - $D$ is the depth of the signal,
        - $H$ is the height of the signal,
        - $W$ is the width of the signal,
    
    the layer first transforms the data to the frequency domain:
        $$x = \mathcal{F} [ x ]$$ 
    
    Then, a convolution is applied (which is multiplication with a weight matrix in the frequency domain).
    The shape of the weight matrix is $(C, O, K_D, K_H, K_W)$, where:
        - $C$ is the number of input channels,
        - $O$ is the number of output channels,
        - $K_D$ is the number of modes considered in depth,
        - $K_H$ is the number of modes considered in height,
        - $K_W$ is the number of modes considered in width.    
    
    $$y_{b, o, k_d, k_h, k_w} = W_{c, o, k_d, k_h, k_w} x_{b, o, k_d, k_h, k_w} $$
    In other words, it's a tensor contraction over the Fourier coefficients of the input signal and the weights. 
    Finally, the output is transformed back to the time domain using the inverse Fourier transform. 
    $$y = \mathcal{F}^{-1}[  y  ]$$
    """

    def __init__(self, in_channels: int, out_channels: int, modes: Tuple[int, int, int]):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            modes (Tuple[int, int, int]): Number of modes to consider in depth and height and width.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Complex weights
        self.weight = torch.nn.Parameter(
            torch.randn(in_channels, out_channels, *modes, dtype=torch.cfloat)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the Spectral Convolution 3D layer.
        """
        batchsize, _, d, h, w = x.shape

        modes_d = min(self.modes[0], d)
        modes_h = min(self.modes[1], h)
        modes_w = min(self.modes[2], w)

        if self.modes[0] > d or self.modes[1] > h or self.modes[2] > w:
            warnings.warn(
                f"Number of modes {self.modes} cannot be greater than input dimensions {(d, h, w)}. Clipping modes..."
            )

        x_ft = torch.fft.fftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros(
            batchsize, self.out_channels, d, h, w, dtype=torch.cfloat, device=x.device
        )
        out_ft[..., :modes_d, :modes_h, :modes_w] = torch.einsum(
            "bixyw,ioxyw->boxyw", x_ft[..., :modes_d, :modes_h, :modes_w], self.weight[..., :modes_d, :modes_h, :modes_w]
        )

        x_out = torch.fft.ifftn(out_ft, dim=(-3, -2, -1)).real
        return x_out
