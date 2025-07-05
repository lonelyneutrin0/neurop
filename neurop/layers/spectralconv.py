import torch

from torch.types import Tensor
from typing import Tuple, List


class SpectralConv1DLayer(torch.nn.Module):
    """
    Spectral Convolution 1 Dimensional Layer

    The input is assumed to have shape (B, C, L) where:\n
        - B is the batch size,
        - C is the number of input channels,
        - L is the length of the signal,
    
    The layer transforms the input to the frequency domain using a 1D FFT along the last dimension,
    $$x_f = \mathcal{F}[x]$$ 
    Then, the layer applies a pointwise multiplication (which is equivalent to a convolution in the time domain) as follows: 
    $$y_{b, o, k} = \sum_{c=0}^{C-1} W_{c, o, k} x_{b, c, k}$$
    where the contraction is along the features. Finally, the output is transformed back to the time domain using the inverse FFT:
    $$y = \mathcal{F}^{-1}[y]$$
    """
    
    in_features: int
    """Number of input channels."""

    out_features: int
    """Number of output channels."""

    modes: int
    """Number of Fourier modes to consider."""
    
    weight: torch.nn.Parameter
    """Learnable weights of the spectral convolution layer, initialized with a complex normal distribution."""

    def __init__(self, in_features: int, out_features: int, modes: int, init_scale: float = 1.0):
        """
        Initializes the SpectralConv1DLayer with the given parameters.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.modes = modes
        
        scale = init_scale / (in_features * out_features)
        self.weight = torch.nn.Parameter(
            torch.randn(in_features, out_features, modes, dtype=torch.cfloat) * scale
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the spectral convolution layer.
        
        Args:
            x (Tensor): Input tensor of shape (B, C, L)
        
        Returns:
            Tensor: Output tensor of shape (B, out_features, L)
        """
        batchsize, c, length = x.shape
        
        # FFTs only produce n//2 + 1 nonredundant modes for real inputs
        modes = min(self.modes, length // 2 + 1)  
        
        x_ft = torch.fft.rfft(x, dim=2, norm='ortho')
        
        out_ft = torch.zeros(
            batchsize, self.out_features, x_ft.size(2), 
            dtype=torch.cfloat, device=x.device
        )
        
        if modes > 0:
            out_ft[:, :, :modes] = torch.einsum(
                "bck,cok->bok", 
                x_ft[:, :, :modes], 
                self.weight[:, :, :modes]
            )
        
        x_out = torch.fft.irfft(out_ft, n=length, dim=2, norm='ortho')
        return x_out

class SpectralConv2DLayer(torch.nn.Module):
    """
    Spectral Convolution 2 Dimensional Layer    
    The input is assumed to have shape (B, C, H, W) where:\n
        - B is the batch size,
        - C is the number of input channels,
        - H is the height of the signal,
        - W is the width of the signal,
    The layer transforms the input to the frequency domain using a 2D FFT along the last two dimensions,
    $$x_f = \mathcal{F}[x]$$
    Then, the layer applies a pointwise multiplication (which is equivalent to a convolution in the time domain) as follows:
    $$y_{b, o, k_h, k_w} = \sum_{c=0}^{C-1} W_{c, o, k_h, k_w} x_{b, c, k_h, k_w}$$
    where the contraction is along the features. Finally, the output is transformed back to the time domain using the inverse FFT:
    $$y = \mathcal{F}^{-1}[y]$$
    """

    in_features: int
    """Number of input channels."""

    out_features: int
    """Number of output channels."""

    mode_h: int
    """Number of Fourier modes to consider in the height dimension."""
    
    mode_w: int
    """Number of Fourier modes to consider in the width dimension."""

    weight: torch.nn.Parameter
    """Learnable weights of the spectral convolution layer, initialized with a complex normal distribution."""

    def __init__(self, in_features: int, out_features: int, mode_h: int, mode_w: int, init_scale: float = 1.0):
        """
        Initializes the SpectralConv2DLayer with the given parameters.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode_h = mode_h
        self.mode_w = mode_w
        
        scale = init_scale / (in_features * out_features)
        self.weight = torch.nn.Parameter(
            torch.randn(in_features, out_features, mode_h, mode_w, dtype=torch.cfloat) * scale
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the Spectral Convolution 2D layer.
        
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) - PyTorch convention
        
        Returns:
            Tensor: Output tensor of shape (B, out_features, H, W)
        """
        batchsize, c, h, w = x.shape
        
        mode_h = min(self.mode_h, h)
        mode_w = min(self.mode_w, w // 2 + 1) 
        
        x_ft = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        
        out_ft = torch.zeros(
            batchsize, self.out_features, h, x_ft.size(3),
            dtype=torch.cfloat, device=x.device
        )
        
        if mode_h > 0 and mode_w > 0:
            out_ft[:, :, :mode_h, :mode_w] = torch.einsum(
                "bchw,cohw->bohw", 
                x_ft[:, :, :mode_h, :mode_w], 
                self.weight[:, :, :mode_h, :mode_w]
            )
        
        x_out = torch.fft.irfft2(out_ft, s=(h, w), dim=(2, 3), norm='ortho')
        return x_out

class SpectralConv3DLayer(torch.nn.Module):
    """
    Spectral Convolution 3 Dimensional Layer
    The input is assumed to have shape (B, C, D, H, W) where:\n
        - B is the batch size,
        - C is the number of input channels,
        - D is the depth of the signal,
        - H is the height of the signal,
        - W is the width of the signal,
    The layer transforms the input to the frequency domain using a 3D FFT along the last three dimensions,
        $$x_f = \mathcal{F}[x]$$
    Then, the layer applies a pointwise multiplication (which is equivalent to a convolution in the time domain) as follows:
        $$y_{b, o, k_d, k_h, k_w} = \sum_{c=0}^{C-1} W_{c, o, k_d, k_h, k_w} x_{b, c, k_d, k_h, k_w}$$
    where the contraction is along the features. Finally, the output is transformed back to the time domain using the inverse FFT:
        $$y = \mathcal{F}^{-1}[y]$$
    """

    in_features: int
    """Number of input channels."""
    
    out_features: int
    """Number of output channels."""

    mode_d: int
    """Number of Fourier modes to consider in the depth dimension."""

    mode_h: int
    """Number of Fourier modes to consider in the height dimension."""

    mode_w: int
    """Number of Fourier modes to consider in the width dimension."""

    weight: torch.nn.Parameter
    """Learnable weights of the spectral convolution layer, initialized with a complex normal distribution."""

    def __init__(self, in_features: int, out_features: int, mode_d: int, mode_h: int, mode_w: int, init_scale: float = 1.0):
        """
        Initializes the SpectralConv3DLayer with the given parameters.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode_d = mode_d
        self.mode_h = mode_h
        self.mode_w = mode_w
        
        scale = init_scale / (in_features * out_features)
        self.weight = torch.nn.Parameter(
            torch.randn(in_features, out_features, mode_d, mode_h, mode_w, dtype=torch.cfloat) * scale
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the Spectral Convolution 3D layer.

        Args:
            x (Tensor): Input tensor of shape (B, C, D, H, W)

        Returns:
            Tensor: Output tensor of shape (B, out_features, D, H, W)
        """
        batchsize, c, d, h, w = x.shape
        
        mode_d = min(self.mode_d, d)
        mode_h = min(self.mode_h, h)
        mode_w = min(self.mode_w, w // 2 + 1)

        x_ft = torch.fft.rfftn(x, dim=(2, 3, 4), norm='ortho')

        out_ft = torch.zeros(
            batchsize, self.out_features, d, h, x_ft.size(4),
            dtype=torch.cfloat, device=x.device
        )

        if mode_d > 0 and mode_h > 0 and mode_w > 0:
            out_ft[:, :, :mode_d, :mode_h, :mode_w] = torch.einsum(
                "bcdhw,codhw->bohdw", 
                x_ft[:, :, :mode_d, :mode_h, :mode_w], 
                self.weight[:, :, :mode_d, :mode_h, :mode_w]
            )
        
        x_out = torch.fft.irfftn(out_ft, s=(d, h, w), dim=(2, 3, 4), norm='ortho')
        return x_out
    
class SpectralConvNDLayer(torch.nn.Module):
    """
    N-Dimensional Spectral Convolution Layer
    
    The input is assumed to have shape (B, C, *spatial_dims) where:
        - B is the batch size,
        - C is the number of input channels,
        - *spatial_dims are the spatial dimensions (D, H, W for 3D, H, W for 2D, etc.)
    
    The layer transforms the input to the frequency domain using an N-D FFT along the spatial dimensions,
        $$x_f = \mathcal{F}[x]$$
    Then, the layer applies a pointwise multiplication (which is equivalent to a convolution in the spatial domain) as follows:
        $$y_{b, o, k_1, k_2, ..., k_N} = \sum_{c=0}^{C-1} W_{c, o, k_1, k_2, ..., k_N} x_{b, c, k_1, k_2, ..., k_N}$$
    where the contraction is along the features. Finally, the output is transformed back to the spatial domain using the inverse FFT:
        $$y = \mathcal{F}^{-1}[y]$$
    """

    in_features: int
    """Number of input channels."""

    out_features: int
    """Number of output channels."""

    ndim: int
    """Number of spatial dimensions"""

    modes: List[int]
    """List of integers representing the number of Fourier modes to consider in each spatial dimension."""

    weight: torch.nn.Parameter
    """Learnable weights of the spectral convolution layer, initialized with a complex normal distribution."""

    spatial_dims: Tuple[int, ...]
    """Tuple of integers representing the indices of the spatial dimensions in the input tensor."""

    def __init__(self, in_features: int, out_features: int, modes: List[int], init_scale: float = 1.0):
        """
        Initializes the N-Dimensional Spectral Convolution Layer with the given parameters.
        """

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if not isinstance(modes, (list)): 
            raise TypeError("Modes must be a list of integers. For dimensions <=3, use the specific SpectralConv1DLayer, SpectralConv2DLayer, or SpectralConv3DLayer.")
        
        self.modes = list(modes)
        self.ndim = len(self.modes)
        
        scale = init_scale / (in_features * out_features)

        self.weight = torch.nn.Parameter(
            torch.randn(in_features, out_features, *self.modes, dtype=torch.cfloat) * scale
        )
        
        # Store spatial dimensions for forward pass
        self.spatial_dims = tuple(range(2, 2 + self.ndim))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the N-dimensional Spectral Convolution layer.
        
        Args:
            x (Tensor): Input tensor of shape (B, C, *spatial_dims)
        
        Returns:
            Tensor: Output tensor of shape (B, out_features, *spatial_dims)
        """
        # Get input dimensions
        batch_size = x.shape[0]
        spatial_shape = x.shape[2:]
        
        if len(spatial_shape) != self.ndim:
            raise ValueError(f"Input has {len(spatial_shape)} spatial dimensions, "
                           f"but layer expects {self.ndim}")
        
        # Clamp to nonredundant modes
        effective_modes = []
        for i, (mode, dim_size) in enumerate(zip(self.modes, spatial_shape)):
            if i == len(self.modes) - 1:  # Last dimension uses rfft
                effective_modes.append(min(mode, dim_size // 2 + 1))
            else:
                effective_modes.append(min(mode, dim_size))
        
        x_ft = torch.fft.rfftn(x, dim=self.spatial_dims, norm='ortho')
        
        out_ft_shape = [batch_size, self.out_features] + list(x_ft.shape[2:])
        out_ft = torch.zeros(out_ft_shape, dtype=torch.cfloat, device=x.device)
        
        if all(mode > 0 for mode in effective_modes):
            
            # Set up [batchsize, channels, :modes, :modes, ...] slices for input and weight
            input_slices = [slice(None), slice(None)] 
            weight_slices = [slice(None), slice(None)] 
            
            for i, mode in enumerate(effective_modes):
                input_slices.append(slice(None, mode))
                weight_slices.append(slice(None, mode))
            
            x_ft_truncated = x_ft[tuple(input_slices)]
            weight_truncated = self.weight[tuple(weight_slices)]

            spatial_chars = 'adefghijklmnopqrstuvwxyz'[:self.ndim] 
            input_indices = 'bc' + spatial_chars
            weight_indices = 'co' + spatial_chars
            output_indices = 'bo' + spatial_chars
            einsum_str = f"{input_indices},{weight_indices}->{output_indices}"
            
            result = torch.einsum(einsum_str, x_ft_truncated, weight_truncated)
            
            output_slices = [slice(None), slice(None)]
            for mode in effective_modes:
                output_slices.append(slice(None, mode))
            
            out_ft[tuple(output_slices)] = result
        
        x_out = torch.fft.irfftn(out_ft, s=spatial_shape, dim=self.spatial_dims, norm='ortho')
        
        return x_out