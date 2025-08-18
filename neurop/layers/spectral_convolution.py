"""Spectral Convolution Layer Implementation."""
import torch
from torch import nn

from typing import Tuple, List, Union
from enum import Enum

from abc import ABC, abstractmethod

class FFTNormType(Enum): 
    BACKWARD = 'backward'
    FORWARD = 'forward'
    ORTHO = 'ortho'

class Conv(nn.Module, ABC): 
    """Spectral Convolution Layer Abstract Class."""

    def __init__(self, *args, **kwargs): 
        pass

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        pass
    
class SpectralConv(Conv):
    r"""N-Dimensional Spectral Convolution Layer.
    
    The input is assumed to have shape (B, C, *spatial_dims) where:
        - B is the batch size,
        - C is the number of input channels,
        - *spatial_dims are the spatial dimensions (D, H, W for 3D, H, W for 2D, etc.)
    
    The layer transforms the input to the frequency domain using an N-D FFT along the spatial dimensions:
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
    """Number of spatial dimensions."""

    modes: List[int]
    """List of integers representing the number of Fourier modes to consider in each spatial dimension."""

    weight: nn.Parameter
    """Learnable weights of the spectral convolution layer, initialized with a complex normal distribution."""

    spatial_dims: Tuple[int, ...]
    """Tuple of integers representing the indices of the spatial dimensions in the input tensor."""

    norm: FFTNormType
    """Normalization type for the FFT operation, can be 'backward', 'forward', or 'ortho'."""   
    def __init__(self,
                in_features: int, 
                out_features: int, 
                modes: Union[int, List[int]], 
                init_scale: float = 1.0,
                dtype: torch.dtype = torch.cfloat,
                *, 
                weight_dtype: torch.dtype = torch.cfloat,
                norm: FFTNormType = FFTNormType.ORTHO):
        """Initialize the N-Dimensional Spectral Convolution Layer with the given parameters."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
            
        if isinstance(modes, int): 
            if modes < 0:
                raise ValueError("Number of modes must be non-negative.")
            self.modes = [modes]
        else:
            self.modes = modes

            for i in modes:
                if i < 0:
                    raise ValueError("Number of modes must be non-negative.")
                
        self.ndim = len(self.modes)
        
        scale = init_scale / (in_features * out_features)

        self.weight = nn.Parameter(
            torch.randn(in_features, out_features, *self.modes, dtype=weight_dtype) * scale
        )
        
        # Store spatial dimensions for forward pass
        self.spatial_dims = tuple(range(2, 2 + self.ndim))
        self.einsum_eq = self._einsum_eq()

        self.norm = norm

    def forward(self, x: torch.Tensor, dtype: torch.dtype = torch.cfloat) -> torch.Tensor:
        """Forward pass for the N-dimensional Spectral Convolution layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, *spatial_dims)
            dtype (torch.dtype): Data type for the output tensor, default is torch.cfloat
        
        Returns:
            torch.Tensor: Output tensor of shape (B, out_features, *spatial_dims)

        """
        if not isinstance(x, torch.Tensor):
            try:
                x = torch.as_tensor(x)
            except Exception as e:
                raise TypeError(f"Input must be a PyTorch tensor or convertible to one. Error: {e}")

        c = x.shape[1]
        spatial_shape = x.shape[2:]

        if c != self.in_features:
            raise ValueError(f"Input has {c} channels, but layer expects {self.in_features} channels.")
        
        if x.numel() == 0:
            raise ValueError("Input tensor is empty.")
    
        if len(spatial_shape) != self.ndim:
            raise ValueError(f"Input has {len(spatial_shape)} spatial dimensions, "
                           f"but layer expects {self.ndim}")
        
        if dtype is None:
            dtype = self.dtype
        
        is_complex = x.is_complex()
        effective_modes =  self._get_modes(spatial_shape, is_complex)

        if is_complex:
            x_ft = torch.fft.fftn(x, dim=self.spatial_dims, norm=self.norm.value)

        else: 
            x_ft = torch.fft.rfftn(x, dim=self.spatial_dims, norm=self.norm.value)
        
        out_ft = self._apply_spectral_convolution(x=x_ft, effective_modes=effective_modes, dtype=dtype)

        if is_complex:
            x_out = torch.fft.ifftn(out_ft, s=spatial_shape, dim=self.spatial_dims, norm=self.norm.value)
        
        else:
            x_out = torch.fft.irfftn(out_ft, s=spatial_shape, dim=self.spatial_dims, norm=self.norm.value)

        return x_out

    def _get_modes(self, shape: Tuple[int, ...], is_complex: bool) -> List[int]: 
        """Calculate effective modes based on input shape and data type.
        
        For real inputs, the last dimension uses rfft, which has size (N//2 + 1).
        For complex inputs, all dimensions use full fft.
        """
        effective_modes = []

        for i, (mode, dim_size) in enumerate(zip(self.modes, shape)): 

            if not is_complex and i == len(self.modes) - 1:
                effective_modes.append(min(mode, dim_size // 2 + 1))
            
            else:
                effective_modes.append(min(mode, dim_size))
        
        return effective_modes

    def _einsum_eq(self) -> str:
            """Return the einsum equation string for the spectral convolution operation."""
            spatial_chars = 'adefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'[:self.ndim] 
            input_indices = 'bc' + spatial_chars
            weight_indices = 'co' + spatial_chars
            output_indices = 'bo' + spatial_chars
            return f"{input_indices},{weight_indices}->{output_indices}"

    def _apply_spectral_convolution(self, x: torch.Tensor, effective_modes: List[int], dtype: torch.dtype) -> torch.Tensor:
        """Apply the spectral convolution operation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, *spatial_dims)
            effective_modes (List[int]): List of effective modes for each spatial dimension.
            dtype (torch.dtype): Data type for the output tensor, default is torch.cfloat
        
        Returns:
            torch.Tensor: Output tensor of shape (B, out_features, *spatial_dims)

        """
        batch_size = x.shape[0]
        
        # Create output tensor
        out_ft_shape = [batch_size, self.out_features] + list(x.shape[2:])
        out_ft = torch.zeros(out_ft_shape, dtype=dtype, device=x.device)
        
        # Skip computation if any effective mode is zero
        if not all(mode > 0 for mode in effective_modes):
            return out_ft
        
        input_slices = [slice(None), slice(None)] 
        weight_slices = [slice(None), slice(None)]  
        
        for mode in effective_modes:
            input_slices.append(slice(None, mode))
            weight_slices.append(slice(None, mode))
        
        x_ft_truncated = x[tuple(input_slices)]
        weight_truncated = self.weight[tuple(weight_slices)]
        
        result = torch.einsum(self.einsum_eq, x_ft_truncated, weight_truncated)
        
        output_slices = [slice(None), slice(None)]
        for mode in effective_modes:
            output_slices.append(slice(None, mode))
        
        out_ft[tuple(output_slices)] = result
        
        return out_ft