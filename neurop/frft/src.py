"""Fractional Fourier Transform (FrFT) implementation using Bluestein's Algorithm."""
import torch

from torch.types import Device
from torch import Tensor
from typing import Union, Optional


def frft(x: Tensor, alpha: Union[float, Tensor], *, dim: int = -1, device: Optional[Device] = None) -> Tensor: 
    """Compute the fractional Fourier transform (FRFT) of a tensor using the Chirp Multiplication Algorithm O(nlog n).
    
    The fractional Fourier transform is a generalization of the Fourier transform that allows for fractional orders. 
    This translates to a rotation in the time-frequency plane by an arbitrary angle. 
    Rotations can be interpreted as a composition of three shear transformations, which can be efficiently computed using the Chirp Multiplication Algorithm.

    Args:
        x (Tensor): The input tensor to perform the FrFT on 
        alpha (Union[float, Tensor]): The fractional order of the FrFT 
        dim (int): The dimension along which to perform the FrFT 
        device (Device): The device to run the computation on. Defaults to x.device

    Returns:
        Tensor 

    """
    if not device: 
        device = x.device

    N = x.size(dim)

    if N % 2 == 1: 
        raise ValueError("Signal size must be even")

    if not isinstance(alpha, torch.Tensor): 
        alpha = torch.Tensor(alpha)
    
    a = torch.fmod(alpha, 4)

    # Shift to [-2, 2] to handle special cases
    # 0 is a no-op 
    # 1 is a Fourier Transform   
    # 2 is a Time Reversal Operation 
    # -2 is a Time Reversal Operation

    if a > 2: 
        a -= 4
    
    elif a < -2: 
        a += 4 
    
    if a == 0.0: 
        return x + a * torch.zeros_like(x, device=device)

    elif a == 2.0 or a == -2.0:
        return _flip(x, dim = dim) + a * torch.zeros_like(x, device=device)
    
    # Zero padding operation
    b = _interp(x, dim = dim)
    z = torch.zeros_like(b, device = device).index_select(dim, torch.arange(0, N, device = device))

    x = torch.cat([z, b, z], dim = dim)

    res = x 

    # Decomposition Property of FrFT
    if (0 < a < 0.5) or (1.5 < a < 2):
        res = _frft_core(x, torch.Tensor(1.0), dim = dim)
        a -= 1
    
    if (-0.5 < a < 0) or (-2 < a < -1.5): 
        res = _frft_core(x, torch.Tensor(-1.0), dim = dim)
        a += 1
    
    # Select and downscale the signal
    res = _frft_core(res, a, dim = dim)
    res = torch.index_select(res, dim = dim, index = torch.arange(N, 3 * N, device = device))
    res = _decim(res, dim = dim)

    # Scale the first slice by two 
    y = torch.ones(res.size(dim), device = device)
    y[0] = 2

    res = _vecmul_ndim(res, y, dim = dim)
    return res 

def ifrft(x: Tensor, alpha: Union[float, Tensor], *, dim: int = -1, device: Optional[Device] = None) -> Tensor:
    """Compute the inverse fractional Fourier transform (iFRFT) of a tensor using the Chirp Multiplication Algorithm O(nlog n).
    
    Args:
        x (Tensor): The input tensor to perform the iFRFT on 
        alpha (Union[float, Tensor]): The fractional order of the iFRFT 
        dim (int): The dimension along which to perform the iFRFT 
        device (Device): The device to run the computation on. Defaults to x.device
    Returns:
        Tensor: The result of the inverse fractional Fourier transform

    """
    return frft(x, -alpha, dim = dim, device = device)

def _flip(x: Tensor, *, dim: int = -1) -> Tensor:
    """Reverse the order of elements, keeping the first slice in place."""
    first, remaining = torch.tensor_split(x, (1,), dim=dim)
    return torch.concat((first, remaining.flip(dim)), dim=dim)

def _decim(x: Tensor, *, dim: int = -1, device: Optional[Device] = None) -> Tensor: 
    """Decimation by 2 operation."""
    if not device:
        device = x.device

    t = torch.arange(0, x.size(dim), 2, device=device)
    return x.index_select(dim, t)

def _interp(x: Tensor, *, dim: int = -1, device: Optional[Device] = None) -> Tensor: 
    """Bandlimited interpolation function (Calls _interp_real for real and complex parts separately)."""
    if not device:
        device = x.device

    if x.is_complex():
        return _interp_real(x.real, dim=dim, device=device) + 1j * _interp_real(x.imag, dim=dim, device=device)
    
    return _interp_real(x, dim=dim, device=device)

def _interp_real(x: Tensor, *, dim: int = -1, device: Optional[Device] = None) -> Tensor: 
    """Prevent aliasing by implementing a low pass filter."""
    if not device:
        device = x.device

    N = x.size(dim)
    N1 = N // 2 + N % 2
    N2 = 2 * N - (N // 2)

    upsampled_signal = _upsample(x, dim=dim)
    xf = torch.fft.fft(upsampled_signal, dim=dim)

    xf = torch.index_fill(xf, dim, torch.arange(N1, N2, device=device), 0)
    return 2 * torch.real(torch.fft.ifft(xf, dim = dim))

def _upsample(x: Tensor, *, dim: int = -1, device: Optional[Device] = None) -> Tensor: 
    """Insert zeros between each element of input data for upsampling."""
    if not device:
        device = x.device
    
    upsampled_signal = x.repeat_interleave(2, dim=dim)
    idx = torch.arange(1, upsampled_signal.size(dim), 2, device=device)
    return torch.index_fill(upsampled_signal, dim, idx, 0)

def _vecmul_ndim(
    tensor: torch.Tensor,
    vector: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Multiply two tensors (`torch.mul()`) along a given dimension."""
    return torch.einsum(_get_mul_dim_einstr(len(tensor.shape), dim), tensor, vector)

def _get_mul_dim_einstr(dim_count: int, req_dim: int) -> str:
    if req_dim < -dim_count or req_dim >= dim_count:
        raise ValueError("Dimension size error.")
    dim = torch.remainder(req_dim, torch.tensor(dim_count))
    diff = dim_count - dim
    remaining_str = "".join([chr(num) for num in range(97, 97 + diff)])
    return f"...{remaining_str},a->...{remaining_str}"

def _frft_core(x: Tensor, a: Union[float, Tensor], *, dim: int = -1, device: Optional[Device] = None) -> Tensor:
    
    if not device:
        device = x.device

    if not isinstance(a, torch.Tensor):
        a = torch.Tensor(a)

    N = x.size(dim)
    N_end = N // 2
    N_start = -(N % 2 + N_end)

    # Compute the factor for energy scaling
    dx = torch.sqrt(torch.tensor(N, device=device))

    # Compute the 
    phi = a * torch.pi / 2
    alpha = -1j * torch.pi * torch.tan(phi/2)
    beta = 1j * torch.pi / torch.sin(phi)

    Aphi_num = torch.exp(-1j * (torch.pi * torch.sign(torch.sin(phi)) / 4 - phi / 2))
    Aphi_denom = torch.sqrt(torch.abs(torch.sin(phi)))
    Aphi = Aphi_num / Aphi_denom

    t = torch.arange(N_start, N_end, device=device) / dx
    chirp = torch.exp(alpha * t**2)
    chirped_x = _vecmul_ndim(x, chirp, dim=dim)

    t_ext = torch.arange(-N + 1, N, device=device) / dx
    chirp_ext = torch.exp(beta * t_ext**2)

    N2 = chirp_ext.size(0)

    # Compute the next power of 2 
    power_two = 2 ** torch.ceil(torch.log2(torch.tensor(N2 + N - 1))).int() 

    Hc = torch.fft.ifft(
        _vecmul_ndim(
            torch.fft.fft(chirped_x, n = power_two, dim=dim),
            torch.fft.fft(chirp_ext, n = power_two)
            ),
            dim = dim
        )
    
    Hc = torch.index_select(Hc, dim, torch.arange(N-1, 2 * N - 1, device=device))

    result = _vecmul_ndim(Hc, Aphi * chirp, dim = dim) / dx

    if N % 2 == 1: 
        return torch.roll(result, -1, dims=(dim,))
    
    return result