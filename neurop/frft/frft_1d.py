import torch

from torch.types import Tensor, Device


def frft_1d(x: Tensor, N: int, device: Device, alpha: float = 1.0) -> Tensor: 
    """
    Compute the fractional Fourier transform (FRFT) of a 1D tensor using the Chirp Multiplication Algorithm O(nlog n).
    
    The fractional Fourier transform is a generalization of the Fourier transform that allows for fractional orders. 
    This translates to a rotation in the time-frequency plane by an arbitrary angle. 
    Rotations can be interpreted as a composition of three shear transformations, which can be efficiently computed using the Chirp Multiplication Algorithm.

    Args:
        x (Tensor): Input tensor to be FrFT'd 
        N (int): Length 
        device (Device): Torch Device to run the computation on 
        alpha (float): Fractional order of the FrFT
    """

    alpha %= 4
    phi = torch.tensor(torch.pi/2 * alpha, device=device)

    # Special Cases
    if abs(alpha) < 1e-12:
        return x.clone()
    elif abs(alpha - 1) < 1e-12:
        return torch.fft.fft(x.to(torch.complex64)) / torch.sqrt(torch.tensor(N, device=device))
    elif abs(alpha - 2) < 1e-12:
        return torch.flip(x, [0])
    elif abs(alpha - 3) < 1e-12:
        return torch.fft.ifft(x.to(torch.complex64)) * torch.sqrt(torch.tensor(N, device=device))

    if not x.is_complex():
        x = x.to(torch.complex64)
    
    cot_phi = 1.0 / torch.tan(phi)
    csc_phi = 1.0 / torch.sin(phi)
    
    t = torch.arange(N, device=device, dtype=torch.float32)

    # Pre-chirp multiplication
    pre_chirp = torch.exp(1j * torch.pi * cot_phi * t**2)
    x_pre = pre_chirp * x

    M = 2 * N - 1
    t_ext = torch.arange(-N+1, N, device=device, dtype=torch.float32)
    h_ext = torch.exp(1j * torch.pi * csc_phi * t_ext**2)

    x_padded = torch.zeros(M, device=device, dtype=torch.complex64)
    x_padded[:N] = x_pre

    scale = torch.sqrt(torch.tensor(M, dtype=torch.float32, device=device))
    x_fft = torch.fft.fft(x_padded) / scale
    h_fft = torch.fft.fft(h_ext) / scale
    y_fft = x_fft * h_fft
    y = torch.fft.ifft(y_fft) * scale
    y_result = y[:N]

    # Post-chirp multiplication
    post_chirp = torch.exp(-1j * torch.pi * cot_phi * t**2)
    norm_factor = torch.sqrt(torch.abs(torch.sin(phi))) * torch.exp(-1j * torch.pi * torch.sign(torch.sin(phi)) / 4)

    return (norm_factor * post_chirp * y_result) / torch.sqrt(torch.tensor(N, device=device))