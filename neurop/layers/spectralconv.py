import torch

from torch import Tensor


class SpectralConv1DLayer(torch.nn.Module):
    """
    Spectral Convolution 1 Dimensional Layer

    The input is assumed to have shape (B, C, L) where:
        - B is the batch size,
        - C is the number of input channels,
        - L is the length of the signal,
    
    The layer transforms the input to the frequency domain using a 1D FFT along the last dimension,
    $$x_f = \mathcal{F}[x]$$ 
    Then, the layer applies a pointwise multiplication (which is equivalent to a convolution in the time domain) as follows: 
    $$y_{b, o, k} = \sum_{c=0}^{C-1} W_{c, o, k} x_{b, c, k}$$
    where the contraction is along the features. Finally, the output is transformed back to the time domain using the inverse FFT:
    $$y = \mathcal{F}^{-1}[y]$$
    
    Args:
        in_features (int): Number of input channels.
        out_features (int): Number of output channels.
        modes (int): Number of Fourier modes to consider.
        init_scale (float): Initialization scale for weights.
    """

    def __init__(self, in_features: int, out_features: int, modes: int, init_scale: float = 1.0):
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
    The input is assumed to have shape (B, C, H, W) where:
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

    def __init__(self, in_features: int, out_features: int, mode_h: int, mode_w: int, init_scale: float = 1.0):
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
    The input is assumed to have shape (B, C, D, H, W) where:
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

    def __init__(self, in_features: int, out_features: int, mode_d: int, mode_h: int, mode_w: int, init_scale: float = 1.0):
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