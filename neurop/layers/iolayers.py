import torch

from ..base import Layer

class ReadinLayer(Layer):
    """
    Reads in input data and projects it to a higher dimensional space.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ReadoutLayer(Layer):
    """
    Reads out data to lower dimensional space.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    