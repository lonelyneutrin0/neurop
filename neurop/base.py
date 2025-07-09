"""Base class for Neural Operators in neurop."""
from abc import ABC, abstractmethod

import torch 
from torch.types import Tensor

class NeuralOperator(torch.nn.Module, ABC):
    """Abstract class for Neural Operators."""

    def __init__(self, *args, **kwargs):
        """__init__ method to initialize NeuralOperator."""
        super().__init__()
        pass
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Forward pass to be implemented by subclasses."""
        pass