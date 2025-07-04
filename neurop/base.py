from abc import ABC, abstractmethod

import torch 
from torch.types import Tensor
from typing import TypedDict


class Metrics(TypedDict, total=False):
    """
    TypedDict to hold metrics for evaluation.
    All fields are optional for flexibility
    """
    loss: float
    accuracy: float
class NeuralOperator(torch.nn.Module, ABC):
    """
    Abstract class for Neural Operators.
    """

    def __init__(self,):
        """
        __init__ method to initialize NeuralOperator
        """
        super().__init__()
        pass
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass to be implemented by subclasses.
        """
        pass