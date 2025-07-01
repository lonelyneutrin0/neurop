from abc import ABC, abstractmethod

import torch 
from torch.types import Tensor
from typing import Callable, Optional, TypedDict

from pathlib import Path

class Metrics(TypedDict, total=False):
    """
    TypedDict to hold metrics for evaluation.
    All fields are optional for flexibility
    """
    loss: float
    accuracy: float
class Layer(torch.nn.Module, ABC):
    """
    Abstract base class for all layers in a NeuralOperator.
    """
    @abstractmethod
    def forward(self, x) -> Tensor:
        pass

class NeuralOperator(torch.nn.Module, ABC):
    """
    Abstract class for Neural Operators.

    Attributes: 
        readin (Layer):
            Reads in input data and projects to higher dimensional space 
        kernel_integral (Layer):
            The kernel integral operator that performs the main computation
        readout (Layer):
            Reads out data to lower dimensional space
        optimizer (torch.optim.Optimizer):
            Optimization algorithm to choose. Defaults to Adam(lr=1e-3)
        activation_function (Callable[[Tensor], Tensor]):
            Activation to introduce nonlinearity between kernel operations
    """

    def __init__(self,
                 readin: Layer,
                 kernel_integral: Layer,
                 readout: Layer,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 activation_function: Callable[[Tensor], Tensor] = torch.relu
                 ):
        """
        __init__ method to initialize NeuralOperator
        """
        super().__init__()

        self.optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=1e-3)
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass to be implemented by subclasses.
        """
        pass

    @abstractmethod
    def loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        """
        Loss function specific to the problem/operator.
        """
        pass

    @abstractmethod
    def train_step(self, x: Tensor, y: Tensor) -> Tensor:
        """
        One training step: forward + loss + backward + optimizer step.
        """
        pass

    @abstractmethod
    def evaluate(self, x: Tensor, y: Tensor) -> float:
        """
        Evaluate model performance on validation/test data.
        """
        pass

    @abstractmethod
    def save(self, path: Path): 
        """
        Write model parameters to a file  
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path):
        """
        Load a neural operator from a file
        """
        pass

    def to_device(self, device: torch.device): 
        """
        Send data to a Torch device 
        """
        self.readin.to(device)
        self.kernel_integral.to(device)
        self.readout.to(device)

    @abstractmethod
    def calculate_metrics(self, ground_truth: Tensor, predicted: Tensor) -> Metrics: 
        """
        Compute the desired metrics and output a TypedDict 
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """
        Brief overview of the model architecture
        """
        pass