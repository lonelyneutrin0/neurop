from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch 
from torch.types import Tensor
from typing import Callable, Optional

from pathlib import Path

@dataclass
class NeuralOperator(torch.nn.Module, ABC):
    """
    Abstract class for Neural Operators.

    Instance Attributes: 

        readin: (torch.nn.Module) 
            Reads in input data and projects to higher dimensional space 

        kernel_integral: (torch.nn.Module) 
            Performs the kernel operator on the data

        readout: (torch.nn.Module)
            Reads out data to lower dimensional space

        optimizer: (torch.optim.Optimizer) 
            Optimization algorithm to choose. Defaults to Adam(lr=1e-3)
        
        parameters: (torch.nn.Parameter)
            Neural operator parameters
        
        activation_function: (Callable[[Tensor], Tensor])
            Activation to introduce nonlinearity between kernel operations
    """

    readin: torch.nn.Module
    kernel_integral: torch.nn.Module
    readout: torch.nn.Module
    optimizer: Optional[torch.optim.Optimizer]
    activation_function: Callable[[Tensor], Tensor] = torch.relu

    def __post_init__(self,):
        super().__init__()

        if not self.optimizer:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    
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

    @abstractmethod
    def load(self, path: Path):
        """
        Load model parameters from a file 
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
    def calculate_metrics(self, ground_truth: Tensor, predicted: Tensor): 
        """
        Compute the desired metrics and output a TypedDict 
        """
        pass