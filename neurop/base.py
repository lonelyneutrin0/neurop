"""Base class for Neural Operators in neurop."""
from abc import ABC, abstractmethod

import torch 

from typing import Self

class NeuralOperator(torch.nn.Module, ABC):
    """Abstract class for Neural Operators."""

    def __init__(self):
        """__init__ method to initialize NeuralOperator."""
        super().__init__()
        pass
    
    @abstractmethod
    def forward(self) -> torch.Tensor:
        """Forward pass to be implemented by subclasses."""
        pass

class NeuralOperatorUnitBuilder(torch.nn.Module, ABC):
    """Abstract class for a single Neural Operator Layer."""

    @abstractmethod
    def __init__(self):
        """__init__ method for the NeuralOperatorUnit."""
        super().__init__()
        pass 

    @abstractmethod
    def set_architecture(self, *args, **kwargs) -> Self: 
        """Set the architecture for the NeuralOperatorUnit.""" 
        pass
    
    @abstractmethod
    def set_activation_function(self, *args, **kwargs) -> Self:
        pass 

    @abstractmethod
    def set_conv_module(self, *args, **kwargs) -> Self:
        """Set the convolution module for the NeuralOperatorUnit."""
        pass

    @abstractmethod
    def set_feature_mlp(self, *args, **kwargs) -> Self:
        """Use a Feature MLP for the Unit."""
        pass

    @abstractmethod
    def set_normalizer(self, *args, **kwargs) -> Self: 
        """Set a normalizer for the NeuralOperatorUnit."""
        pass

    @abstractmethod
    def set_scaling(self, *args, **kwargs) -> Self:
        """Set the initial scaling and bias for the NeuralOperatorUnit."""
        pass