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
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass to be implemented by subclasses."""
        pass

class NeuralOperatorUnit(torch.nn.Module, ABC):
    """Abstract class for Neural Operator Units."""

    def __init__(self): 
        """__init__ method to initialize NeuralOperatorUnit."""
        super().__init__()
        pass 

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass to be implemented by subclasses."""
        pass

class NeuralOperatorBuilder(ABC): 
    """Abstract class for building Neural Operators."""

    @abstractmethod
    def __init__(self): 
        """__init__ method for the NeuralOperatorBuilder."""
        super().__init__()
        pass

    @abstractmethod
    def set_architecture(self, *args, **kwargs) -> Self: 
        """Set the architecture for the Neural Operator.""" 
        pass
    
    @abstractmethod
    def set_activation_function(self, *args, **kwargs) -> Self:
        """Set the activation function for the Neural Operator."""
        pass 

    @abstractmethod
    def set_conv_module(self, *args, **kwargs) -> Self:
        """Set the convolution module for the Neural Operator."""
        pass

    @abstractmethod
    def set_feature_mlp(self, *args, **kwargs) -> Self:
        """Use a Feature MLP for the Neural Operator."""
        pass

    @abstractmethod
    def use_bias(self, *args, **kwargs) -> Self:
        """Toggle bias for the Neural Operator."""
        pass

    @abstractmethod
    def use_learnable_normalizers(self, *args, **kwargs) -> Self:
        """Configure settings for learnable normalizers for the Neural Operator."""
        pass

    @abstractmethod
    def build(self, *args, **kwargs) -> NeuralOperator:
        """Build and return the NeuralOperator."""
        pass

class NeuralOperatorUnitBuilder(ABC):
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
        """Set the activation function for the NeuralOperatorUnit."""
        pass 

    @abstractmethod
    def set_conv_module(self, *args, **kwargs) -> Self:
        """Set the convolution module for the NeuralOperatorUnit."""
        pass

    @abstractmethod
    def set_feature_mlp(self, *args, **kwargs) -> Self:
        """Use a Feature MLP for the NeuralOperatorUnit."""
        pass

    @abstractmethod
    def use_bias(self, *args, **kwargs) -> Self:
        """Toggle bias for the NeuralOperatorUnit."""
        pass

    @abstractmethod
    def use_learnable_normalizers(self, *args, **kwargs) -> Self:
        """Configure settings for learnable normalizers for the NeuralOperatorUnit."""
        pass

    @abstractmethod
    def build(self, *args, **kwargs) -> NeuralOperatorUnit:
        """Build and return the NeuralOperatorUnit."""
        pass