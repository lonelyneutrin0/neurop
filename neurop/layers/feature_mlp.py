"""Feature MLP Layer."""
from torch import nn

from torch.types import Tensor
from typing import Type, List, Optional

class ConvFeatureMLP(nn.Module):
    """Feature MLP Layer.""" 

    in_features: int
    """Number of input features."""

    out_features: int
    """Number of output features."""

    layer_features: Optional[List[int]]
    """Number of features in each layer of the MLP."""

    depth: int
    """Depth of the MLP."""

    n_dim: int
    """Number of dimensions."""

    activation_function: nn.Module
    """Activation function to use in the MLP."""

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 depth: int, 
                 n_dim: int, 
                 activation_function: Type[nn.Module] = nn.ReLU,
                 *,
                layer_features: Optional[List[int]] = None,
                ):
        """Initialize the ConvFeatureMLP layer.
        
        Args:
            in_features (int): Number of input features.
            hidden_features (int): Number of hidden features.
            out_features (int): Number of output features.
            depth (int): Depth of the MLP.
            n_dim (int): Number of dimensions.
            activation_function (Type[nn.Module], optional): Activation function to use. Defaults to nn.ReLU.
            layer_features (List[int]): List of integers representing the number of features in each layer. Used for arbitrary architectures.

        """
        super().__init__()
        
        if not layer_features:
            self._build_default_arch(in_features, hidden_features, out_features, depth)
        else:
            self._build_arbitrary_arch(layer_features, depth)

        self.depth = depth
        self.n_dim = n_dim
        self.activation_function = activation_function()

    def _build_default_arch(self, in_features, hidden_features, out_features, depth):
        """Build the default architecture for the MLP.

        Args:
            in_features (int): Number of input features.
            hidden_features (int): Number of hidden features.
            out_features (int): Number of output features.
            depth (int): Depth of the MLP.

        Returns:
            None

        """
        if depth < 1:
            raise ValueError("Depth must be at least 1.")

        layer_features = [in_features] + [hidden_features] * (depth - 2) + [out_features]

        self._build_arbitrary_arch(layer_features, depth)

    def _build_arbitrary_arch(self, layer_features: List[int], depth: int):
        """Build an arbitrary architecture for the MLP.

        Args:
            layer_features (List[int]): List of integers representing the number of features in each layer.
            depth (int): Depth of the MLP.

        Returns:
            None

        """
        if len(layer_features) != depth+1: 
            raise ValueError("The length of layers must be equal to depth + 1.")
        
        layers = nn.ModuleList()

        for i in range(depth):
            layers.append(
                nn.Conv1d(
                    in_channels=layer_features[i],
                    out_channels=layer_features[i + 1],
                    kernel_size=1,
                )
            )

        self.layers = layers
        self.in_features = layer_features[0]
        self.out_features = layer_features[-1]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the ConvFeatureMLP layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features, d_1, d_2, ...).
        
        Returns:
            Tensor: Output tensor of shape (batch_size, out_features, d_1, d_2, ....).

        """
        x_shape = list(x.shape)

        spatial_dims = len(x_shape[2:])

        if spatial_dims > 1:
            x = x.reshape((*x_shape[:2], -1)) # (batch_size, in_features, d_1 * d_2 * ...)

        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i < (self.depth - 1):
                x = self.activation_function(x)
        
        if spatial_dims > 1:
            x = x.reshape(x_shape[0], self.out_features, *x_shape[2:])

        return x 
    
class LinearFeatureMLP(nn.Module):
    """Feature MLP Layer using Linear layers."""

    in_features: int
    """Number of input features."""

    hidden_features: int
    """Number of hidden features."""

    out_features: int
    """Number of output features."""

    depth: int
    """Depth of the MLP."""

    n_dim: int
    """Number of dimensions."""

    activation_function: nn.Module
    """Activation function to use in the MLP."""

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 depth: int,
                 n_dim: int,
                 activation_function: Type[nn.Module] = nn.ReLU,
                 *,
                layer_features: Optional[List[int]] = None
                 ):
        """Initialize the LinearFeatureMLP layer.

        Args:
            in_features (int): Number of input features.
            hidden_features (int): Number of hidden features.
            out_features (int): Number of output features.
            depth (int): Depth of the MLP.
            n_dim (int): Number of dimensions.
            activation_function (nn.Module, optional): Activation function to use. Defaults to nn.ReLU.
            layer_features (List[int], optional): List of integers representing the number of features in each layer. Used for arbitrary architectures.
        
        """
        super().__init__()
        if not layer_features:
            self._build_default_arch(in_features, hidden_features, out_features, depth)
        else:
            self._build_arbitrary_arch(layer_features, depth)

        self.depth = depth
        self.n_dim = n_dim
        self.activation_function = activation_function()
    
    def _build_default_arch(self, in_features, hidden_features, out_features, depth):
        """Build the default architecture for the LinearFeatureMLP.

        Args:
            in_features (int): Number of input features.
            hidden_features (int): Number of hidden features.
            out_features (int): Number of output features.
            depth (int): Depth of the MLP.

        Returns:
            None

        """
        if depth < 1:
            raise ValueError("Depth must be at least 1.")

        layer_features = [in_features] + [hidden_features] * (depth - 2) + [out_features]

        self._build_arbitrary_arch(layer_features, depth)

    def _build_arbitrary_arch(self, layer_features: List[int], depth: int):
        """Build an arbitrary architecture for the LinearFeatureMLP.

        Args:
            layer_features (List[int]): List of integers representing the number of features in each layer.
            depth (int): Depth of the MLP.

        Returns:
            None

        """
        if len(layer_features) != depth + 1:
            raise ValueError("The length of layers must be equal to depth + 1.")
        
        layers = nn.ModuleList()

        for i in range(depth):
            layers.append(
                nn.Linear(
                    in_features=layer_features[i],
                    out_features=layer_features[i + 1],
                )
            )
        self.in_features = layer_features[0]
        self.out_features = layer_features[-1]

        self.layers = layers
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the LinearFeatureMLP layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features, d_1, d_2, ...).
        
        Returns:
            Tensor: Output tensor of shape (batch_size, out_features, d_1, d_2, ....).

        """
        if x.shape[1] != self.in_features:
            raise ValueError(f"Input tensor must have {self.in_features} features, but got {x.shape[1]}.")
        
        x = x.permute(0, *range(2, x.ndim), 1)
        
        for i, layer in enumerate(self.layers): 
            x = layer(x)

            if i < (self.depth - 1):
                x = self.activation_function(x)

        return x.permute(0, -1, *range(1, x.ndim - 1))
