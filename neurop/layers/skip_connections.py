"""Skip connections for neural operators."""
import torch 
import torch.nn as nn

from torch.types import Tensor
from typing import Literal, Optional, Union

Connection = Union['LinearConnection', 'SoftGatingConnection', 'IdentityConnection', 'ResidualConnection']
ConnectionType = Literal['identity', 'linear', 'soft-gating', 'residual']

class LinearConnection(nn.Module): 
    """Linear skip connection that applies a linear transformation to the input and combines it with the transformed input."""

    skip_weight: nn.Parameter
    """Skip weight to scale original data by"""

    skip_projection: Optional[nn.Linear]
    """Skip projection in case in_features != out_features"""

    transform_weight: nn.Parameter
    """Transform weight to scale the transformed data by"""

    bias: Optional[nn.Parameter]
    """Optional bias to add"""

    def __init__(self, in_features: int, out_features: int, n_dim: int, bias: bool = False):
        """Initialize the linear skip connection.
        
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            n_dim (int): Number of spatial dimensions.
            bias (bool): Whether to include a bias term in the connection.
        
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_dim = n_dim
        
        if in_features != out_features:
            self.skip_projection = nn.Linear(in_features, out_features, bias=False)
        else:
            self.skip_projection = None

        self.skip_weight = nn.Parameter(torch.ones(1, out_features, *(1,) * n_dim))
        self.transform_weight = nn.Parameter(torch.ones(1, out_features, *(1,) * n_dim))

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_features, *(1,) * n_dim))
        else:
            self.bias = None

    def forward(self, x: Tensor, transformed_x: Tensor) -> Tensor: 
        """ 
        Forward pass for linear skip connection.

        Args:
            x (Tensor): The input tensor [B, C, d_1, d_2 ....]
            transformed_x (Tensor): The transformed input tensor [B, C, d_1, d_2, ...]
        
        Returns:
            Tensor (Shape depends on whether a projection is applied)

        """
        if self.skip_projection is not None:

            original_shape = x.shape

            x = x.permute(0, *range(2, len(original_shape)), 1).contiguous() # [B, d_1, d_2, ..., C]

            x = x.view(-1, self.in_features) # [B * spatial_dims, C]
            x = self.skip_projection(x)

            new_shape = (original_shape[0],) + original_shape[2:] + (self.out_features,) # [B, d_1, d_2, ..., C']
            x = x.view(*new_shape)
 
            x = x.permute(0, -1, *range(1, len(original_shape)-1)).contiguous() # [B, C', d_1, d_2, ...]
        
        output = self.skip_weight * x + self.transform_weight * transformed_x

        if self.bias is not None:
            output += self.bias
        
        return output
        
class SoftGatingConnection(nn.Module): 
    """Soft-gating connection that applies a learnable gating mechanism to the skip connection."""

    in_features: int
    """Number of input features"""

    weight: nn.Parameter
    """Weight matrix for the connection"""

    bias: Optional[nn.Parameter]
    """Optional bias to be added to the skip connection layer"""

    def __init__(self, in_features: int, n_dim: int, bias: bool = False):
        """Initialize the soft-gating connection.
        
        Args:
            in_features (int): Number of input features.
            n_dim (int): Number of spatial dimensions.
            bias (bool): Whether to include a bias term in the connection.

        """
        super().__init__() 

        self.in_features = in_features
        self.weight = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, self.in_features, *(1,) * n_dim))
        else:
            self.bias = None
        
    def forward(self, x: Tensor, transformed_x: Tensor) -> Tensor:
        """Forward pass for the soft-gating connection.

        Args:
            x (Tensor): The input tensor [B, C, d_1, d_2, ...]
            transformed_x (Tensor) : The transformed input tensor [B, C, d_1, d_2, ...]

        Returns:
            Tensor [B, C, d_1, d_2, ....]

        """
        gate = torch.sigmoid(self.weight)
        output = gate * x + (1 - gate) * transformed_x

        if self.bias is not None:
            output += self.bias

        return output

class ResidualConnection(nn.Module):
    """A simple residual connection that adds the input to the output.

    This is useful for architectures where you want to add the input back to the output.
    """
    
    def forward(self, x: Tensor, transformed_x: Tensor) -> Tensor:
        """Forward pass for the residual connection.

        Args:
            x (Tensor): Input tensor.
            transformed_x (Tensor): Transformed input tensor.

        Returns:
            Tensor: Output tensor, which is the sum of input and transformed input.

        """
        return x + transformed_x

class IdentityConnection(nn.Module):
    """Identity connection that returns the transformed input.

    This maintains consistent interface with other skip connections.
    """
    
    def forward(self, x: Tensor, transformed_x: Tensor) -> Tensor:
        """Forward pass for the identity connection.

        Args:
            x (Tensor): Input tensor (ignored).
            transformed_x (Tensor): Transformed input tensor.

        Returns:
            Tensor: The transformed input tensor.

        """
        return transformed_x

def create_skip_connection(
    in_features: int, 
    out_features: int, 
    n_dim: int, 
    bias: bool = False, 
    connection_type: ConnectionType = 'soft-gating'
) -> Connection: 
    """Create a skip connection module.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features (optional, defaults to in_features)
        n_dim: Number of spatial dimensions (for soft-gating)
        bias: Whether to include bias parameters
        connection_type: Type of connection ('identity', 'linear', 'soft-gating', 'residual')
    
    Returns:
        The appropriate skip connection module

    """
    if connection_type == 'identity':
        return IdentityConnection()
    
    if connection_type == 'linear': 
        return LinearConnection(in_features=in_features, out_features=out_features, n_dim=n_dim, bias=bias)
    
    if connection_type == 'soft-gating':
        return SoftGatingConnection(in_features=in_features, n_dim=n_dim, bias=bias)

    if connection_type == 'residual':
        return ResidualConnection()
    
    raise ValueError(f"Unknown connection type: {connection_type}")
