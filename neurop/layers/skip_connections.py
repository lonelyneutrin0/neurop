import torch 
import torch.nn as nn

from torch.types import Tensor
from typing import Literal, Optional, Union

Connection = Literal['identity', 'linear', 'soft-gating']

class LinearConnection(nn.Module): 
    skip_weight: nn.Parameter
    """Skip weight to scale original data by"""

    skip_projection: Optional[nn.Linear]
    """Skip projection in case in_features != out_features"""

    transform_weight: nn.Parameter
    """Transform weight to scale the transformed data by"""

    bias: Optional[nn.Parameter]
    """Optional bias to add"""

    def __init__(self, in_features: int, out_features: int, n_dim: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_dim = n_dim
        
        # Handle channel dimension changes for FNO
        if in_features != out_features:
            self.skip_projection = nn.Linear(in_features, out_features, bias=False)
        else:
            self.skip_projection = None
            
        self.skip_weight = nn.Parameter(torch.ones(1,))
        self.transform_weight = nn.Parameter(torch.ones(1,))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features,))
        else:
            self.bias = None

    def forward(self, x: Tensor, transformed_x: Tensor) -> Tensor: 
        """ 
        Forward pass for linear skip connection

        Args:
            x (Tensor) [B, C, d_1, d_2 ....]: The input tensor
            transformed_x (Tensor) [B, C, d_1, d_2, ...] The transformed input tensor 
        
        Returns:
            Tensor (Shape depends on whether a projection is applied)
        """
        if self.skip_projection is not None:

            x = x.transpose(1, -1)  # (B, C, ...) -> (B, ..., C)
            x = self.skip_projection(x)
            x = x.transpose(1, -1)  # (B, ..., C) -> (B, C, ...)
        
        output = self.skip_weight * x + self.transform_weight * transformed_x

        if self.bias is not None:
            output += self.bias
        
        return output

class SoftGatingConnection(nn.Module): 

    in_features: int
    """Number of input features"""

    out_features: Optional[int] 
    """Number of output features"""

    weight: nn.Parameter
    """Weight matrix for the connection"""

    bias: Optional[nn.Parameter]
    """Optional bias to be added to the skip connection layer"""

    def __init__(self, in_features: int, n_dim: int, out_features: Optional[int] = None, bias: bool = False):
        super().__init__() 

        if out_features is not None and in_features != out_features:
            raise ValueError('In-features should equal out-features for soft-gating connections')

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.ones(1, self.in_features, *(1,) * n_dim))

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, self.in_features, *(1,) * n_dim))  # Initialize bias to zero
        else:
            self.bias = None
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the soft-gating connection

        Args:
            x (Tensor) [B, C, d_1, d_2, ...]: The input tensor 

        Returns:
            Tensor [B, C, d_1, d_2, ....]
        """
        if self.bias is not None:
            return self.weight * x + self.bias
        
        return self.weight * x
    
def skip_connection(
    in_features: int, 
    out_features: int, 
    n_dim: int, 
    bias: bool = False, 
    connection_type: Connection = 'soft-gating'
) -> Union[LinearConnection, SoftGatingConnection, nn.Identity]: 
    """
    Create a skip connection module.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features (optional, defaults to in_features)
        n_dim: Number of spatial dimensions (for soft-gating)
        bias: Whether to include bias parameters
        connection_type: Type of connection ('identity', 'linear', 'soft-gating')
    
    Returns:
        The appropriate skip connection module
    """
    
    if connection_type == 'identity':
        return nn.Identity()
    
    if connection_type == 'linear': 
        return LinearConnection(in_features=in_features, out_features=out_features, n_dim=n_dim, bias=bias)
    
    if connection_type == 'soft-gating':
        return SoftGatingConnection(in_features=in_features, n_dim=n_dim, out_features=out_features, bias=bias)
    
    raise ValueError(f"Unknown connection type: {connection_type}")