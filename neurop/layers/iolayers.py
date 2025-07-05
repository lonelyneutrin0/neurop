import torch
from torch.types import Tensor
class ReadinLayer(torch.nn.Module):
    """
    Reads in input data and projects it to a higher dimensional space.
    """

    linear: torch.nn.Linear
    """Linear transformation layer that projects input data to a higher dimensional space."""

    def __init__(self, in_features: int, hidden_features: int):
        """
        Initializes the ReadinLayer with a linear transformation.

        Args:
            in_features (int): Number of features of input data.
            hidden_features (int): Number of hidden featuers of transformed data.
        
        Returns:
            None
        """
        super().__init__()
        self.linear = torch.nn.Linear(in_features, hidden_features)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ReadinLayer.

        Args:
            x (Tensor): Input data of shape (batch_size, in_features, d_1, d_2, .... d_n).
        
        Returns:
            Tensor: Transformed data of shape (batch_size, hidden_features, d_1, d_2, .... d_n).
        """ 

        # Reshape input to (batch_size, d_1, d_2, .... d_n, in_features) 
        x = x.permute(0, *range(2, x.ndim), 1)
        y = self.linear(x)
    
        # Reshape back to (batch_size, hidden_features, d_1, d_2, .... d_n)
        return y.permute(0, -1, *range(1, y.ndim - 1))  
class ReadoutLayer(torch.nn.Module):
    """
    Reads out data to lower dimensional space.
    """

    linear: torch.nn.Linear
    """Linear transformation layer that projects input data to a lower dimensional space."""
    
    def __init__(self, hidden_features: int, output_features: int):
        """
        Initializes the ReadoutLayer with a linear transformation.

        Args:
            hidden_features (int): Dimension of the input data.
            output_features (int): Dimension of the output data after projection.
        
        Returns:
            None
        """
        super().__init__()
        self.linear = torch.nn.Linear(hidden_features, output_features)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ReadoutLayer.

        Args:
            x (Tensor): Input data of shape (batch_size, hidden_features, d_1, d_2, .... d_n).
        
        Returns:
            Tensor: Transformed data of shape (batch_size, output_features, d_1, d_2, .... d_n).
        """

        # Reshape input to (batch_size, d_1, d_2, .... d_n, hidden_features)
        x = x.permute(0, *range(2, x.ndim), 1)
        y = self.linear(x)

        # Reshape back to (batch_size, output_features, d_1, d_2, .... d_n)
        return y.permute(0, -1, *range(1, y.ndim - 1))
    