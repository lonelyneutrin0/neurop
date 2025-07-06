import torch
from ..base import NeuralOperator

from torch.types import Tensor
from typing import List, Type

class DeepONet(NeuralOperator):
    """
    DeepONet is a neural operator that approximates a mapping from functions to functions.
    It consists of a trunk network and a branch network. 
    The trunk network processes the input coordinates to determine basis functions,
    while the branch network processes the input function values to determine coefficients. 
    The output is a weighted sum of the basis functions, where the weights are determined by the coefficients.

    """

    branch_layers: torch.nn.Sequential
    """
    Branch network of the DeepONet, which processes the input function to determine coefficients.
    """

    trunk_layers: torch.nn.Sequential
    """
    Trunk network of the DeepONet, which processes the input coordinates to determine basis functions.
    """

    bias: torch.nn.Parameter
    """
    Bias term added to the output of the DeepONet.
    """

    def __init__(self, 
                 branch_input_dim: int, 
                 trunk_input_dim: int, 
                 latent_dim: int, 
                 hidden_dim: int = 64, 
                 depth: int = 3,
                 activation_function: Type[torch.nn.Module] = torch.nn.ReLU
    ):
        """
        Initializes the DeepONet operator.
        """ 
        
        super().__init__()

        branch_layers: List[torch.nn.Module] = []

        # Branch Network
        branch_layers.append(torch.nn.Linear(branch_input_dim, hidden_dim))
        branch_layers.append(activation_function())

        for _ in range(depth - 1):
            branch_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            branch_layers.append(activation_function())

        branch_layers.append(torch.nn.Linear(hidden_dim, latent_dim))

        self.branch_layers = torch.nn.Sequential(*branch_layers)

        # Trunk Network
        trunk_layers: List[torch.nn.Module] = []

        trunk_layers.append(torch.nn.Linear(trunk_input_dim, hidden_dim))
        trunk_layers.append(activation_function())

        for _ in range(depth - 1):
            trunk_layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            trunk_layers.append(activation_function())
        
        trunk_layers.append(torch.nn.Linear(hidden_dim, latent_dim))

        self.trunk_layers = torch.nn.Sequential(*trunk_layers)

        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Forward pass of the DeepONet operator. Handles both single point and multi-point evaluation.
        
        Args:
            x (Tensor): Input tensor for the branch network, typically representing function values. (batch_size, branch_input_dim)
            y (Tensor): Input tensor for the trunk network, typically representing coordinates or points in space. (batch_size, num_points, trunk_input_dim)
        
        Returns:
            Tensor: Output tensor representing the weighted sum of basis functions, where weights are determined by the branch network.
        """

        if y.dim() == 2: # If y is 2D, we assume it is a single point for each batch
            y = y.unsqueeze(1)

        batch_size, num_points, dim = y.shape 
        
        # Process the branch input
        b_out = self.branch_layers(x)  # (batch_size, latent_dim)  
        b_out = b_out.unsqueeze(1) # shape should be (batch_size, 1, latent_dim) for braodcasting in the sum later

        # Process the trunk input
        y_flattened = y.view(-1, dim) # shape should be (batch_size * num_points, trunk_input_dim)

        t_out = self.trunk_layers(y_flattened)  # (batch_size * num_points, latent_dim)
        t_out = t_out.view(batch_size, num_points, -1)  # shape should be (batch_size, num_points, latent_dim)

        return torch.sum(b_out * t_out, dim=2) + self.bias.to(x.device)  # (batch_size, num_points)
    