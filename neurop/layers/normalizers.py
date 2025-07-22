"""Normalization layers for neural operators."""
from torch import nn 
import torch

class BatchNormalizer(nn.Module):
    """Batch Normalization Layer."""

    tol: float
    """Tolerance for numerical stability in normalization."""

    learnable: bool
    """Whether the normalization parameters are learnable."""

    gamma: nn.Parameter
    """Learnable scale parameter for normalization, initialized to ones."""

    beta: nn.Parameter
    """Learnable shift parameter for normalization, initialized to zeros."""

    def __init__(self, num_features: int, ndim: int, tol: float = 1e-10, learnable: bool = True):
        """Initialize the batch normalizer.
        
        Args:
            num_features (int): Number of features in the input.
            tol (float): Tolerance for numerical stability in normalization.
            ndim (int): Number of dimensions in the input tensor.
            learnable (bool): Whether the normalization parameters are learnable.

        """
        super().__init__()
        self.tol = tol
        self.learnable = learnable

        if self.learnable:
            self.gamma = nn.Parameter(torch.ones(((1, num_features) + (1,) * (ndim - 2)))) 
            self.beta = nn.Parameter(torch.zeros(((1, num_features) + (1,) * (ndim - 2)))) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the input tensor x using batch normalization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, d_1, d_2, ...).
        
        Returns:
            torch.Tensor: Normalized tensor of the same shape as x.

        """
        dims = (0,) + tuple(range(2, x.ndim))
        mean = torch.mean(x, dim=dims, keepdim=True)
        var = torch.var(x, dim=dims, keepdim=True, unbiased=False)
        
        if self.learnable:
            return self.gamma * (x - mean) / torch.sqrt(var + self.tol) + self.beta

        return (x - mean) / torch.sqrt(var + self.tol) 

class LayerNormalizer(nn.Module):
    """Layer Normalization Layer."""

    tol: float
    """Tolerance for numerical stability in normalization."""

    learnable: bool
    """Whether the normalization parameters are learnable."""

    gamma: nn.Parameter
    """Learnable scale parameter for normalization, initialized to ones."""

    beta: nn.Parameter
    """Learnable shift parameter for normalization, initialized to zeros."""

    def __init__(self, num_features: int, ndim: int, tol: float = 1e-10, learnable: bool = True):
        """Initialize the layer normalizer.
        
        Args:
            num_features (int): Number of features in the input.
            tol (float): Tolerance for numerical stability in normalization.
            ndim (int): Number of dimensions in the input tensor.
            learnable (bool): Whether the normalization parameters are learnable.

        """
        super().__init__()
        self.tol = tol
        self.learnable = learnable

        if self.learnable:
            self.gamma = nn.Parameter(torch.ones(((1, num_features) + (1,) * (ndim - 2)))) 
            self.beta = nn.Parameter(torch.zeros(((1, num_features) + (1,) * (ndim - 2)))) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the input tensor x using layer normalization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, d_1, d_2, ...).
        
        Returns:
            torch.Tensor: Normalized tensor of the same shape as x.

        """
        dims = tuple(range(1, x.ndim))
        mean = torch.mean(x, dim=dims, keepdim=True)
        var = torch.var(x, dim=dims, keepdim=True, unbiased=False)

        if self.learnable:
            return self.gamma * (x - mean) / torch.sqrt(var + self.tol) + self.beta

        return (x - mean) / torch.sqrt(var + self.tol)

class InstanceNormalizer(nn.Module):
    """Instance Normalization Layer."""

    tol: float
    """Tolerance for numerical stability in normalization."""

    learnable: bool
    """Whether the normalization parameters are learnable."""

    gamma: nn.Parameter
    """Learnable scale parameter for normalization, initialized to ones."""

    beta: nn.Parameter
    """Learnable shift parameter for normalization, initialized to zeros."""

    def __init__(self, num_features: int, ndim: int, tol: float = 1e-10, learnable: bool = True):
        """Initialize the batch normalizer.
        
        Args:
            num_features (int): Number of features in the input.
            tol (float): Tolerance for numerical stability in normalization.
            ndim (int): Number of dimensions in the input tensor.
            learnable (bool): Whether the normalization parameters are learnable.

        """
        super().__init__()
        self.tol = tol
        self.learnable = learnable

        if self.learnable:
            self.gamma = nn.Parameter(torch.ones(((1, num_features) + (1,) * (ndim - 2)))) 
            self.beta = nn.Parameter(torch.zeros(((1, num_features) + (1,) * (ndim - 2)))) 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the input tensor x using instance normalization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, d_1, d_2, ...).
        
        Returns:
            torch.Tensor: Normalized tensor of the same shape as x.

        """
        dims = tuple(range(2, x.ndim))
        mean = torch.mean(x, dim=dims, keepdim=True)
        var = torch.var(x, dim=dims, keepdim=True, unbiased=False)

        if self.learnable:
            return self.gamma * (x - mean) / torch.sqrt(var + self.tol) + self.beta

        return (x - mean) / torch.sqrt(var + self.tol)
    
class GroupNormalizer(nn.Module):
    """Group Normalization Layer."""

    tol: float
    """Tolerance for numerical stability in normalization."""

    learnable: bool
    """Whether the normalization parameters are learnable."""

    gamma: nn.Parameter
    """Learnable scale parameter for normalization, initialized to ones."""

    beta: nn.Parameter
    """Learnable shift parameter for normalization, initialized to zeros."""

    def __init__(self, num_features: int, num_groups: int, ndim: int, tol: float = 1e-10, learnable: bool = True):
        """Initialize the group normalizer.

        Args:
            num_features (int): Number of features in the input.
            num_groups (int): Number of groups for normalization.
            ndim (int): Number of dimensions in the input tensor.
            tol (float): Tolerance for numerical stability in normalization.
            learnable (bool): Whether the normalization parameters are learnable.
        
        Raises:
            ValueError: If num_features is not divisible by num_groups.

        """
        super().__init__()
        
        if num_features % num_groups != 0:
            raise ValueError(f"num_features ({num_features}) must be divisible by num_groups ({num_groups})")
        
        self.num_features = num_features
        self.tol = tol
        self.learnable = learnable
        self.num_groups = num_groups

        if self.learnable:
            self.gamma = nn.Parameter(torch.ones(((1, num_features) + (1,) * (ndim - 2))))
            self.beta = nn.Parameter(torch.zeros(((1, num_features) + (1,) * (ndim - 2))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the input tensor x using group normalization.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, d_1, d_2, ...).

        Returns:
            torch.Tensor: Normalized tensor of the same shape as x.

        """
        B, C = x.shape[:2]
        group_size = C // self.num_groups
        original_shape = x.shape
        
        x_grouped = x.view(B, self.num_groups, group_size, *x.shape[2:])
        
        dims = tuple(range(2, x_grouped.ndim))
        mean = x_grouped.mean(dim=dims, keepdim=True)
        var = x_grouped.var(dim=dims, keepdim=True, unbiased=False)
        
        x_norm = (x_grouped - mean) / torch.sqrt(var + self.tol)
        
        x_norm = x_norm.view(original_shape)

        if self.learnable:
            return self.gamma * x_norm + self.beta

        return x_norm