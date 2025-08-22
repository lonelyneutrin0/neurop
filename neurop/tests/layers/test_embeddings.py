"""Test cases for Positional Embeddings."""
import torch
import pytest
from neurop.layers.positional_embeddings import GridEmbedding, make_regular_grid

def test_grid_embedding_basic_forward():
    """Test forward pass of GridEmbedding."""
    # 2D example
    batch, channels, height, width = 2, 3, 8, 8
    domain = [[0.0, 1.0], [0.0, 1.0]]
    x = torch.randn(batch, channels, height, width)
    emb = GridEmbedding(in_features=channels, dim=2, domain=domain)
    out = emb(x)
    assert out.shape == (batch, channels + 2, height, width)
    # Check that grid values are within domain
    grid_x = out[0, channels, :, :]
    grid_y = out[0, channels+1, :, :]
    assert grid_x.min() >= domain[0][0] and grid_x.max() <= domain[0][1]
    assert grid_y.min() >= domain[1][0] and grid_y.max() <= domain[1][1]

def test_grid_embedding_3d_forward():
    """Test forward pass of GridEmbedding."""
    batch, channels, d1, d2, d3 = 1, 4, 5, 6, 7
    domain = [[-1, 1], [0, 2], [10, 20]]
    x = torch.randn(batch, channels, d1, d2, d3)
    emb = GridEmbedding(in_features=channels, dim=3, domain=domain)
    out = emb(x)
    assert out.shape == (batch, channels + 3, d1, d2, d3)

def test_make_regular_grid_values_and_shape():
    """Test make_regular_grid function."""
    res = torch.Size([4, 5])
    domain = [[0, 1], [10, 20]]
    grid = make_regular_grid(res, domain)
    assert len(grid) == 2
    assert grid[0].shape == (4, 5)
    assert grid[1].shape == (4, 5)
    # Check grid value ranges
    assert torch.all(grid[0] >= 0) and torch.all(grid[0] <= 1)
    assert torch.all(grid[1] >= 10) and torch.all(grid[1] <= 20)

def test_grid_embedding_device_and_dtype():
    """Test device and dtype consistency in GridEmbedding."""
    batch, channels, height, width = 1, 2, 3, 3
    domain = [[0, 1], [0, 1]]
    x = torch.randn(batch, channels, height, width, device='cpu', dtype=torch.float64)
    emb = GridEmbedding(in_features=channels, dim=2, domain=domain)
    out = emb(x)
    assert out.device == x.device
    assert out.dtype == x.dtype

def test_grid_embedding_repeat_consistency():
    """Test repeat consistency in GridEmbedding."""
    # Check that repeated calls with same shape do not rebuild grid
    batch, channels, height, width = 1, 2, 4, 4
    domain = [[0, 1], [0, 1]]
    x = torch.randn(batch, channels, height, width)
    emb = GridEmbedding(in_features=channels, dim=2, domain=domain)
    out1 = emb(x)
    out2 = emb(x)
    # Should be identical
    assert torch.allclose(out1, out2)

def test_make_regular_grid_invalid_resolution():
    """Test make_regular_grid function with invalid resolution."""
    # Should raise ValueError if resolutions and domain lengths mismatch
    with pytest.raises(ValueError):
        make_regular_grid(torch.Size([2]), [[0, 1], [0, 1]])

def test_grid_embedding_invalid_domain():
    """Test GridEmbedding with invalid domain."""
    # Should raise ValueError if domain and input shape mismatch
    batch, channels, height, width = 1, 2, 4, 4
    domain = [[0, 1]]  # Only one dimension
    x = torch.randn(batch, channels, height, width)
    emb = GridEmbedding(in_features=channels, dim=2, domain=domain)
    with pytest.raises(ValueError):
        emb(x)

def test_grid_embedding_out_channels_property():
    """Test out_channels property of GridEmbedding."""
    emb = GridEmbedding(in_features=5, dim=3, domain=[[0,1],[0,1],[0,1]])
    assert emb.out_channels == 8

def test_grid_embedding_grid_caching():
    """Test grid caching in GridEmbedding."""
    # Should cache grid for repeated calls with same shape
    emb = GridEmbedding(in_features=2, dim=2, domain=[[0,1],[0,1]])
    x = torch.randn(1, 2, 4, 4)
    _ = emb(x)
    cached_grid = emb._grid
    _ = emb(x)
    assert emb._grid is cached_grid
