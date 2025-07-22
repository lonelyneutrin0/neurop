from neurop.layers.skip_connections import ResidualConnection, IdentityConnection, SoftGatingConnection, LinearConnection

import torch 

def test_residual_connection(): 
    """Test the ResidualConnection class."""
    connection = ResidualConnection()
    x = torch.randn(2, 3, 4, 5)  # Example input tensor
    transformed_x = torch.randn(2, 3, 4, 5)  # Example transformed tensor
    output = connection(x, transformed_x)
    
    assert output.shape == x.shape, "Output shape should match input shape"
    assert torch.allclose(output, x + transformed_x), "Output should be the sum of input and transformed input"

def test_identity_connection():
    """Test the IdentityConnection class."""
    connection = IdentityConnection()
    x = torch.randn(2, 3, 4, 5)  # Example input tensor
    transformed_x = torch.randn(2, 3, 4, 5)  # Example transformed tensor
    output = connection(x, transformed_x)

    assert output.shape == x.shape, "Output shape should match input shape"
    assert torch.allclose(output, transformed_x), "Output should be the same as transformed input"

def test_soft_gating_connection():
    """Test the SoftGatingConnection class."""
    connection = SoftGatingConnection(in_features=3, n_dim=2)
    x = torch.randn(2, 3, 4, 5)  # Example input tensor
    transformed_x = torch.randn(2, 3, 4, 5)  # Example transformed tensor
    output = connection(x, transformed_x)

    assert output.shape == x.shape, "Output shape should match input shape"
    assert not torch.allclose(output, x), "Output should not be the same as input"
    assert not torch.allclose(output, transformed_x), "Output should not be the same as transformed input"

def test_linear_connection():
    """Test the LinearConnection class."""
    connection = LinearConnection(in_features=3, out_features=5, n_dim=2)
    x = torch.randn(2, 3, 4, 5)  # Example input tensor
    transformed_x = torch.randn(2, 5, 4, 5)  # Example transformed tensor
    output = connection(x, transformed_x)
    assert output.shape == transformed_x.shape, "Output shape should match (B, out_features, H, W)"
    assert not torch.allclose(output, transformed_x), "Output should not be the same as transformed input"

