from neurop.layers.feature_mlp import LinearFeatureMLP, ConvFeatureMLP
import torch

def test_linear_feature_mlp():
    """Test the LinearFeatureMLP class."""
    model = LinearFeatureMLP(in_features=3, hidden_features=4, out_features=5, depth=4, n_dim=2)
    x = torch.randn(2, 3, 4, 5)  # Example input tensor
    output = model(x)
    assert output.shape == (2, 5, 4, 5), "Output shape should match (B, out_features, H, W)"

def test_conv_feature_mlp():
    """Test the ConvFeatureMLP class."""
    model = ConvFeatureMLP(in_features=3, hidden_features=4, out_features=5, depth=3, n_dim=2)
    x = torch.randn(2, 3, 4, 5)  # Example input tensor
    output = model(x)
    assert output.shape == (2, 5, 4, 5), "Output shape should match (B, out_features, H, W)"