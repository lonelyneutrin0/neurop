from neurop.layers.iolayers import ReadinLayer, ReadoutLayer
import torch

def test_readin_layer():
    """
    Test the ReadinLayer functionality.
    """
    in_features = 5
    hidden_features = 10
    batch_size = 2
    spatial_dims = (4, 4)

    layer = ReadinLayer(in_features, hidden_features)
    x = torch.randn(batch_size, in_features, *spatial_dims)
    
    y = layer(x)
    
    assert y.shape == (batch_size, hidden_features, *spatial_dims), "Output shape mismatch in ReadinLayer"

def test_readout_layer():
    """
    Test the ReadoutLayer functionality.
    """

    hidden_features = 10
    output_features = 3
    batch_size = 2
    spatial_dims = (4, 4)

    layer = ReadoutLayer(hidden_features, output_features)
    x = torch.randn(batch_size, hidden_features, *spatial_dims)
    
    y = layer(x)
    
    assert y.shape == (batch_size, output_features, *spatial_dims), "Output shape mismatch in ReadoutLayer"

