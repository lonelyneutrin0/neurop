from neurop.layers.spectralconv import SpectralConv1DLayer, SpectralConv2DLayer, SpectralConv3DLayer, SpectralConvNDLayer #type: ignore 

import torch

def test_spectral_conv1d_layer():
    """
    Test the SpectralConv1DLayer functionality.
    """
    in_features = 5
    out_features = 10
    modes = 3
    batch_size = 2
    spatial_dims = (4,)

    layer = SpectralConv1DLayer(in_features, out_features, modes)
    x = torch.randn(batch_size, in_features, *spatial_dims)

    y = layer(x)

    assert y.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConv1DLayer"

def test_spectral_conv2d_layer():
    """
    Test the SpectralConv2DLayer functionality.
    """
    in_features = 5
    out_features = 10
    mode_h = 3
    mode_w = 4
    batch_size = 2
    spatial_dims = (4, 4)

    layer = SpectralConv2DLayer(in_features, out_features, mode_h, mode_w)
    x = torch.randn(batch_size, in_features, *spatial_dims)

    y = layer(x)

    assert y.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConv2DLayer"

def test_spectral_conv3d_layer():
    """
    Test the SpectralConv3DLayer functionality.
    """
    in_features = 5
    out_features = 10
    mode_d = 3
    mode_h = 4
    mode_w = 5
    batch_size = 2

    layer = SpectralConv3DLayer(in_features, out_features, mode_d, mode_h, mode_w)
    x = torch.randn(batch_size, in_features, 3, 3, 3,)

    y = layer(x)

    assert y.shape == (batch_size, out_features, 3,3,3), "Output shape mismatch in SpectralConv3DLayer"

def test_spectral_conv_nd_layer():
    """
    Test the SpectralConvNDLayer functionality.
    """
    in_features = 5
    out_features = 10
    modes = [3, 4, 5]  # Example for a 3D case
    batch_size = 2
    spatial_dims = (4, 4, 4)

    layer = SpectralConvNDLayer(in_features, out_features, modes)
    x = torch.randn(batch_size, in_features, *spatial_dims)

    y = layer(x)

    assert y.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConvNDLayer"

