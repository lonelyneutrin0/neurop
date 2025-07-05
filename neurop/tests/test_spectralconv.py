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

def test_spectral_conv1d_layer_complex():
    """
    Test the SpectralConv1DLayer with complex input.
    """
    in_features = 5
    out_features = 10
    modes = 3
    batch_size = 2
    spatial_dims = (4,)

    layer = SpectralConv1DLayer(in_features, out_features, modes)
    x_real = torch.randn(batch_size, in_features, *spatial_dims)
    x_imag = torch.randn(batch_size, in_features, *spatial_dims)
    x_complex = torch.complex(x_real, x_imag)

    y = layer(x_complex)

    assert y.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConv1DLayer with complex input"

def test_spectral_conv2d_layer_complex():
    """
    Test the SpectralConv2DLayer with complex input.
    """
    in_features = 5
    out_features = 10
    mode_h = 3
    mode_w = 4
    batch_size = 2
    spatial_dims = (4, 4)

    layer = SpectralConv2DLayer(in_features, out_features, mode_h, mode_w)
    x_real = torch.randn(batch_size, in_features, *spatial_dims)
    x_imag = torch.randn(batch_size, in_features, *spatial_dims)
    x_complex = torch.complex(x_real, x_imag)

    y = layer(x_complex)

    assert y.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConv2DLayer with complex input"

def test_spectral_conv3d_layer_complex():
    """
    Test the SpectralConv3DLayer with complex input.
    """
    in_features = 5
    out_features = 10
    mode_d = 3
    mode_h = 4
    mode_w = 5
    batch_size = 2

    layer = SpectralConv3DLayer(in_features, out_features, mode_d, mode_h, mode_w)
    x_real = torch.randn(batch_size, in_features, 3, 3, 3)
    x_imag = torch.randn(batch_size, in_features, 3, 3, 3)
    x_complex = torch.complex(x_real, x_imag)

    y = layer(x_complex)

    assert y.shape == (batch_size, out_features, 3, 3, 3), "Output shape mismatch in SpectralConv3DLayer with complex input"

def test_spectral_conv_nd_layer_complex():
    """
    Test the SpectralConvNDLayer with complex input.
    """
    in_features = 5
    out_features = 10
    modes = [3, 4, 5]  # Example for a 3D case
    batch_size = 2
    spatial_dims = (4, 4, 4)

    layer = SpectralConvNDLayer(in_features, out_features, modes)
    x_real = torch.randn(batch_size, in_features, *spatial_dims)
    x_imag = torch.randn(batch_size, in_features, *spatial_dims)
    x_complex = torch.complex(x_real, x_imag)

    y = layer(x_complex)

    assert y.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConvNDLayer with complex input"

def test_spectral_conv1d_layer_invalid_modes():
    """
    Test the SpectralConv1DLayer with invalid modes.
    """
    in_features = 5
    out_features = 10
    modes = -1  # Invalid mode
    batch_size = 2
    spatial_dims = (4,)

    try:
        layer = SpectralConv1DLayer(in_features, out_features, modes)
        assert False, "Expected ValueError for invalid modes in SpectralConv1DLayer"
    except ValueError:
        pass  # Expected

def test_spectral_conv2d_layer_invalid_modes():
    """
    Test the SpectralConv2DLayer with invalid modes.
    """
    in_features = 5
    out_features = 10
    mode_h = -1  # Invalid mode
    mode_w = 4
    batch_size = 2
    spatial_dims = (4, 4)

    try:
        layer = SpectralConv2DLayer(in_features, out_features, mode_h, mode_w)
        assert False, "Expected ValueError for invalid modes in SpectralConv2DLayer"
    except ValueError:
        pass  # Expected

def test_spectral_conv3d_layer_invalid_modes():
    """
    Test the SpectralConv3DLayer with invalid modes.
    """
    in_features = 5
    out_features = 10
    mode_d = -1  # Invalid mode
    mode_h = 4
    mode_w = 5
    batch_size = 2

    try:
        layer = SpectralConv3DLayer(in_features, out_features, mode_d, mode_h, mode_w)
        assert False, "Expected ValueError for invalid modes in SpectralConv3DLayer"
    except ValueError:
        pass  # Expected

def test_spectral_conv_nd_layer_invalid_modes():
    """
    Test the SpectralConvNDLayer with invalid modes.
    """
    in_features = 5
    out_features = 10
    modes = [-1, 4, 5]  # Invalid mode
    batch_size = 2
    spatial_dims = (4, 4, 4)

    try:
        layer = SpectralConvNDLayer(in_features, out_features, modes)
        assert False, "Expected ValueError for invalid modes in SpectralConvNDLayer"
    except ValueError:
        pass  # Expected

def test_spectral_conv1d_layer_zero_modes():
    """
    Test the SpectralConv1DLayer with zero modes.
    """
    in_features = 5
    out_features = 10
    modes = 0  # Zero modes
    batch_size = 2
    spatial_dims = (4,)

    layer = SpectralConv1DLayer(in_features, out_features, modes)
    x = torch.randn(batch_size, in_features, *spatial_dims)

    y = layer(x)

    assert y.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConv1DLayer with zero modes"

def test_spectral_conv2d_layer_zero_modes():
    """
    Test the SpectralConv2DLayer with zero modes.
    """
    in_features = 5
    out_features = 10
    mode_h = 0  # Zero modes
    mode_w = 4
    batch_size = 2
    spatial_dims = (4, 4)

    layer = SpectralConv2DLayer(in_features, out_features, mode_h, mode_w)
    x = torch.randn(batch_size, in_features, *spatial_dims)

    y = layer(x)

    assert y.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConv2DLayer with zero modes"

def test_spectral_conv3d_layer_zero_modes():
    """
    Test the SpectralConv3DLayer with zero modes.
    """
    in_features = 5
    out_features = 10
    mode_d = 0  # Zero modes
    mode_h = 4
    mode_w = 5
    batch_size = 2

    layer = SpectralConv3DLayer(in_features, out_features, mode_d, mode_h, mode_w)
    x = torch.randn(batch_size, in_features, 3, 3, 3)

    y = layer(x)

    assert y.shape == (batch_size, out_features, 3, 3, 3), "Output shape mismatch in SpectralConv3DLayer with zero modes"

def test_spectral_conv_nd_layer_zero_modes():
    """
    Test the SpectralConvNDLayer with zero modes.
    """
    in_features = 5
    out_features = 10
    modes = [0, 4, 5]  # Zero mode in the first dimension
    batch_size = 2
    spatial_dims = (4, 4, 4)

    layer = SpectralConvNDLayer(in_features, out_features, modes)
    x = torch.randn(batch_size, in_features, *spatial_dims)

    y = layer(x)

    assert y.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConvNDLayer with zero modes"

def test_spectral_conv1d_layer_invalid_input_shape():
    """
    Test the SpectralConv1DLayer with invalid input shape.
    """
    in_features = 5
    out_features = 10
    modes = 3
    batch_size = 2
    spatial_dims = (4,)

    layer = SpectralConv1DLayer(in_features, out_features, modes)
    x = torch.randn(batch_size, in_features + 1, *spatial_dims)  # Invalid input shape

    try:
        y = layer(x)
        assert False, "Expected ValueError for invalid input shape in SpectralConv1DLayer"
    except ValueError:
        pass  # Expected

def test_spectral_conv2d_layer_invalid_input_shape():
    """
    Test the SpectralConv2DLayer with invalid input shape.
    """
    in_features = 5
    out_features = 10
    mode_h = 3
    mode_w = 4
    batch_size = 2
    spatial_dims = (4, 4)

    layer = SpectralConv2DLayer(in_features, out_features, mode_h, mode_w)
    x = torch.randn(batch_size, in_features + 1, *spatial_dims)  # Invalid input shape

    try:
        y = layer(x)
        assert False, "Expected ValueError for invalid input shape in SpectralConv2DLayer"
    except ValueError:
        pass  # Expected

def test_spectral_conv3d_layer_invalid_input_shape():
    """
    Test the SpectralConv3DLayer with invalid input shape.
    """
    in_features = 5
    out_features = 10
    mode_d = 3
    mode_h = 4
    mode_w = 5
    batch_size = 2

    layer = SpectralConv3DLayer(in_features, out_features, mode_d, mode_h, mode_w)
    x = torch.randn(batch_size, in_features + 1, 3, 3, 3)  # Invalid input shape

    try:
        y = layer(x)
        assert False, "Expected ValueError for invalid input shape in SpectralConv3DLayer"
    except ValueError:
        pass  # Expected

def test_spectral_conv_nd_layer_invalid_input_shape():
    """
    Test the SpectralConvNDLayer with invalid input shape.
    """
    in_features = 5
    out_features = 10
    modes = [3, 4, 5]  # Example for a 3D case
    batch_size = 2
    spatial_dims = (4, 4, 4)

    layer = SpectralConvNDLayer(in_features, out_features, modes)
    x = torch.randn(batch_size, in_features + 1, *spatial_dims)  # Invalid input shape

    try:
        y = layer(x)
        assert False, "Expected ValueError for invalid input shape in SpectralConvNDLayer"
    except ValueError:
        pass  # Expected

def test_spectral_conv1d_layer_empty_input():
    """
    Test the SpectralConv1DLayer with empty input.
    """
    in_features = 5
    out_features = 10
    modes = 3
    batch_size = 2
    spatial_dims = (0,)  # Empty spatial dimension

    layer = SpectralConv1DLayer(in_features, out_features, modes)
    x = torch.randn(batch_size, in_features, *spatial_dims)

    try:
        y = layer(x)
        assert False, "Expected ValueError for empty input in SpectralConv1DLayer"
    except ValueError:
        pass  # Expected

def test_spectral_conv2d_layer_empty_input():
    """
    Test the SpectralConv2DLayer with empty input.
    """
    in_features = 5
    out_features = 10
    mode_h = 3
    mode_w = 4
    batch_size = 2
    spatial_dims = (0, 0)  # Empty spatial dimensions

    layer = SpectralConv2DLayer(in_features, out_features, mode_h, mode_w)
    x = torch.randn(batch_size, in_features, *spatial_dims)

    try:
        y = layer(x)
        assert False, "Expected ValueError for empty input in SpectralConv2DLayer"
    except ValueError:
        pass  # Expected

def test_spectral_conv3d_layer_empty_input():
    """
    Test the SpectralConv3DLayer with empty input.
    """
    in_features = 5
    out_features = 10
    mode_d = 3
    mode_h = 4
    mode_w = 5
    batch_size = 2

    layer = SpectralConv3DLayer(in_features, out_features, mode_d, mode_h, mode_w)
    x = torch.randn(batch_size, in_features, 0, 0, 0)  # Empty spatial dimensions

    try:
        y = layer(x)
        assert False, "Expected ValueError for empty input in SpectralConv3DLayer"
    except ValueError:
        pass  # Expected

def test_spectral_conv_nd_layer_empty_input():
    """
    Test the SpectralConvNDLayer with empty input.
    """
    in_features = 5
    out_features = 10
    modes = [3, 4, 5]  # Example for a 3D case
    batch_size = 2
    spatial_dims = (0, 0, 0)  # Empty spatial dimensions

    layer = SpectralConvNDLayer(in_features, out_features, modes)
    x = torch.randn(batch_size, in_features, *spatial_dims)

    try:
        y = layer(x)
        assert False, "Expected ValueError for empty input in SpectralConvNDLayer"
    except ValueError:
        pass  # Expected

def test_spectral_conv1d_layer_non_tensor_input():
    """
    Test the SpectralConv1DLayer with non-tensor input.
    """
    in_features = 5
    out_features = 10
    modes = 3
    batch_size = 2
    spatial_dims = (4,)

    layer = SpectralConv1DLayer(in_features, out_features, modes)
    
    try:
        y = layer("invalid input")  # Non-tensor input
        assert False, "Expected TypeError for non-tensor input in SpectralConv1DLayer"
    except TypeError:
        pass  # Expected

def test_spectral_conv2d_layer_non_tensor_input():
    """
    Test the SpectralConv2DLayer with non-tensor input.
    """
    in_features = 5
    out_features = 10
    mode_h = 3
    mode_w = 4
    batch_size = 2
    spatial_dims = (4, 4)

    layer = SpectralConv2DLayer(in_features, out_features, mode_h, mode_w)
    
    try:
        y = layer("invalid input")  # Non-tensor input
        assert False, "Expected TypeError for non-tensor input in SpectralConv2DLayer"
    except TypeError:
        pass  # Expected

def test_spectral_conv3d_layer_non_tensor_input():
    """
    Test the SpectralConv3DLayer with non-tensor input.
    """
    in_features = 5
    out_features = 10
    mode_d = 3
    mode_h = 4
    mode_w = 5
    batch_size = 2

    layer = SpectralConv3DLayer(in_features, out_features, mode_d, mode_h, mode_w)
    
    try:
        y = layer("invalid input")  # Non-tensor input
        assert False, "Expected TypeError for non-tensor input in SpectralConv3DLayer"
    except TypeError:
        pass  # Expected

def test_spectral_conv_nd_layer_non_tensor_input():
    """
    Test the SpectralConvNDLayer with non-tensor input.
    """
    in_features = 5
    out_features = 10
    modes = [3, 4, 5]  # Example for a 3D case
    batch_size = 2
    spatial_dims = (4, 4, 4)

    layer = SpectralConvNDLayer(in_features, out_features, modes)
    
    try:
        y = layer("invalid input")  # Non-tensor input
        assert False, "Expected TypeError for non-tensor input in SpectralConvNDLayer"
    except TypeError:
        pass  # Expected

def test_spectral_conv1d_layer_output_dtype():
    """
    Test the SpectralConv1DLayer with a specified output dtype.
    """
    in_features = 5
    out_features = 10
    modes = 3
    batch_size = 2
    spatial_dims = (4,)

    layer = SpectralConv1DLayer(in_features, out_features, modes)
    x = torch.randn(batch_size, in_features, *spatial_dims)

    y = layer(x, output_dtype=torch.float64)

    assert y.dtype == torch.float64, "Output dtype mismatch in SpectralConv1DLayer with specified output dtype"

def test_spectral_conv2d_layer_output_dtype():
    """
    Test the SpectralConv2DLayer with a specified output dtype.
    """
    in_features = 5
    out_features = 10
    mode_h = 3
    mode_w = 4
    batch_size = 2
    spatial_dims = (4, 4)

    layer = SpectralConv2DLayer(in_features, out_features, mode_h, mode_w)
    x = torch.randn(batch_size, in_features, *spatial_dims)

    y = layer(x, output_dtype=torch.float64)

    assert y.dtype == torch.float64, "Output dtype mismatch in SpectralConv2DLayer with specified output dtype"

def test_spectral_conv3d_layer_output_dtype():
    """
    Test the SpectralConv3DLayer with a specified output dtype.
    """
    in_features = 5
    out_features = 10
    mode_d = 3
    mode_h = 4
    mode_w = 5
    batch_size = 2

    layer = SpectralConv3DLayer(in_features, out_features, mode_d, mode_h, mode_w)
    x = torch.randn(batch_size, in_features, 3, 3, 3)

    y = layer(x, output_dtype=torch.float64)

    assert y.dtype == torch.float64, "Output dtype mismatch in SpectralConv3DLayer with specified output dtype"

def test_spectral_conv_nd_layer_output_dtype():
    """
    Test the SpectralConvNDLayer with a specified output dtype.
    """
    in_features = 5
    out_features = 10
    modes = [3, 4, 5]  # Example for a 3D case
    batch_size = 2
    spatial_dims = (4, 4, 4)

    layer = SpectralConvNDLayer(in_features, out_features, modes)
    x = torch.randn(batch_size, in_features, *spatial_dims)

    y = layer(x, output_dtype=torch.float64)

    assert y.dtype == torch.float64, "Output dtype mismatch in SpectralConvNDLayer with specified output dtype"

