"""Spectral Convolution Unit Tests.

This module contains comprehensive unit tests for the SpectralConv layer,
testing functionality across different dimensions (1D, 2D, 3D), data types,
error conditions, and edge cases.
"""

from neurop.layers.spectral_convolution import SpectralConv
import torch

def test_spectral_conv_1d():
    """Test the SpectralConv functionality for 1D."""
    in_features = 5
    out_features = 10
    modes = 3
    batch_size = 2
    spatial_dims = (4,)

    layer = SpectralConv(in_features, out_features, modes)
    x = torch.randn(batch_size, in_features, *spatial_dims)

    y = layer(x)

    assert y.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConv 1D"

def test_spectral_conv_2d():
    """Test the SpectralConv functionality for 2D."""
    in_features = 5
    out_features = 10
    modes = [3, 4]
    batch_size = 2
    spatial_dims = (8, 8)

    layer = SpectralConv(in_features, out_features, modes)
    x = torch.randn(batch_size, in_features, *spatial_dims)

    y = layer(x)

    assert y.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConv 2D"

def test_spectral_conv_3d():
    """Test the SpectralConv functionality for 3D."""
    in_features = 5
    out_features = 10
    modes = [3, 4, 5]
    batch_size = 2
    spatial_dims = (6, 6, 6)

    layer = SpectralConv(in_features, out_features, modes)
    x = torch.randn(batch_size, in_features, *spatial_dims)

    y = layer(x)

    assert y.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConv 3D"

def test_spectral_conv_1d_complex():
    """Test the SpectralConv with complex input for 1D."""
    in_features = 5
    out_features = 10
    modes = 3
    batch_size = 2
    spatial_dims = (4,)

    layer = SpectralConv(in_features, out_features, modes)
    x_real = torch.randn(batch_size, in_features, *spatial_dims)
    x_imag = torch.randn(batch_size, in_features, *spatial_dims)
    x_complex = torch.complex(x_real, x_imag)

    y = layer(x_complex)

    assert y.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConv 1D with complex input"

def test_spectral_conv_2d_complex():
    """Test the SpectralConv with complex input for 2D."""
    in_features = 5
    out_features = 10
    modes = [3, 4]
    batch_size = 2
    spatial_dims = (8, 8)

    layer = SpectralConv(in_features, out_features, modes)
    x_real = torch.randn(batch_size, in_features, *spatial_dims)
    x_imag = torch.randn(batch_size, in_features, *spatial_dims)
    x_complex = torch.complex(x_real, x_imag)

    y = layer(x_complex)

    assert y.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConv 2D with complex input"

def test_spectral_conv_3d_complex():
    """Test the SpectralConv with complex input for 3D."""
    in_features = 5
    out_features = 10
    modes = [3, 4, 5]
    batch_size = 2
    spatial_dims = (6, 6, 6)

    layer = SpectralConv(in_features, out_features, modes)
    x_real = torch.randn(batch_size, in_features, *spatial_dims)
    x_imag = torch.randn(batch_size, in_features, *spatial_dims)
    x_complex = torch.complex(x_real, x_imag)

    y = layer(x_complex)

    assert y.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConv 3D with complex input"

def test_spectral_conv_invalid_modes():
    """Test the SpectralConv with invalid modes."""
    in_features = 5
    out_features = 10
    modes = [-1, 4, 5]  # Invalid mode

    try:
        SpectralConv(in_features, out_features, modes)
        assert False, "Expected ValueError for invalid modes in SpectralConv"
    except ValueError:
        pass  # Expected

def test_spectral_conv_zero_modes():
    """Test the SpectralConv with zero modes."""
    in_features = 5
    out_features = 10
    modes = [0, 4, 5]  # Zero mode in the first dimension
    batch_size = 2
    spatial_dims = (4, 4, 4)

    layer = SpectralConv(in_features, out_features, modes)
    x = torch.randn(batch_size, in_features, *spatial_dims)

    y = layer(x)

    assert y.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConv with zero modes"

def test_spectral_conv_invalid_input_shape():
    """Test the SpectralConv with invalid input shape."""
    in_features = 5
    out_features = 10
    modes = [3, 4, 5]
    batch_size = 2
    spatial_dims = (4, 4, 4)

    layer = SpectralConv(in_features, out_features, modes)
    x = torch.randn(batch_size, in_features + 1, *spatial_dims)  # Invalid input shape

    try:
        layer(x)
        assert False, "Expected ValueError for invalid input shape in SpectralConv"
    except ValueError:
        pass  # Expected

def test_spectral_conv_empty_input():
    """Test the SpectralConv with empty input."""
    in_features = 5
    out_features = 10
    modes = [3, 4, 5]
    batch_size = 2
    spatial_dims = (0, 0, 0)  # Empty spatial dimensions

    layer = SpectralConv(in_features, out_features, modes)
    x = torch.randn(batch_size, in_features, *spatial_dims)

    try:
        layer(x)
        assert False, "Expected ValueError for empty input in SpectralConv"
    except ValueError:
        pass  # Expected

def test_spectral_conv_non_tensor_input():
    """Test the SpectralConv with non-tensor input."""
    in_features = 5
    out_features = 10
    modes = [3, 4, 5]

    layer = SpectralConv(in_features, out_features, modes)
    
    try:
        layer("invalid input")  # Non-tensor input
        assert False, "Expected TypeError for non-tensor input in SpectralConv"
    except TypeError:
        pass  # Expected

def test_spectral_conv_single_mode():
    """Test the SpectralConv with a single mode (int instead of list)."""
    in_features = 5
    out_features = 10
    modes = 3  # Single mode for 1D
    batch_size = 2
    spatial_dims = (8,)

    layer = SpectralConv(in_features, out_features, modes)
    x = torch.randn(batch_size, in_features, *spatial_dims)

    y = layer(x)

    assert y.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConv with single mode"

def test_spectral_conv_different_dtypes():
    """Test the SpectralConv with different data types."""
    in_features = 5
    out_features = 10
    modes = [3, 4]
    batch_size = 2
    spatial_dims = (8, 8)

    # Test with float32
    layer_float32 = SpectralConv(in_features, out_features, modes, dtype=torch.cfloat, weight_dtype=torch.cfloat)
    x_float32 = torch.randn(batch_size, in_features, *spatial_dims, dtype=torch.float32)
    y_float32 = layer_float32(x_float32)

    # Test with float64
    layer_float64 = SpectralConv(in_features, out_features, modes, dtype=torch.cdouble, weight_dtype=torch.cdouble)
    x_float64 = torch.randn(batch_size, in_features, *spatial_dims, dtype=torch.float64)
    y_float64 = layer_float64(x_float64)

    assert y_float32.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConv with float32"
    assert y_float64.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConv with float64"

def test_spectral_conv_different_init_scales():
    """Test the SpectralConv with different initialization scales."""
    in_features = 5
    out_features = 10
    modes = [3, 4]
    batch_size = 2
    spatial_dims = (8, 8)

    # Test with different init scales
    layer_scale_01 = SpectralConv(in_features, out_features, modes, init_scale=0.1)
    layer_scale_10 = SpectralConv(in_features, out_features, modes, init_scale=10.0)
    
    x = torch.randn(batch_size, in_features, *spatial_dims)
    
    y_01 = layer_scale_01(x)
    y_10 = layer_scale_10(x)

    assert y_01.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConv with init_scale=0.1"
    assert y_10.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConv with init_scale=10.0"

def test_spectral_conv_dimension_mismatch():
    """Test the SpectralConv with mismatched spatial dimensions."""
    in_features = 5
    out_features = 10
    modes = [3, 4, 5]  # 3D modes
    batch_size = 2
    spatial_dims = (8, 8)  # 2D spatial dims

    layer = SpectralConv(in_features, out_features, modes)
    x = torch.randn(batch_size, in_features, *spatial_dims)

    try:
        layer(x)
        assert False, "Expected ValueError for dimension mismatch in SpectralConv"
    except ValueError:
        pass  # Expected

def test_spectral_conv_large_modes():
    """Test the SpectralConv with modes larger than spatial dimensions."""
    in_features = 5
    out_features = 10
    modes = [10, 10]  # Large modes
    batch_size = 2
    spatial_dims = (4, 4)  # Small spatial dims

    layer = SpectralConv(in_features, out_features, modes)
    x = torch.randn(batch_size, in_features, *spatial_dims)

    y = layer(x)

    assert y.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConv with large modes"
    
def test_spectral_conv_device_consistency():
    """Test the SpectralConv device consistency."""
    in_features = 5
    out_features = 10
    modes = [3, 4]
    batch_size = 2
    spatial_dims = (8, 8)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Be explicit about cuda:0
    
    layer = SpectralConv(in_features, out_features, modes).to(device)
    x = torch.randn(batch_size, in_features, *spatial_dims, device=device)

    y = layer(x)

    assert y.device == device, f"Device mismatch in SpectralConv: expected {device}, got {y.device}"
    assert layer.weight.device == device, f"Weight device mismatch in SpectralConv: expected {device}, got {layer.weight.device}"

def test_spectral_conv_mixed_precision():
    """Test the SpectralConv with mixed precision."""
    in_features = 5
    out_features = 10
    modes = [3, 4]
    batch_size = 2
    spatial_dims = (8, 8)

    layer = SpectralConv(in_features, out_features, modes)
    x = torch.randn(batch_size, in_features, *spatial_dims, dtype=torch.float32)

    y = layer(x)

    assert y.shape == (batch_size, out_features, *spatial_dims), "Output shape mismatch in SpectralConv with mixed precision"
