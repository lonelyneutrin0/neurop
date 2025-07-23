"""FNOUnit Tests."""
from neurop.layers.fno_unit import FNOUnit
from neurop.layers.normalizers import BatchNormalizer, LayerNormalizer

import torch
import torch.nn as nn

def test_fno_unit_basic():
    """Test basic FNOUnit functionality."""
    unit = FNOUnit(in_features=3, out_features=3, n_dim=2, modes=5) 
    x = torch.randn(2, 3, 4, 5)  
    output = unit(x)

    assert output.shape == x.shape, "Output shape should match input shape"
    assert output.dtype == torch.float32, "FNOUnit should return real tensor for real inputs"

    assert not torch.allclose(output, x, atol=1e-3), "FNOUnit should transform the signal"

def test_fno_unit_different_output_features():
    """Test FNOUnit with different input and output features."""
    unit = FNOUnit(
        in_features=4, 
        out_features=8, 
        n_dim=2, 
        modes=[3, 3],  
        skip_connection='linear' 
    )
    x = torch.randn(2, 4, 6, 8)
    output = unit(x)
    
    assert output.shape == (2, 8, 6, 8), "Output should have correct feature dimensions"
    assert output.dtype == torch.float32, "FNOUnit should return real tensor for real inputs"

def test_fno_unit_3d():
    """Test FNOUnit with 3D spatial dimensions."""
    unit = FNOUnit(in_features=2, out_features=2, n_dim=3, modes=[2, 3, 4])  
    x = torch.randn(1, 2, 4, 6, 8) 
    output = unit(x)
    
    assert output.shape == (1, 2, 4, 6, 8), "Output should have correct 3D shape"
    assert output.dtype == torch.float32, "FNOUnit should return real tensor for real inputs"

def test_fno_unit_3d_different_features():
    """Test FNOUnit with 3D spatial dimensions and different input/output features."""
    unit = FNOUnit(
        in_features=2, 
        out_features=4, 
        n_dim=3, 
        modes=[2, 3, 4],
        skip_connection='linear'  
    )
    x = torch.randn(1, 2, 4, 6, 8) 
    output = unit(x)
    
    assert output.shape == (1, 4, 4, 6, 8), "Output should have correct 3D shape with feature change"
    assert output.dtype == torch.float32, "FNOUnit should return real tensor for real inputs"

def test_fno_unit_with_activation():
    """Test FNOUnit with activation function."""
    unit = FNOUnit(
        in_features=3, 
        out_features=3, 
        n_dim=2, 
        modes=[4, 4],  
        activation_function=nn.ReLU
    )
    x = torch.randn(2, 3, 4, 5)
    output = unit(x)
    
    assert output.shape == x.shape, "Output shape should match input shape"
    assert output.dtype == torch.float32, "FNOUnit should return real tensor for real inputs"

def test_fno_unit_without_feature_mlp():
    """Test FNOUnit without feature MLP."""
    unit = FNOUnit(
        in_features=3, 
        out_features=3, 
        n_dim=2, 
        modes=[4, 4], 
        use_feature_mlp=False
    )
    x = torch.randn(2, 3, 4, 5)
    output = unit(x)
    
    assert output.shape == x.shape, "Output shape should match input shape"
    assert unit.feature_mlp is None, "Feature MLP should be None when disabled"

def test_fno_unit_with_normalizers():
    """Test FNOUnit with spectral and feature normalizers."""
    unit = FNOUnit(
        in_features=3, 
        out_features=3, 
        n_dim=2, 
        modes=[4, 4],  
        spectral_normalizer=BatchNormalizer,
        feature_normalizer=LayerNormalizer,
        learnable_normalizers=True
    )
    x = torch.randn(2, 3, 4, 5)
    output = unit(x)
    
    assert output.shape == x.shape, "Output shape should match input shape"
    assert unit.spectral_normalizer is not None, "Spectral normalizer should be initialized"
    assert unit.feature_normalizer is not None, "Feature normalizer should be initialized"

def test_fno_unit_skip_connection_types():
    """Test FNOUnit with different skip connection types."""
    unit_identity = FNOUnit(
        in_features=3, 
        out_features=3, 
        n_dim=2, 
        modes=[4, 4],  
        skip_connection='identity',
        feature_mlp_skip_connection='identity'
    )
    
    unit_linear = FNOUnit(
        in_features=3, 
        out_features=5, 
        n_dim=2, 
        modes=[4, 4], 
        skip_connection='linear',
        feature_mlp_skip_connection='linear'
    )
    
    x = torch.randn(2, 3, 4, 5)
    
    output_identity = unit_identity(x)
    output_linear = unit_linear(x)
    
    assert output_identity.shape == x.shape, "Identity skip connection should preserve shape"
    assert output_linear.shape == (2, 5, 4, 5), "Linear skip connection should handle feature change"

def test_fno_unit_input_validation():
    """Test FNOUnit input validation."""
    unit = FNOUnit(in_features=3, out_features=3, n_dim=2, modes=[4, 4])  # 2D requires 2 modes
    
    x_wrong = torch.randn(2, 5, 4, 5)
    
    try:
        unit(x_wrong)
        assert False, "Should raise ValueError for wrong input features"
    except ValueError as e:
        assert "Expected 3 input features, got 5" in str(e), "Should provide clear error message"

def test_fno_unit_parameter_count():
    """Test that FNOUnit has trainable parameters."""
    unit = FNOUnit(in_features=3, out_features=3, n_dim=2, modes=[4, 4])  # 2D requires 2 modes
    
    param_count = sum(p.numel() for p in unit.parameters() if p.requires_grad)
    assert param_count > 0, "FNOUnit should have trainable parameters"
    
    x = torch.randn(2, 3, 4, 5, requires_grad=True)
    output = unit(x)
    loss = output.abs().sum()
    loss.backward()
    
    assert x.grad is not None, "Gradients should flow back to input"

def test_fno_unit_feature_expansion():
    """Test FNOUnit with different feature expansion factors."""
    unit = FNOUnit(
        in_features=4, 
        out_features=4, 
        n_dim=2, 
        modes=[3, 3],  # 2D requires 2 modes
        feature_expansion_factor=2.0  # Expand hidden features by 2x
    )
    x = torch.randn(2, 4, 6, 8)
    output = unit(x)
    
    assert output.shape == x.shape, "Output shape should match input shape"
    # The feature MLP should have expanded hidden dimensions internally
    