"""Fourier Operator Tests."""
import torch 
import torch.nn as nn

from neurop.operators.fourier import FourierOperator, FourierOperatorBuilder

from neurop.layers.skip_connections import IdentityConnection, ConvConnection, SoftGatingConnection
from neurop.layers.feature_mlp import LinearFeatureMLP
from neurop.layers.normalizers import BatchNormalizer, LayerNormalizer
from neurop.layers.spectral_convolution import SpectralConv

def test_fourier_operator_basic():
    """Test basic FourierOperator functionality."""
    operator = FourierOperator(
        in_features=3,
        hidden_features=8,
        out_features=2,
        n_dim=2,
        modes=[4, 4],
        depth=2
    )
    
    x = torch.randn(2, 3, 6, 8)  
    output = operator(x)

    expected_shape = (2, 2, 6, 8)  
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    assert output.dtype == torch.float32, "FourierOperator should return real tensor after readout"
    
    assert not torch.allclose(output, torch.zeros_like(output)), "FourierOperator should produce non-zero output"


def test_fourier_operator_same_features():
    """Test FourierOperator with same input and output features."""
    operator = FourierOperator(
        in_features=4,
        hidden_features=8,
        out_features=4,
        n_dim=2,
        modes=[3, 3],
        depth=1
    )
    
    x = torch.randn(1, 4, 5, 7)
    output = operator(x)

    assert output.shape == x.shape, "Output shape should match input shape when in_features == out_features"
    assert output.dtype == torch.float32, "Output should be real"


def test_fourier_operator_3d():
    """Test FourierOperator with 3D spatial dimensions."""
    operator = FourierOperator(
        in_features=2,
        hidden_features=6,
        out_features=3,
        n_dim=3,
        modes=[2, 3, 4],
        depth=2
    )
    
    x = torch.randn(1, 2, 4, 6, 8)  
    output = operator(x)

    expected_shape = (1, 3, 4, 6, 8)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    assert output.dtype == torch.float32, "Output should be real"


def test_fourier_operator_different_depths():
    """Test FourierOperator with different depths."""
    depths = [1, 3, 5]
    
    for depth in depths:
        operator = FourierOperator(
            in_features=2,
            hidden_features=4,
            out_features=2,
            n_dim=2,
            modes=[2, 2],
            depth=depth
        )
        
        x = torch.randn(1, 2, 4, 4)
        output = operator(x)

        assert output.shape == x.shape, f"Output shape mismatch for depth {depth}"
        assert len(operator.fno_units) == depth, f"Expected {depth} FNO units, got {len(operator.fno_units)}"


def test_fourier_operator_skip_connections():
    """Test FourierOperator with different skip connection types."""
    skip_types = [IdentityConnection, ConvConnection, SoftGatingConnection]
    
    for skip_type in skip_types:
        operator = FourierOperator(
            in_features=3,
            hidden_features=6,
            out_features=3,
            n_dim=2,
            n_kernel=1,
            modes=[2, 2],
            depth=2,
            skip_connections=skip_type,
            feature_mlp_skip_connections=skip_type
        )
        
        x = torch.randn(1, 3, 4, 4)
        output = operator(x)

        assert output.shape == x.shape, f"Output shape mismatch for skip connection {skip_type}"
        assert output.dtype == torch.float32, "Output should be real"


def test_fourier_operator_without_feature_mlp():
    """Test FourierOperator without feature MLP."""
    operator = FourierOperator(
        in_features=2,
        hidden_features=4,
        out_features=2,
        n_dim=2,
        modes=[3, 3],
        depth=2,
        use_feature_mlp=False
    )
    
    x = torch.randn(1, 2, 6, 6)
    output = operator(x)

    assert output.shape == x.shape, "Output shape should match input shape"
    assert output.dtype == torch.float32, "Output should be real"
    
    for unit in operator.fno_units:
        assert unit.feature_mlp is None, "FNO units should not have feature MLPs when use_feature_mlp=False"


def test_fourier_operator_activation_functions():
    """Test FourierOperator with different activation functions."""
    activations = [torch.nn.ReLU, torch.nn.GELU, torch.nn.Tanh]
    
    for activation in activations:
        operator = FourierOperator(
            in_features=2,
            hidden_features=4,
            out_features=2,
            n_dim=2,
            modes=[2, 2],
            depth=2,
            activation_function=activation
        )
        
        x = torch.randn(1, 2, 4, 4)
        output = operator(x)

        assert output.shape == x.shape, f"Output shape mismatch for activation {activation.__name__}"
        assert output.dtype == torch.float32, "Output should be real"


def test_fourier_operator_different_modes():
    """Test FourierOperator with different mode configurations."""
    operator_single = FourierOperator(
        in_features=2,
        hidden_features=4,
        out_features=2,
        n_dim=2,
        modes=3,
        depth=1
    )
    operator_list = FourierOperator(
        in_features=2,
        hidden_features=4,
        out_features=2,
        n_dim=2,
        modes=[2, 4],
        depth=1
    )
    
    x = torch.randn(1, 2, 6, 8)
    
    output_single = operator_single(x)
    output_list = operator_list(x)

    assert output_single.shape == x.shape, "Output shape mismatch for single mode"
    assert output_list.shape == x.shape, "Output shape mismatch for mode list"


def test_fourier_operator_gradient_flow():
    """Test that gradients flow properly through FourierOperator."""
    operator = FourierOperator(
        in_features=2,
        hidden_features=4,
        out_features=1,
        n_dim=2,
        modes=[2, 2],
        depth=2,
        activation_function=nn.ReLU
    )
    
    x = torch.randn(1, 2, 4, 4, requires_grad=True)
    output = operator(x)
    
    loss = output.sum()
    loss.backward()

    assert x.grad is not None, "Input gradients should be computed"
    # assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), "Input gradients should be non-zero"
    
    for param in operator.parameters():
        if param.requires_grad:
            assert param.grad is not None, "Model parameter gradients should be computed"


def test_fourier_operator_batch_processing():
    """Test FourierOperator with different batch sizes."""
    operator = FourierOperator(
        in_features=2,
        hidden_features=4,
        out_features=2,
        n_dim=2,
        modes=[2, 2],
        depth=1
    )
    
    batch_sizes = [1, 4, 8]
    spatial_shape = (6, 8)
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 2, *spatial_shape)
        output = operator(x)

        expected_shape = (batch_size, 2, *spatial_shape)
        assert output.shape == expected_shape, f"Output shape mismatch for batch size {batch_size}"


def test_fourier_operator_determinism():
    """Test that FourierOperator produces deterministic results."""
    torch.manual_seed(42)
    operator = FourierOperator(
        in_features=2,
        hidden_features=4,
        out_features=2,
        n_dim=2,
        modes=[2, 2],
        depth=1
    )
    
    torch.manual_seed(123)
    x = torch.randn(1, 2, 4, 4)
    
    output1 = operator(x)
    output2 = operator(x)

    assert torch.allclose(output1, output2), "FourierOperator should produce deterministic results"



def test_fourier_operator_parameter_count():
    """Test that FourierOperator has reasonable number of parameters."""
    operator = FourierOperator(
        in_features=3,
        hidden_features=8,
        out_features=2,
        n_dim=2,
        modes=[4, 4],
        depth=2
    )
    
    total_params = sum(p.numel() for p in operator.parameters())

    assert total_params > 0, "FourierOperator should have parameters"
    
    readin_params = sum(p.numel() for p in operator.readin.parameters())
    fno_params = sum(p.numel() for p in operator.fno_units.parameters())
    readout_params = sum(p.numel() for p in operator.readout.parameters())
    
    assert readin_params > 0, "Readin layer should have parameters"
    assert fno_params > 0, "FNO units should have parameters"
    assert readout_params > 0, "Readout layer should have parameters"


def test_fourier_operator_builder():
    """Test FourierOperator construction using the builder pattern."""
    builder = FourierOperatorBuilder()\
    .set_architecture(
        in_features=3,
        hidden_features=20,
        out_features=3,
        n_dim=2,
        modes=[4, 4],
        depth=4,
    )\
    .set_activation_function(nn.ReLU)\
    .set_conv_module(
        conv_module=SpectralConv,
        skip_connection=SoftGatingConnection,
        conv_normalizer=BatchNormalizer,
    )\
    .set_feature_mlp(
        feature_mlp_module=LinearFeatureMLP,
        feature_mlp_skip_connection=SoftGatingConnection,
        feature_mlp_normalizer=LayerNormalizer,
        feature_expansion_factor=2.0,
        feature_mlp_depth=4
    )
    operator = builder.build()
    x = torch.randn(1, 3, 4, 4)
    output = operator(x)
    assert output.shape == (1, 3, 4, 4), f"Builder output shape mismatch: {output.shape}"
    assert output.dtype == torch.float32 or output.dtype == torch.float64, "Builder output should be real"