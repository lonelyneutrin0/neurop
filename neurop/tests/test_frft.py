from neurop.frft import frft
import torch    

def test_frft_even_size():
    x = torch.randn(2, 4, 4)  
    output = frft(x, 1)
    assert output.shape == x.shape, "Output shape should match input shape"
    assert output.dtype == torch.complex64, "FRFT should return complex tensor"

def test_frft_odd_size():
    x = torch.randn(2, 3, 5) 
    try:
        frft(x, 1)
        assert False, "Should have raised ValueError for odd size"
    except ValueError as e:
        assert str(e) == "Signal size must be even", "Should raise ValueError for odd signal size"

def test_frft_invalid_size():
    x = torch.randn(2, 3, 4)  
    try:
        frft(x, 1)
    except ValueError as e:
        assert str(e) == "Signal size must be even", "Should raise ValueError for odd signal size"
    
def test_frft_zero_alpha():
    x = torch.randn(2, 4, 4)  
    output = frft(x, 0)
    assert torch.allclose(output, x, atol=1e-4), "Output should be equal to input for alpha=0"
    assert output.dtype == x.dtype, "Alpha=0 should preserve input dtype"

def test_frft_time_reversal():
    x = torch.randn(2, 4, 4)  
    output = frft(x, 2)

    first, remaining = torch.tensor_split(x, (1,), dim=-1)
    expected_output = torch.concat((first, remaining.flip(dims=[-1])), dim=-1)
    assert torch.allclose(output, expected_output, atol=1e-4), "Output should be time-reversed for alpha=2"
    assert output.dtype == x.dtype, "Alpha=2 should preserve input dtype"

def test_frft_negative_time_reversal():
    x = torch.randn(2, 4, 4)  
    output = frft(x, -2)

    first, remaining = torch.tensor_split(x, (1,), dim=-1)
    expected_output = torch.concat((first, remaining.flip(dims=[-1])), dim=-1)
    assert torch.allclose(output, expected_output, atol=1e-4), "Output should be time-reversed for alpha=-2"
    assert output.dtype == x.dtype, "Alpha=-2 should preserve input dtype"

def test_frft_general_case():
    """Test FRFT for general alpha values (not special cases)."""
    x = torch.randn(2, 4, 4)  
    output = frft(x, 0.5)  
    assert output.shape == x.shape, "Output shape should match input shape"
    assert output.dtype == torch.complex64, "General FRFT should return complex tensor"
    
    assert not torch.allclose(output.real, x, atol=1e-3), "FRFT with alpha=0.5 should transform the signal" 