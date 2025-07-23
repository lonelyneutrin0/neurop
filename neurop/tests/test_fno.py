from neurop.operators.fourier import FourierOperator

import torch 

# def test_fourier_operator():
#     """Test the FourierOperator class."""
#     operator = FourierOperator(in_features=3, hidden_features=10, out_features=3, n_dim=2, modes=5)
#     x = torch.randn(2, 3, 4, 5)  # Example input tensor
#     output = operator(x)

#     assert output.shape == x.shape, "Output shape should match input shape"
#     assert output.dtype == torch.complex64, "FourierOperator should return complex tensor"
    
#     # Check if the output is a valid Fourier transform (not just identity)
#     assert not torch.allclose(output.real, x, atol=1e-3), "FourierOperator should transform the signal"