from neurop.operators.fourier import FourierOperator
import torch

def test_fourier_operator():
    """
    Test the FourierOperator functionality.
    """
    batch_size = 2
    channels = 1
    shape = (8, 8, 8, 8)  # 4 spatial dimensions

    # Create a Fourier operator instance
    fourier_op = FourierOperator(in_features=channels, out_features=channels, hidden_features=32, modes=[6, 6, 6, 5], depth=4, activation_function=torch.relu)

    # Generate random input data
    x = torch.randn(batch_size, channels, *shape)

    # Apply the Fourier operator
    y = fourier_op(x)

    # Check output shape
    assert y.shape == (batch_size, channels, *shape), "Output shape mismatch in FourierOperator"