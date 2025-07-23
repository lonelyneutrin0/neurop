"""Normalizers Tests."""
from neurop.layers.normalizers import BatchNormalizer, LayerNormalizer, InstanceNormalizer 

import torch 

def test_batch_normalizer_learnable():
    """Test the BatchNormalizer class."""
    normalizer = BatchNormalizer(num_features=3, ndim=2, learnable=False)
    x = torch.randn(2, 3, 4, 5)  
    output = normalizer(x)
    
    assert torch.allclose(torch.mean(output, dim=(0, 2, 3)), torch.zeros((3,)), atol=1e-4), "Batch Normalizer should center the data"
    assert torch.allclose(torch.var(output, dim=(0, 2, 3), unbiased=False), torch.ones((3,)), atol=1e-4), "Batch Normalizer should scale the data to unit variance"
    assert output.shape == x.shape, "Output shape should match input shape"
    
    normalizer_learnable = BatchNormalizer(num_features=3, ndim=2, learnable=True)
    output_learnable = normalizer_learnable(x)
    assert output_learnable.shape == x.shape, "Output shape should match input shape with learnable parameters"

def test_layer_normalizer_learnable():
    """Test the LayerNormalizer class."""
    normalizer = LayerNormalizer(num_features=3, ndim=2, learnable=False)
    x = torch.randn(2, 3, 4, 5) 
    output = normalizer(x)
    
    assert torch.allclose(torch.mean(output, dim=(1, 2, 3)), torch.zeros((2,)), atol=1e-4), "Layer Normalizer should center the data"
    assert torch.allclose(torch.var(output, dim=(1, 2, 3), unbiased=False), torch.ones((2,)), atol=1e-4), "Layer Normalizer should scale the data to unit variance"
    assert output.shape == x.shape, "Output shape should match input shape"

    normalizer_learnable = LayerNormalizer(num_features=3, ndim=2, learnable=True)
    output_learnable = normalizer_learnable(x)
    assert output_learnable.shape == x.shape, "Output shape should match input shape with learnable parameters"

def test_instance_normalizer_learnable():
    """Test the InstanceNormalizer class."""
    normalizer = InstanceNormalizer(num_features=3, ndim=2, learnable=False)
    x = torch.randn(2, 3, 4, 5)  # Example input tensor
    output = normalizer(x)
    
    assert torch.allclose(torch.mean(output, dim=(2, 3)), torch.zeros((2, 3)), atol=1e-4), "Instance Normalizer should center the data"
    assert torch.allclose(torch.var(output, dim=(2, 3), unbiased=False), torch.ones((2, 3)), atol=1e-4), "Instance Normalizer should scale the data to unit variance"
    assert output.shape == x.shape, "Output shape should match input shape"
    
    normalizer_learnable = InstanceNormalizer(num_features=3, ndim=2, learnable=True)
    output_learnable = normalizer_learnable(x)
    assert output_learnable.shape == x.shape, "Output shape should match input shape with learnable parameters"

def test_instance_normalizer():
    """Test the InstanceNormalizer class."""
    normalizer = InstanceNormalizer(num_features=3, ndim=2, learnable=False)
    x = torch.randn(2, 3, 4, 5) 
    output = normalizer(x)

    assert torch.allclose(torch.mean(output, dim=(2, 3)), torch.zeros((2, 3)), atol=1e-4), "Instance Normalizer should center the data"
    assert torch.allclose(torch.var(output, dim=(2, 3), unbiased=False), torch.ones((2, 3)), atol=1e-4), "Instance Normalizer should scale the data to unit variance"
    assert output.shape == x.shape, "Output shape should match input shape"

def test_batch_normalizer():
    """Test the BatchNormalizer class."""
    normalizer = BatchNormalizer(num_features=3, ndim=2, learnable=False)
    x = torch.randn(2, 3, 4, 5)  
    output = normalizer(x)

    assert torch.allclose(torch.mean(output, dim=(0, 2, 3)), torch.zeros((3,)), atol=1e-4), "Batch Normalizer should center the data"
    assert torch.allclose(torch.var(output, dim=(0, 2, 3), unbiased=False), torch.ones((3,)), atol=1e-4), "Batch Normalizer should scale the data to unit variance"
    assert output.shape == x.shape, "Output shape should match input shape"

def test_layer_normalizer():
    """Test the LayerNormalizer class."""
    normalizer = LayerNormalizer(num_features=3, ndim=2, learnable=False)
    x = torch.randn(2, 3, 4, 5) 
    output = normalizer(x)

    assert torch.allclose(torch.mean(output, dim=(1, 2, 3)), torch.zeros((2,)), atol=1e-4), "Layer Normalizer should center the data"
    assert torch.allclose(torch.var(output, dim=(1, 2, 3), unbiased=False), torch.ones((2,)), atol=1e-4), "Layer Normalizer should scale the data to unit variance"
    assert output.shape == x.shape, "Output shape should match input shape"
