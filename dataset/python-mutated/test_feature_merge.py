import pytest
import torch
from ding.torch_utils.network.merge import TorchBilinearCustomized, TorchBilinear, BilinearGeneral, FiLM

@pytest.mark.unittest
def test_torch_bilinear_customized():
    if False:
        while True:
            i = 10
    batch_size = 10
    in1_features = 20
    in2_features = 30
    out_features = 40
    bilinear_customized = TorchBilinearCustomized(in1_features, in2_features, out_features)
    x = torch.randn(batch_size, in1_features)
    z = torch.randn(batch_size, in2_features)
    out = bilinear_customized(x, z)
    assert out.shape == (batch_size, out_features), 'Output shape does not match expected shape.'

@pytest.mark.unittest
def test_torch_bilinear():
    if False:
        return 10
    batch_size = 10
    in1_features = 20
    in2_features = 30
    out_features = 40
    torch_bilinear = TorchBilinear(in1_features, in2_features, out_features)
    x = torch.randn(batch_size, in1_features)
    z = torch.randn(batch_size, in2_features)
    out = torch_bilinear(x, z)
    assert out.shape == (batch_size, out_features), 'Output shape does not match expected shape.'

@pytest.mark.unittest
def test_bilinear_consistency():
    if False:
        for i in range(10):
            print('nop')
    batch_size = 10
    in1_features = 20
    in2_features = 30
    out_features = 40
    weight = torch.randn(out_features, in1_features, in2_features)
    bias = torch.randn(out_features)
    bilinear_customized = TorchBilinearCustomized(in1_features, in2_features, out_features)
    bilinear_customized.weight.data = weight.clone()
    bilinear_customized.bias.data = bias.clone()
    torch_bilinear = TorchBilinear(in1_features, in2_features, out_features)
    torch_bilinear.weight.data = weight.clone()
    torch_bilinear.bias.data = bias.clone()
    x = torch.randn(batch_size, in1_features)
    z = torch.randn(batch_size, in2_features)
    out_bilinear_customized = bilinear_customized(x, z)
    out_torch_bilinear = torch_bilinear(x, z)
    mse = torch.mean((out_bilinear_customized - out_torch_bilinear) ** 2)
    print(f'Mean Squared Error between outputs: {mse.item()}')

def test_bilinear_general():
    if False:
        i = 10
        return i + 15
    '\n    Overview:\n        Test for the `BilinearGeneral` class.\n    '
    in1_features = 20
    in2_features = 30
    out_features = 40
    batch_size = 10
    bilinear_general = BilinearGeneral(in1_features, in2_features, out_features)
    input1 = torch.randn(batch_size, in1_features)
    input2 = torch.randn(batch_size, in2_features)
    output = bilinear_general(input1, input2)
    assert output.shape == (batch_size, out_features), 'Output shape does not match expected shape.'
    assert bilinear_general.W.shape == (out_features, in1_features, in2_features), 'Weight W shape does not match expected shape.'
    assert bilinear_general.U.shape == (out_features, in2_features), 'Weight U shape does not match expected shape.'
    assert bilinear_general.V.shape == (out_features, in1_features), 'Weight V shape does not match expected shape.'
    assert bilinear_general.b.shape == (out_features,), 'Bias shape does not match expected shape.'
    assert isinstance(bilinear_general.W, torch.nn.Parameter), 'Weight W is not an instance of torch.nn.Parameter.'
    assert isinstance(bilinear_general.U, torch.nn.Parameter), 'Weight U is not an instance of torch.nn.Parameter.'
    assert isinstance(bilinear_general.V, torch.nn.Parameter), 'Weight V is not an instance of torch.nn.Parameter.'
    assert isinstance(bilinear_general.b, torch.nn.Parameter), 'Bias is not an instance of torch.nn.Parameter.'

@pytest.mark.unittest
def test_film_forward():
    if False:
        print('Hello World!')
    feature_dim = 128
    context_dim = 256
    film_layer = FiLM(feature_dim, context_dim)
    feature = torch.randn((32, feature_dim))
    context = torch.randn((32, context_dim))
    conditioned_feature = film_layer(feature, context)
    assert conditioned_feature.shape == feature.shape, f'Expected output shape {feature.shape}, but got {conditioned_feature.shape}'
    assert not torch.all(torch.eq(feature, conditioned_feature)), 'The output feature is the same as the input feature'