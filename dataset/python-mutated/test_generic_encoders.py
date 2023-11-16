import pytest
import torch
from ludwig.constants import ENCODER_OUTPUT
from ludwig.encoders.generic_encoders import DenseEncoder, PassthroughEncoder

@pytest.mark.parametrize('input_size', [1, 2, 10])
@pytest.mark.parametrize('categorical', [True, False])
def test_generic_passthrough_encoder(input_size: int, categorical: bool):
    if False:
        for i in range(10):
            print('nop')
    passthrough_encoder = PassthroughEncoder(input_size)
    if categorical:
        inputs = torch.randint(10, (10, input_size))
    else:
        inputs = torch.rand((10, input_size))
    outputs = passthrough_encoder(inputs)
    assert outputs[ENCODER_OUTPUT].shape[1:] == passthrough_encoder.output_shape

@pytest.mark.parametrize('input_size', [1, 2, 10])
@pytest.mark.parametrize('num_layers', [1, 3, 6])
@pytest.mark.parametrize('output_size', [1, 2, 10, 256])
def test_generic_dense_encoder(input_size: int, num_layers: int, output_size: int):
    if False:
        for i in range(10):
            print('nop')
    dense_encoder = DenseEncoder(input_size, num_layers=num_layers, output_size=output_size)
    inputs = torch.rand((10, input_size))
    outputs = dense_encoder(inputs)
    assert outputs[ENCODER_OUTPUT].shape[1:] == dense_encoder.output_shape