from typing import Optional
import pytest
import torch
from ludwig.modules.tabnet_modules import AttentiveTransformer, FeatureBlock, FeatureTransformer, TabNet
from ludwig.utils.entmax import sparsemax
from tests.integration_tests.parameter_update_utils import check_module_parameters_updated
RANDOM_SEED = 67

@pytest.mark.parametrize('input_tensor', [torch.tensor([[-1.0, 0.0, 1.0], [5.01, 4.0, -2.0]], dtype=torch.float32), torch.tensor([[136762051.0, -136762051.0, 1.59594639e+20], [1.59594639e+37, 13676205.1, 1260000.0]], dtype=torch.float32)])
def test_sparsemax(input_tensor: torch.Tensor) -> None:
    if False:
        return 10
    output_tensor = sparsemax(input_tensor)
    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.equal(torch.tensor([[0, 0, 1], [1, 0, 0]], dtype=torch.float32))

@pytest.mark.parametrize('bn_virtual_bs', [None, 7])
@pytest.mark.parametrize('external_shared_fc_layer', [True, False])
@pytest.mark.parametrize('apply_glu', [True, False])
@pytest.mark.parametrize('size', [4, 12])
@pytest.mark.parametrize('input_size', [2, 6])
@pytest.mark.parametrize('batch_size', [1, 16])
def test_feature_block(input_size, size: int, apply_glu: bool, external_shared_fc_layer: bool, bn_virtual_bs: Optional[int], batch_size: int) -> None:
    if False:
        return 10
    torch.manual_seed(RANDOM_SEED)
    input_tensor = torch.randn([batch_size, input_size], dtype=torch.float32)
    if external_shared_fc_layer:
        shared_fc_layer = torch.nn.Linear(input_size, size * 2 if apply_glu else size, bias=False)
    else:
        shared_fc_layer = None
    feature_block = FeatureBlock(input_size, size, apply_glu=apply_glu, shared_fc_layer=shared_fc_layer, bn_virtual_bs=bn_virtual_bs)
    output_tensor = feature_block(input_tensor)
    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.shape == (batch_size, size)
    assert feature_block.input_shape[-1] == input_size
    assert feature_block.output_shape[-1] == size
    assert feature_block.input_dtype == torch.float32

@pytest.mark.parametrize('num_total_blocks, num_shared_blocks', [(4, 2), (6, 4), (3, 1)])
@pytest.mark.parametrize('virtual_batch_size', [None, 7])
@pytest.mark.parametrize('size', [4, 12])
@pytest.mark.parametrize('input_size', [2, 6])
@pytest.mark.parametrize('batch_size', [1, 16])
def test_feature_transformer(input_size: int, size: int, virtual_batch_size: Optional[int], num_total_blocks: int, num_shared_blocks: int, batch_size: int) -> None:
    if False:
        while True:
            i = 10
    torch.manual_seed(RANDOM_SEED)
    input_tensor = torch.randn([batch_size, input_size], dtype=torch.float32)
    feature_transformer = FeatureTransformer(input_size, size, bn_virtual_bs=virtual_batch_size, num_total_blocks=num_total_blocks, num_shared_blocks=num_shared_blocks)
    output_tensor = feature_transformer(input_tensor)
    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.shape == (batch_size, size)
    assert feature_transformer.input_shape[-1] == input_size
    assert feature_transformer.output_shape[-1] == size
    assert feature_transformer.input_dtype == torch.float32

@pytest.mark.parametrize('virtual_batch_size', [None, 7])
@pytest.mark.parametrize('output_size', [10, 12])
@pytest.mark.parametrize('size', [4, 8])
@pytest.mark.parametrize('input_size', [2, 6])
@pytest.mark.parametrize('entmax_mode', [None, 'entmax15', 'adaptive', 'constant'])
@pytest.mark.parametrize('batch_size', [1, 16])
def test_attentive_transformer(entmax_mode: Optional[str], input_size: int, size: int, output_size: int, virtual_batch_size: Optional[int], batch_size: int) -> None:
    if False:
        while True:
            i = 10
    torch.manual_seed(RANDOM_SEED)
    input_tensor = torch.randn([batch_size, input_size], dtype=torch.float32)
    prior_scales = torch.ones([batch_size, input_size])
    feature_transformer = FeatureTransformer(input_size, size + output_size, bn_virtual_bs=virtual_batch_size)
    attentive_transformer = AttentiveTransformer(size, input_size, bn_virtual_bs=virtual_batch_size, entmax_mode=entmax_mode)
    x = feature_transformer(input_tensor)
    output_tensor = attentive_transformer(x[:, output_size:], prior_scales)
    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.shape == (batch_size, input_size)
    assert attentive_transformer.input_shape[-1] == size
    assert attentive_transformer.output_shape[-1] == input_size
    assert attentive_transformer.input_dtype == torch.float32
    if entmax_mode == 'adaptive':
        assert isinstance(attentive_transformer.trainable_alpha, torch.Tensor)

@pytest.mark.parametrize('virtual_batch_size', [None, 7])
@pytest.mark.parametrize('size', [2, 4, 8])
@pytest.mark.parametrize('output_size', [2, 4, 12])
@pytest.mark.parametrize('input_size', [2])
@pytest.mark.parametrize('entmax_mode', [None, 'entmax15', 'adaptive', 'constant'])
@pytest.mark.parametrize('batch_size', [1, 16])
def test_tabnet(entmax_mode: Optional[str], input_size: int, output_size: int, size: int, virtual_batch_size: Optional[int], batch_size: int) -> None:
    if False:
        i = 10
        return i + 15
    torch.manual_seed(RANDOM_SEED)
    input_tensor = torch.randn([batch_size, input_size], dtype=torch.float32)
    tabnet = TabNet(input_size, size, output_size, num_steps=3, num_total_blocks=4, num_shared_blocks=2, entmax_mode=entmax_mode)
    output = tabnet(input_tensor)
    assert isinstance(output, tuple)
    assert output[0].shape == (batch_size, output_size)
    assert tabnet.input_shape[-1] == input_size
    assert tabnet.output_shape[-1] == output_size
    assert tabnet.input_dtype == torch.float32
    target = torch.randn([batch_size, 1])
    (fpc, tpc, upc, not_updated) = check_module_parameters_updated(tabnet, (input_tensor,), target)
    if batch_size == 1:
        assert upc == 17, f'Updated parameter count not expected value. Parameters not updated: {not_updated}\nModule structure:\n{tabnet}'
    else:
        assert tpc == upc, f'All parameter not updated. Parameters not updated: {not_updated}\nModule structure:\n{tabnet}'