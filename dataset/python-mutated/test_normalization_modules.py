from typing import Optional
import pytest
import torch
from ludwig.modules.normalization_modules import GhostBatchNormalization
BATCH_SIZE = 16
OUTPUT_SIZE = 8

@pytest.mark.parametrize('virtual_batch_size', [None, BATCH_SIZE // 2, BATCH_SIZE - 14, BATCH_SIZE - 10])
@pytest.mark.parametrize('mode', [True, False])
def test_ghostbatchnormalization(mode: bool, virtual_batch_size: Optional[int]) -> None:
    if False:
        print('Hello World!')
    ghost_batch_norm = GhostBatchNormalization(OUTPUT_SIZE, virtual_batch_size=virtual_batch_size)
    ghost_batch_norm.train(mode=mode)
    inputs = torch.randn([BATCH_SIZE, OUTPUT_SIZE], dtype=torch.float32)
    norm_tensor = ghost_batch_norm(inputs)
    assert isinstance(norm_tensor, torch.Tensor)
    assert norm_tensor.shape == (BATCH_SIZE, OUTPUT_SIZE)
    assert ghost_batch_norm.input_shape == inputs.shape[1:]
    assert ghost_batch_norm.output_shape == inputs.shape[1:]
    assert ghost_batch_norm.input_dtype == torch.float32
    assert isinstance(ghost_batch_norm.moving_mean, torch.Tensor)
    assert ghost_batch_norm.moving_mean.shape == (OUTPUT_SIZE,)
    assert isinstance(ghost_batch_norm.moving_variance, torch.Tensor)
    assert ghost_batch_norm.moving_variance.shape == (OUTPUT_SIZE,)

def test_ghostbatchnormalization_chunk_size_2() -> None:
    if False:
        i = 10
        return i + 15
    'Test GhostBatchNormalization with virtual_batch_size=2 and batch_size=7 This creates chunks of size 2, 2, 2,\n    1 which should be handled correctly since we should skip applying batch norm to the last chunk since it is size\n    1.'
    ghost_batch_norm = GhostBatchNormalization(6, virtual_batch_size=2)
    inputs = torch.randn([7, 6], dtype=torch.float32)
    ghost_batch_norm.train(mode=True)
    ghost_batch_norm(inputs)