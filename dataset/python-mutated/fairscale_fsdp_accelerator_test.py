import os
from os import PathLike
from typing import Union, Dict, Optional
import torch
from torch.cuda import amp
from torch.testing import assert_allclose
import pytest
from allennlp.common.testing import AllenNlpTestCase, run_distributed_test, requires_multi_gpu
from allennlp.nn.util import load_state_dict_distributed
from allennlp.nn.parallel import FairScaleFsdpAccelerator, FairScaleFsdpWrappedModel, ShardedModuleMixin

class EncoderDecoderModel(torch.nn.Module):
    """
    Simple model to use for testing. We use an encoder-decoder architecture with tied
    embeddings to make sure we cover enough edge cases.
    """

    def __init__(self, fsdp_wrapper: FairScaleFsdpAccelerator) -> None:
        if False:
            return 10
        super().__init__()
        self.embedding = torch.nn.Embedding(12, 4)
        self.emb_proj = fsdp_wrapper.wrap_module(torch.nn.Linear(4, 4))
        self.encoder = fsdp_wrapper.wrap_module(Encoder())
        self.decoder = Decoder(self.embedding, fsdp_wrapper)
        self.register_buffer('buffer', torch.randn(4, 4))

    def tie_weights(self):
        if False:
            return 10
        '\n        Should be called after loading state dict to make sure embedding weigths are tied.\n        '
        self.decoder.linear.weight = self.embedding.weight

    def forward(self, x):
        if False:
            print('Hello World!')
        x = self.embedding(x)
        x = self.emb_proj(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Encoder(torch.nn.Module):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.ff1 = FeedForward()
        self.ff2 = FeedForward()
        self.register_buffer('buffer', torch.randn(4, 4))

    def forward(self, x):
        if False:
            while True:
                i = 10
        return self.ff2(self.ff1(x))

class Decoder(torch.nn.Module):

    def __init__(self, embedding: torch.nn.Embedding, fsdp_wrapper: FairScaleFsdpAccelerator) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.ff = fsdp_wrapper.wrap_module(FeedForward())
        self.linear = torch.nn.Linear(4, 12, bias=False)
        self.linear.weight = embedding.weight
        self.register_buffer('buffer', torch.randn(4, 4))

    def forward(self, x):
        if False:
            while True:
                i = 10
        return self.linear(self.ff(x))

class FeedForward(torch.nn.Module):

    def __init__(self) -> None:
        if False:
            return 10
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        return self.activation(self.linear(x))

def _dist_load_and_train(global_rank: int, world_size: int, gpu_id: int, test_dir: Union[str, PathLike], mixed_precision: bool, **kwargs):
    if False:
        while True:
            i = 10
    torch.manual_seed(global_rank)
    fsdp_wrapper = FairScaleFsdpAccelerator(local_rank=global_rank, world_size=world_size, cuda_device=gpu_id, mixed_precision=mixed_precision, **kwargs)
    model = EncoderDecoderModel(fsdp_wrapper)
    state_dict: Optional[Dict[str, torch.Tensor]] = None
    if global_rank == 0:
        embedding_weight = torch.randn(12, 4)
        state_dict = {'embedding.weight': embedding_weight, 'emb_proj.weight': torch.randn(4, 4), 'emb_proj.bias': torch.randn(4), 'encoder.ff1.linear.weight': torch.randn(4, 4), 'encoder.ff1.linear.bias': torch.randn(4), 'encoder.ff2.linear.weight': torch.randn(4, 4), 'encoder.ff2.linear.bias': torch.randn(4), 'encoder.buffer': torch.randn(4, 4), 'decoder.ff.linear.weight': torch.randn(4, 4), 'decoder.ff.linear.bias': torch.randn(4), 'decoder.linear.weight': embedding_weight, 'decoder.buffer': torch.randn(4, 4), 'buffer': torch.randn(4, 4)}
        torch.save(state_dict, os.path.join(test_dir, 'state.pt'))
    assert not isinstance(model.embedding, ShardedModuleMixin)
    assert isinstance(model.encoder, ShardedModuleMixin)
    assert isinstance(model.decoder.ff, ShardedModuleMixin)
    (missing_keys, unexpected_keys) = load_state_dict_distributed(model, state_dict)
    assert not missing_keys
    assert not unexpected_keys
    model.tie_weights()
    (model, wrapped_model) = fsdp_wrapper.wrap_model(model)
    scaler: Optional[amp.GradScaler] = None
    worker_state = wrapped_model.state_dict()
    for (name, value) in worker_state['weights'].items():
        if mixed_precision:
            assert value.device == torch.device('cpu')
        else:
            assert value.device == torch.device(gpu_id)
        assert value.dtype == torch.float, f'{name} is {value.dtype}'
    torch.save(worker_state, os.path.join(test_dir, f'state_worker{gpu_id}.pt'))
    optim = torch.optim.Adam(wrapped_model.model.parameters(), lr=0.0001)
    x = torch.randint(12, (2, 6)).to(torch.device(gpu_id))
    with amp.autocast(enabled=mixed_precision):
        x = wrapped_model.model(x)
        loss = x.sum()
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
    else:
        loss.backward()
        optim.step()
    torch.save(wrapped_model.state_dict(), os.path.join(test_dir, f'final_state_worker{gpu_id}.pt'))

class TestFairScaleFsdpAccelerator(AllenNlpTestCase):

    @pytest.mark.parametrize('mixed_precision', (True, False), ids=lambda val: f'amp={val}')
    @pytest.mark.parametrize('flatten_parameters', (True, False), ids=lambda val: f'flatten={val}')
    @requires_multi_gpu
    def test_distributed_loading_and_training(self, mixed_precision, flatten_parameters):
        if False:
            return 10
        run_distributed_test([0, 1], func=_dist_load_and_train, test_dir=self.TEST_DIR, mixed_precision=mixed_precision, flatten_parameters=flatten_parameters)
        original_state = torch.load(self.TEST_DIR / 'state.pt', map_location='cpu')
        consolidated_state = FairScaleFsdpWrappedModel.consolidate_sharded_state([self.TEST_DIR / 'state_worker0.pt', self.TEST_DIR / 'state_worker1.pt'])
        assert set(original_state.keys()) - set(consolidated_state.keys()) == {'decoder.linear.weight'}
        for (key, tensor0) in original_state.items():
            if key not in consolidated_state:
                continue
            tolerance = None if not mixed_precision or 'buffer' not in key else 0.001
            tensor1 = consolidated_state[key]
            assert_allclose(tensor0, tensor1, msg=f'{key} is off in consolidated state.\nExpected:\n{tensor0}\nGot:\n{tensor1}', atol=tolerance, rtol=tolerance)