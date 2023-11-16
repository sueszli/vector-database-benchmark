import torch
from allennlp.nn import util
from allennlp.common.testing import requires_gpu

@requires_gpu
def bench_add_sentence_boundary_token_ids(benchmark):
    if False:
        i = 10
        return i + 15
    device = torch.device('cuda')
    tensor = torch.tensor([[3] * 50] * 32, device=device)
    mask = torch.tensor([[True] * 50, [True] * 30 + [False] * 20] * 16, device=device)
    begin_token = 1
    end_token = 2
    benchmark(util.add_sentence_boundary_token_ids, tensor, mask, begin_token, end_token)

@requires_gpu
def bench_remove_sentence_boundaries(benchmark):
    if False:
        while True:
            i = 10
    device = torch.device('cuda')
    tensor = torch.tensor([[3] * 50] * 32, device=device).unsqueeze(-1)
    mask = torch.tensor([[True] * 50, [True] * 30 + [False] * 20] * 16, device=device)
    benchmark(util.remove_sentence_boundaries, tensor, mask)

@requires_gpu
def bench_create_tensor_then_send_to_device(benchmark):
    if False:
        i = 10
        return i + 15
    device = torch.device('cuda:0')

    def create_tensor():
        if False:
            return 10
        return torch.rand((32, 50)).to(device)
    benchmark(create_tensor)

@requires_gpu
def bench_create_tensor_directly_on_device(benchmark):
    if False:
        return 10
    device = torch.device('cuda:0')

    def create_tensor():
        if False:
            return 10
        return torch.rand((32, 50), device=device)
    benchmark(create_tensor)