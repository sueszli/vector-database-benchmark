import os
import time
from copy import deepcopy
import torch
import torch.distributed
import torch.nn.functional
from lightning.fabric.fabric import Fabric
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from parity_fabric.models import ConvNet
from parity_fabric.utils import cuda_reset, is_cuda_memory_close, is_state_dict_equal, is_timing_close, make_deterministic

def train_torch_ddp(rank, world_size, device=torch.device('cpu'), backend='nccl'):
    if False:
        return 10
    make_deterministic()
    memory_stats = {}
    os.environ['LOCAL_RANK'] = str(rank)
    torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)
    model = ConvNet().to(device)
    initial_state_dict = deepcopy(model.state_dict())
    ddp_model = DistributedDataParallel(model, device_ids=[rank] if device.type == 'cuda' else None)
    dataloader = model.get_dataloader()
    sampler = DistributedSampler(dataloader.dataset, rank=rank, num_replicas=world_size, drop_last=False, shuffle=False)
    dataloader = DataLoader(dataloader.dataset, sampler=sampler, batch_size=model.batch_size)
    optimizer = model.get_optimizer()
    loss_fn = model.get_loss_function()
    memory_stats['start'] = torch.cuda.memory_stats()
    ddp_model.train()
    iteration_timings = []
    iterator = iter(dataloader)
    for _ in range(model.num_steps):
        t0 = time.perf_counter()
        (inputs, labels) = next(iterator)
        (inputs, labels) = (inputs.to(device), labels.to(device))
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        t1 = time.perf_counter()
        iteration_timings.append(t1 - t0)
    memory_stats['end'] = torch.cuda.memory_stats()
    assert not is_state_dict_equal(initial_state_dict, ddp_model.module.state_dict())
    return (ddp_model.module.state_dict(), torch.tensor(iteration_timings), memory_stats)

def train_fabric_ddp(fabric):
    if False:
        print('Hello World!')
    make_deterministic()
    memory_stats = {}
    model = ConvNet()
    initial_state_dict = deepcopy(model.state_dict())
    optimizer = model.get_optimizer()
    (model, optimizer) = fabric.setup(model, optimizer)
    dataloader = model.get_dataloader()
    dataloader = fabric.setup_dataloaders(dataloader)
    loss_fn = model.get_loss_function()
    memory_stats['start'] = torch.cuda.memory_stats()
    model.train()
    iteration_timings = []
    iterator = iter(dataloader)
    for _ in range(model.num_steps):
        t0 = time.perf_counter()
        (inputs, labels) = next(iterator)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        fabric.backward(loss)
        optimizer.step()
        t1 = time.perf_counter()
        iteration_timings.append(t1 - t0)
    memory_stats['end'] = torch.cuda.memory_stats()
    assert not is_state_dict_equal(initial_state_dict, model.state_dict())
    return (model.state_dict(), torch.tensor(iteration_timings), memory_stats)

def run_parity_test(accelerator: str='cpu', devices: int=2, tolerance: float=0.02):
    if False:
        i = 10
        return i + 15
    cuda_reset()
    fabric = Fabric(accelerator=accelerator, strategy='ddp', devices=devices)
    fabric.launch()
    (state_dict_fabric, timings_fabric, memory_fabric) = train_fabric_ddp(fabric)
    fabric.barrier()
    cuda_reset()
    torch.distributed.destroy_process_group()
    time.sleep(3)
    (state_dict_torch, timings_torch, memory_torch) = train_torch_ddp(rank=fabric.global_rank, world_size=fabric.world_size, device=fabric.device, backend=fabric.strategy._process_group_backend)
    assert all(fabric.all_gather(is_state_dict_equal(state_dict_torch, state_dict_fabric)))
    assert all(fabric.all_gather(is_timing_close(timings_torch, timings_fabric, rtol=tolerance, atol=tolerance)))
    if accelerator == 'cuda':
        assert all(fabric.all_gather(is_cuda_memory_close(memory_torch['start'], memory_fabric['start'])))
        assert all(fabric.all_gather(is_cuda_memory_close(memory_torch['end'], memory_fabric['end'])))
if __name__ == '__main__':
    from jsonargparse import CLI
    CLI(run_parity_test)