"""
The following example demonstrates how to use Pytorch Distributed Checkpoint to save a FSDP model.

This is the current recommended way to checkpoint FSDP.
torch.save() and torch.load() is not recommended when checkpointing sharded models.
"""
import os
import shutil
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
CHECKPOINT_DIR = f"/scratch/{os.environ['LOGNAME']}/checkpoint"

def opt_at(opt, idx):
    if False:
        while True:
            i = 10
    return list(opt.state.values())[idx]

def init_model():
    if False:
        for i in range(10):
            print('nop')
    model = FSDP(torch.nn.Linear(4, 4).cuda(dist.get_rank()))
    optim = torch.optim.Adam(model.parameters(), lr=0.1)
    model(torch.rand(4, 4)).sum().backward()
    optim.step()
    return (model, optim)

def print_params(stage, model_1, model_2, optim_1, optim_2):
    if False:
        print('Hello World!')
    with FSDP.summon_full_params(model_1):
        with FSDP.summon_full_params(model_2):
            print(f'{stage} --- rank: {dist.get_rank()}\nmodel.weight: {model_1.weight}\nmodel_2.weight:{model_2.weight}\nmodel.bias: {model_1.bias}\nmodel_2.bias: {model_2.bias}\n')
    print(f"{stage} --- rank: {dist.get_rank()}\noptim exp_avg:{opt_at(optim_1, 0)['exp_avg']}\noptim_2 exp_avg:{opt_at(optim_2, 0)['exp_avg']}\noptim exp_avg_sq:{opt_at(optim_1, 0)['exp_avg_sq']}\noptim_2 exp_avg_sq:{opt_at(optim_2, 0)['exp_avg_sq']}\n")

def run_fsdp_checkpoint_example(rank, world_size):
    if False:
        i = 10
        return i + 15
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('cpu:gloo,cuda:nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    (model_1, optim_1) = init_model()
    with FSDP.state_dict_type(model_1, StateDictType.SHARDED_STATE_DICT):
        state_dict = {'model': model_1.state_dict(), 'optim': FSDP.optim_state_dict(model_1, optim_1)}
        dist_cp.save_state_dict(state_dict=state_dict, storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR))
    (model_2, optim_2) = init_model()
    print_params('Before loading', model_1, model_2, optim_1, optim_2)
    with FSDP.state_dict_type(model_2, StateDictType.SHARDED_STATE_DICT):
        state_dict = {'model': model_2.state_dict()}
        dist_cp.load_state_dict(state_dict=state_dict, storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR))
        model_2.load_state_dict(state_dict['model'])
        optim_state = load_sharded_optimizer_state_dict(model_state_dict=state_dict['model'], optimizer_key='optim', storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR))
        flattened_osd = FSDP.optim_state_dict_to_load(model_2, optim_2, optim_state['optim'])
        optim_2.load_state_dict(flattened_osd)
    print_params('After loading', model_1, model_2, optim_1, optim_2)
    dist.destroy_process_group()
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(f'Running fsdp checkpoint example on {world_size} devices.')
    shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    mp.spawn(run_fsdp_checkpoint_example, args=(world_size,), nprocs=world_size, join=True)