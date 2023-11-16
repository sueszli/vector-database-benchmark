import argparse
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F

def parse_args():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(description='test script')
    parser.add_argument('--init-method', '--init_method', type=str, required=True, help='init_method to pass to `dist.init_process_group()` (e.g. env://)')
    parser.add_argument('--world-size', '--world_size', type=int, default=os.getenv('WORLD_SIZE', -1), help='world_size to pass to `dist.init_process_group()`')
    parser.add_argument('--rank', type=int, default=os.getenv('RANK', -1), help='rank to pass to `dist.init_process_group()`')
    return parser.parse_args()

def main():
    if False:
        for i in range(10):
            print('nop')
    args = parse_args()
    dist.init_process_group(backend='gloo', init_method=args.init_method, world_size=args.world_size, rank=args.rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    t = F.one_hot(torch.tensor(rank), num_classes=world_size)
    dist.all_reduce(t)
    derived_world_size = torch.sum(t).item()
    if derived_world_size != world_size:
        raise RuntimeError(f'Wrong world size derived. Expected: {world_size}, Got: {derived_world_size}')
    print('Done')
if __name__ == '__main__':
    main()