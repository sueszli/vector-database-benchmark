"""
A simple launcher script for TPU training

Inspired by https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py

::
    >>> python xla_spawn.py --num_cores=NUM_CORES_YOU_HAVE
               YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
               arguments of your training script)

"""
import importlib
import sys
from argparse import REMAINDER, ArgumentParser
from pathlib import Path
import torch_xla.distributed.xla_multiprocessing as xmp

def parse_args():
    if False:
        while True:
            i = 10
    '\n    Helper function parsing the command line options\n    @retval ArgumentParser\n    '
    parser = ArgumentParser(description='PyTorch TPU distributed training launch helper utility that will spawn up multiple distributed processes')
    parser.add_argument('--num_cores', type=int, default=1, help='Number of TPU cores to use (1 or 8).')
    parser.add_argument('training_script', type=str, help='The full path to the single TPU training program/script to be launched in parallel, followed by all the arguments for the training script')
    parser.add_argument('training_script_args', nargs=REMAINDER)
    return parser.parse_args()

def main():
    if False:
        for i in range(10):
            print('nop')
    args = parse_args()
    script_fpath = Path(args.training_script)
    sys.path.append(str(script_fpath.parent.resolve()))
    mod_name = script_fpath.stem
    mod = importlib.import_module(mod_name)
    sys.argv = [args.training_script] + args.training_script_args + ['--tpu_num_cores', str(args.num_cores)]
    xmp.spawn(mod._mp_fn, args=(), nprocs=args.num_cores)
if __name__ == '__main__':
    main()