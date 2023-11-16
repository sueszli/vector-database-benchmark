import logging
from typing import List
logger = logging.getLogger(__name__)

def uniform(dataset_sizes: List[int]):
    if False:
        return 10
    return [1.0] * len(dataset_sizes)

def temperature_sampling(dataset_sizes, temp):
    if False:
        for i in range(10):
            print('nop')
    total_size = sum(dataset_sizes)
    return [(size / total_size) ** (1.0 / temp) for size in dataset_sizes]

def make_temperature_sampling(temp=1.0):
    if False:
        print('Hello World!')

    def sampling_func(dataset_sizes):
        if False:
            i = 10
            return i + 15
        return temperature_sampling(dataset_sizes, temp)
    return sampling_func

def make_ratio_sampling(ratios):
    if False:
        return 10

    def sampling_func(dataset_sizes):
        if False:
            for i in range(10):
                print('nop')
        return ratios
    return sampling_func

class SamplingMethod:

    @staticmethod
    def add_arguments(parser):
        if False:
            print('Hello World!')
        parser.add_argument('--sampling-method', choices=['uniform', 'temperature', 'concat', 'RoundRobin'], type=str, default='concat', help='The method to sample data per language pairs')
        parser.add_argument('--sampling-temperature', default=1.5, type=float, help='only work with --sampling-method temperature')

    @staticmethod
    def build_sampler(args, task):
        if False:
            print('Hello World!')
        return SamplingMethod(args, task)

    def __init__(self, args, task):
        if False:
            return 10
        self.args = args
        self.task = task

    def is_adaptive(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def sampling_method_selector(self):
        if False:
            i = 10
            return i + 15
        args = self.args
        logger.info(f'selected sampler: {args.sampling_method}')
        if args.sampling_method == 'uniform':
            return uniform
        elif args.sampling_method == 'temperature' or self.is_adaptive():
            return make_temperature_sampling(float(args.sampling_temperature))
        else:
            return None