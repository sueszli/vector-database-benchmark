import torch
from torch.autograd import Variable
from torch.nn.parallel._functions import Scatter

def scatter(inputs, target_gpus, dim=0, chunk_sizes=None):
    if False:
        return 10
    '\n    Slices variables into approximately equal chunks and\n    distributes them across given GPUs. Duplicates\n    references to objects that are not variables. Does not\n    support Tensors.\n    '

    def scatter_map(obj):
        if False:
            while True:
                i = 10
        if isinstance(obj, Variable):
            return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
        assert not torch.is_tensor(obj), 'Tensors not supported in scatter.'
        if isinstance(obj, tuple):
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list):
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict):
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]
    return scatter_map(inputs)

def scatter_kwargs(inputs, kwargs, target_gpus, dim=0, chunk_sizes=None):
    if False:
        print('Hello World!')
    'Scatter with support for kwargs dictionary'
    inputs = scatter(inputs, target_gpus, dim, chunk_sizes) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim, chunk_sizes) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return (inputs, kwargs)