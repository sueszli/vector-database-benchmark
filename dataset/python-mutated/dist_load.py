import copy
import re
import paddle
import paddle.distributed as dist
from paddle.base.framework import dygraph_only
from paddle.distributed import fleet

@dygraph_only
def load(path, **configs):
    if False:
        i = 10
        return i + 15
    '\n    Load an object can be used in paddle from specified path.\n    The file is saved by distributed.save\n\n    Note:\n        The file to load must be saved bu the API paddle.incubate.distributed.utils.io.save\n\n    Args:\n        path(str|BytesIO) : The path/buffer to load the target object. Generally, the path is the target\n            file path. When loading state_dict from the saved result of the API used to save\n            the inference model, the path may be a file prefix or directory.\n        **configs (dict, optional): other load configuration options for compatibility. We do not\n            recommend using these configurations, they may be removed in the future. If not necessary,\n            DO NOT use them. Default None.\n            The following options are currently supported:\n                (1) place: where to place the loaded state dict.\n                     If the state dict is too large, the palce should be set \'cpu\'.\n            Note:\n                Other config value may cause some error.Please don\'t use any more config options.\n    Returns:\n        Object(Object): a target object can be used in paddle\n\n    Examples:\n        import paddle\n        paddle.distributed.init_process_group(backend=\'nccl\')\n        paddle.distributed.fleet.init(is_collective=True)\n\n        model = build_model()\n        optimizer = build_optimizer(model)\n\n        dist_model = paddle.distributed_optimizer(model)\n        dist_optimizer = paddle.distributed_optimizer(optimizer)\n\n\n        # load model state dict\n        model_state_dict = paddle.incubate.distributed.utils.io.load(path="path/to/load.pdparams")\n        dist_model.set_state_dict(model_state_dict)\n\n        # load optimizer satte dict\n        optimizer_state_dict = paddle.incubate.distributed.utils.io.load(path="path/to/load.pdopt")\n        dist_optimizer.set_state_dict(optimizer_state_dict)\n\n    '
    if dist.get_world_size() == 1:
        return paddle.load(path, **configs)
    hcg = fleet.get_hybrid_communicate_group()
    assert hcg.get_model_parallel_world_size() == 1 and hcg.get_pipe_parallel_world_size() == 1, 'Sharding and DP are supported only now'
    if 'place' not in configs:
        configs['place'] = 'cpu'
    place = configs['place']
    assert isinstance(place, str), f'configs[place] must be a str, but this is a {type(place)}'
    assert re.search('^(cpu|gpu:[0-9]*)$', place), 'configs[place] must be cpu, gpu:0, gpu:1 ...'
    return load_with_place(path, **configs)

def load_with_place(path, **configs):
    if False:
        i = 10
        return i + 15
    place = configs['place']
    if place is None:
        return paddle.load(path)
    origin_place = paddle.get_device()
    paddle.set_device(place)
    configs = _remove_not_supported_itmes(configs)
    state_dict = paddle.load(path, **configs)
    paddle.set_device(origin_place)
    return state_dict

def _remove_not_supported_itmes(configs):
    if False:
        return 10
    __supported_by_load__ = ['model_filename', 'params_filename', 'return_numpy']
    _configs = copy.copy(configs)
    for k in configs.keys():
        if k not in __supported_by_load__:
            _configs.pop(k, None)
    return _configs