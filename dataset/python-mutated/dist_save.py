import copy
import re
import sys
import paddle
import paddle.distributed as dist
from paddle.base.framework import dygraph_only
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.log_util import logger
from .save_for_auto import save_for_auto_inference
__all__ = ['save', 'save_for_auto_inference']

@dygraph_only
def save(state_dict, path, **configs):
    if False:
        print('Hello World!')
    '\n    Save a state dict to the specified path in both distributed and single-card environment.\n\n    Note:\n        Now supports saving ``state_dict`` of Layer/Optimizer, Tensor and nested structure containing Tensor, Program.\n\n    Note:\n        Different from ``paddle.jit.save``, since the save result of ``paddle.save`` is a single file,\n        there is no need to distinguish multiple saved files by adding a suffix. The argument ``path``\n        of ``paddle.save`` will be directly used as the saved file name instead of a prefix.\n        In order to unify the saved file name format, we recommend using the paddle standard suffix:\n        1. for ``Layer.state_dict`` , recommend to use ``.pdparams`` ;\n        2. for ``Optimizer.state_dict`` , recommend to use ``.pdopt`` .\n        For specific examples, please refer to API code examples.\n\n    Args:\n        obj(Object) : The object to be saved.\n        path(str|BytesIO) : The path/buffer of the object to be saved.\n          If saved in the current directory, the input path string will be used as the file name.\n        protocol(int, optional): The protocol version of pickle module must be greater than 1 and less than 5.\n                                 Default: 4\n        **configs(dict, optional): optional keyword arguments. The following options are currently supported:\n          (1)use_binary_format(bool):\n            To be used in paddle.save. When the saved object is static graph variable, you can specify ``use_binary_for_var``.\n            If True, save the file in the c++ binary format when saving a single static graph variable; otherwise, save it in pickle format.\n            Default: False\n          (2)gather_to(int|list|tuple|None):\n            To specify which global rank to save in.Defalut is None.\n            None value means distributed saving with no gathering to a single card.\n          (3)state_type(str):\n            Value can be \'params\' or \'opt\', specifying to save parametres or optimizer state.\n          (4)max_grouped_size(str|int):\n            To limit the max size(how many bits) a object group to be transfered a time.\n            If str, the format must be as num+\'G/M/K\', for example, 3G, 2K, 10M, etc. Default is 3G.\n    Returns:\n        None\n    Examples:\n        import paddle\n        paddle.distributed.init_process_group(backend=\'nccl\')\n        paddle.distributed.fleet.init(is_collective=True)\n\n        model = build_model()\n        optimizer = build_optimizer(model)\n\n        dist_optimizer = paddle.distributed_optimizer(optimizer)\n        dist_model = paddle.distributed_optimizer(model)\n\n        # gather params to rank 0 and then save\n        paddle.incubate.distributed.utils.io.save(model.state_dict(), path="path/to/save.pdparams", gather_to=[0], state_type="params")\n\n        # save whoe params on all ranks\n        paddle.incubate.distributed.utils.io.save(model.state_dict(), path="path/to/save.pdparams", gather_to=[0,1], state_type="params")\n\n        # save optimizer state dict on rank 0\n        paddle.incubate.distributed.utils.io.save(optimizer.state_dict(), path="path/to/save.pdopt", gather=0, state_type="opt")\n\n    '
    gather_to = configs.get('gather_to', None)
    if dist.get_world_size() == 1 or gather_to is None:
        configs = _remove_not_supported_conf(configs)
        return paddle.save(state_dict, path, **configs)
    state_type = configs.get('state_type', None)
    assert isinstance(state_type, str), "must pass an arg state_type='params' or state_type='opt' to specify whether to save model state_dict or optimizer state_dict"
    assert state_type in ['params', 'opt'], "must pass an arg state_type='params' or state_type='opt'"
    if re.search(f'{state_type}$', path) is None:
        logger.warning(f'You are saving {state_type}, while the path({path} does not end with {state_type})')
    hcg = fleet.get_hybrid_communicate_group()
    assert hcg.get_model_parallel_world_size() == 1 and hcg.get_pipe_parallel_world_size() == 1, f'Only DP and Sharding is supported now. However, current MP={hcg.get_model_parallel_world_size()} , PP={hcg.get_pipe_parallel_world_size()}'
    sharding_group = hcg.get_sharding_parallel_group()
    dp_group = hcg.get_data_parallel_group()
    if state_type == 'params':
        if dp_group.nranks > 1:
            assert _same_keys(state_dict, dp_group), 'only sharding stage 1/2 and DP are supported now'
        if sharding_group.nranks > 1:
            assert _same_keys(state_dict, sharding_group), 'only sharding stage 1/2 and DP are supported now'
        configs = _remove_not_supported_conf(configs)
        return paddle.save(state_dict, path, **configs)
    if sharding_group.nranks == 1:
        configs = _remove_not_supported_conf(configs)
        return paddle.save(state_dict, path, **configs)
    if _same_keys(state_dict, sharding_group):
        return paddle.save(state_dict, path, **configs)
    assert isinstance(gather_to, (list, tuple, int))
    if isinstance(gather_to, int):
        gather_to = [gather_to]
    max_size = configs.get('max_grouped_size', '3G')
    try:
        logger.info('state_dict_keys:' + str(state_dict.keys()))
        gathered_state_dict = _gather_state_dict(state_dict, gather_to, sharding_group, max_size=max_size)
        logger.info('gathered_state_dict_keys:' + str(state_dict.keys()))
        if dist.get_rank() in gather_to:
            configs = _remove_not_supported_conf(configs)
            paddle.save(gathered_state_dict, path, **configs)
    except:
        raise RuntimeError(f'Saving failed. Follwing are some suggestions:\n    1) pass the param max_grouped_size to turn the grouped size smaller (current value of max_grouped_size is {max_size})\n    2) if sharding stage is 1, use paddle.save rather than paddle.distributed.save\n    3) Concat the developers\n')

def _state_dict_groups(state_dict, max_size):
    if False:
        for i in range(10):
            print('nop')
    '\n    Description:\n        Generator of state dict groups to transfer.the size of each group is less than max_size.\n    '
    max_tensor_size = 0
    for (k, v) in state_dict.items():
        if max_tensor_size < sys.getsizeof(v) + sys.getsizeof(k):
            max_tensor_size = sys.getsizeof(v) + sys.getsizeof(k)
    max_size = max(max_size, max_tensor_size)
    logger.debug(f'max tensor size: {max_size}')
    state_group = {}
    k_list = list(state_dict.keys())
    index = 0
    bits = 0
    while index < len(k_list):
        bsize = sys.getsizeof(state_dict[k_list[index]]) + sys.getsizeof(k_list[index])
        if bits + bsize >= max_size:
            yield state_group
            state_group = {}
            bits = 0
        state_group[k_list[index]] = state_dict[k_list[index]]
        index += 1
        bits += bsize
        if index == len(k_list) and bits > 0:
            yield state_group

def all_empty(dict_list):
    if False:
        print('Hello World!')
    '\n    Check if all items are empty\n    '
    for v in dict_list:
        if len(v) > 0:
            return False
    return True

def _parse_mem_size_to_bits(max_size):
    if False:
        for i in range(10):
            print('nop')
    '\n    Parse an integer or a mem size str to an integer\n    convert xxxG to xxx * 1024^3\n    convert xxxM to xxx * 1024^2\n    convert xxxK to xxx * 1024^1\n    '
    assert isinstance(max_size, (int, str))
    if isinstance(max_size, str):
        assert re.search('^[0-9]*[GMK]$', max_size), f"Wrong max_size 's format, the format ust be like 10K, 9M, 200G , etc, or an integer. However this is {max_size}"
        num = int(max_size[:-1])
        if max_size[-1] == 'G':
            max_size = num * 1024 ** 3
        elif max_size[-1] == 'M':
            max_size = num * 1024 ** 2
        else:
            max_size = num * 1024
    return max_size

def _gather_state_dict(state_dict, dst, group, max_size='3G'):
    if False:
        while True:
            i = 10
    "\n    Description:\n        Gather state dicts across all group ranks to dst, Depiring the same elements. including LR_Scheduler.\n    Args:\n        state_dict(dict):\n            local state dict\n        dst(int|list|tuple):\n            ranks the state dicts are gathered to\n        group(ProcessGroup):\n            group across which the state dicts are gathered\n        max_size(int|str):\n            The max limitation of the gathered tensor group size transformered a time. Default is 3G bits.\n            Each rank 's max tensor group before gathering is max_size // group.size\n    Returns:\n        Gathered state dict\n    "
    assert isinstance(dst, (list, tuple, int)), "dst' type must be one of int, list and tuple"
    if isinstance(dst, int):
        dst = [dst]
    max_size = _parse_mem_size_to_bits(max_size)
    max_size //= dist.get_world_size(group)
    logger.debug('len state_dict: len(state_dict)')
    state_dict_ = copy.copy(state_dict)
    mw = None
    has_mw = False
    has_lr = False
    if 'master_weights' in state_dict_:
        mw = state_dict_.pop('master_weights', None)
        has_mw = True
    if 'LR_Scheduler' in state_dict_:
        lr = state_dict_.pop('LR_Scheduler', None)
        has_lr = True
    output = _grouped_gather_data_dict(state_dict_, dst, group, max_size)
    if isinstance(mw, dict):
        masters = _grouped_gather_data_dict(mw, dst, group, max_size)
    else:
        assert mw is None, f'Wrong type of master weights . type: {type(mw)}'
    if has_mw:
        output['master_weights'] = masters
    if has_lr:
        output['LR_Scheduler'] = lr
    return output

def _grouped_gather_data_dict(state_data_dict, dst, group, max_size):
    if False:
        while True:
            i = 10
    "\n    Description:\n        Gather state data dict by groups.\n    Args:\n        state__data_dict(dict):\n            local dict to transfer.The state_data_dict only contains the mapping: str->paddle.Tensor\n        dst(int|list|tuple):\n            ranks the state dicts are gathered to\n        group(ProcessGroup):\n            group across which the state dicts are gathered\n        max_size(int|str):\n            The max limitation of the gathered tensor group size transformered a time. Default is 3G bits.\n            Each rank 's max tensor group before gathering is max_size // group.size\n    Returns:\n        Gatherd state_data_dict\n\n    "
    numpy_dict = {}
    logger.debug(f'len state_tict_ : {len(state_data_dict)}')
    for (k, v) in state_data_dict.items():
        try:
            numpy_dict[k] = v.numpy()
        except:
            raise TypeError(f"the object (type of {type(v)}) of '{k}' is neither tensor nor parameter")
    total = 0
    output_state = {}
    logger.info('start all gather ...')
    for state in _state_dict_groups(numpy_dict, max_size):
        s_list = []
        total += len(state)
        logger.info(f'gen to gather: {total} / {len(numpy_dict)}')
        dist.all_gather_object(s_list, state, group)
        if dist.get_rank() in dst:
            for s in s_list:
                for (k, v) in s.items():
                    logger.debug(f'gathered: {k}, {v.shape}')
                output_state.update(s)
        logger.debug(f's list size: {sum((len(s) for s in s_list))} output: {len(output_state)}')
    while True:
        s_list = []
        state = {}
        logger.debug('while True')
        dist.all_gather_object(s_list, state, group)
        if all_empty(s_list):
            break
        if dist.get_rank() in dst:
            for s in s_list:
                for (k, v) in s.items():
                    logger.debug(f'gathered: {k}, {v.shape}')
                output_state.update(s)
        logger.debug(f's list size: {sum((len(s) for s in s_list))} output: {len(output_state)}')
    logger.debug('all gathered ...')
    if dist.get_rank() in dst:
        place = paddle.CPUPlace()
        for k in output_state.keys():
            output_state[k] = paddle.to_tensor(output_state[k], place=place)
            output_state[k].name = k
        return output_state
    return {}

def _same_keys(state_dict, group):
    if False:
        while True:
            i = 10
    '\n    Check whther all keys in each dict in the group are the same.\n    Used in sharding strategy to determine whether a dict needs to be gathered.\n    '
    keys = list(state_dict.keys())
    key_list = []
    logger.info(keys)
    dist.all_gather_object(key_list, keys, group=group)
    for k in key_list:
        if not k == keys:
            return False
    return True

def _remove_not_supported_conf(configs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Remove the config values not supported by paddle.save\n    '
    __supported_by_save__ = ['use_binary_format']
    configs_ = copy.copy(configs)
    for k in configs.keys():
        if k not in __supported_by_save__:
            configs_.pop(k, None)
    return configs_