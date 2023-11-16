import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
from typing import Dict, Iterator, List, Set, Tuple
import torch
import torch.distributed.fsdp._flat_param as flat_param_file
from torch.distributed.fsdp._common_utils import _apply_to_modules, _get_module_fsdp_state, clean_tensor_name
logger = logging.getLogger(__name__)

class SimpleProfiler:

    class Type(str, Enum):
        ALL = 'all'
        ALLGATHER = 'all_gather'
        ALLGATHER_OBJ = 'all_gather_object'
        RESHARDING = 'resharding'
        H2D = 'H2D'
        D2H = 'D2H'
    results: Dict[str, float] = defaultdict(float)
    profiling: Set[str] = set()

    @classmethod
    def reset(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        cls.results.clear()
        cls.profiling.clear()

    @classmethod
    @contextmanager
    def profile(cls, profile_type: str) -> Iterator[None]:
        if False:
            return 10
        assert profile_type not in cls.profiling, f'{profile_type} is already being profiled. SimpleProfiler does not support profiling multiple instances at the same time. '
        cls.profiling.add(profile_type)
        begin = time.monotonic()
        try:
            yield
        finally:
            end = time.monotonic()
            cls.results[profile_type] += end - begin
            cls.profiling.remove(profile_type)

    @classmethod
    def dump_and_reset(cls, msg: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        logger.warning('%s %s', msg, str(cls.results))
        cls.reset()

def _get_sharded_module_tree_with_module_name_to_fqns(model: torch.nn.Module) -> Tuple[str, Dict[str, List[str]]]:
    if False:
        print('Hello World!')
    "\n    It is used for composable fully_shard() code path, it returns\n      1. sharded module tree info: each line reprents a submodule name that contats the\n    submodule's FQN and its submodule class name, if the submodule is sharded by `fully_shard`,\n    the submodule name will add a postfix with ' FULLY SHARDED'. Each increased tree\n    level adds 4 spaces before the printed name. A printed sharded module tree info for a toy model\n    is like this:\n        [CompositeModel] FULLY SHARDED\n            l1[Linear]\n            u1[UnitModule] FULLY SHARDED\n                u1.l1[Linear]\n                u1.seq[Sequential]\n                    u1.seq.0[ReLU]\n                    u1.seq.1[Linear]\n                    u1.seq.2[ReLU]\n                u1.l2[Linear]\n            u2[UnitModule] FULLY SHARDED\n                u2.l1[Linear]\n                u2.seq[Sequential]\n                    u2.seq.0[ReLU]\n                    u2.seq.1[Linear]\n                    u2.seq.2[ReLU]\n                u2.l2[Linear]\n            l2[Linear]\n      2. a dict mapping from the concated module FQN and class name to a list of its managed\n    original parameters' FQNs. An example of the dict for the above toy sharded model is like this:\n            {'[CompositeModel]': ['l1.weight', 'l1.bias', 'l2.weight', 'l2.bias'],\n             'u1[UnitModule]': ['u1.l1.weight', 'u1.l1.bias', 'u1.seq.1.weight', 'u1.seq.1.bias', 'u1.l2.weight', 'u1.l2.bias'],\n             'u2[UnitModule]': ['u2.l1.weight', 'u2.l1.bias', 'u2.seq.1.weight', 'u2.seq.1.bias', 'u2.l2.weight', 'u2.l2.bias']\n            }\n    All FQNs are prefixed starting from ``model``.\n\n    Args:\n        model (torch.nn.Module): Root module (which may or may not be passed to\n                                 composable `fully_shard()`).\n    "

    def module_fn(module, prefix, tree_level, sharded_tree_info, sharded_module_name_to_fqns):
        if False:
            while True:
                i = 10
        num_spaces = tree_level * 4
        trimed_prefix = prefix[:-1] if len(prefix) > 0 and prefix[-1] == '.' else prefix
        prefixed_module_name = trimed_prefix + '[' + module.__class__.__name__ + ']'
        printed_prefixed_module_name = ' ' * num_spaces + prefixed_module_name
        state = _get_module_fsdp_state(module)
        if state is None:
            sharded_tree_info[0] += printed_prefixed_module_name + '\n'
            return
        handle = state._fully_sharded_module_to_handle.get(module, None)
        if handle:
            sharded_tree_info[0] += printed_prefixed_module_name + ' FULLY SHARDED' + '\n'
        else:
            sharded_tree_info[0] += printed_prefixed_module_name + '\n'
        if handle:
            param = handle.flat_param
            assert isinstance(param, flat_param_file.FlatParameter)
            global_fqns = [clean_tensor_name(prefix + name) for name in param._fqns]
            if prefixed_module_name in sharded_module_name_to_fqns:
                sharded_module_name_to_fqns[prefixed_module_name].extend(global_fqns)
            else:
                sharded_module_name_to_fqns[prefixed_module_name] = global_fqns

    def return_fn(sharded_tree_info, sharded_module_name_to_fqns):
        if False:
            for i in range(10):
                print('nop')
        return (sharded_tree_info[0], sharded_module_name_to_fqns)
    sharded_tree_info: List[str] = ['']
    sharded_module_name_to_fqns: Dict[str, List[str]] = {}
    return _apply_to_modules(model, module_fn, return_fn, [key for (key, _) in model.named_parameters()], sharded_tree_info, sharded_module_name_to_fqns)