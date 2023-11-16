import functools
import importlib
from .utils import hashable
from .variables import TorchCtxManagerClassVariable
"\nMap of torch objects to their tracing rules (Dynamo variables).\n* TorchVariable: The functions should be put into the FX graph or can be constant folded. E.g.,\n  - torch.add: should be put into the FX graph.\n  - torch.is_floating_point: constant folded.\n* TorchCtxManagerClassVariable: The context manager classes are supported by Dynamo. E.g., torch.no_grad\n* SkipFilesVariable: The objects should be skipped from tracing.\n* UserFunctionVariable: The functions should be inlined.\n\nWe explicitly list torch objects which should be wrapped as TorchCtxManagerClassVariable.\nThe initial list comes from the heuristic in test/dynamo/test_trace_rules.py:generate_allow_list.\n\nFor developers: If you add/remove a torch level API, it may trigger failures from\ntest/dynamo/test_trace_rules.py:test_torch_name_rule_map. To fix the failures:\nIf you are adding a new torch level API or Dynamo implementation:\n* Add the name with TorchCtxManagerClassVariable to this map\n  if you are adding Dynamo implementation for that context manager.\n* Remove the object name from test/dynamo/test_trace_rules.ignored_torch_name_rule_set if it's there.\n\nIf you are removing an existing torch level API:\n* Remove the entry represented the API from this map or test/dynamo/test_trace_rules.ignored_torch_name_rule_set\n  depends on where it is.\n\nTODO: Add torch object names mapping to TorchVariable for in graph and constant fold functions.\nTODO: We would consolidate the skipfiles.check rules into trace_rules.lookup later.\nTODO: We would support explictly list objects treated as skip/inline after the skipfiles.check\nand trace_rules.lookup consolidation is done. Then the explicit listing of skip/inline objects have\na higher priority, which can be used to override the skipfiles.check rules in some cases.\n"
torch_name_rule_map = {'torch._C.DisableTorchFunctionSubclass': TorchCtxManagerClassVariable, 'torch.amp.autocast_mode.autocast': TorchCtxManagerClassVariable, 'torch.autograd.grad_mode.enable_grad': TorchCtxManagerClassVariable, 'torch.autograd.grad_mode.inference_mode': TorchCtxManagerClassVariable, 'torch.autograd.grad_mode.no_grad': TorchCtxManagerClassVariable, 'torch.autograd.grad_mode.set_grad_enabled': TorchCtxManagerClassVariable, 'torch.autograd.profiler.profile': TorchCtxManagerClassVariable, 'torch.autograd.profiler.record_function': TorchCtxManagerClassVariable, 'torch.cpu.amp.autocast_mode.autocast': TorchCtxManagerClassVariable, 'torch.cuda.amp.autocast_mode.autocast': TorchCtxManagerClassVariable, 'torch.profiler.profiler.profile': TorchCtxManagerClassVariable}

@functools.lru_cache(None)
def get_torch_obj_rule_map():
    if False:
        while True:
            i = 10
    d = dict()
    for (k, v) in torch_name_rule_map.items():
        obj = load_object(k)
        assert obj not in d
        d[obj] = v
    return d

def load_object(name):
    if False:
        for i in range(10):
            print('nop')
    (mod_name, obj_name) = name.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, obj_name)
    return obj

def lookup(obj):
    if False:
        i = 10
        return i + 15
    if not hashable(obj):
        return None
    rule = get_torch_obj_rule_map().get(obj, None)
    return rule