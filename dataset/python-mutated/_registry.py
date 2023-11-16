from nvidia.dali import backend as _b
_cpu_ops = set({})
_gpu_ops = set({})
_mixed_ops = set({})

def cpu_ops():
    if False:
        i = 10
        return i + 15
    'Get the set of the names of all registered CPU operators'
    return _cpu_ops

def gpu_ops():
    if False:
        for i in range(10):
            print('nop')
    'Get the set of the names of all registered GPU operators'
    return _gpu_ops

def mixed_ops():
    if False:
        print('Hello World!')
    'Get the set of the names of all registered Mixed operators'
    return _mixed_ops

def _all_registered_ops():
    if False:
        while True:
            i = 10
    'Return the set of the names of all registered operators'
    return _cpu_ops.union(_gpu_ops).union(_mixed_ops)

def register_cpu_op(name):
    if False:
        while True:
            i = 10
    'Add new CPU op name to the registry.'
    global _cpu_ops
    _cpu_ops = _cpu_ops.union({name})

def register_gpu_op(name):
    if False:
        while True:
            i = 10
    'Add new GPU op name to the registry'
    global _gpu_ops
    _gpu_ops = _gpu_ops.union({name})

def _discover_ops():
    if False:
        i = 10
        return i + 15
    'Query the backend for all registered operator names, update the Python-side registry of\n    operator names.'
    global _cpu_ops
    global _gpu_ops
    global _mixed_ops
    _cpu_ops = _cpu_ops.union(set(_b.RegisteredCPUOps()))
    _gpu_ops = _gpu_ops.union(set(_b.RegisteredGPUOps()))
    _mixed_ops = _mixed_ops.union(set(_b.RegisteredMixedOps()))