import torch._subclasses

def is_builtin(op):
    if False:
        return 10
    return op.namespace in ('aten', 'prims', 'prim')

def fake_check(op, args, kwargs):
    if False:
        while True:
            i = 10
    with torch._subclasses.CrossRefFakeMode(ignore_op_fn=is_builtin):
        op(*args, **kwargs)