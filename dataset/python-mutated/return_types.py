import torch
import inspect
__all__ = ['pytree_register_structseq']
return_types = torch._C._return_types

def pytree_register_structseq(cls):
    if False:
        return 10

    def structseq_flatten(structseq):
        if False:
            return 10
        return (list(structseq), None)

    def structseq_unflatten(values, context):
        if False:
            for i in range(10):
                print('nop')
        return cls(values)
    torch.utils._pytree._register_pytree_node(cls, structseq_flatten, structseq_unflatten)
for name in dir(return_types):
    if name.startswith('__'):
        continue
    _attr = getattr(return_types, name)
    globals()[name] = _attr
    if not name.startswith('_'):
        __all__.append(name)
    if inspect.isclass(_attr) and issubclass(_attr, tuple):
        pytree_register_structseq(_attr)