from torch.fx.proxy import Proxy
from ._compatibility import compatibility

@compatibility(is_backward_compatible=False)
def annotate(val, type):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(val, Proxy):
        if val.node.type:
            raise RuntimeError(f'Tried to annotate a value that already had a type on it! Existing type is {val.node.type} and new type is {type}. This could happen if you tried to annotate a function parameter value (in which case you should use the type slot on the function signature) or you called annotate on the same value twice')
        else:
            val.node.type = type
        return val
    else:
        return val