import torch
from torch._ops import OpOverload, OpOverloadPacket

def _register_decomposition(op: OpOverload, graph: torch._C.Graph):
    if False:
        for i in range(10):
            print('nop')
    assert not isinstance(op, OpOverloadPacket), f'Must pass specific op overload, not overload packet, found {op}'
    assert isinstance(op, OpOverload)
    torch._C._jit_register_decomposition_for_schema(op._schema, graph)