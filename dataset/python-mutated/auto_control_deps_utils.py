"""Utilities for AutomaticControlDependencies."""
from tensorflow.python.framework import dtypes
from tensorflow.python.util import object_identity
READ_ONLY_RESOURCE_INPUTS_ATTR = '_read_only_resource_inputs'
RESOURCE_READ_OPS = set()
COLLECTIVE_MANAGER_IDS = '_collective_manager_ids'

def register_read_only_resource_op(op_type):
    if False:
        print('Hello World!')
    'Declares that `op_type` does not update its touched resource.'
    RESOURCE_READ_OPS.add(op_type)

def get_read_only_resource_input_indices_graph(func_graph):
    if False:
        i = 10
        return i + 15
    'Returns sorted list of read-only resource indices in func_graph.inputs.'
    result = []
    op_read_only_resource_inputs = {}
    for (input_index, t) in enumerate(func_graph.inputs):
        if t.dtype != dtypes.resource:
            continue
        read_only = True
        for op in t.consumers():
            if op in op_read_only_resource_inputs:
                if t not in op_read_only_resource_inputs[op]:
                    read_only = False
                    break
            else:
                indices = _get_read_only_resource_input_indices_op(op)
                op_read_only_resource_inputs[op] = object_identity.ObjectIdentitySet([op.inputs[i] for i in indices])
                if t not in op_read_only_resource_inputs[op]:
                    read_only = False
                    break
        if read_only:
            result.append(input_index)
    return result

def _get_read_only_resource_input_indices_op(op):
    if False:
        return 10
    'Returns sorted list of read-only resource indices in op.inputs.'
    if op.type in RESOURCE_READ_OPS:
        return [i for (i, t) in enumerate(op.inputs) if t.dtype == dtypes.resource]
    try:
        read_only_input_indices = op.get_attr(READ_ONLY_RESOURCE_INPUTS_ATTR)
    except ValueError:
        return []
    read_only_index = 0
    result = []
    for (i, t) in enumerate(op.inputs):
        if read_only_index >= len(read_only_input_indices):
            break
        if op.inputs[i].dtype != dtypes.resource:
            continue
        if read_only_index < len(read_only_input_indices) and i == read_only_input_indices[read_only_index]:
            result.append(i)
            read_only_index += 1
    return result

def get_read_write_resource_inputs(op):
    if False:
        return 10
    'Returns a tuple of resource reads, writes in op.inputs.\n\n  Args:\n    op: Operation\n\n  Returns:\n    A 2-tuple of ObjectIdentitySets, the first entry containing read-only\n    resource handles and the second containing read-write resource handles in\n    `op.inputs`.\n  '
    reads = object_identity.ObjectIdentitySet()
    writes = object_identity.ObjectIdentitySet()
    if op.type in RESOURCE_READ_OPS:
        reads.update((t for t in op.inputs if t.dtype == dtypes.resource))
        return (reads, writes)
    try:
        read_only_input_indices = op.get_attr(READ_ONLY_RESOURCE_INPUTS_ATTR)
    except ValueError:
        writes.update((t for t in op.inputs if t.dtype == dtypes.resource))
        return (reads, writes)
    read_only_index = 0
    for (i, t) in enumerate(op.inputs):
        if op.inputs[i].dtype != dtypes.resource:
            continue
        if read_only_index < len(read_only_input_indices) and i == read_only_input_indices[read_only_index]:
            reads.add(op.inputs[i])
            read_only_index += 1
        else:
            writes.add(op.inputs[i])
    return (reads, writes)

def _op_writes_to_resource(handle, op):
    if False:
        print('Hello World!')
    'Returns whether op writes to resource handle.\n\n  Args:\n    handle: Resource handle. Must be an input of `op`.\n    op: Operation.\n\n  Returns:\n    Returns False if op is a read-only op registered using\n    `register_read_only_resource_op` or if `handle` is an input at one of\n    the indices in the `READ_ONLY_RESOURCE_INPUTS_ATTR` attr of the op, True\n    otherwise.\n\n  Raises:\n    ValueError: if `handle` is not an input of `op`.\n  '
    if op.type in RESOURCE_READ_OPS:
        return False
    input_index = _input_index(op, handle)
    try:
        read_only_input_indices = op.get_attr(READ_ONLY_RESOURCE_INPUTS_ATTR)
    except ValueError:
        return True
    return input_index not in read_only_input_indices

def _input_index(op, handle):
    if False:
        for i in range(10):
            print('nop')
    'Returns the index of `handle` in `op.inputs`.\n\n  Args:\n    op: Operation.\n    handle: Resource handle.\n\n  Returns:\n    Index in `op.inputs` receiving the resource `handle`.\n\n  Raises:\n    ValueError: If handle and its replicated input are both not found in\n    `op.inputs`.\n  '
    for (i, t) in enumerate(op.inputs):
        if handle is t:
            return i
    raise ValueError(f'{handle!s} not in list of inputs for op: {op!r}')