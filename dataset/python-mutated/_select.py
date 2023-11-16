from typing import List
from nvidia.dali.data_node import DataNode as _DataNode
from nvidia.dali.auto_aug.core._augmentation import Augmentation

def split_samples_among_ops(op_range_lo: int, op_range_hi: int, ops: List[Augmentation], selected_op_idx: _DataNode, op_args, op_kwargs):
    if False:
        while True:
            i = 10
    assert op_range_lo <= op_range_hi
    if op_range_lo == op_range_hi:
        return ops[op_range_lo](*op_args, **op_kwargs)
    mid = (op_range_lo + op_range_hi) // 2
    if selected_op_idx <= mid:
        return split_samples_among_ops(op_range_lo, mid, ops, selected_op_idx, op_args, op_kwargs)
    else:
        return split_samples_among_ops(mid + 1, op_range_hi, ops, selected_op_idx, op_args, op_kwargs)

def select(ops: List[Augmentation], selected_op_idx: _DataNode, *op_args, **op_kwargs):
    if False:
        print('Hello World!')
    '\n    Applies the operator from the operators list based on the provided index as if by calling\n    `ops[selected_op_idx](**op_kwargs)`.\n\n    The `selected_op_idx` must be a batch of indices from [0, len(ops) - 1] range. The `op_kwargs`\n    can contain other data nodes, they will be split into partial batches accordingly.\n    '
    return split_samples_among_ops(0, len(ops) - 1, ops, selected_op_idx, op_args, op_kwargs)