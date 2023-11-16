from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil import Builder as mb
import numpy as np

def remove_reshape(reshape_op, block):
    if False:
        print('Hello World!')
    input_var = reshape_op.x
    input_op = input_var.op
    reshape_op.enclosing_block.replace_uses_of_var_after_op(anchor_op=input_op, old_var=reshape_op.outputs[0], new_var=input_var)
    block.remove_ops([reshape_op])
    return True

def remove_split(split_op, block):
    if False:
        while True:
            i = 10
    input_var = split_op.x
    input_op = input_var.op
    split_op.enclosing_block.replace_uses_of_var_after_op(anchor_op=input_op, old_var=split_op.outputs[0], new_var=input_var)
    block.remove_ops([split_op])
    return True

def remove_slice(slice_op, block):
    if False:
        while True:
            i = 10
    input_var = slice_op.x
    input_op = input_var.op
    slice_op.enclosing_block.replace_uses_of_var_after_op(anchor_op=input_op, old_var=slice_op.outputs[0], new_var=input_var)
    block.remove_ops([slice_op])
    return True
op_to_removal_fn = {'reshape': remove_reshape, 'split': remove_split, 'slice_by_index': remove_slice, 'slice_by_size': remove_slice}

def match_pattern(op):
    if False:
        i = 10
        return i + 15
    if op.outputs[0] in op.enclosing_block.outputs:
        return None
    if op.op_type in {'reshape', 'split', 'slice_by_index', 'slice_by_size'}:
        input_shape = op.x.sym_type
        if len(op.outputs) != 1:
            return None
        output_shape = op.outputs[0].sym_type
        if input_shape != output_shape:
            return None
        return op_to_removal_fn[op.op_type]
    return None

def noop_elimination_block(block):
    if False:
        print('Hello World!')
    for op in list(block.operations):
        for b in op.blocks:
            block_changed = True
            while block_changed:
                block_changed = noop_elimination_block(b)
        if len(op.blocks) > 0:
            continue
        remove_fn = match_pattern(op)
        if remove_fn is not None:
            with block:
                status = remove_fn(op, block)
            if status:
                return status
    return False

@register_pass(namespace='common')
def noop_elimination(prog):
    if False:
        while True:
            i = 10
    "\n    We remove reshape/slice/split if it's a no-op\n\n    Given:\n        %1 (1, 96, 128, 64, fp32) = ...\n        %2 (1, 96, 128, 64, fp32) = reshape(%1)\n        ...\n        %3 (1, 96, 128, 64, fp32) = add(%2, constant)\n        ...\n\n    Result:\n        %1 (1, 96, 128, 64, fp32) = ...\n        %3 (1, 96, 128, 64, fp32) = add(%1, constant)\n        ...\n\n    "
    for (f_name, f) in prog.functions.items():
        block_changed = True
        while block_changed:
            block_changed = noop_elimination_block(f)