from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np
import logging

@register_pass(namespace='tensorflow')
def tf_lstm_to_core_lstm(prog):
    if False:
        i = 10
        return i + 15
    '\n    Try to map TF dialect ops `tf_lstm_block` and `tf_lstm_block_cell` to\n    `lstm` in the core op set if compatible. They are compatible if all of the\n    followings are satisfied:\n\n    - If tf_lstm_block: only h output is consumed. tf_lstm_block has 7\n      sequence outputs: [i, cs, f, o, ci, co, h]. Each of them (e.g., i) has\n      shape [seq_len, batch, hidden_dim] (see tf_lstm_block op doc string).\n      core lstm only supports sequence output for hidden state h, and thus if\n      any outputs other than `h` is consumed, we cannot convert to lstm in the\n      core op set.\n\n    - If tf_lstm_block_cell: only cs, h output (outputs[1], outputs[6])\n      are consumed. Similar to above.\n\n    - batch size == 1 (due to bugs in core lstm backend impl rdar://62475041)\n\n    Inputs:\n\n        prog: Program\n    '
    for (f_name, f) in prog.functions.items():
        tf_lstm_to_core_lstm_block(f)

def tf_lstm_to_core_lstm_block(block):
    if False:
        i = 10
        return i + 15
    for op in block.operations[:]:
        for b in op.blocks:
            tf_lstm_to_core_lstm_block(b)
        if op.op_type in ['tf_lstm_block_cell', 'tf_lstm_block']:
            if try_replace_with_core_lstm(op):
                logging.info('Successfully map {} to lstm'.format(op.op_type))
            else:
                logging.info('Unable to map {} to lstm'.format(op.op_type))

def try_replace_with_core_lstm(op):
    if False:
        return 10
    "\n    Inputs:\n\n    op (Operation): op.op_type must be 'tf_lstm_block_cell' or `tf_lstm_block`\n\n    Returns:\n\n    True if op can be represented by mb.lstm op in SSA. False otherwise\n    "
    if op.op_type == 'tf_lstm_block_cell':
        batch = op.x.shape[0]
    else:
        batch = op.x.shape[1]
    if op.use_peephole.val:
        return False
    if op.cell_clip is not None:
        return False
    (i, cs, f, o, ci, co, h) = op.outputs
    if op.op_type == 'tf_lstm_block_cell':
        unsupported_outputs = [i, f, o, ci, co]
        for ov in unsupported_outputs:
            if len(ov.child_ops) > 0 or len(ov.consuming_blocks) > 0:
                return False
    else:
        unsupported_outputs = [i, cs, f, o, ci, co]
        for ov in unsupported_outputs:
            if len(ov.child_ops) > 0 or len(ov.consuming_blocks) > 0:
                return False
    hidden_dim = op.c_prev.shape[1]
    mb_peep = None
    if op.use_peephole.val:
        mb_peep = np.stack([op.weight_peep_i.val, op.weight_peep_f.val, op.weight_peep_o.val])
    tf_w = op.weight.val
    (tf_w_i, tf_w_c, tf_w_f, tf_w_o) = np.split(tf_w, 4, axis=1)
    w = np.concatenate([tf_w_i, tf_w_f, tf_w_o, tf_w_c], axis=1)
    tf_b = op.bias.val
    (tf_b_i, tf_b_c, tf_b_f, tf_b_o) = np.split(tf_b, 4, axis=0)
    tf_b_f += op.forget_bias.val
    bias = np.concatenate([tf_b_i, tf_b_f, tf_b_o, tf_b_c], axis=0)
    bias = np.stack([np.zeros_like(bias), bias])
    cell_clip = None if op.cell_clip is None else op.cell_clip.val
    output_sequence = op.op_type == 'tf_lstm_block'
    block = op.enclosing_block
    with block:
        if op.op_type == 'tf_lstm_block_cell':
            x = mb.expand_dims(x=op.x, axes=[0], before_op=op)
        else:
            x = op.x
        (new_h_all, new_h, new_cs) = mb.lstm(x=x, initial_c=op.c_prev, initial_h=op.h_prev, weight=w, bias=bias, activations=('sigmoid', 'tanh', 'tanh'), peephole=mb_peep, clip=cell_clip, output_sequence=output_sequence, name=op.name, before_op=op)
    if op.op_type == 'tf_lstm_block_cell':
        block.replace_uses_of_var_after_op(anchor_op=op, old_var=cs, new_var=new_cs)
        block.replace_uses_of_var_after_op(anchor_op=op, old_var=h, new_var=new_h)
    else:
        block.replace_uses_of_var_after_op(anchor_op=op, old_var=h, new_var=new_h_all)
    block.remove_ops([op])
    return True