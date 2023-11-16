from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np
import logging

@register_pass(namespace='tensorflow')
def expand_tf_lstm(prog):
    if False:
        return 10
    '\n    Expand tf_lstm_block_cell to fine-grained SSA ops following:\n\n    xh = [x, h_prev]\n    [i, ci, f, o] = xh * w + b\n    f = f + forget_bias\n    if not use_peephole:\n      wci = wcf = wco = 0\n    i = sigmoid(cs_prev .* wci + i)\n    f = sigmoid(cs_prev .* wcf + f)\n    ci = tanh(ci)\n    cs = ci .* i + cs_prev .* f\n    cs = clip(cs, cell_clip)\n    o = sigmoid(cs * wco + o)\n    co = tanh(cs)\n    h = co .* o\n\n    Inputs:\n\n        prog: Program\n    '
    for (f_name, f) in prog.functions.items():
        expand_tf_lstm_helper(f)

def expand_tf_lstm_helper(block):
    if False:
        print('Hello World!')
    for op in block.operations[:]:
        for b in op.blocks:
            expand_tf_lstm_helper(b)
        if op.op_type == 'tf_lstm_block_cell':
            expand_tf_lstm_block_cell(op)
            logging.info('Expanding {} (op_type: {})'.format(op.name, op.op_type))
        if op.op_type == 'tf_lstm_block':
            (i, cs, f, o, ci, co, h) = op.outputs
            if all([len(ov.child_ops) <= 0 and len(ov.consuming_blocks) <= 0 for ov in [i, f, o, ci, co]]):
                expand_tf_lstm_block(op)
                logging.info('Expanding {} (op_type: {})'.format(op.name, op.op_type))

def _lstm_cell_builder(op, x, h_prev, cs_prev, before_op=None):
    if False:
        for i in range(10):
            print('nop')
    b = op.bias
    forget_bias = op.forget_bias.val
    xh = mb.concat(values=[x, h_prev], axis=-1, before_op=before_op)
    w = np.transpose(op.weight.val)
    icfo = mb.linear(x=xh, weight=w, bias=b, before_op=before_op)
    (i, ci, f, o) = mb.split(x=icfo, num_splits=4, axis=-1, before_op=before_op)
    if op.forget_bias.val != 0:
        f = mb.add(x=f, y=forget_bias, before_op=before_op)
    if op.use_peephole.val:
        wci = op.weight_peep_i.val
        wcf = op.weight_peep_f.val
        x = mb.mul(x=cs_prev, y=wci, before_op=before_op)
        pre_i = mb.add(x=x, y=i, before_op=before_op)
        x = mb.mul(x=cs_prev, y=wcf, before_op=before_op)
        pre_f = mb.add(x=x, y=f, before_op=before_op)
    else:
        pre_i = i
        pre_f = f
    i = mb.sigmoid(x=pre_i, before_op=before_op)
    f = mb.sigmoid(x=pre_f, before_op=before_op)
    ci = mb.tanh(x=ci, before_op=before_op)
    x = mb.mul(x=ci, y=i, before_op=before_op)
    y = mb.mul(x=cs_prev, y=f, before_op=before_op)
    cs = mb.add(x=x, y=y, before_op=before_op)
    if op.cell_clip is not None:
        clip_val = op.cell_clip.val
        cs = mb.clip(x=cs, alpha=-clip_val, beta=clip_val, before_op=before_op)
    if op.use_peephole.val:
        wco = op.weight_peep_o.val
        x = mb.mul(x=cs, y=wco, before_op=before_op)
        pre_o = mb.add(x=x, y=o, before_op=before_op)
    else:
        pre_o = o
    o = mb.sigmoid(x=pre_o, before_op=before_op)
    co = mb.tanh(x=cs, before_op=before_op)
    h = mb.mul(x=co, y=o, before_op=before_op)
    return [i, cs, f, o, ci, co, h]

def expand_tf_lstm_block_cell(op):
    if False:
        return 10
    if op.op_type != 'tf_lstm_block_cell':
        raise ValueError()
    with op.enclosing_block as block:
        x = op.x
        h_prev = op.h_prev
        cs_prev = op.c_prev
        (i, cs, f, o, ci, co, h) = _lstm_cell_builder(op, x, h_prev, cs_prev, before_op=op)
        new_outputs = [i, cs, f, o, ci, co, h]
        for (old_v, new_v) in zip(op.outputs, new_outputs):
            block.replace_uses_of_var_after_op(anchor_op=op, old_var=old_v, new_var=new_v)
        block.remove_ops([op])

def expand_tf_lstm_block(op):
    if False:
        i = 10
        return i + 15
    if op.op_type != 'tf_lstm_block':
        raise ValueError()
    with op.enclosing_block as block:
        x = op.x
        h_prev = op.h_prev
        cs_prev = op.c_prev
        x_shape = mb.shape(x=x, before_op=op)
        length = mb.slice_by_index(x=x_shape, begin=[0], end=[1], before_op=op)
        h_shape = mb.shape(x=h_prev, before_op=op)
        list_shape = mb.concat(values=[length, h_shape], axis=0, before_op=op)
        cs_list = mb.fill(shape=list_shape, before_op=op)
        h_list = mb.fill(shape=list_shape, before_op=op)
        cs_prev = mb.expand_dims(x=cs_prev, axes=[0], before_op=op)
        cs_list = mb.concat(values=[cs_prev, cs_list], axis=0, before_op=op)
        h_prev = mb.expand_dims(x=h_prev, axes=[0], before_op=op)
        h_list = mb.concat(values=[h_prev, h_list], axis=0, before_op=op)

        def cond(i, cs_list, h_list):
            if False:
                print('Hello World!')
            return mb.less(x=i, y=length)

        def body(i, cs_list, h_list):
            if False:
                return 10
            xi = mb.gather(x=x, indices=i, axis=0)
            h_prev = mb.gather(x=h_list, indices=i, axis=0)
            cs_prev = mb.gather(x=cs_list, indices=i, axis=0)
            (ig, cs, fg, og, ci, co, h) = _lstm_cell_builder(op, xi, h_prev, cs_prev)
            counter = mb.add(x=i, y=1)
            return (counter, mb.scatter(data=cs_list, indices=counter, updates=cs), mb.scatter(data=h_list, indices=counter, updates=h))
        (_, cs_list, h_list) = mb.while_loop(_cond=cond, _body=body, loop_vars=([0], cs_list, h_list), before_op=op)
        (begin, end) = ([1, 0, 0], [0, 0, 0])
        begin_mask = [False, True, True]
        end_mask = [True, True, True]
        cs = mb.slice_by_index(x=cs_list, begin=begin, end=end, begin_mask=begin_mask, end_mask=end_mask, before_op=op)
        h = mb.slice_by_index(x=h_list, begin=begin, end=end, begin_mask=begin_mask, end_mask=end_mask, before_op=op)
        new_outputs = [cs, h]
        for (old_v, new_v) in zip([ov for (index, ov) in enumerate(op.outputs) if index in [1, 6]], new_outputs):
            block.replace_uses_of_var_after_op(anchor_op=op, old_var=old_v, new_var=new_v)
        block.remove_ops([op])