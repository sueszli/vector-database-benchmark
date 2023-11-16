from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
import logging

def remove_vacuous_cond_block(block):
    if False:
        print('Hello World!')
    num_changes = 0
    for op in list(block.operations):
        for b in op.blocks:
            num_changes += remove_vacuous_cond_block(b)
        if op.op_type != 'cond':
            continue
        then_ops = op.blocks[0].operations
        else_ops = op.blocks[1].operations
        if len(then_ops) > 1 or len(else_ops) > 1:
            continue
        if len(then_ops) == 0 and len(else_ops) == 0:
            if op.pred.op.op_type not in {'less_equal', 'greater_equal'}:
                continue
            pred_x = op.pred.op.x.op
            pred_y = op.pred.op.y.op
            if pred_x is None and pred_y is None:
                continue
            if op.pred.op.op_type == 'less_equal':
                if pred_x.op_type != 'list_length':
                    continue
                new_var = pred_x.ls
            else:
                if pred_y.op_type != 'list_length':
                    continue
                new_var = pred_y.ls
            with block:
                op.enclosing_block.replace_uses_of_var_after_op(anchor_op=op, old_var=op.outputs[0], new_var=new_var)
                block.remove_ops([op])
            num_changes += 1
        if len(then_ops) == 1 and len(then_ops) == 1:
            if then_ops[0].op_type != 'identity' or else_ops[0].op_type != 'identity':
                continue
            if then_ops[0].x != else_ops[0].x:
                continue
            with block:
                new_var = mb.identity(x=then_ops[0].x, before_op=op, name=op.name)
                op.enclosing_block.replace_uses_of_var_after_op(anchor_op=op, old_var=op.outputs[0], new_var=new_var)
                block.remove_ops([op])
            num_changes += 1
    return num_changes

@register_pass(namespace='tensorflow2')
def remove_vacuous_cond(prog):
    if False:
        for i in range(10):
            print('nop')
    '\n    Remove cond op and it\'s sub-graphs that produces identity on both then and\n    else branch. One example use case is the TensorListReverse op, in Core ML,\n    we dynamically resize in write operations, and thus, both branches of the\n    cond op will be a skip (identity) op.\n\n    Given:\n\n        main(%a: (1, bool),\n         %b: (2, 3, fp32)) {\n          block0() {\n            %squeeze_0: (bool) = squeeze(x=%a, name="squeeze_0")\n            %cond_0: (2, 3, fp32) = cond(pred=%squeeze_0, name="cond_0")\n              cond_0_true() {\n                %identity_0: (2, 3, fp32) = identity(x=%b, name="identity_0")\n              } -> (%identity_0)\n              cond_0_false() {\n                %identity_1: (2, 3, fp32) = identity(x=%b, name="identity_1")\n              } -> (%identity_1)\n          } -> (%cond_0)\n        }\n\n    Result:\n\n        main(%a: (1, bool),\n             %b: (2, 3, fp32)) {\n          block0() {\n            %squeeze_0: (bool) = squeeze(x=%a, name="squeeze_0")\n            %cond_0: (2, 3, fp32) = identity(x=%b, name="cond_0")\n          } -> (%cond_0)\n        }\n    '
    for (f_name, f) in prog.functions.items():
        num_changes = remove_vacuous_cond_block(f)
        msg = "remove_vacuous_cond: changed {} ops in function '{}'"
        logging.info(msg.format(num_changes, f_name))