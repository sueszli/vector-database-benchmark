from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import types

@register_pass(namespace='tensorflow')
def backfill_make_list_elem_type(prog):
    if False:
        print('Hello World!')
    "\n    TF's TensorArrayV3 (represented as make_list in mil) doesn't necessarily\n    contain elem shape/type, which is known when write is performed. We\n    backfill elem type info to make_list\n\n    Inputs:\n\n        prog: Program\n    "
    for (f_name, f) in prog.functions.items():
        backfill_make_list_elem_type_block(f)

def backfill_make_list_elem_type_block(block):
    if False:
        for i in range(10):
            print('nop')
    for op in block.operations[:]:
        for b in op.blocks:
            backfill_make_list_elem_type_block(b)
        if op.op_type != 'tf_make_list':
            continue
        if op.outputs[0].elem_type != types.unknown:
            continue
        list_var = op.outputs[0]
        elem_type = infer_elem_type(list_var)
        if elem_type is None:
            msg = 'No list_write or list_scatter op to infer make_list ' + "'{}' element type. Block:\n{}"
            raise ValueError(msg.format(op.name, op.enclosing_block))
        with block:
            new_list = mb.make_list(init_length=op.init_length, dynamic_length=op.dynamic_length, elem_shape=elem_type.get_shape(), dtype=op.inputs['dtype'], before_op=op, name=op.name)
        block.replace_uses_of_var_after_op(anchor_op=op, old_var=op.outputs[0], new_var=new_list)
        block.remove_ops([op])

def infer_elem_type(list_var):
    if False:
        i = 10
        return i + 15
    '\n    Returns types.tensor. None if failed to infer element type.\n    Example:\n\n    Given:\n\n    main(%update: (2,fp32)) {\n      block0() {\n        %list: List[unknown] = tf_make_list(...) # unknown elem type\n        %while_loop_0:0: (i32), %while_loop_0:1: List[(2,fp32)] = while_loop(loop_vars=(...))\n          while_loop_0_body(...) {\n            %list_write_0: List[(2,fp32)] = list_write(index=..., ls=%list, value=%update)\n          } -> (%add_0, %list_write_0)\n\n        Result:\n\n        main(%update: (2,fp32)) {\n          block0() {\n        %list: List[(2,fp32)] = tf_make_list(...) # Get the elem type from list_write\n        %while_loop_0:0: (i32), %while_loop_0:1: List[(2,fp32)] = while_loop(loop_vars=(...))\n          while_loop_0_body(...) {\n            %list_write_0: List[(2,fp32)] = list_write(index=..., ls=%list, value=%update)\n          } -> (%add_0, %list_write_0)\n    '
    for o in list_var.child_ops:
        if o.op_type in ['list_write', 'list_scatter']:
            return o.outputs[0].elem_type
        if o.op_type == 'while_loop':
            idx = list(o.loop_vars).index(list_var)
            block = o.blocks[0]
            block_var = block.inputs[idx]
            elem_type = infer_elem_type(block_var)
            if elem_type is not None:
                return elem_type
    return None