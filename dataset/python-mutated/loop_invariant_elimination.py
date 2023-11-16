from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import numpy as np
import six
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

def detect_loop_invariants(while_op):
    if False:
        for i in range(10):
            print('nop')
    block = while_op.blocks[0]
    loop_invariant_ids = []
    for (i, vx_in) in enumerate(block.inputs):
        vx_out = block.outputs[i + 1]
        return_input_as_output = vx_in == vx_out
        output_from_outside_of_block = vx_out in block._visible_vars_from_enclosing_block()
        if return_input_as_output or output_from_outside_of_block:
            loop_invariant_ids.append(i)
    return loop_invariant_ids

def loop_invariant_elimination_block(block):
    if False:
        for i in range(10):
            print('nop')
    output_rename = []
    for op in list(block.operations):
        for b in op.blocks:
            loop_invariant_elimination_block(b)
        if op.op_type != 'while_loop':
            continue
        loop_invariant_ids = detect_loop_invariants(op)
        for i in loop_invariant_ids:
            output_rename.append((op.loop_vars[i], op.outputs[i], op))
        if len(loop_invariant_ids) > 0:
            op.name = op.name + '_renamed'
    for (v_src, v_tgt, op) in output_rename:
        if v_tgt in block.outputs:
            with block:
                res = mb.identity(x=v_src, before_op=op, name=v_tgt.name)
                op.enclosing_block.replace_uses_of_var_after_op(anchor_op=op, old_var=v_tgt, new_var=res)
    for op in list(block.operations):
        if op.op_type != 'while_loop':
            continue
        block = op.blocks[0]
        loop_invariant_ids = detect_loop_invariants(op)
        loop_variant_vars = []
        for i in loop_invariant_ids:
            block.replace_uses_of_var_after_op(anchor_op=None, old_var=block.inputs[i], new_var=op.loop_vars[i])
        block.remove_inputs([block.inputs[i] for i in loop_invariant_ids])
        for i in loop_invariant_ids:
            op.enclosing_block.replace_uses_of_var_after_op(anchor_op=op, old_var=op.outputs[i], new_var=op.loop_vars[i])
        for i in loop_invariant_ids:
            op.loop_vars[i].remove_child_op(op)
        op.loop_vars = tuple((v for (i, v) in enumerate(op.loop_vars) if i not in loop_invariant_ids))
        op._input_vars['loop_vars'] = op.loop_vars
        block.set_outputs([block.outputs[0]] + [v for (i, v) in enumerate(block.outputs[1:]) if i not in loop_invariant_ids])
        op._output_vars = [v for (i, v) in enumerate(op._output_vars) if i not in loop_invariant_ids]
        op.enclosing_block.validate()

@register_pass(namespace='common')
def loop_invariant_elimination(prog):
    if False:
        return 10
    '\n    prog: Program\n\n    # When a block does not modify a block input var, eliminate that block\n    # input var and use the corresponding var in the outer scope. Example:\n    #\n    # Given:\n    #    main(%a: (1, 2, fp32),\n    #         %b: (1, 2, fp32)) {\n    #      block0() {\n    #        %loop:0: (1, 2, fp32), %loop:1: (1, 2, fp32) =     #        while_loop(loop_vars=(%a, %b))\n    #          loop_cond(%a.x, %b.x) {\n    #            %cond_var: (bool) = some_op(x=%a.x, y=%b.x)\n    #          } -> (%cond_var)\n    #          loop_body(%a.x, %b.x) {\n    #            %add_0: (1, 2, fp32) = add(x=%a.x, y=%b.x)\n    #          } -> (%add_0, %b.x)\n    #      } -> (%loop:0, %loop:1)\n    #    }\n    #\n    # (Notice that %b.x is constant through while loop iterates)\n    #\n    # Result:\n    #    main(%a: (1, 2, fp32),\n    #         %b: (1, 2, fp32)) {\n    #      block0() {\n    #        %loop:1: (1, 2, fp32) = identity(x=%b)\n    #        %loop:0: (1, 2, fp32) =     #        while_loop(loop_vars=(%a))\n    #          loop_cond(%a.x) {\n    #            %cond_var: (bool) = some_op(x=%a.x, y=%b)\n    #          } -> (%cond_var)\n    #          loop_body(%a.x) {\n    #            %add_0: (1, 2, fp32) = add(x=%a.x, y=%b)\n    #          } -> (%add_0)\n    #      } -> (%loop:0, %loop:1)\n    #    }\n    #\n    # where we eliminate loop invariant %b.x from while_loop, which returns 1\n    # instead of 2 outputs. We also preserve the return var names with\n    # identity.\n    '
    for (f_name, f) in prog.functions.items():
        loop_invariant_elimination_block(f)