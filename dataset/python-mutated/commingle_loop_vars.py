from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

def commingle_loop_vars_block(block):
    if False:
        while True:
            i = 10
    for op in list(block.operations):
        for b in op.blocks:
            commingle_loop_vars_block(b)
        if op.op_type != 'while_loop':
            continue
        block = op.blocks[0]
        for (v_out, vx_in) in zip(op.outputs, block.inputs):
            block.replace_uses_of_var_after_op(anchor_op=None, old_var=vx_in, new_var=v_out, no_check_var_visibility=True)
        block._block_inputs = op.outputs

@register_pass(namespace='nn_backend')
def commingle_loop_vars(prog):
    if False:
        print('Hello World!')
    '\n    prog: Program\n\n    # NN backend expects output vars as loop vars. Example:\n    #\n    # Given:\n    #    main(%a: (1, 2, fp32),\n    #         %b: (1, 2, fp32)) {\n    #      block0() {\n    #        %loop:0: (1, 2, fp32), %loop:1: (1, 2, fp32) =     #        while_loop(loop_vars=(%a, %b))\n    #          loop_cond(%a.x, %b.x) {\n    #            %cond_var: (bool) = some_op(x=%a.x, y=%b.x)\n    #          } -> (%cond_var)\n    #          loop_body(%a.x, %b.x) {\n    #            %add_0: (1, 2, fp32) = add(x=%a.x, y=%b.x)\n    #          } -> (%add_0, %b.x)\n    #      } -> (%loop:0, %loop:1)\n    #    }\n    #\n    # Result:\n    #    main(%a: (1, 2, fp32),\n    #         %b: (1, 2, fp32)) {\n    #      block0() {\n    #        %loop:0: (1, 2, fp32), %loop:1: (1, 2, fp32) =     #        while_loop(loop_vars=(%a, %b))\n    #          loop_cond(%loop:0, %loop:1) {\n    #            %cond_var: (bool) = some_op(x=%loop:0, y=%loop:1)\n    #          } -> (%cond_var)\n    #          loop_body(%loop:0, %loop:1) {\n    #            %add_0: (1, 2, fp32) = add(x=%loop:0, y=%loop:1)\n    #          } -> (%add_0, %loop:1)\n    #      } -> (%loop:0, %loop:1)\n    #    }\n    #\n    # Comment: The resulting program is no longer SSA (multiple assignments on\n    # %loop:0).\n    '
    for (f_name, f) in prog.functions.items():
        commingle_loop_vars_block(f)