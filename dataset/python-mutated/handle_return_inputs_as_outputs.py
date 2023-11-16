from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

def handle_return_inputs_as_outputs_func(f):
    if False:
        while True:
            i = 10
    returned_inputs = []
    for (v_name, v) in f.inputs.items():
        if v not in f.outputs:
            continue
        returned_inputs.append(v)
    with f:
        for v in returned_inputs:
            v_tmp = mb.identity(x=v, name=v.name + '_tmp')
            res = mb.identity(x=v_tmp, name=v.name)
            res.op.enclosing_block.replace_uses_of_var_after_op(anchor_op=res.op, old_var=v, new_var=res)

@register_pass(namespace='nn_backend')
def handle_return_inputs_as_outputs(prog):
    if False:
        i = 10
        return i + 15
    '\n    prog: Program\n\n    # NN cannot handle returning input as output. Insert an identity op for\n    # those cases. Example:\n    #\n    # Given:\n    #    main(%a: (1, 2, fp32),\n    #         %b: (1, 2, fp32)) {\n    #      block0() {\n    #        %mul_0_y_0: (i32)* = const(val=2)\n    #        %mul_0: (1, 2, fp64) = mul(x=%a, y=%mul_0_y_0)\n    #      } -> (%mul_0, %b)\n    #    }\n    #\n    # (Notice that %b is returned from input. This causes error in NN)\n    #\n    # Result:\n    #    main(%a: (1, 2, fp32),\n    #         %b: (1, 2, fp32)) {\n    #      block0() {\n    #        %mul_0_y_0: (i32)* = const(val=2)\n    #        %mul_0: (1, 2, fp64) = mul(x=%a, y=%mul_0_y_0)\n    #        %b_tmp: (1, 2, fp32) = identity(x=%b)\n    #        %b: (1, 2, fp32) = identity(x=%b_tmp)\n    #      } -> (%mul_0, %b)\n    #    }\n    #\n    # where identity is applied twice since NN layer cannot have\n    # input name == output name\n    '
    for (f_name, f) in prog.functions.items():
        handle_return_inputs_as_outputs_func(f)