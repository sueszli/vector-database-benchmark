from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

def handle_unused_inputs_func(f):
    if False:
        for i in range(10):
            print('nop')
    unused_inputs = [v for (v_name, v) in f.inputs.items() if len(v.child_ops) == 0]
    with f:
        for v in unused_inputs:
            v_tmp = mb.identity(x=v, name=v.name + '_tmp')

@register_pass(namespace='nn_backend')
def handle_unused_inputs(prog):
    if False:
        return 10
    '\n    prog: Program\n\n    # NN doesn\'t allow unused inputs. Insert an identity op to consume\n    # inputs (though its outputs are not used.). This pass must come after\n    # dead code elimination as all inserted code are "dead code". Example:\n    #\n    # Given:\n    #\n    #    main(%x: (2, 3, fp32)) {\n    #      block0() {\n    #        %shape_0_const: (2,i32)* = const(val=[4, 7])\n    #      } -> (%shape_0_const)\n    #    }\n    #\n    # (Notice that input %x is not consumed. This causes error in NN.)\n    #\n    # Result:\n    #\n    #    main(%x: (2, 3, fp32)) {\n    #      block0() {\n    #        %unused_var: (2, 3, fp32) = identity(x=%x)\n    #        %shape_0_const: (2,i32)* = const(val=[4, 7])\n    #      } -> (%shape_0_const)\n    #    }\n    '
    for (f_name, f) in prog.functions.items():
        handle_unused_inputs_func(f)