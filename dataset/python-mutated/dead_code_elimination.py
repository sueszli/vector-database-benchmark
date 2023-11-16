from coremltools.converters.mil.mil.passes.pass_registry import register_pass
import logging

def dead_code_elimination_block(block):
    if False:
        print('Hello World!')
    used_vars = set()
    ops_to_remove = list()
    used_vars.update(block.outputs)
    for op in reversed(block.operations):
        if not set(op.outputs).intersection(used_vars):
            ops_to_remove.append(op)
            continue
        for (_, input_var) in op.inputs.items():
            if isinstance(input_var, (tuple, list)):
                used_vars.update(list(input_var))
            else:
                used_vars.update([input_var])
        for b in op.blocks:
            used_in_block = dead_code_elimination_block(b)
            used_vars.update(used_in_block)
    for op in ops_to_remove:
        logging.info('Removing op "{}" (type: {})'.format(op.name, op.op_type))
        op.remove_from_block()
    return used_vars

@register_pass(namespace='common')
def dead_code_elimination(program):
    if False:
        while True:
            i = 10
    "\n    Eliminate unused ops in program.\n\n    Parameters\n    ----------\n    program: Program SSA Program before graph pass\n\n    Returns\n    -------\n    program: Program SSA Program after graph pass\n\n    Example\n    -------\n\n        Given:\n        main(%x: (2, 4, fp32)) {\n          block0() {\n            %const_2: (4, 2, fp32)* = const(val=[...])\n            %const_3: (4, fp32)* = const(val=[...])\n            %tx_0: (bool)* = const(val=False)\n            %ty_0: (bool)* = const(val=False)\n            %matmul_0: (2, 2, fp32) = matmul(x=%x, y=%const_2, transpose_x=%tx_0, transpose_y=%ty_0)\n            %linear_0: (2, 4, fp32) = linear(x=%x, weight=%const_2, bias=%const_3)\n          } -> (%linear_0)\n        }\n\n        Result:\n        main(%x: (2, 4, fp32)) {\n          block0() {\n            %const_2: (4, 2, fp32)* = const(val=[...])\n            %const_3: (4, fp32)* = const(val=[...])\n            %linear_0: (2, 4, fp32) = linear(x=%x, weight=%const_2, bias=%const_3)\n          } -> (%linear_0)\n        }\n\n    Ops whose outputs are not contributed to final outputs will be deleted.\n    In this example, %matmul_0 is an op that's not used in the computation,\n    this op and its input ops (%tx_0 and %ty_0) are eliminated in this pass.\n    "
    for (name, f) in program.functions.items():
        dead_code_elimination_block(f)