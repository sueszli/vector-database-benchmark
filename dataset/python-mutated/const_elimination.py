from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import numpy as np
import six
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil.passes.pass_registry import register_pass

def get_const_mode(val):
    if False:
        return 10
    if isinstance(val, six.string_types):
        return 'immediate_value'
    if isinstance(val, (np.generic, np.ndarray)):
        if val.size > 10:
            return 'file_value'
        return 'immediate_value'
    raise ValueError('val {} not recognized.'.format(val))

def const_elimination_block(block):
    if False:
        while True:
            i = 10
    for op in list(block.operations):
        if op.op_type == 'const':
            continue
        for b in op.blocks:
            const_elimination_block(b)
        all_outputs_are_const = True
        for (i, o) in enumerate(op.outputs):
            if o.val is not None:
                with block:
                    res = mb.const(val=o.val, mode=get_const_mode(o.val), before_op=op, name=o.name)
                op.enclosing_block.replace_uses_of_var_after_op(anchor_op=op, old_var=o, new_var=res)
                o.set_name(o.name + '_ignored')
            else:
                all_outputs_are_const = False
        if all_outputs_are_const:
            op.remove_from_block()

@register_pass(namespace='common')
def const_elimination(prog):
    if False:
        for i in range(10):
            print('nop')
    "\n    prog: Program\n\n    # Replace non-const ops that have const Var\n    # outputs replaced with const op. Example:\n    #\n    # Given:\n    #   %2, %3 = non_const_op(...)  # %2 is const, %3 isn't const\n    #   %4 = other_op(%2, %3)\n    #\n    # Result:\n    #   _, %3 = non_const_op(...)  # _ is the ignored output\n    #   %2_const = const(mode=m)  # %2_const name is for illustration only\n    #   %4 = other_op(%2_const, %3)\n    #\n    # where m is 'file_value' / 'immediate_value' depending on heuristics\n    # in get_const_mode.\n    "
    for (f_name, f) in prog.functions.items():
        const_elimination_block(f)