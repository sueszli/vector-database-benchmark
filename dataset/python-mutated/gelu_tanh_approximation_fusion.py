from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil import Builder as mb
import numpy as np

def _check_var_scalar_value(x, val, tol=0.001):
    if False:
        for i in range(10):
            print('nop')
    '\n    :param x: var\n    :param val: a scalar value\n    :return: True if the value of var is equal to val otherwise return False\n    '
    if x.val is None:
        return False
    if not isinstance(x.val, (np.ndarray, np.generic)):
        return False
    if isinstance(x.val, np.ndarray):
        if x.val.size != 1:
            return False
        x_val = x.val[:][0]
    else:
        x_val = x.val
    if abs(x_val - val) < tol:
        return True
    return False

def _check_child_op_type(op, child_op_type):
    if False:
        i = 10
        return i + 15
    '\n    :param op: operation\n    :param child_op_type: str\n    :return: Return True if op has 1 child and type of that child matches child_op_type\n    '
    if len(op.outputs) != 1:
        return False
    child_ops = list(op.outputs[0].child_ops)
    if len(child_ops) != 1:
        return False
    if child_ops[0].op_type == child_op_type:
        return True
    return False

def try_to_transform(pow_op, block):
    if False:
        print('Hello World!')
    all_ops = [pow_op]
    root_var = pow_op.x
    if len(list(root_var.child_ops)) != 3:
        return False
    if not _check_child_op_type(pow_op, 'mul'):
        return False
    mul_op1 = list(pow_op.outputs[0].child_ops)[0]
    if not (mul_op1.x == pow_op.outputs[0] and _check_var_scalar_value(mul_op1.y, 0.044715) or (mul_op1.y == pow_op.outputs[0] and _check_var_scalar_value(mul_op1.x, 0.044715))):
        return False
    all_ops.append(mul_op1)
    if not _check_child_op_type(mul_op1, 'add'):
        return False
    add_op1 = list(mul_op1.outputs[0].child_ops)[0]
    if not (add_op1.x == mul_op1.outputs[0] and add_op1.y == root_var or (add_op1.y == mul_op1.outputs[0] and add_op1.x == root_var)):
        return False
    all_ops.append(add_op1)
    if not _check_child_op_type(add_op1, 'mul'):
        return False
    mul_op2 = list(add_op1.outputs[0].child_ops)[0]
    if not (mul_op2.x == add_op1.outputs[0] and _check_var_scalar_value(mul_op2.y, 0.79788) or (mul_op2.y == add_op1.outputs[0] and _check_var_scalar_value(mul_op2.x, 0.79788))):
        return False
    all_ops.append(mul_op2)
    if not _check_child_op_type(mul_op2, 'tanh'):
        return False
    tanh_op = list(mul_op2.outputs[0].child_ops)[0]
    all_ops.append(tanh_op)
    if not _check_child_op_type(tanh_op, 'add'):
        return False
    add_op2 = list(tanh_op.outputs[0].child_ops)[0]
    if not (add_op2.x == tanh_op.outputs[0] and _check_var_scalar_value(add_op2.y, 1) or (add_op2.y == tanh_op.outputs[0] and _check_var_scalar_value(add_op2.x, 1))):
        return False
    all_ops.append(add_op2)
    if not _check_child_op_type(add_op2, 'mul'):
        return False
    mul_op3 = list(add_op2.outputs[0].child_ops)[0]
    if not (mul_op3.x == add_op2.outputs[0] and _check_var_scalar_value(mul_op3.y, 0.5) or (mul_op3.y == add_op2.outputs[0] and _check_var_scalar_value(mul_op3.x, 0.5))):
        return False
    all_ops.append(mul_op3)
    if not _check_child_op_type(mul_op3, 'mul'):
        return False
    mul_op4 = list(mul_op3.outputs[0].child_ops)[0]
    if not (mul_op4.x == mul_op3.outputs[0] and mul_op4.y == root_var or (mul_op4.y == mul_op3.outputs[0] and mul_op4.x == root_var)):
        return False
    all_ops.append(mul_op4)
    for (i, op) in enumerate(all_ops):
        if i == len(all_ops) - 1:
            continue
        for out in op.outputs:
            if out in block.outputs:
                return False
    out_name = mul_op4.outputs[0].name
    x = mb.gelu(x=root_var, mode='TANH_APPROXIMATION', name=out_name, before_op=pow_op)
    mul_op4.enclosing_block.replace_uses_of_var_after_op(anchor_op=mul_op4, old_var=mul_op4.outputs[0], new_var=x)
    block.remove_ops(all_ops)
    return True

def fuse_gelu_tanh_block(block):
    if False:
        print('Hello World!')
    fusion_status = False
    for op in list(block.operations):
        for b in op.blocks:
            block_changed = True
            while block_changed:
                block_changed = fuse_gelu_tanh_block(b)
        if len(op.blocks) > 0:
            continue
        if op.op_type == 'pow':
            if _check_var_scalar_value(op.y, 3):
                with block:
                    fusion_status = try_to_transform(op, block)
                if fusion_status:
                    return fusion_status
    return fusion_status

@register_pass(namespace='common')
def fuse_gelu_tanh_approximation(prog):
    if False:
        return 10
    '\n    Identify the pattern that corresponds to the tanh approximate version of gelu, and replace it with a single\n    gelu layer with mode=TANH_APPROXIMATION\n\n    y = ( tanh((.0447)x^3 + x ) * (sqrt(2/pi)) + 1 ) * 0.5 * x\n\n    [...] -----> pow (3) ----> mul (.044715) ---> add -----> mul (sqrt(2/pi)) ---> tanh ----> add (1) ----> mul (0.5) -----> mul ---> [...]\n      |                                            ^                                                                          ^\n      |                                            |                                                                          |\n      |------------------------------------------------------------------------------------------------------------------------\n\n\n    '
    for (f_name, f) in prog.functions.items():
        block_changed = True
        while block_changed:
            block_changed = fuse_gelu_tanh_block(f)