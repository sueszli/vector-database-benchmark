import paddle
from paddle.base.framework import Variable
from paddle.utils import gast, is_sequence, map_structure
from .utils import UndefinedVar, create_undefined_variable
__all__ = []

def create_undefined_var(name):
    if False:
        while True:
            i = 10
    func_code = f"{name} = _jst.UndefinedVar('{name}')"
    return gast.parse(func_code).body[0]

def create_fill_constant_node(name, value=0):
    if False:
        return 10
    func_code = f'{name} = paddle.full(shape=[1], '
    if isinstance(value, bool):
        func_code += f"dtype='bool', fill_value={value}, name='{name}')"
        return gast.parse(func_code).body[0]
    if isinstance(value, float):
        func_code += f"dtype='float64', fill_value={value}, name='{name}')"
        return gast.parse(func_code).body[0]
    if isinstance(value, int):
        func_code += f"dtype='int64', fill_value={value}, name='{name}')"
        return gast.parse(func_code).body[0]

def to_static_variable(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Translate a Python Tensor to PaddlePaddle static graph Tensor\n    '
    if isinstance(x, bool):
        return paddle.full(shape=[], dtype='bool', fill_value=x)
    if isinstance(x, float):
        return paddle.full(shape=[], dtype='float64', fill_value=x)
    if isinstance(x, int):
        return paddle.full(shape=[], dtype='int64', fill_value=x)
    if isinstance(x, UndefinedVar) or x is None:
        '\n        for early return case, we need a variable to represent None, current we use data_layer_not_check.\n        '
        return create_undefined_variable()
    if is_sequence(x):
        return map_structure(to_static_variable, x)
    return x

def create_bool_as_type(x, value=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a bool variable, which type is the same as x.\n    '
    if isinstance(x, Variable):
        return paddle.full(shape=[1], fill_value=value, dtype='bool')
    else:
        return value

def create_bool_node(name, value):
    if False:
        i = 10
        return i + 15
    '\n    Create a assign stmt for name = value .\n    '
    assert isinstance(value, bool)
    node = f'{name} = {value}'
    return gast.parse(node).body[0]