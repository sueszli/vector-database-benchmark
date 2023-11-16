class Registry:
    """A general registry object."""
    __slots__ = ['name', 'tab']

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.tab = {}

    def register(self, name, value):
        if False:
            return 10
        assert name not in self.tab, f'name "{name}" should not be registered before.'
        self.tab[name] = value

    def lookup(self, name):
        if False:
            return 10
        return self.tab.get(name)
_primop_fn = Registry('primop_fn')
_orig2prim = Registry('orig2prim')
_prim2orig = Registry('prim2orig')
_primop_jvp = Registry('primop_jvp')
_primop_transpose = Registry('primop_transpose')
_primop_position_argnames = Registry('primop_position_argnames')
_composite_ops = Registry('composite')

def lookup_fn(optype):
    if False:
        while True:
            i = 10
    return _primop_fn.lookup(optype)

def lookup_orig2prim(optype):
    if False:
        for i in range(10):
            print('nop')
    return _orig2prim.lookup(optype)

def lookup_prim2orig(optype):
    if False:
        while True:
            i = 10
    return _prim2orig.lookup(optype)

def lookup_jvp(optype):
    if False:
        while True:
            i = 10
    return _primop_jvp.lookup(optype)

def lookup_transpose(optype):
    if False:
        return 10
    return _primop_transpose.lookup(optype)

def lookup_composite(optype):
    if False:
        return 10
    return _composite_ops.lookup(optype)

def op_position_inputs(op):
    if False:
        while True:
            i = 10
    "\n    Returns the position inputs of `op` as registered with REGISTER_FN.\n\n    Args:\n        op(Operator): The op that needs to get the inputs\n\n    Returns:\n        Tensor(s): Inputs of the op\n\n    Examples:\n        .. code-block:: python\n\n            >>> from paddle.incubate.autograd.primops import _simple_binop\n            >>> from paddle.base.layer_helper import LayerHelper\n            >>> from paddle.incubate.autograd.primreg import REGISTER_FN\n\n            >>> # doctest: +SKIP('Depends on external code.')\n            >>> @REGISTER_FN('div_p', 'X', 'Y', 'Z')\n            >>> def div(x, y, out=None):\n            ...     return _simple_binop(LayerHelper('div_p', **locals()))\n\n    The registered inputs are ['X', 'Y'] for div_p and accordingly this\n    function will return inputs in the order of X then Y.\n\n    "
    args = _primop_position_argnames.lookup(op.type)
    assert args is not None, f'args of {op.type} should not be None in op_position_inputs().'
    (*input_names, _) = args
    inputs = []
    for name in input_names:
        vars = list(map(op.block.var, op.input(name)))
        assert len(vars) >= 0, f'len(vars) should be greater than or equal to 0, but len(vars)={len(vars)}.'
        if len(vars) > 1:
            inputs.append(vars)
        else:
            inputs.append(vars[0])
    return inputs

def op_position_output(op):
    if False:
        i = 10
        return i + 15
    "\n    Returns the output of `op` as registered with REGISTER_FN.\n\n    Args:\n        op(Operator): The op that needs to get the output\n\n    Returns:\n        Tensor(s): Output of the op\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +SKIP('Depends on external code.')\n            >>> from paddle.incubate.autograd.primops import _simple_binop\n            >>> from paddle.base.layer_helper import LayerHelper\n            >>> from paddle.incubate.autograd.primreg import REGISTER_FN\n\n            >>> @REGISTER_FN('div_p', 'X', 'Y', 'Z')\n            >>> def div(x, y, out=None):\n            ...     return _simple_binop(LayerHelper('div_p', **locals()))\n\n    The registered output is ['Z'] for div_p and accordingly this\n    function will return output Z.\n\n    "
    args = _primop_position_argnames.lookup(op.type)
    assert args is not None, 'args should not be None in op_position_output().'
    (*_, output_name) = args
    outvars = list(map(op.block.var, op.output(output_name)))
    assert len(outvars) >= 0, f'len(outvars) should be greater than or equal to 0, but len(outvars)={len(outvars)}.'
    if len(outvars) > 1:
        output = outvars
    else:
        output = outvars[0]
    return output

def REGISTER_FN(op_type, *position_argnames):
    if False:
        i = 10
        return i + 15
    "\n    Decorator for registering the Python function for a primitive op.\n\n    Args:\n        op_type(str): The op name\n        position_argnames(list[str]): Input and output names of the op\n\n    Returns:\n        wrapper: Inner wrapper function\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +SKIP('Depends on external code.')\n            >>> from paddle.incubate.autograd.primops import _simple_binop\n            >>> from paddle.base.layer_helper import LayerHelper\n            >>> from paddle.incubate.autograd.primreg import REGISTER_FN\n\n            >>> @REGISTER_FN('tanh_p', 'X', 'Y')\n            >>> def tanh(x, out=None):\n            ...    return _simple_unop(LayerHelper('tanh_p', **locals()))\n\n    "
    if not isinstance(op_type, str):
        raise TypeError(f'op_type must be str, but got {type(op_type)}.')
    _primop_position_argnames.register(op_type, position_argnames)

    def wrapper(f):
        if False:
            print('Hello World!')
        _primop_fn.register(op_type, f)
        return f
    return wrapper

def REGISTER_ORIG2PRIM(op_type):
    if False:
        while True:
            i = 10
    "\n    Decorator for registering the lower function for an original op into sequence of primitive ops.\n\n    Args:\n        op_type(str): The op name\n\n    Returns:\n        wrapper: Inner wrapper function\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +SKIP('Depends on external code.')\n            >>> from paddle.base.layer_helper import LayerHelper\n            >>> from paddle.incubate.autograd.utils import get_input_var_list\n            >>> from paddle.incubate.autograd import primops\n            >>> from paddle.incubate.autograd.primreg import REGISTER_ORIG2PRIM\n\n            >>> @REGISTER_ORIG2PRIM('tanh')\n            >>> def tanh_orig2prim(op):\n            ...     x, = get_input_var_list(op)\n            ...     return primops.tanh(x)\n\n    "
    if not isinstance(op_type, str):
        raise TypeError(f'op_type must be str, but got {type(op_type)}.')

    def wrapper(f):
        if False:
            i = 10
            return i + 15

        def _lower(op, *args, **kwargs):
            if False:
                while True:
                    i = 10
            assert op.type == op_type, f'op.type should be equal to op_type, but op.type is {op.type} and op_type is {op_type}'
            return f(op, *args, **kwargs)
        _orig2prim.register(op_type, _lower)
    return wrapper

def REGISTER_COMPOSITE(op_type):
    if False:
        for i in range(10):
            print('nop')
    "\n    Decorator for registering the lower function for an original op into sequence of primitive ops.\n\n    Args:\n        op_type(str): The op name\n\n    Returns:\n        wrapper: Inner wrapper function\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +SKIP('Depends on external code.')\n            >>> import paddle\n            >>> from paddle.incubate.autograd.primreg import REGISTER_COMPOSITE\n\n            >>> @REGISTER_COMPOSITE('softmax')\n            >>> def softmax_composite(x, axis):\n            ...     molecular = paddle.exp(x)\n            ...     denominator = paddle.broadcast_to(sum(molecular, axis=axis, keepdim=True), x.shape)\n            ...     res = paddle.divide(molecular, denominator)\n            ...     return res\n\n    "
    if not isinstance(op_type, str):
        raise TypeError(f'op_type must be str, but got {type(op_type)}.')

    def wrapper(f):
        if False:
            for i in range(10):
                print('nop')

        def _lower(op, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            assert op.type == op_type, f'op.type should be equal to op_type, but op.type is {op.type} and op_type is {op_type}'
            return f(*args, **kwargs)
        _composite_ops.register(op_type, _lower)
    return wrapper

def REGISTER_PRIM2ORIG(op_type):
    if False:
        while True:
            i = 10
    "\n    Decorator for registering the lower function for an primitive op into sequence of original ops.\n\n    Args:\n        op_type(str): The op name\n\n    Returns:\n        wrapper: Inner wrapper function\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +SKIP('Depends on external code.')\n            >>> import paddle\n            >>> from paddle.incubate.autograd.primreg import REGISTER_PRIM2ORIG\n            >>> from paddle.incubate.autograd.utils import get_input_var_list\n\n            >>> @REGISTER_PRIM2ORIG('tanh_p')\n            >>> def tanh_prim2orig(op):\n            ...     x, = get_input_var_list(op)\n            ...     return paddle.tanh(x)\n            ...\n    "
    if not isinstance(op_type, str):
        raise TypeError(f'op_type must be str, but got {type(op_type)}.')

    def wrapper(f):
        if False:
            i = 10
            return i + 15

        def _lower(op, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            assert op.type == op_type, f'op.type should be equal to op_type, but op.type is {op.type} and op_type is {op_type}'
            return f(op, *args, **kwargs)
        _prim2orig.register(op_type, _lower)
    return wrapper

def REGISTER_JVP(op_type):
    if False:
        i = 10
        return i + 15
    "\n    Decorator for registering the JVP function for a primitive op.\n\n    Args:\n        op_type(str): The op name\n\n    Returns:\n        wrapper: Inner wrapper function\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +SKIP('Depends on external code.')\n            >>> from paddle.incubate.autograd import primops\n            >>> from paddle.incubate.autograd.primreg import REGISTER_JVP\n\n            >>> @REGISTER_JVP('add_p')\n            >>> def add_jvp(op, x_dot, y_dot):\n            ...     return primops.add(x_dot, y_dot)\n\n    "
    if not isinstance(op_type, str):
        raise TypeError(f'op_type must be str, but got {type(op_type)}.')

    def wrapper(f):
        if False:
            print('Hello World!')

        def _jvp(op, *args, **kwargs):
            if False:
                while True:
                    i = 10
            assert op.type == op_type, f'op.type should be equal to op_type, but op.type is {op.type} and op_type is {op_type}'
            return f(op, *args, **kwargs)
        _primop_jvp.register(op_type, _jvp)
        return f
    return wrapper

def REGISTER_TRANSPOSE(op_type):
    if False:
        for i in range(10):
            print('nop')
    "\n    Decorator for registering the transpose function for a primitive op\n    that denotes a linear operation in the forward AD graph.\n\n    Args:\n        op_type(str): The op name\n\n    Returns:\n        wrapper: Inner wrapper function\n\n    Examples:\n        .. code-block:: python\n\n            >>> # doctest: +SKIP('Depends on external code.')\n            >>> from paddle.incubate.autograd.primreg import REGISTER_TRANSPOSE\n\n            >>> @REGISTER_TRANSPOSE('add_p')\n            >>> def add_transpose(op, z_bar):\n            ...     return z_bar, z_bar\n\n    "
    if not isinstance(op_type, str):
        raise TypeError(f'op_type must be str, but got {type(op_type)}.')

    def wrapper(f):
        if False:
            i = 10
            return i + 15

        def _transpose(op, dot_checker, *args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            assert op.type == op_type, f'op.type should be equal to op_type, but op.type is {op.type} and op_type is {op_type}'
            return f(op, dot_checker, *args, **kwargs)
        _primop_transpose.register(op_type, _transpose)
        return f
    return wrapper