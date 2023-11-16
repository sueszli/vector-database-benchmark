import inspect
import chainer

class WrappedFunctionNode(chainer.FunctionNode):
    """Wrap the target function and operate as ``FunctionNode``

    Arguments:
        name (str): name of the function node
        func (func): the target function
        args (list): args for the function
        kwargs (dict): kwargs for the function
        arg_vars (list): list of `chainer.Variable`s in `args` and `kwargs`
        attributes (list): parameters to be set node's attributes
    """

    def __init__(self, name, func, args, kwargs, arg_vars, attributes=None):
        if False:
            while True:
                i = 10
        self.custom_function_node_name = name
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.arg_vars = arg_vars
        self.internal_results = None
        if attributes is not None:
            for (k, v) in attributes.items():
                setattr(self, k, v)

    def forward(self, xs):
        if False:
            print('Hello World!')
        assert len(xs) == len(self.arg_vars)
        self.xs = xs
        results = self.func(*self.args, **self.kwargs)
        (self.skeleton, flattened_results) = self._flatten_return_value(results)
        dummy_results = tuple((_unwrap_var(ret) for ret in flattened_results))
        if all([_is_var(ret) for ret in flattened_results]):
            self.internal_results = flattened_results
        if not chainer.is_arrays_compatible(dummy_results):
            raise ValueError("returned values from the function wrapped by 'as_funcnode' must consist only array, function name: {}".format(self.custom_function_node_name))
        return dummy_results

    def backward(self, target_input_indexes, grad_outputs):
        if False:
            print('Hello World!')
        if self.internal_results is None:
            raise ValueError('the target function does not support backward, propagation is failed')
        grad_inputs = chainer.grad(self.internal_results, self.arg_vars, grad_outputs=grad_outputs)
        assert len(self.arg_vars) == len(grad_inputs)
        return tuple((grad_input if i in target_input_indexes else None for (i, grad_input) in enumerate(grad_inputs)))

    def _flatten_return_value(self, x):
        if False:
            return 10
        outputs = []

        def skeletonize(r):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(r, tuple):
                return tuple((skeletonize(e) for e in r))
            elif isinstance(r, list):
                return [skeletonize(e) for e in r]
            elif isinstance(r, dict):
                return {k: skeletonize(v) for (k, v) in r.items()}
            else:
                index = len(outputs)
                outputs.append(r)
                return index
        skeleton = skeletonize(x)
        return (skeleton, outputs)

    def reconstruct_return_value(self, outputs):
        if False:
            print('Hello World!')

        def f(skeleton):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(skeleton, tuple):
                return tuple((f(e) for e in skeleton))
            elif isinstance(skeleton, list):
                return [f(e) for e in skeleton]
            elif isinstance(skeleton, dict):
                return {k: f(v) for (k, v) in skeleton.items()}
            else:
                return outputs[skeleton]
        return f(self.skeleton)

def fake_as_funcnode(alt_func, name, rename_attributes=None, experimental_warning=True):
    if False:
        for i in range(10):
            print('nop')
    'The target function fakes FunctionNode\n\n    The target function is replaced to the alternative function to connect\n    variable node by acting function node. ``alt_func`` must satisfy the\n    following restrictions.\n\n    1. Inputs includes one or more ``chainer.Variable`` to trace variables.\n    2. Output consists nothing but ``ndarray`` or ``chainer.Variable``\n\n    Even if ``alt_func`` returns ``ndarray``, the value forced to be converted\n    to ``chainer.Variable``. A caller of the target function have to care\n    both cases, returning ``ndarray`` and ``chainer.Variable``.\n\n    When ``alt_func`` returns ``list`` of variable, the wrapped function will\n    also returns multiple variables as ``tuple``. However ``dict`` cannot\n    be return, the wrapped function breaks down the returned values as\n    ``tuple`` of values, keys will be ignored.\n\n    Arguments of ``alt_func`` except for ``chainer.Variable`` are set as\n    function attributes. Attribute names are set ``argN`` (N is index\n    number) or keyword on default.\n\n    Example:\n\n       >>> def func(x, a, b, c=1, d=2): pass\n       >>> # x is variable\n       >>> func = onnx_chainer.replace_func.fake_as_funcnode(\n       ...     func, \'CustomNode\',\n       ...     rename_attributes=[(1, \'value\'), (\'c\', \'y\')])\n\n    Then ``func`` will be operated as a function node named "CustomNode", and\n    ``\'value\'``, ``\'b\'``, ``\'y\'``, ``\'d\'`` are set as function\'s attributes.\n    See tests/test_replace_func.py more details.\n\n    Args:\n        alt_func (func): actual called function. There are some constrains, see\n            the above documentation.\n        name (str): function name. This name is used for what ONNX operator\n            to be assigned.\n        rename_attributes (list or tuple): rename attribute name, set list\n            of ``tuple(index_of_args, new_name)`` or\n            ``tuple(kwargs_name, new_name)``\n        experimental_warning: this function is experimental utility, if set\n            ``False``, run without experimental warning.\n\n    Returns:\n        func: wrapped function, called on exporting.\n    '

    def _wrapper(*args, **kwargs):
        if False:
            while True:
                i = 10
        inputs = []
        attributes = {}
        rename_attr_dict = {}
        if rename_attributes is not None:
            rename_attr_dict = {attr[0]: attr[1] for attr in rename_attributes}
        arg_spec = inspect.signature(alt_func)
        bound = arg_spec.bind(*args, **kwargs)
        bound.apply_defaults()
        for (i, (k, v)) in enumerate(bound.arguments.items()):
            if i < len(args):
                continue
            kwargs[k] = v

        def set_attr(key, value):
            if False:
                print('Hello World!')
            default_name = key if isinstance(key, str) else 'arg{}'.format(key)
            attributes[rename_attr_dict.get(key, default_name)] = value

        def expand_args(args_iter):
            if False:
                return 10
            for (i, a) in args_iter:
                if _is_var(a):
                    inputs.append(a)
                elif isinstance(a, (tuple, list)):
                    flatten_arg = _flatten(a)
                    var_or_not = map(_is_var, flatten_arg)
                    if all(var_or_not):
                        inputs.extend(flatten_arg)
                    elif not any(var_or_not):
                        set_attr(i, a)
                    else:
                        raise ValueError('arguments mixed variable and other type are not supported')
                else:
                    set_attr(i, a)
        expand_args(enumerate(args))
        expand_args(kwargs.items())
        if not inputs:
            raise ValueError("arguments of the function wrapped by 'as_funcnode' must include at least one chainer.Variable, function name: {}".format(name))
        wrapped = WrappedFunctionNode(name, alt_func, args, kwargs, inputs, attributes=attributes)
        ret = wrapped.apply(inputs)
        return wrapped.reconstruct_return_value(ret)
    if experimental_warning:
        chainer.utils.experimental('as_funcnode')
    return _wrapper

def as_funcnode(name, rename_attributes=None):
    if False:
        i = 10
        return i + 15
    "The target function fakes FunctionNode\n\n    The target function is overwrapped to connect variable node by acting\n    function node. Expected to be used as decorator. More detail, see\n    ``fake_as_funcnode`` documentation.\n\n    Example:\n\n       >>> @onnx_chainer.replace_func.as_funcnode(\n       ...     'CustomNode', rename_attributes=[(1, 'value'), ('c', 'y')])\n       ... def func(x, a, b, c=1, d=2): pass\n\n    Args:\n        name (str): function name. This name is used for what ONNX operator\n            to be assigned.\n        rename_attributes (list or tuple): rename attribute name, set list\n            of ``tuple(index_of_args, new_name)`` or\n            ``tuple(kwargs_name, new_name)``\n    "

    def _wrapper(fn):
        if False:
            while True:
                i = 10
        return fake_as_funcnode(fn, name, rename_attributes=rename_attributes)
    return _wrapper

def _unwrap_var(var):
    if False:
        print('Hello World!')
    return var.array if _is_var(var) else var

def _is_var(array):
    if False:
        i = 10
        return i + 15
    return isinstance(array, chainer.Variable)

def _is_array(v):
    if False:
        for i in range(10):
            print('nop')
    return not isinstance(v, (list, tuple))

def _flatten(xs):
    if False:
        for i in range(10):
            print('nop')
    if _is_array(xs):
        return [xs]
    o = []
    for x in xs:
        if _is_array(x):
            o.append(x)
        else:
            o.extend(_flatten(x))
    return o