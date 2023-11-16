import chainer
from chainer import function
from chainer import function_node
from chainer import variable

def _call_func(func, xs):
    if False:
        while True:
            i = 10
    outs = func(*xs)
    if isinstance(outs, tuple):
        for (i, out) in enumerate(outs):
            if isinstance(out, variable.Variable):
                continue
            n = i + 1
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n if n < 20 else n % 10, 'th')
            msg = '{}{} element of a returned tuple is not Variable, but is {}'.format(n, suffix, type(out))
            raise RuntimeError(msg)
    elif isinstance(outs, variable.Variable):
        outs = (outs,)
    else:
        msg = 'A tuple of Variables or a Variable are expected, but {} is returned.'.format(type(outs))
        raise RuntimeError(msg)
    return outs

class Forget(function_node.FunctionNode):

    def __init__(self, func):
        if False:
            print('Hello World!')
        if not callable(func):
            raise TypeError('func must be callable')
        self.func = func

    def forward(self, inputs):
        if False:
            return 10
        self.retain_inputs(tuple(range(len(inputs))))
        with function.no_backprop_mode(), chainer.using_config('_will_recompute', True):
            xs = [variable.Variable(x) for x in inputs]
            outs = _call_func(self.func, xs)
        return tuple((out.data for out in outs))

    def backward(self, indexes, grad_outputs):
        if False:
            return 10
        if chainer.config.enable_backprop:
            raise RuntimeError('double backpropagation in functions.forget is not allowed.')
        inputs = self.get_retained_inputs()
        dummy_inputs = tuple([variable.Variable(inp.array) for inp in inputs])
        with function.force_backprop_mode(), chainer.using_config('in_recomputing', True):
            outs = _call_func(self.func, dummy_inputs)
            assert len(outs) == len(grad_outputs)
        output_tuples = []
        for (out, grad_output) in zip(outs, grad_outputs):
            if grad_output is not None:
                output_tuples.append((out.node, grad_output))
        chainer._backprop._backprop_to_all(output_tuples, False, None)
        return tuple([inp.grad_var for inp in dummy_inputs])

def forget(func, *xs):
    if False:
        for i in range(10):
            print('nop')
    "Calls a function without storing intermediate results.\n\n    On a forward propagation, Chainer normally stores all intermediate results\n    of :class:`~chainer.variable.VariableNode`\\ s on a computational graph as\n    they are required on backward propagation.\n    Sometimes these results consume too much memory.\n    ``F.forget`` *forgets* such intermediate results on forward propagation,\n    and still supports backpropagation with recalculation.\n\n    On a forward propagation, ``F.forget`` calls a given function with given\n    variables without creating a computational graph. That means, no\n    intermediate results are stored.\n    On a backward propagation, ``F.forget`` calls the given function again to\n    create a computational graph for backpropagation.\n\n    ``F.forget`` reduces internal memory usage, whereas it requires more\n    calculation time as it calls the function twice.\n\n    .. admonition:: Example\n\n       Let ``f`` be a function defined as:\n\n       >>> def f(a, b):\n       ...   return (a + b) * a\n\n       and, ``x`` and ``y`` be :class:`~chainer.Variable`\\ s:\n\n       >>> x = chainer.Variable(np.random.uniform(-1, 1, 5).astype(np.float32))\n       >>> y = chainer.Variable(np.random.uniform(-1, 1, 5).astype(np.float32))\n\n       When ``z`` is calculated as ``z = f(x, y)``, its intermediate result\n       ``x + y`` is stored in memory. Instead, if you call ``f`` with\n       ``F.forget``:\n\n       >>> z = F.forget(f, x, y)\n\n       intermediate ``x + y`` is forgotten.\n\n    .. note::\n\n        ``F.forget`` does not support functions which behave differently in\n        multiple calls with the same inputs, such as\n        :meth:`F.dropout() <chainer.functions.dropout>` and\n        :meth:`F.negative_sampling() <chainer.functions.negative_sampling>`.\n\n    .. note::\n\n        In case input argument variables are of :ref:`ndarray` objects,\n        arguments will automatically be\n        converted to :class:`~chainer.Variable`\\ s.\n        This conversion takes place to ensure that this function is included\n        in the computational graph to enable backward computations.\n\n    .. note::\n\n        ``F.forget`` does not support double backpropagation.\n\n    .. note::\n\n        If you want to use ``F.forget`` to a link which updates the link's\n        internal information every time the forward computation is called,\n        please ensure that the information is updated just once in a single\n        iteration. You may use the ``chainer.config.in_recomputing`` flag to\n        check if the forward computation is the first call in an iteration.\n        Please see the implementation of\n        :class:`~chainer.links.BatchNormalization` for detail.\n\n    Args:\n        func (callable): A function to call. It needs to be called with\n            :class:`~chainer.Variable` object(s) and to return a\n            :class:`~chainer.Variable` object or a tuple of\n            :class:`~chainer.Variable` objects.\n        xs (:class:`tuple` of :class:`~chainer.Variable` or :ref:`ndarray`):\n            Argument variables of the function.\n\n    Returns:\n        ~chainer.Variable: A variable ``func`` returns. If it returns a tuple,\n        the method returns a tuple too.\n\n    "
    xs = tuple((x if isinstance(x, variable.Variable) else variable.Variable(x, requires_grad=True) for x in xs))
    y = Forget(func).apply(xs)
    if len(y) == 1:
        (y,) = y
    return y