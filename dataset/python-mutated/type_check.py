import contextlib
import functools
import operator
import sys
import threading
import numpy
import six
import chainer
from chainer.backends import cuda
_thread_local = threading.local()

@contextlib.contextmanager
def get_function_check_context(f):
    if False:
        return 10
    try:
        default = _thread_local.current_function
    except AttributeError:
        default = None
    _thread_local.current_function = f
    try:
        yield
    finally:
        _thread_local.current_function = default

class TypeInfo(object):
    """Type information of an input/gradient array.

    It contains type information of an array, such as the shape of array and
    the number of dimensions.
    This information is independent of CPU or GPU array.
    """

    def __init__(self, shape, dtype):
        if False:
            for i in range(10):
                print('nop')
        self.shape = shape
        self.dtype = dtype
        self.ndim = len(shape)

    @property
    def size(self):
        if False:
            while True:
                i = 10
        return functools.reduce(operator.mul, self.shape, 1)

class TypeInfoTuple(tuple):
    """Type information of input/gradient tuples.

    It is a sub-class of tuple containing :class:`TypeInfo`. The i-th element
    of this object contains type information of the i-th input/gradient data.
    As each element is :class:`Expr`, you can easily check its validity.
    """

    def size(self):
        if False:
            return 10
        'Returns an expression representing its length.\n\n        Returns:\n            Expr: An expression object representing length of the tuple.\n        '
        return Variable(len(self), '{0}.size'.format(self.name))

class LightTypeInfoTuple(tuple):
    """Type information of input/gradient tuples for light-weight check.

    It is a sub-class of tuple containing :class:`TypeInfo`. The i-th element
    of this object contains type information of the i-th input/gradient data.
    """

    def size(self):
        if False:
            print('Hello World!')
        'Returns its length.\n\n        Returns:\n            int: Length of the tuple.\n        '
        return len(self)

def get_types(data, name, accept_none, *, shapes=None):
    if False:
        while True:
            i = 10
    assert isinstance(data, tuple)
    if shapes is None:
        shapes = tuple([x.shape for x in data])
    info = TypeInfoTuple((_get_type(name, i, x, accept_none, shape) for (i, (x, shape)) in enumerate(zip(data, shapes))))
    info.name = name
    return info

def get_light_types(data, *, shapes=None):
    if False:
        print('Hello World!')
    assert isinstance(data, tuple)
    if shapes is None:
        data_ = data
    else:
        data_ = tuple([TypeInfo(shape, x.dtype) for (x, shape) in zip(data, shapes)])
    return LightTypeInfoTuple(data_)

def _get_type(name, index, array, accept_none, shape):
    if False:
        for i in range(10):
            print('nop')
    var = '{0}[{1}]'.format(name, index)
    if accept_none and array is None:
        return Variable(TypeInfo((), None), var)
    assert isinstance(array, chainer.get_array_types())
    return Variable(TypeInfo(shape, array.dtype), var)

def _make_un_operator(exp, priority, func):
    if False:
        while True:
            i = 10

    def f(x):
        if False:
            print('Hello World!')
        return UnaryOperator(priority, x, exp, func)
    return f

def _make_bin_operator(exp, priority, func, right_associative=False):
    if False:
        i = 10
        return i + 15

    def f(x, y):
        if False:
            i = 10
            return i + 15
        return BinaryOperator(priority, x, y, exp, func, right_associative)
    return f

def _make_bool_operator(exp, inv, func):
    if False:
        print('Hello World!')

    def f(x, y):
        if False:
            print('Hello World!')
        return BoolBinaryOperator(x, y, exp, inv, func)
    return f

def _flip(f):
    if False:
        for i in range(10):
            print('nop')
    return lambda x, y: f(y, x)

class Expr(object):
    """Abstract syntax tree of an expression.

    It represents an abstract syntax tree, and isn't a value. You can get its
    actual value with :meth:`eval` function, and get syntax representation with
    the :meth:`__str__` method.
    Each comparison operator (e.g. ``==``) generates a new :class:`Expr` object
    which represents the result of comparison between two expressions.

    .. admonition:: Example

       Let ``x`` and ``y`` be instances of :class:`Expr`, then ::

          >>> x = Variable(1, 'x')
          >>> y = Variable(1, 'y')
          >>> c = (x == y)

       is also an instance of :class:`Expr`. To evaluate and get its value,
       call :meth:`eval` method::

          >>> c.eval()
          True

       Call ``str`` function to get a representation of the original
       equation::

          >>> str(c)
          'x == y'

       You can actually compare an expression with a value::

          >>> (x == 1).eval()
          True

       Note that you can't use boolean operators such as ``and``, as they try
       to cast expressions to boolean values::

          >>> z = Variable(1, 'z')
          >>> x == y and y == z  # raises an error
          Traceback (most recent call last):
          RuntimeError: Don't convert Expr to bool. Please call Expr.eval method to evaluate expression.


    """

    def __init__(self, priority):
        if False:
            print('Hello World!')
        self.priority = priority

    def eval(self):
        if False:
            print('Hello World!')
        'Evaluates the tree to get actual value.\n\n        Behavior of this function depends on an implementation class.\n        For example, a binary operator ``+`` calls the ``__add__`` function\n        with the two results of :meth:`eval` function.\n        '
        raise NotImplementedError()

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        return GetAttr(self, name)

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        return GetItem(self, key)

    def __call__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return Call(self, args)

    def __nonzero__(self):
        if False:
            for i in range(10):
                print('nop')
        msg = 'An Expr instance cannot be evaluated as bool. Please use chainer.utils.type_check.eval() to evaluate an expression.'
        raise RuntimeError(msg)

    def __bool__(self):
        if False:
            while True:
                i = 10
        self.__nonzero__()
    __eq__ = _make_bool_operator('==', '!=', operator.__eq__)
    __ne__ = _make_bool_operator('!=', '==', operator.__ne__)
    __lt__ = _make_bool_operator('<', '>=', operator.__lt__)
    __le__ = _make_bool_operator('<=', '>', operator.__le__)
    __gt__ = _make_bool_operator('>', '<=', operator.__gt__)
    __ge__ = _make_bool_operator('>=', '<', operator.__ge__)
    __add__ = _make_bin_operator('+', 4, operator.__add__)
    __radd__ = _flip(__add__)
    __sub__ = _make_bin_operator('-', 4, operator.__sub__)
    __rsub__ = _flip(__sub__)
    __mul__ = _make_bin_operator('*', 5, operator.__mul__)
    __rmul__ = _flip(__mul__)
    if sys.version_info < (3, 0, 0):
        __div__ = _make_bin_operator('/', 5, operator.__div__)
        __rdiv__ = _flip(__div__)
    else:
        __truediv__ = _make_bin_operator('/', 5, operator.__truediv__)
        __rtruediv__ = _flip(__truediv__)
    __floordiv__ = _make_bin_operator('//', 5, operator.__floordiv__)
    __rfloordiv__ = _flip(__floordiv__)
    __mod__ = _make_bin_operator('%', 5, operator.__mod__)
    __rmod__ = _flip(__mod__)
    __pow__ = _make_bin_operator('**', 7, operator.__mod__, right_associative=True)
    __lshift__ = _make_bin_operator('<<', 3, operator.__lshift__)
    __rlshift__ = _flip(__lshift__)
    __rshift__ = _make_bin_operator('>>', 3, operator.__rshift__)
    __rrshift__ = _flip(__rshift__)
    __and__ = _make_bin_operator('&', 2, operator.__and__)
    __rand__ = _flip(__and__)
    __xor__ = _make_bin_operator('^', 1, operator.__xor__)
    __rxor__ = _flip(__xor__)
    __or__ = _make_bin_operator('|', 0, operator.__or__)
    __ror__ = _flip(__or__)
    __neg__ = _make_un_operator('-', 6, operator.__neg__)
    __pos__ = _make_un_operator('+', 6, operator.__pos__)
    __invert__ = _make_un_operator('~', 6, operator.__invert__)

def _eval_expr(v):
    if False:
        return 10
    if isinstance(v, Expr):
        return v.eval()
    elif isinstance(v, list):
        return list(map(_eval_expr, v))
    elif isinstance(v, tuple):
        return tuple(map(_eval_expr, v))
    else:
        return v

def _repr(v):
    if False:
        return 10
    if isinstance(v, Expr):
        return str(v)
    elif isinstance(v, list):
        return '[{0}]'.format(', '.join(map(_repr, v)))
    elif isinstance(v, tuple):
        if len(v) == 0:
            return '()'
        elif len(v) == 1:
            return '({0},)'.format(_repr(v[0]))
        else:
            return '({0})'.format(', '.join(map(_repr, v)))
    else:
        return repr(v)

class Atom(Expr):

    def __init__(self):
        if False:
            print('Hello World!')
        super(Atom, self).__init__(8)

class Constant(Atom):

    def __init__(self, value):
        if False:
            while True:
                i = 10
        super(Constant, self).__init__()
        self.value = value

    def __str__(self):
        if False:
            while True:
                i = 10
        return _repr(self.value)

    def eval(self):
        if False:
            print('Hello World!')
        return self.value

class Variable(Atom):

    def __init__(self, value, name):
        if False:
            i = 10
            return i + 15
        super(Variable, self).__init__()
        self.value = value
        self.name = name

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.name

    def eval(self):
        if False:
            for i in range(10):
                print('nop')
        return self.value

class GetAttr(Atom):

    def __init__(self, obj, name):
        if False:
            print('Hello World!')
        super(GetAttr, self).__init__()
        self.obj = obj
        self.name = name

    def __str__(self):
        if False:
            while True:
                i = 10
        if isinstance(self.name, str):
            return '{0}.{1}'.format(_repr(self.obj), self.name)
        elif isinstance(self.name, Constant) and isinstance(self.name.value, str):
            return '{0}.{1}'.format(_repr(self.obj), self.name.value)
        else:
            return 'getattr({0}, {1})'.format(_repr(self.obj), _repr(self.name))

    def eval(self):
        if False:
            return 10
        return getattr(_eval_expr(self.obj), _eval_expr(self.name))

def _str_subscript(exp):
    if False:
        i = 10
        return i + 15
    if exp is Ellipsis:
        return '...'
    elif isinstance(exp, slice):

        def key_str(v):
            if False:
                i = 10
                return i + 15
            return '' if v is None else _repr(v)
        if exp.step is None:
            return '{0}:{1}'.format(key_str(exp.start), key_str(exp.stop))
        else:
            return '{0}:{1}:{2}'.format(key_str(exp.start), key_str(exp.stop), key_str(exp.step))
    elif isinstance(exp, tuple):
        return ', '.join(map(_str_subscript, exp))
    else:
        return _repr(exp)

class GetItem(Atom):

    def __init__(self, obj, key):
        if False:
            i = 10
            return i + 15
        super(GetItem, self).__init__()
        self.obj = obj
        self.key = key

    def __str__(self):
        if False:
            return 10
        key = _str_subscript(self.key)
        return '{0}[{1}]'.format(_repr(self.obj), key)

    def eval(self):
        if False:
            while True:
                i = 10
        return _eval_expr(self.obj)[_eval_expr(self.key)]

class Call(Atom):

    def __init__(self, obj, args):
        if False:
            print('Hello World!')
        assert isinstance(args, tuple)
        super(Call, self).__init__()
        self.obj = obj
        self.args = args

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '{0}({1})'.format(_repr(self.obj), ', '.join(map(_repr, self.args)))

    def eval(self):
        if False:
            return 10
        args = map(_eval_expr, self.args)
        func = _eval_expr(self.obj)
        return func(*args)

class UnaryOperator(Expr):

    def __init__(self, priority, term, exp, func):
        if False:
            i = 10
            return i + 15
        super(UnaryOperator, self).__init__(priority)
        self.term = term
        self.exp = exp
        self.func = func

    def eval(self):
        if False:
            while True:
                i = 10
        return self.func(_eval_expr(self.term))

    def __str__(self):
        if False:
            i = 10
            return i + 15
        exp = _repr(self.term)
        if isinstance(self.term, Expr) and self.term.priority < self.priority:
            exp = '(' + exp + ')'
        return self.exp + exp

class BinaryOperator(Expr):

    def __init__(self, priority, lhs, rhs, exp, func, right_associative=False):
        if False:
            print('Hello World!')
        super(BinaryOperator, self).__init__(priority)
        self.lhs = lhs
        self.rhs = rhs
        self.exp = exp
        self.func = func
        self.right_associative = right_associative

    def eval(self):
        if False:
            for i in range(10):
                print('nop')
        left = self._eval_left()
        right = self._eval_right()
        return self.func(left, right)

    def _eval_left(self):
        if False:
            return 10
        return _eval_expr(self.lhs)

    def _eval_right(self):
        if False:
            for i in range(10):
                print('nop')
        return _eval_expr(self.rhs)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        left = _repr(self.lhs)
        if isinstance(self.lhs, Expr) and (self.priority > self.lhs.priority or (self.right_associative and self.priority == self.lhs.priority)):
            left = '(' + left + ')'
        right = _repr(self.rhs)
        if isinstance(self.rhs, Expr) and (self.priority > self.rhs.priority or (not self.right_associative and self.priority == self.rhs.priority)):
            right = '(' + right + ')'
        return '{0} {2} {1}'.format(left, right, self.exp)

class Testable(object):

    def expect(self):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError()

class BoolBinaryOperator(BinaryOperator, Testable):

    def __init__(self, lhs, rhs, exp, inv, func):
        if False:
            return 10
        BinaryOperator.__init__(self, -1, lhs, rhs, exp, func)
        self.inv = inv

    def expect(self):
        if False:
            print('Hello World!')
        left = self._eval_left()
        right = self._eval_right()
        if not self.func(left, right):
            raise InvalidType('{0} {1} {2}'.format(self.lhs, self.exp, self.rhs), '{0} {1} {2}'.format(left, self.inv, right))

class InvalidType(Exception):
    """Raised when types of data for forward/backward are invalid.

    """

    def __init__(self, expect, actual, msg=None):
        if False:
            return 10
        if msg is None:
            msg = 'Expect: {0}\nActual: {1}'.format(expect, actual)
            if hasattr(_thread_local, 'current_function') and _thread_local.current_function is not None:
                msg = '\nInvalid operation is performed in: {0} (Forward)\n\n{1}'.format(_thread_local.current_function.label, msg)
        super(InvalidType, self).__init__(msg)
        self.expect = expect
        self.actual = actual

    def __reduce__(self):
        if False:
            while True:
                i = 10
        (msg,) = self.args
        return (InvalidType, (self.expect, self.actual, msg))

def _argname(in_types, names):
    if False:
        print('Hello World!')
    'Assigns user friendly names for the input types.\n\n    This function also asserts that lengths of in_types and names are the\n    same.\n\n    Args:\n        in_types (tuple of TypeInfoTuple): Tuple of type information to assign\n            name to.\n        names (tuple of str): Human-readable names of ``in_types``.\n    '
    if len(in_types) != len(names):
        raise InvalidType('{} argument(s)'.format(str(len(names))), '{} argument(s)'.format(str(len(in_types))), 'Invalid number of arguments')
    for (in_type, name) in zip(in_types, names):
        if isinstance(in_type, Variable):
            in_type.name = name

def expect(*bool_exprs):
    if False:
        i = 10
        return i + 15
    'Evaluates and tests all given expressions.\n\n    This function evaluates given boolean expressions in order. When at least\n    one expression is evaluated as ``False``, that means the given condition is\n    not satisfied.\n    You can check conditions with this function.\n\n    Args:\n        bool_exprs (tuple of Bool expressions): Bool expressions you want to\n            evaluate.\n    '
    if in_light_mode():
        if not all(bool_exprs):
            raise InvalidType('', '')
    else:
        for expr in bool_exprs:
            assert isinstance(expr, Testable)
            expr.expect()

def same_types(*arrays):
    if False:
        for i in range(10):
            print('nop')
    for x in arrays:
        if not isinstance(x, numpy.ndarray):
            break
    else:
        return True
    for x in arrays:
        if not isinstance(x, cuda.ndarray):
            return False
    return True

def eval(exp):
    if False:
        while True:
            i = 10
    if in_light_mode():
        return exp
    else:
        return exp.eval()

def make_variable(value, name):
    if False:
        while True:
            i = 10
    if in_light_mode():
        return value
    else:
        return Variable(value, name)

def _make_variable_from_array(array, name):
    if False:
        i = 10
        return i + 15
    if not isinstance(array, chainer.get_array_types()):
        raise InvalidType('isinstance({}, ndarray)'.format(name), 'type({}) == {}'.format(name, type(array)))
    if in_light_mode():
        return array
    else:
        return Variable(TypeInfo(array.shape, array.dtype), name)

class LightMode(object):

    def __enter__(self):
        if False:
            return 10
        _thread_local.light_mode = True

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            print('Hello World!')
        _thread_local.light_mode = False

def _prod_impl(xs):
    if False:
        print('Hello World!')
    result = 1
    for x in xs:
        result *= x
    return result
_prod = Variable(_prod_impl, 'prod')
light_mode = LightMode()

def in_light_mode():
    if False:
        i = 10
        return i + 15
    try:
        return _thread_local.light_mode
    except AttributeError:
        _thread_local.light_mode = False
    return False

def prod(xs):
    if False:
        for i in range(10):
            print('nop')
    if in_light_mode():
        return _prod_impl(xs)
    else:
        return _prod(xs)

def expect_broadcast_shapes(*shape_types):
    if False:
        i = 10
        return i + 15
    'Checks if shapes can be broadcasted together.\n\n    Args:\n        shapes_types: Type-checked shapes of the arrays to broadcast.\n\n    '
    shapes = [eval(s) for s in shape_types]
    error = None
    try:
        numpy.broadcast(*[numpy.empty(s + (0,)) for s in shapes])
    except ValueError:
        msgs = ['cannot broadcast inputs of the following shapes:']
        for (shape_type, shape) in six.moves.zip(shape_types, shapes):
            msgs.append('{} = {}'.format(shape_type, shape))
        error = InvalidType('', '', msg='\n'.join(msgs))
    if error is not None:
        raise error