from __future__ import annotations
from typing import Any
from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.utilities.iterables import is_sequence
import sympy
from functools import partial
aesara = import_module('aesara')
if aesara:
    aes = aesara.scalar
    aet = aesara.tensor
    from aesara.tensor import nlinalg
    from aesara.tensor.elemwise import Elemwise
    from aesara.tensor.elemwise import DimShuffle
    true_divide = getattr(aet, 'true_divide', None)
    if true_divide is None:
        true_divide = aet.true_div
    mapping = {sympy.Add: aet.add, sympy.Mul: aet.mul, sympy.Abs: aet.abs, sympy.sign: aet.sgn, sympy.ceiling: aet.ceil, sympy.floor: aet.floor, sympy.log: aet.log, sympy.exp: aet.exp, sympy.sqrt: aet.sqrt, sympy.cos: aet.cos, sympy.acos: aet.arccos, sympy.sin: aet.sin, sympy.asin: aet.arcsin, sympy.tan: aet.tan, sympy.atan: aet.arctan, sympy.atan2: aet.arctan2, sympy.cosh: aet.cosh, sympy.acosh: aet.arccosh, sympy.sinh: aet.sinh, sympy.asinh: aet.arcsinh, sympy.tanh: aet.tanh, sympy.atanh: aet.arctanh, sympy.re: aet.real, sympy.im: aet.imag, sympy.arg: aet.angle, sympy.erf: aet.erf, sympy.gamma: aet.gamma, sympy.loggamma: aet.gammaln, sympy.Pow: aet.pow, sympy.Eq: aet.eq, sympy.StrictGreaterThan: aet.gt, sympy.StrictLessThan: aet.lt, sympy.LessThan: aet.le, sympy.GreaterThan: aet.ge, sympy.And: aet.bitwise_and, sympy.Or: aet.bitwise_or, sympy.Not: aet.invert, sympy.Xor: aet.bitwise_xor, sympy.Max: aet.maximum, sympy.Min: aet.minimum, sympy.conjugate: aet.conj, sympy.core.numbers.ImaginaryUnit: lambda : aet.complex(0, 1), sympy.MatAdd: Elemwise(aes.add), sympy.HadamardProduct: Elemwise(aes.mul), sympy.Trace: nlinalg.trace, sympy.Determinant: nlinalg.det, sympy.Inverse: nlinalg.matrix_inverse, sympy.Transpose: DimShuffle((False, False), [1, 0])}

class AesaraPrinter(Printer):
    """ Code printer which creates Aesara symbolic expression graphs.

    Parameters
    ==========

    cache : dict
        Cache dictionary to use. If None (default) will use
        the global cache. To create a printer which does not depend on or alter
        global state pass an empty dictionary. Note: the dictionary is not
        copied on initialization of the printer and will be updated in-place,
        so using the same dict object when creating multiple printers or making
        multiple calls to :func:`.aesara_code` or :func:`.aesara_function` means
        the cache is shared between all these applications.

    Attributes
    ==========

    cache : dict
        A cache of Aesara variables which have been created for SymPy
        symbol-like objects (e.g. :class:`sympy.core.symbol.Symbol` or
        :class:`sympy.matrices.expressions.MatrixSymbol`). This is used to
        ensure that all references to a given symbol in an expression (or
        multiple expressions) are printed as the same Aesara variable, which is
        created only once. Symbols are differentiated only by name and type. The
        format of the cache's contents should be considered opaque to the user.
    """
    printmethod = '_aesara'

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.cache = kwargs.pop('cache', {})
        super().__init__(*args, **kwargs)

    def _get_key(self, s, name=None, dtype=None, broadcastable=None):
        if False:
            print('Hello World!')
        ' Get the cache key for a SymPy object.\n\n        Parameters\n        ==========\n\n        s : sympy.core.basic.Basic\n            SymPy object to get key for.\n\n        name : str\n            Name of object, if it does not have a ``name`` attribute.\n        '
        if name is None:
            name = s.name
        return (name, type(s), s.args, dtype, broadcastable)

    def _get_or_create(self, s, name=None, dtype=None, broadcastable=None):
        if False:
            i = 10
            return i + 15
        '\n        Get the Aesara variable for a SymPy symbol from the cache, or create it\n        if it does not exist.\n        '
        if name is None:
            name = s.name
        if dtype is None:
            dtype = 'floatX'
        if broadcastable is None:
            broadcastable = ()
        key = self._get_key(s, name, dtype=dtype, broadcastable=broadcastable)
        if key in self.cache:
            return self.cache[key]
        value = aet.tensor(name=name, dtype=dtype, shape=broadcastable)
        self.cache[key] = value
        return value

    def _print_Symbol(self, s, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        dtype = kwargs.get('dtypes', {}).get(s)
        bc = kwargs.get('broadcastables', {}).get(s)
        return self._get_or_create(s, dtype=dtype, broadcastable=bc)

    def _print_AppliedUndef(self, s, **kwargs):
        if False:
            print('Hello World!')
        name = str(type(s)) + '_' + str(s.args[0])
        dtype = kwargs.get('dtypes', {}).get(s)
        bc = kwargs.get('broadcastables', {}).get(s)
        return self._get_or_create(s, name=name, dtype=dtype, broadcastable=bc)

    def _print_Basic(self, expr, **kwargs):
        if False:
            print('Hello World!')
        op = mapping[type(expr)]
        children = [self._print(arg, **kwargs) for arg in expr.args]
        return op(*children)

    def _print_Number(self, n, **kwargs):
        if False:
            while True:
                i = 10
        return float(n.evalf())

    def _print_MatrixSymbol(self, X, **kwargs):
        if False:
            print('Hello World!')
        dtype = kwargs.get('dtypes', {}).get(X)
        return self._get_or_create(X, dtype=dtype, broadcastable=(None, None))

    def _print_DenseMatrix(self, X, **kwargs):
        if False:
            while True:
                i = 10
        if not hasattr(aet, 'stacklists'):
            raise NotImplementedError('Matrix translation not yet supported in this version of Aesara')
        return aet.stacklists([[self._print(arg, **kwargs) for arg in L] for L in X.tolist()])
    _print_ImmutableMatrix = _print_ImmutableDenseMatrix = _print_DenseMatrix

    def _print_MatMul(self, expr, **kwargs):
        if False:
            i = 10
            return i + 15
        children = [self._print(arg, **kwargs) for arg in expr.args]
        result = children[0]
        for child in children[1:]:
            result = aet.dot(result, child)
        return result

    def _print_MatPow(self, expr, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        children = [self._print(arg, **kwargs) for arg in expr.args]
        result = 1
        if isinstance(children[1], int) and children[1] > 0:
            for i in range(children[1]):
                result = aet.dot(result, children[0])
        else:
            raise NotImplementedError('Only non-negative integer\n           powers of matrices can be handled by Aesara at the moment')
        return result

    def _print_MatrixSlice(self, expr, **kwargs):
        if False:
            print('Hello World!')
        parent = self._print(expr.parent, **kwargs)
        rowslice = self._print(slice(*expr.rowslice), **kwargs)
        colslice = self._print(slice(*expr.colslice), **kwargs)
        return parent[rowslice, colslice]

    def _print_BlockMatrix(self, expr, **kwargs):
        if False:
            print('Hello World!')
        (nrows, ncols) = expr.blocks.shape
        blocks = [[self._print(expr.blocks[r, c], **kwargs) for c in range(ncols)] for r in range(nrows)]
        return aet.join(0, *[aet.join(1, *row) for row in blocks])

    def _print_slice(self, expr, **kwargs):
        if False:
            return 10
        return slice(*[self._print(i, **kwargs) if isinstance(i, sympy.Basic) else i for i in (expr.start, expr.stop, expr.step)])

    def _print_Pi(self, expr, **kwargs):
        if False:
            while True:
                i = 10
        return 3.141592653589793

    def _print_Piecewise(self, expr, **kwargs):
        if False:
            return 10
        import numpy as np
        (e, cond) = expr.args[0].args
        p_cond = self._print(cond, **kwargs)
        p_e = self._print(e, **kwargs)
        if len(expr.args) == 1:
            return aet.switch(p_cond, p_e, np.nan)
        p_remaining = self._print(sympy.Piecewise(*expr.args[1:]), **kwargs)
        return aet.switch(p_cond, p_e, p_remaining)

    def _print_Rational(self, expr, **kwargs):
        if False:
            while True:
                i = 10
        return true_divide(self._print(expr.p, **kwargs), self._print(expr.q, **kwargs))

    def _print_Integer(self, expr, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return expr.p

    def _print_factorial(self, expr, **kwargs):
        if False:
            print('Hello World!')
        return self._print(sympy.gamma(expr.args[0] + 1), **kwargs)

    def _print_Derivative(self, deriv, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        from aesara.gradient import Rop
        rv = self._print(deriv.expr, **kwargs)
        for var in deriv.variables:
            var = self._print(var, **kwargs)
            rv = Rop(rv, var, aet.ones_like(var))
        return rv

    def emptyPrinter(self, expr):
        if False:
            i = 10
            return i + 15
        return expr

    def doprint(self, expr, dtypes=None, broadcastables=None):
        if False:
            print('Hello World!')
        " Convert a SymPy expression to a Aesara graph variable.\n\n        The ``dtypes`` and ``broadcastables`` arguments are used to specify the\n        data type, dimension, and broadcasting behavior of the Aesara variables\n        corresponding to the free symbols in ``expr``. Each is a mapping from\n        SymPy symbols to the value of the corresponding argument to\n        ``aesara.tensor.var.TensorVariable``.\n\n        See the corresponding `documentation page`__ for more information on\n        broadcasting in Aesara.\n\n\n        .. __: https://aesara.readthedocs.io/en/latest/reference/tensor/shapes.html#broadcasting\n\n        Parameters\n        ==========\n\n        expr : sympy.core.expr.Expr\n            SymPy expression to print.\n\n        dtypes : dict\n            Mapping from SymPy symbols to Aesara datatypes to use when creating\n            new Aesara variables for those symbols. Corresponds to the ``dtype``\n            argument to ``aesara.tensor.var.TensorVariable``. Defaults to ``'floatX'``\n            for symbols not included in the mapping.\n\n        broadcastables : dict\n            Mapping from SymPy symbols to the value of the ``broadcastable``\n            argument to ``aesara.tensor.var.TensorVariable`` to use when creating Aesara\n            variables for those symbols. Defaults to the empty tuple for symbols\n            not included in the mapping (resulting in a scalar).\n\n        Returns\n        =======\n\n        aesara.graph.basic.Variable\n            A variable corresponding to the expression's value in a Aesara\n            symbolic expression graph.\n\n        "
        if dtypes is None:
            dtypes = {}
        if broadcastables is None:
            broadcastables = {}
        return self._print(expr, dtypes=dtypes, broadcastables=broadcastables)
global_cache: dict[Any, Any] = {}

def aesara_code(expr, cache=None, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    Convert a SymPy expression into a Aesara graph variable.\n\n    Parameters\n    ==========\n\n    expr : sympy.core.expr.Expr\n        SymPy expression object to convert.\n\n    cache : dict\n        Cached Aesara variables (see :class:`AesaraPrinter.cache\n        <AesaraPrinter>`). Defaults to the module-level global cache.\n\n    dtypes : dict\n        Passed to :meth:`.AesaraPrinter.doprint`.\n\n    broadcastables : dict\n        Passed to :meth:`.AesaraPrinter.doprint`.\n\n    Returns\n    =======\n\n    aesara.graph.basic.Variable\n        A variable corresponding to the expression's value in a Aesara symbolic\n        expression graph.\n\n    "
    if not aesara:
        raise ImportError('aesara is required for aesara_code')
    if cache is None:
        cache = global_cache
    return AesaraPrinter(cache=cache, settings={}).doprint(expr, **kwargs)

def dim_handling(inputs, dim=None, dims=None, broadcastables=None):
    if False:
        while True:
            i = 10
    '\n    Get value of ``broadcastables`` argument to :func:`.aesara_code` from\n    keyword arguments to :func:`.aesara_function`.\n\n    Included for backwards compatibility.\n\n    Parameters\n    ==========\n\n    inputs\n        Sequence of input symbols.\n\n    dim : int\n        Common number of dimensions for all inputs. Overrides other arguments\n        if given.\n\n    dims : dict\n        Mapping from input symbols to number of dimensions. Overrides\n        ``broadcastables`` argument if given.\n\n    broadcastables : dict\n        Explicit value of ``broadcastables`` argument to\n        :meth:`.AesaraPrinter.doprint`. If not None function will return this value unchanged.\n\n    Returns\n    =======\n    dict\n        Dictionary mapping elements of ``inputs`` to their "broadcastable"\n        values (tuple of ``bool``\\ s).\n    '
    if dim is not None:
        return {s: (False,) * dim for s in inputs}
    if dims is not None:
        maxdim = max(dims.values())
        return {s: (False,) * d + (True,) * (maxdim - d) for (s, d) in dims.items()}
    if broadcastables is not None:
        return broadcastables
    return {}

def aesara_function(inputs, outputs, scalar=False, *, dim=None, dims=None, broadcastables=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a Aesara function from SymPy expressions.\n\n    The inputs and outputs are converted to Aesara variables using\n    :func:`.aesara_code` and then passed to ``aesara.function``.\n\n    Parameters\n    ==========\n\n    inputs\n        Sequence of symbols which constitute the inputs of the function.\n\n    outputs\n        Sequence of expressions which constitute the outputs(s) of the\n        function. The free symbols of each expression must be a subset of\n        ``inputs``.\n\n    scalar : bool\n        Convert 0-dimensional arrays in output to scalars. This will return a\n        Python wrapper function around the Aesara function object.\n\n    cache : dict\n        Cached Aesara variables (see :class:`AesaraPrinter.cache\n        <AesaraPrinter>`). Defaults to the module-level global cache.\n\n    dtypes : dict\n        Passed to :meth:`.AesaraPrinter.doprint`.\n\n    broadcastables : dict\n        Passed to :meth:`.AesaraPrinter.doprint`.\n\n    dims : dict\n        Alternative to ``broadcastables`` argument. Mapping from elements of\n        ``inputs`` to integers indicating the dimension of their associated\n        arrays/tensors. Overrides ``broadcastables`` argument if given.\n\n    dim : int\n        Another alternative to the ``broadcastables`` argument. Common number of\n        dimensions to use for all arrays/tensors.\n        ``aesara_function([x, y], [...], dim=2)`` is equivalent to using\n        ``broadcastables={x: (False, False), y: (False, False)}``.\n\n    Returns\n    =======\n    callable\n        A callable object which takes values of ``inputs`` as positional\n        arguments and returns an output array for each of the expressions\n        in ``outputs``. If ``outputs`` is a single expression the function will\n        return a Numpy array, if it is a list of multiple expressions the\n        function will return a list of arrays. See description of the ``squeeze``\n        argument above for the behavior when a single output is passed in a list.\n        The returned object will either be an instance of\n        ``aesara.compile.function.types.Function`` or a Python wrapper\n        function around one. In both cases, the returned value will have a\n        ``aesara_function`` attribute which points to the return value of\n        ``aesara.function``.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import x, y, z\n    >>> from sympy.printing.aesaracode import aesara_function\n\n    A simple function with one input and one output:\n\n    >>> f1 = aesara_function([x], [x**2 - 1], scalar=True)\n    >>> f1(3)\n    8.0\n\n    A function with multiple inputs and one output:\n\n    >>> f2 = aesara_function([x, y, z], [(x**z + y**z)**(1/z)], scalar=True)\n    >>> f2(3, 4, 2)\n    5.0\n\n    A function with multiple inputs and multiple outputs:\n\n    >>> f3 = aesara_function([x, y], [x**2 + y**2, x**2 - y**2], scalar=True)\n    >>> f3(2, 3)\n    [13.0, -5.0]\n\n    See also\n    ========\n\n    dim_handling\n\n    '
    if not aesara:
        raise ImportError('Aesara is required for aesara_function')
    cache = kwargs.pop('cache', {})
    dtypes = kwargs.pop('dtypes', {})
    broadcastables = dim_handling(inputs, dim=dim, dims=dims, broadcastables=broadcastables)
    code = partial(aesara_code, cache=cache, dtypes=dtypes, broadcastables=broadcastables)
    tinputs = list(map(code, inputs))
    toutputs = list(map(code, outputs))
    toutputs = [output if isinstance(output, aesara.graph.basic.Variable) else aet.as_tensor_variable(output) for output in toutputs]
    if len(toutputs) == 1:
        toutputs = toutputs[0]
    func = aesara.function(tinputs, toutputs, **kwargs)
    is_0d = [len(o.variable.broadcastable) == 0 for o in func.outputs]
    if not scalar or not any(is_0d):
        func.aesara_function = func
        return func

    def wrapper(*args):
        if False:
            return 10
        out = func(*args)
        if is_sequence(out):
            return [o[()] if is_0d[i] else o for (i, o) in enumerate(out)]
        else:
            return out[()]
    wrapper.__wrapped__ = func
    wrapper.__doc__ = func.__doc__
    wrapper.aesara_function = func
    return wrapper