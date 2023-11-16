"""
This module provides convenient functions to transform SymPy expressions to
lambda functions which can be used to calculate numerical values very fast.
"""
from __future__ import annotations
from typing import Any
import builtins
import inspect
import keyword
import textwrap
import linecache
from sympy.external import import_module
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import is_sequence, iterable, NotIterable, flatten
from sympy.utilities.misc import filldedent
__doctest_requires__ = {('lambdify',): ['numpy', 'tensorflow']}
MATH_DEFAULT: dict[str, Any] = {}
MPMATH_DEFAULT: dict[str, Any] = {}
NUMPY_DEFAULT: dict[str, Any] = {'I': 1j}
SCIPY_DEFAULT: dict[str, Any] = {'I': 1j}
CUPY_DEFAULT: dict[str, Any] = {'I': 1j}
JAX_DEFAULT: dict[str, Any] = {'I': 1j}
TENSORFLOW_DEFAULT: dict[str, Any] = {}
SYMPY_DEFAULT: dict[str, Any] = {}
NUMEXPR_DEFAULT: dict[str, Any] = {}
MATH = MATH_DEFAULT.copy()
MPMATH = MPMATH_DEFAULT.copy()
NUMPY = NUMPY_DEFAULT.copy()
SCIPY = SCIPY_DEFAULT.copy()
CUPY = CUPY_DEFAULT.copy()
JAX = JAX_DEFAULT.copy()
TENSORFLOW = TENSORFLOW_DEFAULT.copy()
SYMPY = SYMPY_DEFAULT.copy()
NUMEXPR = NUMEXPR_DEFAULT.copy()
MATH_TRANSLATIONS = {'ceiling': 'ceil', 'E': 'e', 'ln': 'log'}
MPMATH_TRANSLATIONS = {'Abs': 'fabs', 'elliptic_k': 'ellipk', 'elliptic_f': 'ellipf', 'elliptic_e': 'ellipe', 'elliptic_pi': 'ellippi', 'ceiling': 'ceil', 'chebyshevt': 'chebyt', 'chebyshevu': 'chebyu', 'E': 'e', 'I': 'j', 'ln': 'log', 'oo': 'inf', 'LambertW': 'lambertw', 'MutableDenseMatrix': 'matrix', 'ImmutableDenseMatrix': 'matrix', 'conjugate': 'conj', 'dirichlet_eta': 'altzeta', 'Ei': 'ei', 'Shi': 'shi', 'Chi': 'chi', 'Si': 'si', 'Ci': 'ci', 'RisingFactorial': 'rf', 'FallingFactorial': 'ff', 'betainc_regularized': 'betainc'}
NUMPY_TRANSLATIONS: dict[str, str] = {'Heaviside': 'heaviside'}
SCIPY_TRANSLATIONS: dict[str, str] = {}
CUPY_TRANSLATIONS: dict[str, str] = {}
JAX_TRANSLATIONS: dict[str, str] = {}
TENSORFLOW_TRANSLATIONS: dict[str, str] = {}
NUMEXPR_TRANSLATIONS: dict[str, str] = {}
MODULES = {'math': (MATH, MATH_DEFAULT, MATH_TRANSLATIONS, ('from math import *',)), 'mpmath': (MPMATH, MPMATH_DEFAULT, MPMATH_TRANSLATIONS, ('from mpmath import *',)), 'numpy': (NUMPY, NUMPY_DEFAULT, NUMPY_TRANSLATIONS, ('import numpy; from numpy import *; from numpy.linalg import *',)), 'scipy': (SCIPY, SCIPY_DEFAULT, SCIPY_TRANSLATIONS, ('import scipy; import numpy; from scipy.special import *',)), 'cupy': (CUPY, CUPY_DEFAULT, CUPY_TRANSLATIONS, ('import cupy',)), 'jax': (JAX, JAX_DEFAULT, JAX_TRANSLATIONS, ('import jax',)), 'tensorflow': (TENSORFLOW, TENSORFLOW_DEFAULT, TENSORFLOW_TRANSLATIONS, ('import tensorflow',)), 'sympy': (SYMPY, SYMPY_DEFAULT, {}, ('from sympy.functions import *', 'from sympy.matrices import *', 'from sympy import Integral, pi, oo, nan, zoo, E, I')), 'numexpr': (NUMEXPR, NUMEXPR_DEFAULT, NUMEXPR_TRANSLATIONS, ("import_module('numexpr')",))}

def _import(module, reload=False):
    if False:
        i = 10
        return i + 15
    '\n    Creates a global translation dictionary for module.\n\n    The argument module has to be one of the following strings: "math",\n    "mpmath", "numpy", "sympy", "tensorflow", "jax".\n    These dictionaries map names of Python functions to their equivalent in\n    other modules.\n    '
    try:
        (namespace, namespace_default, translations, import_commands) = MODULES[module]
    except KeyError:
        raise NameError("'%s' module cannot be used for lambdification" % module)
    if namespace != namespace_default:
        if reload:
            namespace.clear()
            namespace.update(namespace_default)
        else:
            return
    for import_command in import_commands:
        if import_command.startswith('import_module'):
            module = eval(import_command)
            if module is not None:
                namespace.update(module.__dict__)
                continue
        else:
            try:
                exec(import_command, {}, namespace)
                continue
            except ImportError:
                pass
        raise ImportError("Cannot import '%s' with '%s' command" % (module, import_command))
    for (sympyname, translation) in translations.items():
        namespace[sympyname] = namespace[translation]
    if 'Abs' not in namespace:
        namespace['Abs'] = abs
_lambdify_generated_counter = 1

@doctest_depends_on(modules=('numpy', 'scipy', 'tensorflow'), python_version=(3,))
def lambdify(args, expr, modules=None, printer=None, use_imps=True, dummify=False, cse=False, docstring_limit=1000):
    if False:
        while True:
            i = 10
    'Convert a SymPy expression into a function that allows for fast\n    numeric evaluation.\n\n    .. warning::\n       This function uses ``exec``, and thus should not be used on\n       unsanitized input.\n\n    .. deprecated:: 1.7\n       Passing a set for the *args* parameter is deprecated as sets are\n       unordered. Use an ordered iterable such as a list or tuple.\n\n    Explanation\n    ===========\n\n    For example, to convert the SymPy expression ``sin(x) + cos(x)`` to an\n    equivalent NumPy function that numerically evaluates it:\n\n    >>> from sympy import sin, cos, symbols, lambdify\n    >>> import numpy as np\n    >>> x = symbols(\'x\')\n    >>> expr = sin(x) + cos(x)\n    >>> expr\n    sin(x) + cos(x)\n    >>> f = lambdify(x, expr, \'numpy\')\n    >>> a = np.array([1, 2])\n    >>> f(a)\n    [1.38177329 0.49315059]\n\n    The primary purpose of this function is to provide a bridge from SymPy\n    expressions to numerical libraries such as NumPy, SciPy, NumExpr, mpmath,\n    and tensorflow. In general, SymPy functions do not work with objects from\n    other libraries, such as NumPy arrays, and functions from numeric\n    libraries like NumPy or mpmath do not work on SymPy expressions.\n    ``lambdify`` bridges the two by converting a SymPy expression to an\n    equivalent numeric function.\n\n    The basic workflow with ``lambdify`` is to first create a SymPy expression\n    representing whatever mathematical function you wish to evaluate. This\n    should be done using only SymPy functions and expressions. Then, use\n    ``lambdify`` to convert this to an equivalent function for numerical\n    evaluation. For instance, above we created ``expr`` using the SymPy symbol\n    ``x`` and SymPy functions ``sin`` and ``cos``, then converted it to an\n    equivalent NumPy function ``f``, and called it on a NumPy array ``a``.\n\n    Parameters\n    ==========\n\n    args : List[Symbol]\n        A variable or a list of variables whose nesting represents the\n        nesting of the arguments that will be passed to the function.\n\n        Variables can be symbols, undefined functions, or matrix symbols.\n\n        >>> from sympy import Eq\n        >>> from sympy.abc import x, y, z\n\n        The list of variables should match the structure of how the\n        arguments will be passed to the function. Simply enclose the\n        parameters as they will be passed in a list.\n\n        To call a function like ``f(x)`` then ``[x]``\n        should be the first argument to ``lambdify``; for this\n        case a single ``x`` can also be used:\n\n        >>> f = lambdify(x, x + 1)\n        >>> f(1)\n        2\n        >>> f = lambdify([x], x + 1)\n        >>> f(1)\n        2\n\n        To call a function like ``f(x, y)`` then ``[x, y]`` will\n        be the first argument of the ``lambdify``:\n\n        >>> f = lambdify([x, y], x + y)\n        >>> f(1, 1)\n        2\n\n        To call a function with a single 3-element tuple like\n        ``f((x, y, z))`` then ``[(x, y, z)]`` will be the first\n        argument of the ``lambdify``:\n\n        >>> f = lambdify([(x, y, z)], Eq(z**2, x**2 + y**2))\n        >>> f((3, 4, 5))\n        True\n\n        If two args will be passed and the first is a scalar but\n        the second is a tuple with two arguments then the items\n        in the list should match that structure:\n\n        >>> f = lambdify([x, (y, z)], x + y + z)\n        >>> f(1, (2, 3))\n        6\n\n    expr : Expr\n        An expression, list of expressions, or matrix to be evaluated.\n\n        Lists may be nested.\n        If the expression is a list, the output will also be a list.\n\n        >>> f = lambdify(x, [x, [x + 1, x + 2]])\n        >>> f(1)\n        [1, [2, 3]]\n\n        If it is a matrix, an array will be returned (for the NumPy module).\n\n        >>> from sympy import Matrix\n        >>> f = lambdify(x, Matrix([x, x + 1]))\n        >>> f(1)\n        [[1]\n        [2]]\n\n        Note that the argument order here (variables then expression) is used\n        to emulate the Python ``lambda`` keyword. ``lambdify(x, expr)`` works\n        (roughly) like ``lambda x: expr``\n        (see :ref:`lambdify-how-it-works` below).\n\n    modules : str, optional\n        Specifies the numeric library to use.\n\n        If not specified, *modules* defaults to:\n\n        - ``["scipy", "numpy"]`` if SciPy is installed\n        - ``["numpy"]`` if only NumPy is installed\n        - ``["math", "mpmath", "sympy"]`` if neither is installed.\n\n        That is, SymPy functions are replaced as far as possible by\n        either ``scipy`` or ``numpy`` functions if available, and Python\'s\n        standard library ``math``, or ``mpmath`` functions otherwise.\n\n        *modules* can be one of the following types:\n\n        - The strings ``"math"``, ``"mpmath"``, ``"numpy"``, ``"numexpr"``,\n          ``"scipy"``, ``"sympy"``, or ``"tensorflow"`` or ``"jax"``. This uses the\n          corresponding printer and namespace mapping for that module.\n        - A module (e.g., ``math``). This uses the global namespace of the\n          module. If the module is one of the above known modules, it will\n          also use the corresponding printer and namespace mapping\n          (i.e., ``modules=numpy`` is equivalent to ``modules="numpy"``).\n        - A dictionary that maps names of SymPy functions to arbitrary\n          functions\n          (e.g., ``{\'sin\': custom_sin}``).\n        - A list that contains a mix of the arguments above, with higher\n          priority given to entries appearing first\n          (e.g., to use the NumPy module but override the ``sin`` function\n          with a custom version, you can use\n          ``[{\'sin\': custom_sin}, \'numpy\']``).\n\n    dummify : bool, optional\n        Whether or not the variables in the provided expression that are not\n        valid Python identifiers are substituted with dummy symbols.\n\n        This allows for undefined functions like ``Function(\'f\')(t)`` to be\n        supplied as arguments. By default, the variables are only dummified\n        if they are not valid Python identifiers.\n\n        Set ``dummify=True`` to replace all arguments with dummy symbols\n        (if ``args`` is not a string) - for example, to ensure that the\n        arguments do not redefine any built-in names.\n\n    cse : bool, or callable, optional\n        Large expressions can be computed more efficiently when\n        common subexpressions are identified and precomputed before\n        being used multiple time. Finding the subexpressions will make\n        creation of the \'lambdify\' function slower, however.\n\n        When ``True``, ``sympy.simplify.cse`` is used, otherwise (the default)\n        the user may pass a function matching the ``cse`` signature.\n\n    docstring_limit : int or None\n        When lambdifying large expressions, a significant proportion of the time\n        spent inside ``lambdify`` is spent producing a string representation of\n        the expression for use in the automatically generated docstring of the\n        returned function. For expressions containing hundreds or more nodes the\n        resulting docstring often becomes so long and dense that it is difficult\n        to read. To reduce the runtime of lambdify, the rendering of the full\n        expression inside the docstring can be disabled.\n\n        When ``None``, the full expression is rendered in the docstring. When\n        ``0`` or a negative ``int``, an ellipsis is rendering in the docstring\n        instead of the expression. When a strictly positive ``int``, if the\n        number of nodes in the expression exceeds ``docstring_limit`` an\n        ellipsis is rendered in the docstring, otherwise a string representation\n        of the expression is rendered as normal. The default is ``1000``.\n\n    Examples\n    ========\n\n    >>> from sympy.utilities.lambdify import implemented_function\n    >>> from sympy import sqrt, sin, Matrix\n    >>> from sympy import Function\n    >>> from sympy.abc import w, x, y, z\n\n    >>> f = lambdify(x, x**2)\n    >>> f(2)\n    4\n    >>> f = lambdify((x, y, z), [z, y, x])\n    >>> f(1,2,3)\n    [3, 2, 1]\n    >>> f = lambdify(x, sqrt(x))\n    >>> f(4)\n    2.0\n    >>> f = lambdify((x, y), sin(x*y)**2)\n    >>> f(0, 5)\n    0.0\n    >>> row = lambdify((x, y), Matrix((x, x + y)).T, modules=\'sympy\')\n    >>> row(1, 2)\n    Matrix([[1, 3]])\n\n    ``lambdify`` can be used to translate SymPy expressions into mpmath\n    functions. This may be preferable to using ``evalf`` (which uses mpmath on\n    the backend) in some cases.\n\n    >>> f = lambdify(x, sin(x), \'mpmath\')\n    >>> f(1)\n    0.8414709848078965\n\n    Tuple arguments are handled and the lambdified function should\n    be called with the same type of arguments as were used to create\n    the function:\n\n    >>> f = lambdify((x, (y, z)), x + y)\n    >>> f(1, (2, 4))\n    3\n\n    The ``flatten`` function can be used to always work with flattened\n    arguments:\n\n    >>> from sympy.utilities.iterables import flatten\n    >>> args = w, (x, (y, z))\n    >>> vals = 1, (2, (3, 4))\n    >>> f = lambdify(flatten(args), w + x + y + z)\n    >>> f(*flatten(vals))\n    10\n\n    Functions present in ``expr`` can also carry their own numerical\n    implementations, in a callable attached to the ``_imp_`` attribute. This\n    can be used with undefined functions using the ``implemented_function``\n    factory:\n\n    >>> f = implemented_function(Function(\'f\'), lambda x: x+1)\n    >>> func = lambdify(x, f(x))\n    >>> func(4)\n    5\n\n    ``lambdify`` always prefers ``_imp_`` implementations to implementations\n    in other namespaces, unless the ``use_imps`` input parameter is False.\n\n    Usage with Tensorflow:\n\n    >>> import tensorflow as tf\n    >>> from sympy import Max, sin, lambdify\n    >>> from sympy.abc import x\n\n    >>> f = Max(x, sin(x))\n    >>> func = lambdify(x, f, \'tensorflow\')\n\n    After tensorflow v2, eager execution is enabled by default.\n    If you want to get the compatible result across tensorflow v1 and v2\n    as same as this tutorial, run this line.\n\n    >>> tf.compat.v1.enable_eager_execution()\n\n    If you have eager execution enabled, you can get the result out\n    immediately as you can use numpy.\n\n    If you pass tensorflow objects, you may get an ``EagerTensor``\n    object instead of value.\n\n    >>> result = func(tf.constant(1.0))\n    >>> print(result)\n    tf.Tensor(1.0, shape=(), dtype=float32)\n    >>> print(result.__class__)\n    <class \'tensorflow.python.framework.ops.EagerTensor\'>\n\n    You can use ``.numpy()`` to get the numpy value of the tensor.\n\n    >>> result.numpy()\n    1.0\n\n    >>> var = tf.Variable(2.0)\n    >>> result = func(var) # also works for tf.Variable and tf.Placeholder\n    >>> result.numpy()\n    2.0\n\n    And it works with any shape array.\n\n    >>> tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])\n    >>> result = func(tensor)\n    >>> result.numpy()\n    [[1. 2.]\n     [3. 4.]]\n\n    Notes\n    =====\n\n    - For functions involving large array calculations, numexpr can provide a\n      significant speedup over numpy. Please note that the available functions\n      for numexpr are more limited than numpy but can be expanded with\n      ``implemented_function`` and user defined subclasses of Function. If\n      specified, numexpr may be the only option in modules. The official list\n      of numexpr functions can be found at:\n      https://numexpr.readthedocs.io/en/latest/user_guide.html#supported-functions\n\n    - In the above examples, the generated functions can accept scalar\n      values or numpy arrays as arguments.  However, in some cases\n      the generated function relies on the input being a numpy array:\n\n      >>> import numpy\n      >>> from sympy import Piecewise\n      >>> from sympy.testing.pytest import ignore_warnings\n      >>> f = lambdify(x, Piecewise((x, x <= 1), (1/x, x > 1)), "numpy")\n\n      >>> with ignore_warnings(RuntimeWarning):\n      ...     f(numpy.array([-1, 0, 1, 2]))\n      [-1.   0.   1.   0.5]\n\n      >>> f(0)\n      Traceback (most recent call last):\n          ...\n      ZeroDivisionError: division by zero\n\n      In such cases, the input should be wrapped in a numpy array:\n\n      >>> with ignore_warnings(RuntimeWarning):\n      ...     float(f(numpy.array([0])))\n      0.0\n\n      Or if numpy functionality is not required another module can be used:\n\n      >>> f = lambdify(x, Piecewise((x, x <= 1), (1/x, x > 1)), "math")\n      >>> f(0)\n      0\n\n    .. _lambdify-how-it-works:\n\n    How it works\n    ============\n\n    When using this function, it helps a great deal to have an idea of what it\n    is doing. At its core, lambdify is nothing more than a namespace\n    translation, on top of a special printer that makes some corner cases work\n    properly.\n\n    To understand lambdify, first we must properly understand how Python\n    namespaces work. Say we had two files. One called ``sin_cos_sympy.py``,\n    with\n\n    .. code:: python\n\n        # sin_cos_sympy.py\n\n        from sympy.functions.elementary.trigonometric import (cos, sin)\n\n        def sin_cos(x):\n            return sin(x) + cos(x)\n\n\n    and one called ``sin_cos_numpy.py`` with\n\n    .. code:: python\n\n        # sin_cos_numpy.py\n\n        from numpy import sin, cos\n\n        def sin_cos(x):\n            return sin(x) + cos(x)\n\n    The two files define an identical function ``sin_cos``. However, in the\n    first file, ``sin`` and ``cos`` are defined as the SymPy ``sin`` and\n    ``cos``. In the second, they are defined as the NumPy versions.\n\n    If we were to import the first file and use the ``sin_cos`` function, we\n    would get something like\n\n    >>> from sin_cos_sympy import sin_cos # doctest: +SKIP\n    >>> sin_cos(1) # doctest: +SKIP\n    cos(1) + sin(1)\n\n    On the other hand, if we imported ``sin_cos`` from the second file, we\n    would get\n\n    >>> from sin_cos_numpy import sin_cos # doctest: +SKIP\n    >>> sin_cos(1) # doctest: +SKIP\n    1.38177329068\n\n    In the first case we got a symbolic output, because it used the symbolic\n    ``sin`` and ``cos`` functions from SymPy. In the second, we got a numeric\n    result, because ``sin_cos`` used the numeric ``sin`` and ``cos`` functions\n    from NumPy. But notice that the versions of ``sin`` and ``cos`` that were\n    used was not inherent to the ``sin_cos`` function definition. Both\n    ``sin_cos`` definitions are exactly the same. Rather, it was based on the\n    names defined at the module where the ``sin_cos`` function was defined.\n\n    The key point here is that when function in Python references a name that\n    is not defined in the function, that name is looked up in the "global"\n    namespace of the module where that function is defined.\n\n    Now, in Python, we can emulate this behavior without actually writing a\n    file to disk using the ``exec`` function. ``exec`` takes a string\n    containing a block of Python code, and a dictionary that should contain\n    the global variables of the module. It then executes the code "in" that\n    dictionary, as if it were the module globals. The following is equivalent\n    to the ``sin_cos`` defined in ``sin_cos_sympy.py``:\n\n    >>> import sympy\n    >>> module_dictionary = {\'sin\': sympy.sin, \'cos\': sympy.cos}\n    >>> exec(\'\'\'\n    ... def sin_cos(x):\n    ...     return sin(x) + cos(x)\n    ... \'\'\', module_dictionary)\n    >>> sin_cos = module_dictionary[\'sin_cos\']\n    >>> sin_cos(1)\n    cos(1) + sin(1)\n\n    and similarly with ``sin_cos_numpy``:\n\n    >>> import numpy\n    >>> module_dictionary = {\'sin\': numpy.sin, \'cos\': numpy.cos}\n    >>> exec(\'\'\'\n    ... def sin_cos(x):\n    ...     return sin(x) + cos(x)\n    ... \'\'\', module_dictionary)\n    >>> sin_cos = module_dictionary[\'sin_cos\']\n    >>> sin_cos(1)\n    1.38177329068\n\n    So now we can get an idea of how ``lambdify`` works. The name "lambdify"\n    comes from the fact that we can think of something like ``lambdify(x,\n    sin(x) + cos(x), \'numpy\')`` as ``lambda x: sin(x) + cos(x)``, where\n    ``sin`` and ``cos`` come from the ``numpy`` namespace. This is also why\n    the symbols argument is first in ``lambdify``, as opposed to most SymPy\n    functions where it comes after the expression: to better mimic the\n    ``lambda`` keyword.\n\n    ``lambdify`` takes the input expression (like ``sin(x) + cos(x)``) and\n\n    1. Converts it to a string\n    2. Creates a module globals dictionary based on the modules that are\n       passed in (by default, it uses the NumPy module)\n    3. Creates the string ``"def func({vars}): return {expr}"``, where ``{vars}`` is the\n       list of variables separated by commas, and ``{expr}`` is the string\n       created in step 1., then ``exec``s that string with the module globals\n       namespace and returns ``func``.\n\n    In fact, functions returned by ``lambdify`` support inspection. So you can\n    see exactly how they are defined by using ``inspect.getsource``, or ``??`` if you\n    are using IPython or the Jupyter notebook.\n\n    >>> f = lambdify(x, sin(x) + cos(x))\n    >>> import inspect\n    >>> print(inspect.getsource(f))\n    def _lambdifygenerated(x):\n        return sin(x) + cos(x)\n\n    This shows us the source code of the function, but not the namespace it\n    was defined in. We can inspect that by looking at the ``__globals__``\n    attribute of ``f``:\n\n    >>> f.__globals__[\'sin\']\n    <ufunc \'sin\'>\n    >>> f.__globals__[\'cos\']\n    <ufunc \'cos\'>\n    >>> f.__globals__[\'sin\'] is numpy.sin\n    True\n\n    This shows us that ``sin`` and ``cos`` in the namespace of ``f`` will be\n    ``numpy.sin`` and ``numpy.cos``.\n\n    Note that there are some convenience layers in each of these steps, but at\n    the core, this is how ``lambdify`` works. Step 1 is done using the\n    ``LambdaPrinter`` printers defined in the printing module (see\n    :mod:`sympy.printing.lambdarepr`). This allows different SymPy expressions\n    to define how they should be converted to a string for different modules.\n    You can change which printer ``lambdify`` uses by passing a custom printer\n    in to the ``printer`` argument.\n\n    Step 2 is augmented by certain translations. There are default\n    translations for each module, but you can provide your own by passing a\n    list to the ``modules`` argument. For instance,\n\n    >>> def mysin(x):\n    ...     print(\'taking the sin of\', x)\n    ...     return numpy.sin(x)\n    ...\n    >>> f = lambdify(x, sin(x), [{\'sin\': mysin}, \'numpy\'])\n    >>> f(1)\n    taking the sin of 1\n    0.8414709848078965\n\n    The globals dictionary is generated from the list by merging the\n    dictionary ``{\'sin\': mysin}`` and the module dictionary for NumPy. The\n    merging is done so that earlier items take precedence, which is why\n    ``mysin`` is used above instead of ``numpy.sin``.\n\n    If you want to modify the way ``lambdify`` works for a given function, it\n    is usually easiest to do so by modifying the globals dictionary as such.\n    In more complicated cases, it may be necessary to create and pass in a\n    custom printer.\n\n    Finally, step 3 is augmented with certain convenience operations, such as\n    the addition of a docstring.\n\n    Understanding how ``lambdify`` works can make it easier to avoid certain\n    gotchas when using it. For instance, a common mistake is to create a\n    lambdified function for one module (say, NumPy), and pass it objects from\n    another (say, a SymPy expression).\n\n    For instance, say we create\n\n    >>> from sympy.abc import x\n    >>> f = lambdify(x, x + 1, \'numpy\')\n\n    Now if we pass in a NumPy array, we get that array plus 1\n\n    >>> import numpy\n    >>> a = numpy.array([1, 2])\n    >>> f(a)\n    [2 3]\n\n    But what happens if you make the mistake of passing in a SymPy expression\n    instead of a NumPy array:\n\n    >>> f(x + 1)\n    x + 2\n\n    This worked, but it was only by accident. Now take a different lambdified\n    function:\n\n    >>> from sympy import sin\n    >>> g = lambdify(x, x + sin(x), \'numpy\')\n\n    This works as expected on NumPy arrays:\n\n    >>> g(a)\n    [1.84147098 2.90929743]\n\n    But if we try to pass in a SymPy expression, it fails\n\n    >>> g(x + 1)\n    Traceback (most recent call last):\n    ...\n    TypeError: loop of ufunc does not support argument 0 of type Add which has\n               no callable sin method\n\n    Now, let\'s look at what happened. The reason this fails is that ``g``\n    calls ``numpy.sin`` on the input expression, and ``numpy.sin`` does not\n    know how to operate on a SymPy object. **As a general rule, NumPy\n    functions do not know how to operate on SymPy expressions, and SymPy\n    functions do not know how to operate on NumPy arrays. This is why lambdify\n    exists: to provide a bridge between SymPy and NumPy.**\n\n    However, why is it that ``f`` did work? That\'s because ``f`` does not call\n    any functions, it only adds 1. So the resulting function that is created,\n    ``def _lambdifygenerated(x): return x + 1`` does not depend on the globals\n    namespace it is defined in. Thus it works, but only by accident. A future\n    version of ``lambdify`` may remove this behavior.\n\n    Be aware that certain implementation details described here may change in\n    future versions of SymPy. The API of passing in custom modules and\n    printers will not change, but the details of how a lambda function is\n    created may change. However, the basic idea will remain the same, and\n    understanding it will be helpful to understanding the behavior of\n    lambdify.\n\n    **In general: you should create lambdified functions for one module (say,\n    NumPy), and only pass it input types that are compatible with that module\n    (say, NumPy arrays).** Remember that by default, if the ``module``\n    argument is not provided, ``lambdify`` creates functions using the NumPy\n    and SciPy namespaces.\n    '
    from sympy.core.symbol import Symbol
    from sympy.core.expr import Expr
    if modules is None:
        try:
            _import('scipy')
        except ImportError:
            try:
                _import('numpy')
            except ImportError:
                modules = ['math', 'mpmath', 'sympy']
            else:
                modules = ['numpy']
        else:
            modules = ['numpy', 'scipy']
    namespaces = []
    if use_imps:
        namespaces.append(_imp_namespace(expr))
    if isinstance(modules, (dict, str)) or not hasattr(modules, '__iter__'):
        namespaces.append(modules)
    else:
        if _module_present('numexpr', modules) and len(modules) > 1:
            raise TypeError("numexpr must be the only item in 'modules'")
        namespaces += list(modules)
    namespace = {}
    for m in namespaces[::-1]:
        buf = _get_namespace(m)
        namespace.update(buf)
    if hasattr(expr, 'atoms'):
        syms = expr.atoms(Symbol)
        for term in syms:
            namespace.update({str(term): term})
    if printer is None:
        if _module_present('mpmath', namespaces):
            from sympy.printing.pycode import MpmathPrinter as Printer
        elif _module_present('scipy', namespaces):
            from sympy.printing.numpy import SciPyPrinter as Printer
        elif _module_present('numpy', namespaces):
            from sympy.printing.numpy import NumPyPrinter as Printer
        elif _module_present('cupy', namespaces):
            from sympy.printing.numpy import CuPyPrinter as Printer
        elif _module_present('jax', namespaces):
            from sympy.printing.numpy import JaxPrinter as Printer
        elif _module_present('numexpr', namespaces):
            from sympy.printing.lambdarepr import NumExprPrinter as Printer
        elif _module_present('tensorflow', namespaces):
            from sympy.printing.tensorflow import TensorflowPrinter as Printer
        elif _module_present('sympy', namespaces):
            from sympy.printing.pycode import SymPyPrinter as Printer
        else:
            from sympy.printing.pycode import PythonCodePrinter as Printer
        user_functions = {}
        for m in namespaces[::-1]:
            if isinstance(m, dict):
                for k in m:
                    user_functions[k] = k
        printer = Printer({'fully_qualified_modules': False, 'inline': True, 'allow_unknown_functions': True, 'user_functions': user_functions})
    if isinstance(args, set):
        sympy_deprecation_warning('\nPassing the function arguments to lambdify() as a set is deprecated. This\nleads to unpredictable results since sets are unordered. Instead, use a list\nor tuple for the function arguments.\n            ', deprecated_since_version='1.6.3', active_deprecations_target='deprecated-lambdify-arguments-set')
    iterable_args = (args,) if isinstance(args, Expr) else args
    names = []
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    for (n, var) in enumerate(iterable_args):
        if hasattr(var, 'name'):
            names.append(var.name)
        else:
            name_list = [var_name for (var_name, var_val) in callers_local_vars if var_val is var]
            if len(name_list) == 1:
                names.append(name_list[0])
            else:
                names.append('arg_' + str(n))
    funcname = '_lambdifygenerated'
    if _module_present('tensorflow', namespaces):
        funcprinter = _TensorflowEvaluatorPrinter(printer, dummify)
    else:
        funcprinter = _EvaluatorPrinter(printer, dummify)
    if cse == True:
        from sympy.simplify.cse_main import cse as _cse
        (cses, _expr) = _cse(expr, list=False)
    elif callable(cse):
        (cses, _expr) = cse(expr)
    else:
        (cses, _expr) = ((), expr)
    funcstr = funcprinter.doprint(funcname, iterable_args, _expr, cses=cses)
    imp_mod_lines = []
    for (mod, keys) in (getattr(printer, 'module_imports', None) or {}).items():
        for k in keys:
            if k not in namespace:
                ln = 'from %s import %s' % (mod, k)
                try:
                    exec(ln, {}, namespace)
                except ImportError:
                    ln = '%s = %s.%s' % (k, mod, k)
                    exec(ln, {}, namespace)
                imp_mod_lines.append(ln)
    namespace.update({'builtins': builtins, 'range': range})
    funclocals = {}
    global _lambdify_generated_counter
    filename = '<lambdifygenerated-%s>' % _lambdify_generated_counter
    _lambdify_generated_counter += 1
    c = compile(funcstr, filename, 'exec')
    exec(c, namespace, funclocals)
    linecache.cache[filename] = (len(funcstr), None, funcstr.splitlines(True), filename)
    func = funclocals[funcname]
    sig = 'func({})'.format(', '.join((str(i) for i in names)))
    sig = textwrap.fill(sig, subsequent_indent=' ' * 8)
    if _too_large_for_docstring(expr, docstring_limit):
        expr_str = "EXPRESSION REDACTED DUE TO LENGTH, (see lambdify's `docstring_limit`)"
        src_str = "SOURCE CODE REDACTED DUE TO LENGTH, (see lambdify's `docstring_limit`)"
    else:
        expr_str = str(expr)
        if len(expr_str) > 78:
            expr_str = textwrap.wrap(expr_str, 75)[0] + '...'
        src_str = funcstr
    func.__doc__ = 'Created with lambdify. Signature:\n\n{sig}\n\nExpression:\n\n{expr}\n\nSource code:\n\n{src}\n\nImported modules:\n\n{imp_mods}'.format(sig=sig, expr=expr_str, src=src_str, imp_mods='\n'.join(imp_mod_lines))
    return func

def _module_present(modname, modlist):
    if False:
        i = 10
        return i + 15
    if modname in modlist:
        return True
    for m in modlist:
        if hasattr(m, '__name__') and m.__name__ == modname:
            return True
    return False

def _get_namespace(m):
    if False:
        for i in range(10):
            print('nop')
    '\n    This is used by _lambdify to parse its arguments.\n    '
    if isinstance(m, str):
        _import(m)
        return MODULES[m][0]
    elif isinstance(m, dict):
        return m
    elif hasattr(m, '__dict__'):
        return m.__dict__
    else:
        raise TypeError('Argument must be either a string, dict or module but it is: %s' % m)

def _recursive_to_string(doprint, arg):
    if False:
        print('Hello World!')
    'Functions in lambdify accept both SymPy types and non-SymPy types such as python\n    lists and tuples. This method ensures that we only call the doprint method of the\n    printer with SymPy types (so that the printer safely can use SymPy-methods).'
    from sympy.matrices.common import MatrixOperations
    from sympy.core.basic import Basic
    if isinstance(arg, (Basic, MatrixOperations)):
        return doprint(arg)
    elif iterable(arg):
        if isinstance(arg, list):
            (left, right) = ('[', ']')
        elif isinstance(arg, tuple):
            (left, right) = ('(', ',)')
        else:
            raise NotImplementedError('unhandled type: %s, %s' % (type(arg), arg))
        return left + ', '.join((_recursive_to_string(doprint, e) for e in arg)) + right
    elif isinstance(arg, str):
        return arg
    else:
        return doprint(arg)

def lambdastr(args, expr, printer=None, dummify=None):
    if False:
        return 10
    "\n    Returns a string that can be evaluated to a lambda function.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import x, y, z\n    >>> from sympy.utilities.lambdify import lambdastr\n    >>> lambdastr(x, x**2)\n    'lambda x: (x**2)'\n    >>> lambdastr((x,y,z), [z,y,x])\n    'lambda x,y,z: ([z, y, x])'\n\n    Although tuples may not appear as arguments to lambda in Python 3,\n    lambdastr will create a lambda function that will unpack the original\n    arguments so that nested arguments can be handled:\n\n    >>> lambdastr((x, (y, z)), x + y)\n    'lambda _0,_1: (lambda x,y,z: (x + y))(_0,_1[0],_1[1])'\n    "
    from sympy.matrices import DeferredVector
    from sympy.core.basic import Basic
    from sympy.core.function import Derivative, Function
    from sympy.core.symbol import Dummy, Symbol
    from sympy.core.sympify import sympify
    if printer is not None:
        if inspect.isfunction(printer):
            lambdarepr = printer
        elif inspect.isclass(printer):
            lambdarepr = lambda expr: printer().doprint(expr)
        else:
            lambdarepr = lambda expr: printer.doprint(expr)
    else:
        from sympy.printing.lambdarepr import lambdarepr

    def sub_args(args, dummies_dict):
        if False:
            return 10
        if isinstance(args, str):
            return args
        elif isinstance(args, DeferredVector):
            return str(args)
        elif iterable(args):
            dummies = flatten([sub_args(a, dummies_dict) for a in args])
            return ','.join((str(a) for a in dummies))
        elif isinstance(args, (Function, Symbol, Derivative)):
            dummies = Dummy()
            dummies_dict.update({args: dummies})
            return str(dummies)
        else:
            return str(args)

    def sub_expr(expr, dummies_dict):
        if False:
            print('Hello World!')
        expr = sympify(expr)
        if isinstance(expr, Basic):
            expr = expr.xreplace(dummies_dict)
        elif isinstance(expr, list):
            expr = [sub_expr(a, dummies_dict) for a in expr]
        return expr

    def isiter(l):
        if False:
            return 10
        return iterable(l, exclude=(str, DeferredVector, NotIterable))

    def flat_indexes(iterable):
        if False:
            for i in range(10):
                print('nop')
        n = 0
        for el in iterable:
            if isiter(el):
                for ndeep in flat_indexes(el):
                    yield ((n,) + ndeep)
            else:
                yield (n,)
            n += 1
    if dummify is None:
        dummify = any((isinstance(a, Basic) and a.atoms(Function, Derivative) for a in (args if isiter(args) else [args])))
    if isiter(args) and any((isiter(i) for i in args)):
        dum_args = [str(Dummy(str(i))) for i in range(len(args))]
        indexed_args = ','.join([dum_args[ind[0]] + ''.join(['[%s]' % k for k in ind[1:]]) for ind in flat_indexes(args)])
        lstr = lambdastr(flatten(args), expr, printer=printer, dummify=dummify)
        return 'lambda %s: (%s)(%s)' % (','.join(dum_args), lstr, indexed_args)
    dummies_dict = {}
    if dummify:
        args = sub_args(args, dummies_dict)
    elif isinstance(args, str):
        pass
    elif iterable(args, exclude=DeferredVector):
        args = ','.join((str(a) for a in args))
    if dummify:
        if isinstance(expr, str):
            pass
        else:
            expr = sub_expr(expr, dummies_dict)
    expr = _recursive_to_string(lambdarepr, expr)
    return 'lambda %s: (%s)' % (args, expr)

class _EvaluatorPrinter:

    def __init__(self, printer=None, dummify=False):
        if False:
            i = 10
            return i + 15
        self._dummify = dummify
        from sympy.printing.lambdarepr import LambdaPrinter
        if printer is None:
            printer = LambdaPrinter()
        if inspect.isfunction(printer):
            self._exprrepr = printer
        else:
            if inspect.isclass(printer):
                printer = printer()
            self._exprrepr = printer.doprint
        self._argrepr = LambdaPrinter().doprint

    def doprint(self, funcname, args, expr, *, cses=()):
        if False:
            i = 10
            return i + 15
        '\n        Returns the function definition code as a string.\n        '
        from sympy.core.symbol import Dummy
        funcbody = []
        if not iterable(args):
            args = [args]
        if cses:
            (subvars, subexprs) = zip(*cses)
            exprs = [expr] + list(subexprs)
            (argstrs, exprs) = self._preprocess(args, exprs)
            (expr, subexprs) = (exprs[0], exprs[1:])
            cses = zip(subvars, subexprs)
        else:
            (argstrs, expr) = self._preprocess(args, expr)
        funcargs = []
        unpackings = []
        for argstr in argstrs:
            if iterable(argstr):
                funcargs.append(self._argrepr(Dummy()))
                unpackings.extend(self._print_unpacking(argstr, funcargs[-1]))
            else:
                funcargs.append(argstr)
        funcsig = 'def {}({}):'.format(funcname, ', '.join(funcargs))
        funcbody.extend(self._print_funcargwrapping(funcargs))
        funcbody.extend(unpackings)
        for (s, e) in cses:
            if e is None:
                funcbody.append('del {}'.format(self._exprrepr(s)))
            else:
                funcbody.append('{} = {}'.format(self._exprrepr(s), self._exprrepr(e)))
        str_expr = _recursive_to_string(self._exprrepr, expr)
        if '\n' in str_expr:
            str_expr = '({})'.format(str_expr)
        funcbody.append('return {}'.format(str_expr))
        funclines = [funcsig]
        funclines.extend(['    ' + line for line in funcbody])
        return '\n'.join(funclines) + '\n'

    @classmethod
    def _is_safe_ident(cls, ident):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(ident, str) and ident.isidentifier() and (not keyword.iskeyword(ident))

    def _preprocess(self, args, expr):
        if False:
            i = 10
            return i + 15
        'Preprocess args, expr to replace arguments that do not map\n        to valid Python identifiers.\n\n        Returns string form of args, and updated expr.\n        '
        from sympy.core.basic import Basic
        from sympy.core.sorting import ordered
        from sympy.core.function import Derivative, Function
        from sympy.core.symbol import Dummy, uniquely_named_symbol
        from sympy.matrices import DeferredVector
        from sympy.core.expr import Expr
        dummify = self._dummify or any((isinstance(arg, Dummy) for arg in flatten(args)))
        argstrs = [None] * len(args)
        for (arg, i) in reversed(list(ordered(zip(args, range(len(args)))))):
            if iterable(arg):
                (s, expr) = self._preprocess(arg, expr)
            elif isinstance(arg, DeferredVector):
                s = str(arg)
            elif isinstance(arg, Basic) and arg.is_symbol:
                s = self._argrepr(arg)
                if dummify or not self._is_safe_ident(s):
                    dummy = Dummy()
                    if isinstance(expr, Expr):
                        dummy = uniquely_named_symbol(dummy.name, expr, modify=lambda s: '_' + s)
                    s = self._argrepr(dummy)
                    expr = self._subexpr(expr, {arg: dummy})
            elif dummify or isinstance(arg, (Function, Derivative)):
                dummy = Dummy()
                s = self._argrepr(dummy)
                expr = self._subexpr(expr, {arg: dummy})
            else:
                s = str(arg)
            argstrs[i] = s
        return (argstrs, expr)

    def _subexpr(self, expr, dummies_dict):
        if False:
            for i in range(10):
                print('nop')
        from sympy.matrices import DeferredVector
        from sympy.core.sympify import sympify
        expr = sympify(expr)
        xreplace = getattr(expr, 'xreplace', None)
        if xreplace is not None:
            expr = xreplace(dummies_dict)
        elif isinstance(expr, DeferredVector):
            pass
        elif isinstance(expr, dict):
            k = [self._subexpr(sympify(a), dummies_dict) for a in expr.keys()]
            v = [self._subexpr(sympify(a), dummies_dict) for a in expr.values()]
            expr = dict(zip(k, v))
        elif isinstance(expr, tuple):
            expr = tuple((self._subexpr(sympify(a), dummies_dict) for a in expr))
        elif isinstance(expr, list):
            expr = [self._subexpr(sympify(a), dummies_dict) for a in expr]
        return expr

    def _print_funcargwrapping(self, args):
        if False:
            return 10
        'Generate argument wrapping code.\n\n        args is the argument list of the generated function (strings).\n\n        Return value is a list of lines of code that will be inserted  at\n        the beginning of the function definition.\n        '
        return []

    def _print_unpacking(self, unpackto, arg):
        if False:
            print('Hello World!')
        'Generate argument unpacking code.\n\n        arg is the function argument to be unpacked (a string), and\n        unpackto is a list or nested lists of the variable names (strings) to\n        unpack to.\n        '

        def unpack_lhs(lvalues):
            if False:
                i = 10
                return i + 15
            return '[{}]'.format(', '.join((unpack_lhs(val) if iterable(val) else val for val in lvalues)))
        return ['{} = {}'.format(unpack_lhs(unpackto), arg)]

class _TensorflowEvaluatorPrinter(_EvaluatorPrinter):

    def _print_unpacking(self, lvalues, rvalue):
        if False:
            while True:
                i = 10
        'Generate argument unpacking code.\n\n        This method is used when the input value is not interable,\n        but can be indexed (see issue #14655).\n        '

        def flat_indexes(elems):
            if False:
                return 10
            n = 0
            for el in elems:
                if iterable(el):
                    for ndeep in flat_indexes(el):
                        yield ((n,) + ndeep)
                else:
                    yield (n,)
                n += 1
        indexed = ', '.join(('{}[{}]'.format(rvalue, ']['.join(map(str, ind))) for ind in flat_indexes(lvalues)))
        return ['[{}] = [{}]'.format(', '.join(flatten(lvalues)), indexed)]

def _imp_namespace(expr, namespace=None):
    if False:
        print('Hello World!')
    " Return namespace dict with function implementations\n\n    We need to search for functions in anything that can be thrown at\n    us - that is - anything that could be passed as ``expr``.  Examples\n    include SymPy expressions, as well as tuples, lists and dicts that may\n    contain SymPy expressions.\n\n    Parameters\n    ----------\n    expr : object\n       Something passed to lambdify, that will generate valid code from\n       ``str(expr)``.\n    namespace : None or mapping\n       Namespace to fill.  None results in new empty dict\n\n    Returns\n    -------\n    namespace : dict\n       dict with keys of implemented function names within ``expr`` and\n       corresponding values being the numerical implementation of\n       function\n\n    Examples\n    ========\n\n    >>> from sympy.abc import x\n    >>> from sympy.utilities.lambdify import implemented_function, _imp_namespace\n    >>> from sympy import Function\n    >>> f = implemented_function(Function('f'), lambda x: x+1)\n    >>> g = implemented_function(Function('g'), lambda x: x*10)\n    >>> namespace = _imp_namespace(f(g(x)))\n    >>> sorted(namespace.keys())\n    ['f', 'g']\n    "
    from sympy.core.function import FunctionClass
    if namespace is None:
        namespace = {}
    if is_sequence(expr):
        for arg in expr:
            _imp_namespace(arg, namespace)
        return namespace
    elif isinstance(expr, dict):
        for (key, val) in expr.items():
            _imp_namespace(key, namespace)
            _imp_namespace(val, namespace)
        return namespace
    func = getattr(expr, 'func', None)
    if isinstance(func, FunctionClass):
        imp = getattr(func, '_imp_', None)
        if imp is not None:
            name = expr.func.__name__
            if name in namespace and namespace[name] != imp:
                raise ValueError('We found more than one implementation with name "%s"' % name)
            namespace[name] = imp
    if hasattr(expr, 'args'):
        for arg in expr.args:
            _imp_namespace(arg, namespace)
    return namespace

def implemented_function(symfunc, implementation):
    if False:
        return 10
    " Add numerical ``implementation`` to function ``symfunc``.\n\n    ``symfunc`` can be an ``UndefinedFunction`` instance, or a name string.\n    In the latter case we create an ``UndefinedFunction`` instance with that\n    name.\n\n    Be aware that this is a quick workaround, not a general method to create\n    special symbolic functions. If you want to create a symbolic function to be\n    used by all the machinery of SymPy you should subclass the ``Function``\n    class.\n\n    Parameters\n    ----------\n    symfunc : ``str`` or ``UndefinedFunction`` instance\n       If ``str``, then create new ``UndefinedFunction`` with this as\n       name.  If ``symfunc`` is an Undefined function, create a new function\n       with the same name and the implemented function attached.\n    implementation : callable\n       numerical implementation to be called by ``evalf()`` or ``lambdify``\n\n    Returns\n    -------\n    afunc : sympy.FunctionClass instance\n       function with attached implementation\n\n    Examples\n    ========\n\n    >>> from sympy.abc import x\n    >>> from sympy.utilities.lambdify import implemented_function\n    >>> from sympy import lambdify\n    >>> f = implemented_function('f', lambda x: x+1)\n    >>> lam_f = lambdify(x, f(x))\n    >>> lam_f(4)\n    5\n    "
    from sympy.core.function import UndefinedFunction
    kwargs = {}
    if isinstance(symfunc, UndefinedFunction):
        kwargs = symfunc._kwargs
        symfunc = symfunc.__name__
    if isinstance(symfunc, str):
        symfunc = UndefinedFunction(symfunc, _imp_=staticmethod(implementation), **kwargs)
    elif not isinstance(symfunc, UndefinedFunction):
        raise ValueError(filldedent('\n            symfunc should be either a string or\n            an UndefinedFunction instance.'))
    return symfunc

def _too_large_for_docstring(expr, limit):
    if False:
        i = 10
        return i + 15
    'Decide whether an ``Expr`` is too large to be fully rendered in a\n    ``lambdify`` docstring.\n\n    This is a fast alternative to ``count_ops``, which can become prohibitively\n    slow for large expressions, because in this instance we only care whether\n    ``limit`` is exceeded rather than counting the exact number of nodes in the\n    expression.\n\n    Parameters\n    ==========\n    expr : ``Expr``, (nested) ``list`` of ``Expr``, or ``Matrix``\n        The same objects that can be passed to the ``expr`` argument of\n        ``lambdify``.\n    limit : ``int`` or ``None``\n        The threshold above which an expression contains too many nodes to be\n        usefully rendered in the docstring. If ``None`` then there is no limit.\n\n    Returns\n    =======\n    bool\n        ``True`` if the number of nodes in the expression exceeds the limit,\n        ``False`` otherwise.\n\n    Examples\n    ========\n\n    >>> from sympy.abc import x, y, z\n    >>> from sympy.utilities.lambdify import _too_large_for_docstring\n    >>> expr = x\n    >>> _too_large_for_docstring(expr, None)\n    False\n    >>> _too_large_for_docstring(expr, 100)\n    False\n    >>> _too_large_for_docstring(expr, 1)\n    False\n    >>> _too_large_for_docstring(expr, 0)\n    True\n    >>> _too_large_for_docstring(expr, -1)\n    True\n\n    Does this split it?\n\n    >>> expr = [x, y, z]\n    >>> _too_large_for_docstring(expr, None)\n    False\n    >>> _too_large_for_docstring(expr, 100)\n    False\n    >>> _too_large_for_docstring(expr, 1)\n    True\n    >>> _too_large_for_docstring(expr, 0)\n    True\n    >>> _too_large_for_docstring(expr, -1)\n    True\n\n    >>> expr = [x, [y], z, [[x+y], [x*y*z, [x+y+z]]]]\n    >>> _too_large_for_docstring(expr, None)\n    False\n    >>> _too_large_for_docstring(expr, 100)\n    False\n    >>> _too_large_for_docstring(expr, 1)\n    True\n    >>> _too_large_for_docstring(expr, 0)\n    True\n    >>> _too_large_for_docstring(expr, -1)\n    True\n\n    >>> expr = ((x + y + z)**5).expand()\n    >>> _too_large_for_docstring(expr, None)\n    False\n    >>> _too_large_for_docstring(expr, 100)\n    True\n    >>> _too_large_for_docstring(expr, 1)\n    True\n    >>> _too_large_for_docstring(expr, 0)\n    True\n    >>> _too_large_for_docstring(expr, -1)\n    True\n\n    >>> from sympy import Matrix\n    >>> expr = Matrix([[(x + y + z), ((x + y + z)**2).expand(),\n    ...                 ((x + y + z)**3).expand(), ((x + y + z)**4).expand()]])\n    >>> _too_large_for_docstring(expr, None)\n    False\n    >>> _too_large_for_docstring(expr, 1000)\n    False\n    >>> _too_large_for_docstring(expr, 100)\n    True\n    >>> _too_large_for_docstring(expr, 1)\n    True\n    >>> _too_large_for_docstring(expr, 0)\n    True\n    >>> _too_large_for_docstring(expr, -1)\n    True\n\n    '
    from sympy.core.traversal import postorder_traversal
    if limit is None:
        return False
    i = 0
    for _ in postorder_traversal(expr):
        i += 1
        if i > limit:
            return True
    return False