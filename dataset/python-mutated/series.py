from collections.abc import Callable
from sympy.calculus.util import continuous_domain
from sympy.concrete import Sum, Product
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import arity
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.functions import atan2, zeta, frac, ceiling, floor, im
from sympy.core.relational import Equality, GreaterThan, LessThan, Relational, Ne
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.logic.boolalg import BooleanFunction
from sympy.plotting.utils import _get_free_symbols, extract_solution
from sympy.printing.latex import latex
from sympy.printing.pycode import PythonCodePrinter
from sympy.printing.precedence import precedence
from sympy.sets.sets import Set, Interval, Union
from sympy.simplify.simplify import nsimplify
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.lambdify import lambdify
from .intervalmath import interval
import warnings

class IntervalMathPrinter(PythonCodePrinter):
    """A printer to be used inside `plot_implicit` when `adaptive=True`,
    in which case the interval arithmetic module is going to be used, which
    requires the following edits.
    """

    def _print_And(self, expr):
        if False:
            for i in range(10):
                print('nop')
        PREC = precedence(expr)
        return ' & '.join((self.parenthesize(a, PREC) for a in sorted(expr.args, key=default_sort_key)))

    def _print_Or(self, expr):
        if False:
            for i in range(10):
                print('nop')
        PREC = precedence(expr)
        return ' | '.join((self.parenthesize(a, PREC) for a in sorted(expr.args, key=default_sort_key)))

def _uniform_eval(f1, f2, *args, modules=None, force_real_eval=False, has_sum=False):
    if False:
        while True:
            i = 10
    '\n    Note: this is an experimental function, as such it is prone to changes.\n    Please, do not use it in your code.\n    '
    np = import_module('numpy')

    def wrapper_func(func, *args):
        if False:
            for i in range(10):
                print('nop')
        try:
            return complex(func(*args))
        except (ZeroDivisionError, OverflowError):
            return complex(np.nan, np.nan)
    wrapper_func = np.vectorize(wrapper_func, otypes=[complex])

    def _eval_with_sympy(err=None):
        if False:
            return 10
        if f2 is None:
            msg = 'Impossible to evaluate the provided numerical function'
            if err is None:
                msg += '.'
            else:
                msg += 'because the following exception was raised:\n'
                '{}: {}'.format(type(err).__name__, err)
            raise RuntimeError(msg)
        if err:
            warnings.warn('The evaluation with %s failed.\n' % ('NumPy/SciPy' if not modules else modules) + '{}: {}\n'.format(type(err).__name__, err) + 'Trying to evaluate the expression with Sympy, but it might be a slow operation.')
        return wrapper_func(f2, *args)
    if modules == 'sympy':
        return _eval_with_sympy()
    try:
        return wrapper_func(f1, *args)
    except Exception as err:
        return _eval_with_sympy(err)

def _adaptive_eval(f, x):
    if False:
        print('Hello World!')
    'Evaluate f(x) with an adaptive algorithm. Post-process the result.\n    If a symbolic expression is evaluated with SymPy, it might returns\n    another symbolic expression, containing additions, ...\n    Force evaluation to a float.\n\n    Parameters\n    ==========\n    f : callable\n    x : float\n    '
    np = import_module('numpy')
    y = f(x)
    if isinstance(y, Expr) and (not y.is_Number):
        y = y.evalf()
    y = complex(y)
    if y.imag > 1e-08:
        return np.nan
    return y.real

def _get_wrapper_for_expr(ret):
    if False:
        for i in range(10):
            print('nop')
    wrapper = '%s'
    if ret == 'real':
        wrapper = 're(%s)'
    elif ret == 'imag':
        wrapper = 'im(%s)'
    elif ret == 'abs':
        wrapper = 'abs(%s)'
    elif ret == 'arg':
        wrapper = 'arg(%s)'
    return wrapper

class BaseSeries:
    """Base class for the data objects containing stuff to be plotted.

    Notes
    =====

    The backend should check if it supports the data series that is given.
    (e.g. TextBackend supports only LineOver1DRangeSeries).
    It is the backend responsibility to know how to use the class of
    data series that is given.

    Some data series classes are grouped (using a class attribute like is_2Dline)
    according to the api they present (based only on convention). The backend is
    not obliged to use that api (e.g. LineOver1DRangeSeries belongs to the
    is_2Dline group and presents the get_points method, but the
    TextBackend does not use the get_points method).

    BaseSeries
    """
    is_2Dline = False
    is_3Dline = False
    is_3Dsurface = False
    is_contour = False
    is_implicit = False
    is_interactive = False
    is_parametric = False
    is_generic = False
    is_vector = False
    is_2Dvector = False
    is_3Dvector = False
    _N = 100

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        kwargs = _set_discretization_points(kwargs.copy(), type(self))
        self.only_integers = kwargs.get('only_integers', False)
        self.modules = kwargs.get('modules', None)
        self.show_in_legend = kwargs.get('show_in_legend', True)
        self.colorbar = kwargs.get('colorbar', True)
        self.use_cm = kwargs.get('use_cm', False)
        self.is_polar = kwargs.get('is_polar', kwargs.get('polar', False))
        self.is_point = kwargs.get('is_point', kwargs.get('point', False))
        self._label = self._latex_label = ''
        self._ranges = []
        self._n = [int(kwargs.get('n1', self._N)), int(kwargs.get('n2', self._N)), int(kwargs.get('n3', self._N))]
        self._scales = [kwargs.get('xscale', 'linear'), kwargs.get('yscale', 'linear'), kwargs.get('zscale', 'linear')]
        self._params = kwargs.get('params', {})
        if not isinstance(self._params, dict):
            raise TypeError('`params` must be a dictionary mapping symbols to numeric values.')
        if len(self._params) > 0:
            self.is_interactive = True
        self.rendering_kw = kwargs.get('rendering_kw', {})
        self._tx = kwargs.get('tx', None)
        self._ty = kwargs.get('ty', None)
        self._tz = kwargs.get('tz', None)
        self._tp = kwargs.get('tp', None)
        if not all((callable(t) or t is None for t in [self._tx, self._ty, self._tz, self._tp])):
            raise TypeError('`tx`, `ty`, `tz`, `tp` must be functions.')
        self._functions = []
        self._signature = []
        self._force_real_eval = kwargs.get('force_real_eval', None)
        self._discretized_domain = None
        self._interactive_ranges = False
        self._needs_to_be_int = []
        self.color_func = None
        self._eval_color_func_with_signature = False

    def _block_lambda_functions(self, *exprs):
        if False:
            return 10
        'Some data series can be used to plot numerical functions, others\n        cannot. Execute this method inside the `__init__` to prevent the\n        processing of numerical functions.\n        '
        if any((callable(e) for e in exprs)):
            raise TypeError(type(self).__name__ + ' requires a symbolic expression.')

    def _check_fs(self):
        if False:
            for i in range(10):
                print('nop')
        ' Checks if there are enogh parameters and free symbols.\n        '
        (exprs, ranges) = (self.expr, self.ranges)
        (params, label) = (self.params, self.label)
        exprs = exprs if hasattr(exprs, '__iter__') else [exprs]
        if any((callable(e) for e in exprs)):
            return
        fs = _get_free_symbols(exprs)
        fs = fs.difference(params.keys())
        if ranges is not None:
            fs = fs.difference([r[0] for r in ranges])
        if len(fs) > 0:
            raise ValueError('Incompatible expression and parameters.\n' + 'Expression: {}\n'.format((exprs, ranges, label) if ranges is not None else (exprs, label)) + 'params: {}\n'.format(params) + 'Specify what these symbols represent: {}\n'.format(fs) + 'Are they ranges or parameters?')
        range_symbols = [r[0] for r in ranges]
        for r in ranges:
            fs = set().union(*[e.free_symbols for e in r[1:]])
            if any((t in fs for t in range_symbols)):
                raise ValueError("Range symbols can't be included into minimum and maximum of a range. Received range: %s" % str(r))
            if len(fs) > 0:
                self._interactive_ranges = True
            remaining_fs = fs.difference(params.keys())
            if len(remaining_fs) > 0:
                raise ValueError('Unkown symbols found in plotting range: %s. ' % (r,) + 'Are the following parameters? %s' % remaining_fs)

    def _create_lambda_func(self):
        if False:
            print('Hello World!')
        "Create the lambda functions to be used by the uniform meshing\n        strategy.\n\n        Notes\n        =====\n        The old sympy.plotting used experimental_lambdify. It created one\n        lambda function each time an evaluation was requested. If that failed,\n        it went on to create a different lambda function and evaluated it,\n        and so on.\n\n        This new module changes strategy: it creates right away the default\n        lambda function as well as the backup one. The reason is that the\n        series could be interactive, hence the numerical function will be\n        evaluated multiple times. So, let's create the functions just once.\n\n        This approach works fine for the majority of cases, in which the\n        symbolic expression is relatively short, hence the lambdification\n        is fast. If the expression is very long, this approach takes twice\n        the time to create the lambda functions. Be aware of that!\n        "
        exprs = self.expr if hasattr(self.expr, '__iter__') else [self.expr]
        if not any((callable(e) for e in exprs)):
            fs = _get_free_symbols(exprs)
            self._signature = sorted(fs, key=lambda t: t.name)
            self._functions = []
            for e in exprs:
                self._functions.append([lambdify(self._signature, e, modules=self.modules), lambdify(self._signature, e, modules='sympy', dummify=True)])
        else:
            self._signature = sorted([r[0] for r in self.ranges], key=lambda t: t.name)
            self._functions = [(e, None) for e in exprs]
        if isinstance(self.color_func, Expr):
            self.color_func = lambdify(self._signature, self.color_func)
            self._eval_color_func_with_signature = True

    def _update_range_value(self, t):
        if False:
            i = 10
            return i + 15
        'If the value of a plotting range is a symbolic expression,\n        substitute the parameters in order to get a numerical value.\n        '
        if not self._interactive_ranges:
            return complex(t)
        return complex(t.subs(self.params))

    def _create_discretized_domain(self):
        if False:
            for i in range(10):
                print('nop')
        'Discretize the ranges for uniform meshing strategy.\n        '
        discr_symbols = []
        discretizations = []
        for (i, r) in enumerate(self.ranges):
            discr_symbols.append(r[0])
            c_start = self._update_range_value(r[1])
            c_end = self._update_range_value(r[2])
            start = c_start.real if c_start.imag == c_end.imag == 0 else c_start
            end = c_end.real if c_start.imag == c_end.imag == 0 else c_end
            needs_integer_discr = self.only_integers or r[0] in self._needs_to_be_int
            d = BaseSeries._discretize(start, end, self.n[i], scale=self.scales[i], only_integers=needs_integer_discr)
            if not self._force_real_eval and (not needs_integer_discr) and (d.dtype != 'complex'):
                d = d + 1j * c_start.imag
            if needs_integer_discr:
                d = d.astype(int)
            discretizations.append(d)
        self._create_discretized_domain_helper(discr_symbols, discretizations)

    def _create_discretized_domain_helper(self, discr_symbols, discretizations):
        if False:
            for i in range(10):
                print('nop')
        'Create 2D or 3D discretized grids.\n\n        Subclasses should override this method in order to implement a\n        different behaviour.\n        '
        np = import_module('numpy')
        indexing = 'xy'
        if self.is_3Dvector or (self.is_3Dsurface and self.is_implicit):
            indexing = 'ij'
        meshes = np.meshgrid(*discretizations, indexing=indexing)
        self._discretized_domain = dict(zip(discr_symbols, meshes))

    def _evaluate(self, cast_to_real=True):
        if False:
            i = 10
            return i + 15
        'Evaluation of the symbolic expression (or expressions) with the\n        uniform meshing strategy, based on current values of the parameters.\n        '
        np = import_module('numpy')
        if not self._functions:
            self._create_lambda_func()
        if not self._discretized_domain or self._interactive_ranges:
            self._create_discretized_domain()
        discr = [self._discretized_domain[s[0]] for s in self.ranges]
        args = self._aggregate_args()
        results = []
        for f in self._functions:
            r = _uniform_eval(*f, *args)
            r = self._correct_shape(np.array(r), discr[0])
            r = r.astype(complex)
            results.append(r)
        if cast_to_real:
            discr = [np.real(d.astype(complex)) for d in discr]
        return [*discr, *results]

    def _aggregate_args(self):
        if False:
            return 10
        'Create a list of arguments to be passed to the lambda function,\n        sorted accoring to self._signature.\n        '
        args = []
        for s in self._signature:
            if s in self._params.keys():
                args.append(int(self._params[s]) if s in self._needs_to_be_int else self._params[s] if self._force_real_eval else complex(self._params[s]))
            else:
                args.append(self._discretized_domain[s])
        return args

    @property
    def expr(self):
        if False:
            return 10
        'Return the expression (or expressions) of the series.'
        return self._expr

    @expr.setter
    def expr(self, e):
        if False:
            for i in range(10):
                print('nop')
        'Set the expression (or expressions) of the series.'
        is_iter = hasattr(e, '__iter__')
        is_callable = callable(e) if not is_iter else any((callable(t) for t in e))
        if is_callable:
            self._expr = e
        else:
            self._expr = sympify(e) if not is_iter else Tuple(*e)
            s = set()
            for e in self._expr.atoms(Sum, Product):
                for a in e.args[1:]:
                    if isinstance(a[-1], Symbol):
                        s.add(a[-1])
            self._needs_to_be_int = list(s)
            pf = [ceiling, floor, atan2, frac, zeta]
            if self._force_real_eval is not True:
                check_res = [self._expr.has(f) for f in pf]
                self._force_real_eval = any(check_res)
                if self._force_real_eval and (self.modules is None or (isinstance(self.modules, str) and 'numpy' in self.modules)):
                    funcs = [f for (f, c) in zip(pf, check_res) if c]
                    warnings.warn('NumPy is unable to evaluate with complex numbers some of the functions included in this symbolic expression: %s. ' % funcs + 'Hence, the evaluation will use real numbers. If you believe the resulting plot is incorrect, change the evaluation module by setting the `modules` keyword argument.')
            if self._functions:
                self._create_lambda_func()

    @property
    def is_3D(self):
        if False:
            while True:
                i = 10
        flags3D = [self.is_3Dline, self.is_3Dsurface, self.is_3Dvector]
        return any(flags3D)

    @property
    def is_line(self):
        if False:
            i = 10
            return i + 15
        flagslines = [self.is_2Dline, self.is_3Dline]
        return any(flagslines)

    def _line_surface_color(self, prop, val):
        if False:
            i = 10
            return i + 15
        'This method enables back-compatibility with old sympy.plotting'
        setattr(self, prop, val)
        if callable(val) or isinstance(val, Expr):
            self.color_func = val
            setattr(self, prop, None)
        elif val is not None:
            self.color_func = None

    @property
    def line_color(self):
        if False:
            print('Hello World!')
        return self._line_color

    @line_color.setter
    def line_color(self, val):
        if False:
            i = 10
            return i + 15
        self._line_surface_color('_line_color', val)

    @property
    def n(self):
        if False:
            while True:
                i = 10
        'Returns a list [n1, n2, n3] of numbers of discratization points.\n        '
        return self._n

    @n.setter
    def n(self, v):
        if False:
            i = 10
            return i + 15
        'Set the numbers of discretization points. ``v`` must be an int or\n        a list.\n\n        Let ``s`` be a series. Then:\n\n        * to set the number of discretization points along the x direction (or\n          first parameter): ``s.n = 10``\n        * to set the number of discretization points along the x and y\n          directions (or first and second parameters): ``s.n = [10, 15]``\n        * to set the number of discretization points along the x, y and z\n          directions: ``s.n = [10, 15, 20]``\n\n        The following is highly unreccomended, because it prevents\n        the execution of necessary code in order to keep updated data:\n        ``s.n[1] = 15``\n        '
        if not hasattr(v, '__iter__'):
            self._n[0] = v
        else:
            self._n[:len(v)] = v
        if self._discretized_domain:
            self._create_discretized_domain()

    @property
    def params(self):
        if False:
            return 10
        'Get or set the current parameters dictionary.\n\n        Parameters\n        ==========\n\n        p : dict\n\n            * key: symbol associated to the parameter\n            * val: the numeric value\n        '
        return self._params

    @params.setter
    def params(self, p):
        if False:
            return 10
        self._params = p

    def _post_init(self):
        if False:
            print('Hello World!')
        exprs = self.expr if hasattr(self.expr, '__iter__') else [self.expr]
        if any((callable(e) for e in exprs)) and self.params:
            raise TypeError('`params` was provided, hence an interactive plot is expected. However, interactive plots do not support user-provided numerical functions.')
        if any((callable(e) for e in exprs)):
            if self._label == str(self.expr):
                self.label = ''
        self._check_fs()
        if hasattr(self, 'adaptive') and self.adaptive and self.params:
            warnings.warn('`params` was provided, hence an interactive plot is expected. However, interactive plots do not support adaptive evaluation. Automatically switched to adaptive=False.')
            self.adaptive = False

    @property
    def scales(self):
        if False:
            i = 10
            return i + 15
        return self._scales

    @scales.setter
    def scales(self, v):
        if False:
            while True:
                i = 10
        if isinstance(v, str):
            self._scales[0] = v
        else:
            self._scales[:len(v)] = v

    @property
    def surface_color(self):
        if False:
            return 10
        return self._surface_color

    @surface_color.setter
    def surface_color(self, val):
        if False:
            return 10
        self._line_surface_color('_surface_color', val)

    @property
    def rendering_kw(self):
        if False:
            print('Hello World!')
        return self._rendering_kw

    @rendering_kw.setter
    def rendering_kw(self, kwargs):
        if False:
            return 10
        if isinstance(kwargs, dict):
            self._rendering_kw = kwargs
        else:
            self._rendering_kw = {}
            if kwargs is not None:
                warnings.warn('`rendering_kw` must be a dictionary, instead an object of type %s was received. ' % type(kwargs) + 'Automatically setting `rendering_kw` to an empty dictionary')

    @staticmethod
    def _discretize(start, end, N, scale='linear', only_integers=False):
        if False:
            return 10
        "Discretize a 1D domain.\n\n        Returns\n        =======\n\n        domain : np.ndarray with dtype=float or complex\n            The domain's dtype will be float or complex (depending on the\n            type of start/end) even if only_integers=True. It is left for\n            the downstream code to perform further casting, if necessary.\n        "
        np = import_module('numpy')
        if only_integers is True:
            (start, end) = (int(start), int(end))
            N = end - start + 1
        if scale == 'linear':
            return np.linspace(start, end, N)
        return np.geomspace(start, end, N)

    @staticmethod
    def _correct_shape(a, b):
        if False:
            while True:
                i = 10
        'Convert ``a`` to a np.ndarray of the same shape of ``b``.\n\n        Parameters\n        ==========\n\n        a : int, float, complex, np.ndarray\n            Usually, this is the result of a numerical evaluation of a\n            symbolic expression. Even if a discretized domain was used to\n            evaluate the function, the result can be a scalar (int, float,\n            complex). Think for example to ``expr = Float(2)`` and\n            ``f = lambdify(x, expr)``. No matter the shape of the numerical\n            array representing x, the result of the evaluation will be\n            a single value.\n\n        b : np.ndarray\n            It represents the correct shape that ``a`` should have.\n\n        Returns\n        =======\n        new_a : np.ndarray\n            An array with the correct shape.\n        '
        np = import_module('numpy')
        if not isinstance(a, np.ndarray):
            a = np.array(a)
        if a.shape != b.shape:
            if a.shape == ():
                a = a * np.ones_like(b)
            else:
                a = a.reshape(b.shape)
        return a

    def eval_color_func(self, *args):
        if False:
            return 10
        'Evaluate the color function.\n\n        Parameters\n        ==========\n\n        args : tuple\n            Arguments to be passed to the coloring function. Can be coordinates\n            or parameters or both.\n\n        Notes\n        =====\n\n        The backend will request the data series to generate the numerical\n        data. Depending on the data series, either the data series itself or\n        the backend will eventually execute this function to generate the\n        appropriate coloring value.\n        '
        np = import_module('numpy')
        if self.color_func is None:
            warnings.warn('This is likely not the result you were looking for. Please, re-execute the plot command, this time with the appropriate an appropriate value to line_color or surface_color.')
            return np.ones_like(args[0])
        if self._eval_color_func_with_signature:
            args = self._aggregate_args()
            color = self.color_func(*args)
            (_re, _im) = (np.real(color), np.imag(color))
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            return _re
        nargs = arity(self.color_func)
        if nargs == 1:
            if self.is_2Dline and self.is_parametric:
                if len(args) == 2:
                    return self._correct_shape(self.color_func(args[0]), args[0])
                return self._correct_shape(self.color_func(args[2]), args[2])
            elif self.is_3Dline and self.is_parametric:
                return self._correct_shape(self.color_func(args[3]), args[3])
            elif self.is_3Dsurface and self.is_parametric:
                return self._correct_shape(self.color_func(args[3]), args[3])
            return self._correct_shape(self.color_func(args[0]), args[0])
        elif nargs == 2:
            if self.is_3Dsurface and self.is_parametric:
                return self._correct_shape(self.color_func(*args[3:]), args[3])
            return self._correct_shape(self.color_func(*args[:2]), args[0])
        return self._correct_shape(self.color_func(*args[:nargs]), args[0])

    def get_data(self):
        if False:
            return 10
        'Compute and returns the numerical data.\n\n        The number of parameters returned by this method depends on the\n        specific instance. If ``s`` is the series, make sure to read\n        ``help(s.get_data)`` to understand what it returns.\n        '
        raise NotImplementedError

    def _get_wrapped_label(self, label, wrapper):
        if False:
            return 10
        'Given a latex representation of an expression, wrap it inside\n        some characters. Matplotlib needs "$%s%$", K3D-Jupyter needs "%s".\n        '
        return wrapper % label

    def get_label(self, use_latex=False, wrapper='$%s$'):
        if False:
            while True:
                i = 10
        'Return the label to be used to display the expression.\n\n        Parameters\n        ==========\n        use_latex : bool\n            If False, the string representation of the expression is returned.\n            If True, the latex representation is returned.\n        wrapper : str\n            The backend might need the latex representation to be wrapped by\n            some characters. Default to ``"$%s$"``.\n\n        Returns\n        =======\n        label : str\n        '
        if use_latex is False:
            return self._label
        if self._label == str(self.expr):
            return self._get_wrapped_label(self._latex_label, wrapper)
        return self._latex_label

    @property
    def label(self):
        if False:
            i = 10
            return i + 15
        return self.get_label()

    @label.setter
    def label(self, val):
        if False:
            for i in range(10):
                print('nop')
        'Set the labels associated to this series.'
        self._label = self._latex_label = val

    @property
    def ranges(self):
        if False:
            print('Hello World!')
        return self._ranges

    @ranges.setter
    def ranges(self, val):
        if False:
            print('Hello World!')
        new_vals = []
        for v in val:
            if v is not None:
                new_vals.append(tuple([sympify(t) for t in v]))
        self._ranges = new_vals

    def _apply_transform(self, *args):
        if False:
            return 10
        'Apply transformations to the results of numerical evaluation.\n\n        Parameters\n        ==========\n        args : tuple\n            Results of numerical evaluation.\n\n        Returns\n        =======\n        transformed_args : tuple\n            Tuple containing the transformed results.\n        '
        t = lambda x, transform: x if transform is None else transform(x)
        (x, y, z) = (None, None, None)
        if len(args) == 2:
            (x, y) = args
            return (t(x, self._tx), t(y, self._ty))
        elif len(args) == 3 and isinstance(self, Parametric2DLineSeries):
            (x, y, u) = args
            return (t(x, self._tx), t(y, self._ty), t(u, self._tp))
        elif len(args) == 3:
            (x, y, z) = args
            return (t(x, self._tx), t(y, self._ty), t(z, self._tz))
        elif len(args) == 4 and isinstance(self, Parametric3DLineSeries):
            (x, y, z, u) = args
            return (t(x, self._tx), t(y, self._ty), t(z, self._tz), t(u, self._tp))
        elif len(args) == 4:
            (x, y, u, v) = args
            return (t(x, self._tx), t(y, self._ty), t(u, self._tx), t(v, self._ty))
        elif len(args) == 5 and isinstance(self, ParametricSurfaceSeries):
            (x, y, z, u, v) = args
            return (t(x, self._tx), t(y, self._ty), t(z, self._tz), u, v)
        elif len(args) == 6 and self.is_3Dvector:
            (x, y, z, u, v, w) = args
            return (t(x, self._tx), t(y, self._ty), t(z, self._tz), t(u, self._tx), t(v, self._ty), t(w, self._tz))
        elif len(args) == 6:
            (x, y, _abs, _arg, img, colors) = args
            return (x, y, t(_abs, self._tz), _arg, img, colors)
        return args

    def _str_helper(self, s):
        if False:
            while True:
                i = 10
        (pre, post) = ('', '')
        if self.is_interactive:
            pre = 'interactive '
            post = ' and parameters ' + str(tuple(self.params.keys()))
        return pre + s + post

def _detect_poles_numerical_helper(x, y, eps=0.01, expr=None, symb=None, symbolic=False):
    if False:
        print('Hello World!')
    "Compute the steepness of each segment. If it's greater than a\n    threshold, set the right-point y-value non NaN and record the\n    corresponding x-location for further processing.\n\n    Returns\n    =======\n    x : np.ndarray\n        Unchanged x-data.\n    yy : np.ndarray\n        Modified y-data with NaN values.\n    "
    np = import_module('numpy')
    yy = y.copy()
    threshold = np.pi / 2 - eps
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        dy = abs(y[i + 1] - y[i])
        angle = np.arctan(dy / dx)
        if abs(angle) >= threshold:
            yy[i + 1] = np.nan
    return (x, yy)

def _detect_poles_symbolic_helper(expr, symb, start, end):
    if False:
        for i in range(10):
            print('nop')
    'Attempts to compute symbolic discontinuities.\n\n    Returns\n    =======\n    pole : list\n        List of symbolic poles, possibily empty.\n    '
    poles = []
    interval = Interval(nsimplify(start), nsimplify(end))
    res = continuous_domain(expr, symb, interval)
    res = res.simplify()
    if res == interval:
        pass
    elif isinstance(res, Union) and all((isinstance(t, Interval) for t in res.args)):
        poles = []
        for s in res.args:
            if s.left_open:
                poles.append(s.left)
            if s.right_open:
                poles.append(s.right)
        poles = list(set(poles))
    else:
        raise ValueError(f'Could not parse the following object: {res} .\nPlease, submit this as a bug. Consider also to set `detect_poles=True`.')
    return poles

class Line2DBaseSeries(BaseSeries):
    """A base class for 2D lines.

    - adding the label, steps and only_integers options
    - making is_2Dline true
    - defining get_segments and get_color_array
    """
    is_2Dline = True
    _dim = 2
    _N = 1000

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.steps = kwargs.get('steps', False)
        self.is_point = kwargs.get('is_point', kwargs.get('point', False))
        self.is_filled = kwargs.get('is_filled', kwargs.get('fill', True))
        self.adaptive = kwargs.get('adaptive', False)
        self.depth = kwargs.get('depth', 12)
        self.use_cm = kwargs.get('use_cm', False)
        self.color_func = kwargs.get('color_func', None)
        self.line_color = kwargs.get('line_color', None)
        self.detect_poles = kwargs.get('detect_poles', False)
        self.eps = kwargs.get('eps', 0.01)
        self.is_polar = kwargs.get('is_polar', kwargs.get('polar', False))
        self.unwrap = kwargs.get('unwrap', False)
        self.poles_locations = []
        exclude = kwargs.get('exclude', [])
        if isinstance(exclude, Set):
            exclude = list(extract_solution(exclude, n=100))
        if not hasattr(exclude, '__iter__'):
            exclude = [exclude]
        exclude = [float(e) for e in exclude]
        self.exclude = sorted(exclude)

    def get_data(self):
        if False:
            return 10
        'Return coordinates for plotting the line.\n\n        Returns\n        =======\n\n        x: np.ndarray\n            x-coordinates\n\n        y: np.ndarray\n            y-coordinates\n\n        z: np.ndarray (optional)\n            z-coordinates in case of Parametric3DLineSeries,\n            Parametric3DLineInteractiveSeries\n\n        param : np.ndarray (optional)\n            The parameter in case of Parametric2DLineSeries,\n            Parametric3DLineSeries or AbsArgLineSeries (and their\n            corresponding interactive series).\n        '
        np = import_module('numpy')
        points = self._get_data_helper()
        if isinstance(self, LineOver1DRangeSeries) and self.detect_poles == 'symbolic':
            poles = _detect_poles_symbolic_helper(self.expr.subs(self.params), *self.ranges[0])
            poles = np.array([float(t) for t in poles])
            t = lambda x, transform: x if transform is None else transform(x)
            self.poles_locations = t(np.array(poles), self._tx)
        points = self._apply_transform(*points)
        if self.is_2Dline and self.detect_poles:
            if len(points) == 2:
                (x, y) = points
                (x, y) = _detect_poles_numerical_helper(x, y, self.eps)
                points = (x, y)
            else:
                (x, y, p) = points
                (x, y) = _detect_poles_numerical_helper(x, y, self.eps)
                points = (x, y, p)
        if self.unwrap:
            kw = {}
            if self.unwrap is not True:
                kw = self.unwrap
            if self.is_2Dline:
                if len(points) == 2:
                    (x, y) = points
                    y = np.unwrap(y, **kw)
                    points = (x, y)
                else:
                    (x, y, p) = points
                    y = np.unwrap(y, **kw)
                    points = (x, y, p)
        if self.steps is True:
            if self.is_2Dline:
                (x, y) = (points[0], points[1])
                x = np.array((x, x)).T.flatten()[1:]
                y = np.array((y, y)).T.flatten()[:-1]
                if self.is_parametric:
                    points = (x, y, points[2])
                else:
                    points = (x, y)
            elif self.is_3Dline:
                x = np.repeat(points[0], 3)[2:]
                y = np.repeat(points[1], 3)[:-2]
                z = np.repeat(points[2], 3)[1:-1]
                if len(points) > 3:
                    points = (x, y, z, points[3])
                else:
                    points = (x, y, z)
        if len(self.exclude) > 0:
            points = self._insert_exclusions(points)
        return points

    def get_segments(self):
        if False:
            while True:
                i = 10
        sympy_deprecation_warning('\n            The Line2DBaseSeries.get_segments() method is deprecated.\n\n            Instead, use the MatplotlibBackend.get_segments() method, or use\n            The get_points() or get_data() methods.\n            ', deprecated_since_version='1.9', active_deprecations_target='deprecated-get-segments')
        np = import_module('numpy')
        points = type(self).get_data(self)
        points = np.ma.array(points).T.reshape(-1, 1, self._dim)
        return np.ma.concatenate([points[:-1], points[1:]], axis=1)

    def _insert_exclusions(self, points):
        if False:
            while True:
                i = 10
        "Add NaN to each of the exclusion point. Practically, this adds a\n        NaN to the exlusion point, plus two other nearby points evaluated with\n        the numerical functions associated to this data series.\n        These nearby points are important when the number of discretization\n        points is low, or the scale is logarithm.\n\n        NOTE: it would be easier to just add exclusion points to the\n        discretized domain before evaluation, then after evaluation add NaN\n        to the exclusion points. But that's only work with adaptive=False.\n        The following approach work even with adaptive=True.\n        "
        np = import_module('numpy')
        points = list(points)
        n = len(points)
        k = n - 1
        if n == 2:
            k = 0
        j_indeces = sorted(set(range(n)).difference([k]))
        funcs = [f[0] for f in self._functions]
        for e in self.exclude:
            res = points[k] - e >= 0
            if any(res) and any(~res):
                idx = np.nanargmax(res)
                idx -= 1
                if idx > 0 and idx < len(points[k]) - 1:
                    delta_prev = abs(e - points[k][idx])
                    delta_post = abs(e - points[k][idx + 1])
                    delta = min(delta_prev, delta_post) / 100
                    prev = e - delta
                    post = e + delta
                    points[k] = np.concatenate((points[k][:idx], [prev, e, post], points[k][idx + 1:]))
                    c = 0
                    for j in j_indeces:
                        values = funcs[c](np.array([prev, post]))
                        c += 1
                        points[j] = np.concatenate((points[j][:idx], [values[0], np.nan, values[1]], points[j][idx + 1:]))
        return points

    @property
    def var(self):
        if False:
            for i in range(10):
                print('nop')
        return None if not self.ranges else self.ranges[0][0]

    @property
    def start(self):
        if False:
            return 10
        if not self.ranges:
            return None
        try:
            return self._cast(self.ranges[0][1])
        except TypeError:
            return self.ranges[0][1]

    @property
    def end(self):
        if False:
            print('Hello World!')
        if not self.ranges:
            return None
        try:
            return self._cast(self.ranges[0][2])
        except TypeError:
            return self.ranges[0][2]

    @property
    def xscale(self):
        if False:
            while True:
                i = 10
        return self._scales[0]

    @xscale.setter
    def xscale(self, v):
        if False:
            i = 10
            return i + 15
        self.scales = v

    def get_color_array(self):
        if False:
            while True:
                i = 10
        np = import_module('numpy')
        c = self.line_color
        if hasattr(c, '__call__'):
            f = np.vectorize(c)
            nargs = arity(c)
            if nargs == 1 and self.is_parametric:
                x = self.get_parameter_points()
                return f(centers_of_segments(x))
            else:
                variables = list(map(centers_of_segments, self.get_points()))
                if nargs == 1:
                    return f(variables[0])
                elif nargs == 2:
                    return f(*variables[:2])
                else:
                    return f(*variables)
        else:
            return c * np.ones(self.nb_of_points)

class List2DSeries(Line2DBaseSeries):
    """Representation for a line consisting of list of points."""

    def __init__(self, list_x, list_y, label='', **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        np = import_module('numpy')
        if len(list_x) != len(list_y):
            raise ValueError('The two lists of coordinates must have the same number of elements.\nReceived: len(list_x) = {} '.format(len(list_x)) + 'and len(list_y) = {}'.format(len(list_y)))
        self._block_lambda_functions(list_x, list_y)
        check = lambda l: [isinstance(t, Expr) and (not t.is_number) for t in l]
        if any(check(list_x) + check(list_y)) or self.params:
            if not self.params:
                raise ValueError("Some or all elements of the provided lists are symbolic expressions, but the ``params`` dictionary was not provided: those elements can't be evaluated.")
            self.list_x = Tuple(*list_x)
            self.list_y = Tuple(*list_y)
        else:
            self.list_x = np.array(list_x, dtype=np.float64)
            self.list_y = np.array(list_y, dtype=np.float64)
        self._expr = (self.list_x, self.list_y)
        if not any((isinstance(t, np.ndarray) for t in [self.list_x, self.list_y])):
            self._check_fs()
        self.is_polar = kwargs.get('is_polar', kwargs.get('polar', False))
        self.label = label
        self.rendering_kw = kwargs.get('rendering_kw', {})
        if self.use_cm and self.color_func:
            self.is_parametric = True
            if isinstance(self.color_func, Expr):
                raise TypeError("%s don't support symbolic " % self.__class__.__name__ + 'expression for `color_func`.')

    def __str__(self):
        if False:
            while True:
                i = 10
        return '2D list plot'

    def _get_data_helper(self):
        if False:
            i = 10
            return i + 15
        'Returns coordinates that needs to be postprocessed.'
        (lx, ly) = (self.list_x, self.list_y)
        if not self.is_interactive:
            return self._eval_color_func_and_return(lx, ly)
        np = import_module('numpy')
        lx = np.array([t.evalf(subs=self.params) for t in lx], dtype=float)
        ly = np.array([t.evalf(subs=self.params) for t in ly], dtype=float)
        return self._eval_color_func_and_return(lx, ly)

    def _eval_color_func_and_return(self, *data):
        if False:
            i = 10
            return i + 15
        if self.use_cm and callable(self.color_func):
            return [*data, self.eval_color_func(*data)]
        return data

class LineOver1DRangeSeries(Line2DBaseSeries):
    """Representation for a line consisting of a SymPy expression over a range."""

    def __init__(self, expr, var_start_end, label='', **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.expr = expr if callable(expr) else sympify(expr)
        self._label = str(self.expr) if label is None else label
        self._latex_label = latex(self.expr) if label is None else label
        self.ranges = [var_start_end]
        self._cast = complex
        self._return = kwargs.get('return', None)
        self._post_init()
        if not self._interactive_ranges:
            (start, end) = [complex(t) for t in self.ranges[0][1:]]
            if im(start) != im(end):
                raise ValueError('%s requires the imaginary ' % self.__class__.__name__ + 'part of the start and end values of the range to be the same.')
        if self.adaptive and self._return:
            warnings.warn('The adaptive algorithm is unable to deal with complex numbers. Automatically switching to uniform meshing.')
            self.adaptive = False

    @property
    def nb_of_points(self):
        if False:
            for i in range(10):
                print('nop')
        return self.n[0]

    @nb_of_points.setter
    def nb_of_points(self, v):
        if False:
            for i in range(10):
                print('nop')
        self.n = v

    def __str__(self):
        if False:
            i = 10
            return i + 15

        def f(t):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(t, complex):
                if t.imag != 0:
                    return t
                return t.real
            return t
        pre = 'interactive ' if self.is_interactive else ''
        post = ''
        if self.is_interactive:
            post = ' and parameters ' + str(tuple(self.params.keys()))
        wrapper = _get_wrapper_for_expr(self._return)
        return pre + 'cartesian line: %s for %s over %s' % (wrapper % self.expr, str(self.var), str((f(self.start), f(self.end)))) + post

    def get_points(self):
        if False:
            return 10
        'Return lists of coordinates for plotting. Depending on the\n        ``adaptive`` option, this function will either use an adaptive algorithm\n        or it will uniformly sample the expression over the provided range.\n\n        This function is available for back-compatibility purposes. Consider\n        using ``get_data()`` instead.\n\n        Returns\n        =======\n            x : list\n                List of x-coordinates\n\n            y : list\n                List of y-coordinates\n        '
        return self._get_data_helper()

    def _adaptive_sampling(self):
        if False:
            return 10
        try:
            if callable(self.expr):
                f = self.expr
            else:
                f = lambdify([self.var], self.expr, self.modules)
            (x, y) = self._adaptive_sampling_helper(f)
        except Exception as err:
            warnings.warn('The evaluation with %s failed.\n' % ('NumPy/SciPy' if not self.modules else self.modules) + '{}: {}\n'.format(type(err).__name__, err) + 'Trying to evaluate the expression with Sympy, but it might be a slow operation.')
            f = lambdify([self.var], self.expr, 'sympy')
            (x, y) = self._adaptive_sampling_helper(f)
        return (x, y)

    def _adaptive_sampling_helper(self, f):
        if False:
            print('Hello World!')
        'The adaptive sampling is done by recursively checking if three\n        points are almost collinear. If they are not collinear, then more\n        points are added between those points.\n\n        References\n        ==========\n\n        .. [1] Adaptive polygonal approximation of parametric curves,\n               Luiz Henrique de Figueiredo.\n        '
        np = import_module('numpy')
        x_coords = []
        y_coords = []

        def sample(p, q, depth):
            if False:
                i = 10
                return i + 15
            ' Samples recursively if three points are almost collinear.\n            For depth < 6, points are added irrespective of whether they\n            satisfy the collinearity condition or not. The maximum depth\n            allowed is 12.\n            '
            random = 0.45 + np.random.rand() * 0.1
            if self.xscale == 'log':
                xnew = 10 ** (np.log10(p[0]) + random * (np.log10(q[0]) - np.log10(p[0])))
            else:
                xnew = p[0] + random * (q[0] - p[0])
            ynew = _adaptive_eval(f, xnew)
            new_point = np.array([xnew, ynew])
            if depth > self.depth:
                x_coords.append(q[0])
                y_coords.append(q[1])
            elif depth < 6:
                sample(p, new_point, depth + 1)
                sample(new_point, q, depth + 1)
            elif p[1] is None and q[1] is None:
                if self.xscale == 'log':
                    xarray = np.logspace(p[0], q[0], 10)
                else:
                    xarray = np.linspace(p[0], q[0], 10)
                yarray = list(map(f, xarray))
                if not all((y is None for y in yarray)):
                    for i in range(len(yarray) - 1):
                        if not (yarray[i] is None and yarray[i + 1] is None):
                            sample([xarray[i], yarray[i]], [xarray[i + 1], yarray[i + 1]], depth + 1)
            elif p[1] is None or q[1] is None or new_point[1] is None or (not flat(p, new_point, q)):
                sample(p, new_point, depth + 1)
                sample(new_point, q, depth + 1)
            else:
                x_coords.append(q[0])
                y_coords.append(q[1])
        f_start = _adaptive_eval(f, self.start.real)
        f_end = _adaptive_eval(f, self.end.real)
        x_coords.append(self.start.real)
        y_coords.append(f_start)
        sample(np.array([self.start.real, f_start]), np.array([self.end.real, f_end]), 0)
        return (x_coords, y_coords)

    def _uniform_sampling(self):
        if False:
            for i in range(10):
                print('nop')
        np = import_module('numpy')
        (x, result) = self._evaluate()
        (_re, _im) = (np.real(result), np.imag(result))
        _re = self._correct_shape(_re, x)
        _im = self._correct_shape(_im, x)
        return (x, _re, _im)

    def _get_data_helper(self):
        if False:
            return 10
        'Returns coordinates that needs to be postprocessed.\n        '
        np = import_module('numpy')
        if self.adaptive and (not self.only_integers):
            (x, y) = self._adaptive_sampling()
            return [np.array(t) for t in [x, y]]
        (x, _re, _im) = self._uniform_sampling()
        if self._return is None:
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
        elif self._return == 'real':
            pass
        elif self._return == 'imag':
            _re = _im
        elif self._return == 'abs':
            _re = np.sqrt(_re ** 2 + _im ** 2)
        elif self._return == 'arg':
            _re = np.arctan2(_im, _re)
        else:
            raise ValueError('`_return` not recognized. Received: %s' % self._return)
        return (x, _re)

class ParametricLineBaseSeries(Line2DBaseSeries):
    is_parametric = True

    def _set_parametric_line_label(self, label):
        if False:
            return 10
        'Logic to set the correct label to be shown on the plot.\n        If `use_cm=True` there will be a colorbar, so we show the parameter.\n        If `use_cm=False`, there might be a legend, so we show the expressions.\n\n        Parameters\n        ==========\n        label : str\n            label passed in by the pre-processor or the user\n        '
        self._label = str(self.var) if label is None else label
        self._latex_label = latex(self.var) if label is None else label
        if self.use_cm is False and self._label == str(self.var):
            self._label = str(self.expr)
            self._latex_label = latex(self.expr)
        if any((callable(e) for e in self.expr)) and (not self.use_cm):
            if self._label == str(self.expr):
                self._label = ''

    def get_label(self, use_latex=False, wrapper='$%s$'):
        if False:
            return 10
        if self.use_cm:
            if str(self.var) == self._label:
                if use_latex:
                    return self._get_wrapped_label(latex(self.var), wrapper)
                return str(self.var)
            return self._label
        if use_latex:
            if self._label != str(self.expr):
                return self._latex_label
            return self._get_wrapped_label(self._latex_label, wrapper)
        return self._label

    def _get_data_helper(self):
        if False:
            i = 10
            return i + 15
        'Returns coordinates that needs to be postprocessed.\n        Depending on the `adaptive` option, this function will either use an\n        adaptive algorithm or it will uniformly sample the expression over the\n        provided range.\n        '
        if self.adaptive:
            np = import_module('numpy')
            coords = self._adaptive_sampling()
            coords = [np.array(t) for t in coords]
        else:
            coords = self._uniform_sampling()
        if self.is_2Dline and self.is_polar:
            np = import_module('numpy')
            (x, y, _) = coords
            r = np.sqrt(x ** 2 + y ** 2)
            t = np.arctan2(y, x)
            coords = [t, r, coords[-1]]
        if callable(self.color_func):
            coords = list(coords)
            coords[-1] = self.eval_color_func(*coords)
        return coords

    def _uniform_sampling(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns coordinates that needs to be postprocessed.'
        np = import_module('numpy')
        results = self._evaluate()
        for (i, r) in enumerate(results):
            (_re, _im) = (np.real(r), np.imag(r))
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            results[i] = _re
        return [*results[1:], results[0]]

    def get_parameter_points(self):
        if False:
            for i in range(10):
                print('nop')
        return self.get_data()[-1]

    def get_points(self):
        if False:
            for i in range(10):
                print('nop')
        ' Return lists of coordinates for plotting. Depending on the\n        ``adaptive`` option, this function will either use an adaptive algorithm\n        or it will uniformly sample the expression over the provided range.\n\n        This function is available for back-compatibility purposes. Consider\n        using ``get_data()`` instead.\n\n        Returns\n        =======\n            x : list\n                List of x-coordinates\n            y : list\n                List of y-coordinates\n            z : list\n                List of z-coordinates, only for 3D parametric line plot.\n        '
        return self._get_data_helper()[:-1]

    @property
    def nb_of_points(self):
        if False:
            for i in range(10):
                print('nop')
        return self.n[0]

    @nb_of_points.setter
    def nb_of_points(self, v):
        if False:
            for i in range(10):
                print('nop')
        self.n = v

class Parametric2DLineSeries(ParametricLineBaseSeries):
    """Representation for a line consisting of two parametric SymPy expressions
    over a range."""
    is_2Dline = True

    def __init__(self, expr_x, expr_y, var_start_end, label='', **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.expr_x = expr_x if callable(expr_x) else sympify(expr_x)
        self.expr_y = expr_y if callable(expr_y) else sympify(expr_y)
        self.expr = (self.expr_x, self.expr_y)
        self.ranges = [var_start_end]
        self._cast = float
        self.use_cm = kwargs.get('use_cm', True)
        self._set_parametric_line_label(label)
        self._post_init()

    def __str__(self):
        if False:
            while True:
                i = 10
        return self._str_helper('parametric cartesian line: (%s, %s) for %s over %s' % (str(self.expr_x), str(self.expr_y), str(self.var), str((self.start, self.end))))

    def _adaptive_sampling(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            if callable(self.expr_x) and callable(self.expr_y):
                f_x = self.expr_x
                f_y = self.expr_y
            else:
                f_x = lambdify([self.var], self.expr_x)
                f_y = lambdify([self.var], self.expr_y)
            (x, y, p) = self._adaptive_sampling_helper(f_x, f_y)
        except Exception as err:
            warnings.warn('The evaluation with %s failed.\n' % ('NumPy/SciPy' if not self.modules else self.modules) + '{}: {}\n'.format(type(err).__name__, err) + 'Trying to evaluate the expression with Sympy, but it might be a slow operation.')
            f_x = lambdify([self.var], self.expr_x, 'sympy')
            f_y = lambdify([self.var], self.expr_y, 'sympy')
            (x, y, p) = self._adaptive_sampling_helper(f_x, f_y)
        return (x, y, p)

    def _adaptive_sampling_helper(self, f_x, f_y):
        if False:
            print('Hello World!')
        'The adaptive sampling is done by recursively checking if three\n        points are almost collinear. If they are not collinear, then more\n        points are added between those points.\n\n        References\n        ==========\n\n        .. [1] Adaptive polygonal approximation of parametric curves,\n            Luiz Henrique de Figueiredo.\n        '
        x_coords = []
        y_coords = []
        param = []

        def sample(param_p, param_q, p, q, depth):
            if False:
                for i in range(10):
                    print('nop')
            ' Samples recursively if three points are almost collinear.\n            For depth < 6, points are added irrespective of whether they\n            satisfy the collinearity condition or not. The maximum depth\n            allowed is 12.\n            '
            np = import_module('numpy')
            random = 0.45 + np.random.rand() * 0.1
            param_new = param_p + random * (param_q - param_p)
            xnew = _adaptive_eval(f_x, param_new)
            ynew = _adaptive_eval(f_y, param_new)
            new_point = np.array([xnew, ynew])
            if depth > self.depth:
                x_coords.append(q[0])
                y_coords.append(q[1])
                param.append(param_p)
            elif depth < 6:
                sample(param_p, param_new, p, new_point, depth + 1)
                sample(param_new, param_q, new_point, q, depth + 1)
            elif p[0] is None and q[1] is None or (p[1] is None and q[1] is None):
                param_array = np.linspace(param_p, param_q, 10)
                x_array = [_adaptive_eval(f_x, t) for t in param_array]
                y_array = [_adaptive_eval(f_y, t) for t in param_array]
                if not all((x is None and y is None for (x, y) in zip(x_array, y_array))):
                    for i in range(len(y_array) - 1):
                        if x_array[i] is not None and y_array[i] is not None or (x_array[i + 1] is not None and y_array[i + 1] is not None):
                            point_a = [x_array[i], y_array[i]]
                            point_b = [x_array[i + 1], y_array[i + 1]]
                            sample(param_array[i], param_array[i], point_a, point_b, depth + 1)
            elif p[0] is None or p[1] is None or q[1] is None or (q[0] is None) or (not flat(p, new_point, q)):
                sample(param_p, param_new, p, new_point, depth + 1)
                sample(param_new, param_q, new_point, q, depth + 1)
            else:
                x_coords.append(q[0])
                y_coords.append(q[1])
                param.append(param_p)
        f_start_x = _adaptive_eval(f_x, self.start)
        f_start_y = _adaptive_eval(f_y, self.start)
        start = [f_start_x, f_start_y]
        f_end_x = _adaptive_eval(f_x, self.end)
        f_end_y = _adaptive_eval(f_y, self.end)
        end = [f_end_x, f_end_y]
        x_coords.append(f_start_x)
        y_coords.append(f_start_y)
        param.append(self.start)
        sample(self.start, self.end, start, end, 0)
        return (x_coords, y_coords, param)

class Line3DBaseSeries(Line2DBaseSeries):
    """A base class for 3D lines.

    Most of the stuff is derived from Line2DBaseSeries."""
    is_2Dline = False
    is_3Dline = True
    _dim = 3

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()

class Parametric3DLineSeries(ParametricLineBaseSeries):
    """Representation for a 3D line consisting of three parametric SymPy
    expressions and a range."""
    is_2Dline = False
    is_3Dline = True

    def __init__(self, expr_x, expr_y, expr_z, var_start_end, label='', **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.expr_x = expr_x if callable(expr_x) else sympify(expr_x)
        self.expr_y = expr_y if callable(expr_y) else sympify(expr_y)
        self.expr_z = expr_z if callable(expr_z) else sympify(expr_z)
        self.expr = (self.expr_x, self.expr_y, self.expr_z)
        self.ranges = [var_start_end]
        self._cast = float
        self.adaptive = False
        self.use_cm = kwargs.get('use_cm', True)
        self._set_parametric_line_label(label)
        self._post_init()
        self._xlim = None
        self._ylim = None
        self._zlim = None

    def __str__(self):
        if False:
            print('Hello World!')
        return self._str_helper('3D parametric cartesian line: (%s, %s, %s) for %s over %s' % (str(self.expr_x), str(self.expr_y), str(self.expr_z), str(self.var), str((self.start, self.end))))

    def get_data(self):
        if False:
            for i in range(10):
                print('nop')
        np = import_module('numpy')
        (x, y, z, p) = super().get_data()
        self._xlim = (np.amin(x), np.amax(x))
        self._ylim = (np.amin(y), np.amax(y))
        self._zlim = (np.amin(z), np.amax(z))
        return (x, y, z, p)

class SurfaceBaseSeries(BaseSeries):
    """A base class for 3D surfaces."""
    is_3Dsurface = True

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.use_cm = kwargs.get('use_cm', False)
        self.is_polar = kwargs.get('is_polar', kwargs.get('polar', False))
        self.surface_color = kwargs.get('surface_color', None)
        self.color_func = kwargs.get('color_func', lambda x, y, z: z)
        if callable(self.surface_color):
            self.color_func = self.surface_color
            self.surface_color = None

    def _set_surface_label(self, label):
        if False:
            while True:
                i = 10
        exprs = self.expr
        self._label = str(exprs) if label is None else label
        self._latex_label = latex(exprs) if label is None else label
        is_lambda = callable(exprs) if not hasattr(exprs, '__iter__') else any((callable(e) for e in exprs))
        if is_lambda and self._label == str(exprs):
            self._label = ''
            self._latex_label = ''

    def get_color_array(self):
        if False:
            i = 10
            return i + 15
        np = import_module('numpy')
        c = self.surface_color
        if isinstance(c, Callable):
            f = np.vectorize(c)
            nargs = arity(c)
            if self.is_parametric:
                variables = list(map(centers_of_faces, self.get_parameter_meshes()))
                if nargs == 1:
                    return f(variables[0])
                elif nargs == 2:
                    return f(*variables)
            variables = list(map(centers_of_faces, self.get_meshes()))
            if nargs == 1:
                return f(variables[0])
            elif nargs == 2:
                return f(*variables[:2])
            else:
                return f(*variables)
        elif isinstance(self, SurfaceOver2DRangeSeries):
            return c * np.ones(min(self.nb_of_points_x, self.nb_of_points_y))
        else:
            return c * np.ones(min(self.nb_of_points_u, self.nb_of_points_v))

class SurfaceOver2DRangeSeries(SurfaceBaseSeries):
    """Representation for a 3D surface consisting of a SymPy expression and 2D
    range."""

    def __init__(self, expr, var_start_end_x, var_start_end_y, label='', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.expr = expr if callable(expr) else sympify(expr)
        self.ranges = [var_start_end_x, var_start_end_y]
        self._set_surface_label(label)
        self._post_init()
        self._xlim = (self.start_x, self.end_x)
        self._ylim = (self.start_y, self.end_y)

    @property
    def var_x(self):
        if False:
            print('Hello World!')
        return self.ranges[0][0]

    @property
    def var_y(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ranges[1][0]

    @property
    def start_x(self):
        if False:
            print('Hello World!')
        try:
            return float(self.ranges[0][1])
        except TypeError:
            return self.ranges[0][1]

    @property
    def end_x(self):
        if False:
            print('Hello World!')
        try:
            return float(self.ranges[0][2])
        except TypeError:
            return self.ranges[0][2]

    @property
    def start_y(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return float(self.ranges[1][1])
        except TypeError:
            return self.ranges[1][1]

    @property
    def end_y(self):
        if False:
            return 10
        try:
            return float(self.ranges[1][2])
        except TypeError:
            return self.ranges[1][2]

    @property
    def nb_of_points_x(self):
        if False:
            print('Hello World!')
        return self.n[0]

    @nb_of_points_x.setter
    def nb_of_points_x(self, v):
        if False:
            return 10
        n = self.n
        self.n = [v, n[1:]]

    @property
    def nb_of_points_y(self):
        if False:
            for i in range(10):
                print('nop')
        return self.n[1]

    @nb_of_points_y.setter
    def nb_of_points_y(self, v):
        if False:
            for i in range(10):
                print('nop')
        n = self.n
        self.n = [n[0], v, n[2]]

    def __str__(self):
        if False:
            return 10
        series_type = 'cartesian surface' if self.is_3Dsurface else 'contour'
        return self._str_helper(series_type + ': %s for %s over %s and %s over %s' % (str(self.expr), str(self.var_x), str((self.start_x, self.end_x)), str(self.var_y), str((self.start_y, self.end_y))))

    def get_meshes(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the x,y,z coordinates for plotting the surface.\n        This function is available for back-compatibility purposes. Consider\n        using ``get_data()`` instead.\n        '
        return self.get_data()

    def get_data(self):
        if False:
            print('Hello World!')
        'Return arrays of coordinates for plotting.\n\n        Returns\n        =======\n        mesh_x : np.ndarray\n            Discretized x-domain.\n        mesh_y : np.ndarray\n            Discretized y-domain.\n        mesh_z : np.ndarray\n            Results of the evaluation.\n        '
        np = import_module('numpy')
        results = self._evaluate()
        for (i, r) in enumerate(results):
            (_re, _im) = (np.real(r), np.imag(r))
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            results[i] = _re
        (x, y, z) = results
        if self.is_polar and self.is_3Dsurface:
            r = x.copy()
            x = r * np.cos(y)
            y = r * np.sin(y)
        self._zlim = (np.amin(z), np.amax(z))
        return self._apply_transform(x, y, z)

class ParametricSurfaceSeries(SurfaceBaseSeries):
    """Representation for a 3D surface consisting of three parametric SymPy
    expressions and a range."""
    is_parametric = True

    def __init__(self, expr_x, expr_y, expr_z, var_start_end_u, var_start_end_v, label='', **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.expr_x = expr_x if callable(expr_x) else sympify(expr_x)
        self.expr_y = expr_y if callable(expr_y) else sympify(expr_y)
        self.expr_z = expr_z if callable(expr_z) else sympify(expr_z)
        self.expr = (self.expr_x, self.expr_y, self.expr_z)
        self.ranges = [var_start_end_u, var_start_end_v]
        self.color_func = kwargs.get('color_func', lambda x, y, z, u, v: z)
        self._set_surface_label(label)
        self._post_init()

    @property
    def var_u(self):
        if False:
            i = 10
            return i + 15
        return self.ranges[0][0]

    @property
    def var_v(self):
        if False:
            while True:
                i = 10
        return self.ranges[1][0]

    @property
    def start_u(self):
        if False:
            while True:
                i = 10
        try:
            return float(self.ranges[0][1])
        except TypeError:
            return self.ranges[0][1]

    @property
    def end_u(self):
        if False:
            print('Hello World!')
        try:
            return float(self.ranges[0][2])
        except TypeError:
            return self.ranges[0][2]

    @property
    def start_v(self):
        if False:
            return 10
        try:
            return float(self.ranges[1][1])
        except TypeError:
            return self.ranges[1][1]

    @property
    def end_v(self):
        if False:
            return 10
        try:
            return float(self.ranges[1][2])
        except TypeError:
            return self.ranges[1][2]

    @property
    def nb_of_points_u(self):
        if False:
            while True:
                i = 10
        return self.n[0]

    @nb_of_points_u.setter
    def nb_of_points_u(self, v):
        if False:
            print('Hello World!')
        n = self.n
        self.n = [v, n[1:]]

    @property
    def nb_of_points_v(self):
        if False:
            i = 10
            return i + 15
        return self.n[1]

    @nb_of_points_v.setter
    def nb_of_points_v(self, v):
        if False:
            while True:
                i = 10
        n = self.n
        self.n = [n[0], v, n[2]]

    def __str__(self):
        if False:
            return 10
        return self._str_helper('parametric cartesian surface: (%s, %s, %s) for %s over %s and %s over %s' % (str(self.expr_x), str(self.expr_y), str(self.expr_z), str(self.var_u), str((self.start_u, self.end_u)), str(self.var_v), str((self.start_v, self.end_v))))

    def get_parameter_meshes(self):
        if False:
            i = 10
            return i + 15
        return self.get_data()[3:]

    def get_meshes(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the x,y,z coordinates for plotting the surface.\n        This function is available for back-compatibility purposes. Consider\n        using ``get_data()`` instead.\n        '
        return self.get_data()[:3]

    def get_data(self):
        if False:
            return 10
        'Return arrays of coordinates for plotting.\n\n        Returns\n        =======\n        x : np.ndarray [n2 x n1]\n            x-coordinates.\n        y : np.ndarray [n2 x n1]\n            y-coordinates.\n        z : np.ndarray [n2 x n1]\n            z-coordinates.\n        mesh_u : np.ndarray [n2 x n1]\n            Discretized u range.\n        mesh_v : np.ndarray [n2 x n1]\n            Discretized v range.\n        '
        np = import_module('numpy')
        results = self._evaluate()
        for (i, r) in enumerate(results):
            (_re, _im) = (np.real(r), np.imag(r))
            _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
            results[i] = _re
        (x, y, z) = results[2:]
        self._xlim = (np.amin(x), np.amax(x))
        self._ylim = (np.amin(y), np.amax(y))
        self._zlim = (np.amin(z), np.amax(z))
        return self._apply_transform(*results[2:], *results[:2])

class ContourSeries(SurfaceOver2DRangeSeries):
    """Representation for a contour plot."""
    is_3Dsurface = False
    is_contour = True

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.is_filled = kwargs.get('is_filled', kwargs.get('fill', True))
        self.show_clabels = kwargs.get('clabels', True)
        self.rendering_kw = kwargs.get('contour_kw', kwargs.get('rendering_kw', {}))

class GenericDataSeries(BaseSeries):
    """Represents generic numerical data.

    Notes
    =====
    This class serves the purpose of back-compatibility with the "markers,
    annotations, fill, rectangles" keyword arguments that represent
    user-provided numerical data. In particular, it solves the problem of
    combining together two or more plot-objects with the ``extend`` or
    ``append`` methods: user-provided numerical data is also taken into
    consideration because it is stored in this series class.

    Also note that the current implementation is far from optimal, as each
    keyword argument is stored into an attribute in the ``Plot`` class, which
    requires a hard-coded if-statement in the ``MatplotlibBackend`` class.
    The implementation suggests that it is ok to add attributes and
    if-statements to provide more and more functionalities for user-provided
    numerical data (e.g. adding horizontal lines, or vertical lines, or bar
    plots, etc). However, in doing so one would reinvent the wheel: plotting
    libraries (like Matplotlib) already implements the necessary API.

    Instead of adding more keyword arguments and attributes, users interested
    in adding custom numerical data to a plot should retrieve the figure
    created by this plotting module. For example, this code:

    .. plot::
       :context: close-figs
       :include-source: True

       from sympy import Symbol, plot, cos
       x = Symbol("x")
       p = plot(cos(x), markers=[{"args": [[0, 1, 2], [0, 1, -1], "*"]}])

    Becomes:

    .. plot::
       :context: close-figs
       :include-source: True

       p = plot(cos(x), backend="matplotlib")
       fig, ax = p._backend.fig, p._backend.ax[0]
       ax.plot([0, 1, 2], [0, 1, -1], "*")
       fig

    Which is far better in terms of readibility. Also, it gives access to the
    full plotting library capabilities, without the need to reinvent the wheel.
    """
    is_generic = True

    def __init__(self, tp, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.type = tp
        self.args = args
        self.rendering_kw = kwargs

    def get_data(self):
        if False:
            while True:
                i = 10
        return self.args

class ImplicitSeries(BaseSeries):
    """Representation for 2D Implicit plot."""
    is_implicit = True
    use_cm = False
    _N = 100

    def __init__(self, expr, var_start_end_x, var_start_end_y, label='', **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.adaptive = kwargs.get('adaptive', False)
        self.expr = expr
        self._label = str(expr) if label is None else label
        self._latex_label = latex(expr) if label is None else label
        self.ranges = [var_start_end_x, var_start_end_y]
        (self.var_x, self.start_x, self.end_x) = self.ranges[0]
        (self.var_y, self.start_y, self.end_y) = self.ranges[1]
        self._color = kwargs.get('color', kwargs.get('line_color', None))
        if self.is_interactive and self.adaptive:
            raise NotImplementedError('Interactive plot with `adaptive=True` is not supported.')
        depth = kwargs.get('depth', 0)
        if depth > 4:
            depth = 4
        elif depth < 0:
            depth = 0
        self.depth = 4 + depth
        self._post_init()

    @property
    def expr(self):
        if False:
            i = 10
            return i + 15
        if self.adaptive:
            return self._adaptive_expr
        return self._non_adaptive_expr

    @expr.setter
    def expr(self, expr):
        if False:
            print('Hello World!')
        self._block_lambda_functions(expr)
        (expr, has_equality) = self._has_equality(sympify(expr))
        self._adaptive_expr = expr
        self.has_equality = has_equality
        self._label = str(expr)
        self._latex_label = latex(expr)
        if isinstance(expr, (BooleanFunction, Ne)) and (not self.adaptive):
            self.adaptive = True
            msg = 'contains Boolean functions. '
            if isinstance(expr, Ne):
                msg = 'is an unequality. '
            warnings.warn('The provided expression ' + msg + 'In order to plot the expression, the algorithm ' + 'automatically switched to an adaptive sampling.')
        if isinstance(expr, BooleanFunction):
            self._non_adaptive_expr = None
            self._is_equality = False
        else:
            (expr, is_equality) = self._preprocess_meshgrid_expression(expr, self.adaptive)
            self._non_adaptive_expr = expr
            self._is_equality = is_equality

    @property
    def line_color(self):
        if False:
            return 10
        return self._color

    @line_color.setter
    def line_color(self, v):
        if False:
            for i in range(10):
                print('nop')
        self._color = v
    color = line_color

    def _has_equality(self, expr):
        if False:
            print('Hello World!')
        has_equality = False

        def arg_expand(bool_expr):
            if False:
                while True:
                    i = 10
            'Recursively expands the arguments of an Boolean Function'
            for arg in bool_expr.args:
                if isinstance(arg, BooleanFunction):
                    arg_expand(arg)
                elif isinstance(arg, Relational):
                    arg_list.append(arg)
        arg_list = []
        if isinstance(expr, BooleanFunction):
            arg_expand(expr)
            if any((isinstance(e, (Equality, GreaterThan, LessThan)) for e in arg_list)):
                has_equality = True
        elif not isinstance(expr, Relational):
            expr = Equality(expr, 0)
            has_equality = True
        elif isinstance(expr, (Equality, GreaterThan, LessThan)):
            has_equality = True
        return (expr, has_equality)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        f = lambda t: float(t) if len(t.free_symbols) == 0 else t
        return self._str_helper('Implicit expression: %s for %s over %s and %s over %s') % (str(self._adaptive_expr), str(self.var_x), str((f(self.start_x), f(self.end_x))), str(self.var_y), str((f(self.start_y), f(self.end_y))))

    def get_data(self):
        if False:
            i = 10
            return i + 15
        'Returns numerical data.\n\n        Returns\n        =======\n\n        If the series is evaluated with the `adaptive=True` it returns:\n\n        interval_list : list\n            List of bounding rectangular intervals to be postprocessed and\n            eventually used with Matplotlib\'s ``fill`` command.\n        dummy : str\n            A string containing ``"fill"``.\n\n        Otherwise, it returns 2D numpy arrays to be used with Matplotlib\'s\n        ``contour`` or ``contourf`` commands:\n\n        x_array : np.ndarray\n        y_array : np.ndarray\n        z_array : np.ndarray\n        plot_type : str\n            A string specifying which plot command to use, ``"contour"``\n            or ``"contourf"``.\n        '
        if self.adaptive:
            data = self._adaptive_eval()
            if data is not None:
                return data
        return self._get_meshes_grid()

    def _adaptive_eval(self):
        if False:
            i = 10
            return i + 15
        "\n        References\n        ==========\n\n        .. [1] Jeffrey Allen Tupper. Reliable Two-Dimensional Graphing Methods for\n        Mathematical Formulae with Two Free Variables.\n\n        .. [2] Jeffrey Allen Tupper. Graphing Equations with Generalized Interval\n        Arithmetic. Master's thesis. University of Toronto, 1996\n        "
        import sympy.plotting.intervalmath.lib_interval as li
        user_functions = {}
        printer = IntervalMathPrinter({'fully_qualified_modules': False, 'inline': True, 'allow_unknown_functions': True, 'user_functions': user_functions})
        keys = [t for t in dir(li) if '__' not in t and t not in ['import_module', 'interval']]
        vals = [getattr(li, k) for k in keys]
        d = dict(zip(keys, vals))
        func = lambdify((self.var_x, self.var_y), self.expr, modules=[d], printer=printer)
        data = None
        try:
            data = self._get_raster_interval(func)
        except NameError as err:
            warnings.warn('Adaptive meshing could not be applied to the expression, as some functions are not yet implemented in the interval math module:\n\nNameError: %s\n\n' % err + 'Proceeding with uniform meshing.')
            self.adaptive = False
        except TypeError:
            warnings.warn('Adaptive meshing could not be applied to the expression. Using uniform meshing.')
            self.adaptive = False
        return data

    def _get_raster_interval(self, func):
        if False:
            while True:
                i = 10
        'Uses interval math to adaptively mesh and obtain the plot'
        np = import_module('numpy')
        k = self.depth
        interval_list = []
        (sx, sy) = [float(t) for t in [self.start_x, self.start_y]]
        (ex, ey) = [float(t) for t in [self.end_x, self.end_y]]
        xsample = np.linspace(sx, ex, 33)
        ysample = np.linspace(sy, ey, 33)
        jitterx = (np.random.rand(len(xsample)) * 2 - 1) * (ex - sx) / 2 ** 20
        jittery = (np.random.rand(len(ysample)) * 2 - 1) * (ey - sy) / 2 ** 20
        xsample += jitterx
        ysample += jittery
        xinter = [interval(x1, x2) for (x1, x2) in zip(xsample[:-1], xsample[1:])]
        yinter = [interval(y1, y2) for (y1, y2) in zip(ysample[:-1], ysample[1:])]
        interval_list = [[x, y] for x in xinter for y in yinter]
        plot_list = []

        def refine_pixels(interval_list):
            if False:
                i = 10
                return i + 15
            'Evaluates the intervals and subdivides the interval if the\n            expression is partially satisfied.'
            temp_interval_list = []
            plot_list = []
            for intervals in interval_list:
                intervalx = intervals[0]
                intervaly = intervals[1]
                func_eval = func(intervalx, intervaly)
                if func_eval[1] is False or func_eval[0] is False:
                    pass
                elif func_eval == (True, True):
                    plot_list.append([intervalx, intervaly])
                elif func_eval[1] is None or func_eval[0] is None:
                    avgx = intervalx.mid
                    avgy = intervaly.mid
                    a = interval(intervalx.start, avgx)
                    b = interval(avgx, intervalx.end)
                    c = interval(intervaly.start, avgy)
                    d = interval(avgy, intervaly.end)
                    temp_interval_list.append([a, c])
                    temp_interval_list.append([a, d])
                    temp_interval_list.append([b, c])
                    temp_interval_list.append([b, d])
            return (temp_interval_list, plot_list)
        while k >= 0 and len(interval_list):
            (interval_list, plot_list_temp) = refine_pixels(interval_list)
            plot_list.extend(plot_list_temp)
            k = k - 1
        if self.has_equality:
            for intervals in interval_list:
                intervalx = intervals[0]
                intervaly = intervals[1]
                func_eval = func(intervalx, intervaly)
                if func_eval[1] and func_eval[0] is not False:
                    plot_list.append([intervalx, intervaly])
        return (plot_list, 'fill')

    def _get_meshes_grid(self):
        if False:
            while True:
                i = 10
        "Generates the mesh for generating a contour.\n\n        In the case of equality, ``contour`` function of matplotlib can\n        be used. In other cases, matplotlib's ``contourf`` is used.\n        "
        np = import_module('numpy')
        (xarray, yarray, z_grid) = self._evaluate()
        (_re, _im) = (np.real(z_grid), np.imag(z_grid))
        _re[np.invert(np.isclose(_im, np.zeros_like(_im)))] = np.nan
        if self._is_equality:
            return (xarray, yarray, _re, 'contour')
        return (xarray, yarray, _re, 'contourf')

    @staticmethod
    def _preprocess_meshgrid_expression(expr, adaptive):
        if False:
            return 10
        'If the expression is a Relational, rewrite it as a single\n        expression.\n\n        Returns\n        =======\n\n        expr : Expr\n            The rewritten expression\n\n        equality : Boolean\n            Wheter the original expression was an Equality or not.\n        '
        equality = False
        if isinstance(expr, Equality):
            expr = expr.lhs - expr.rhs
            equality = True
        elif isinstance(expr, Relational):
            expr = expr.gts - expr.lts
        elif not adaptive:
            raise NotImplementedError('The expression is not supported for plotting in uniform meshed plot.')
        return (expr, equality)

    def get_label(self, use_latex=False, wrapper='$%s$'):
        if False:
            print('Hello World!')
        'Return the label to be used to display the expression.\n\n        Parameters\n        ==========\n        use_latex : bool\n            If False, the string representation of the expression is returned.\n            If True, the latex representation is returned.\n        wrapper : str\n            The backend might need the latex representation to be wrapped by\n            some characters. Default to ``"$%s$"``.\n\n        Returns\n        =======\n        label : str\n        '
        if use_latex is False:
            return self._label
        if self._label == str(self._adaptive_expr):
            return self._get_wrapped_label(self._latex_label, wrapper)
        return self._latex_label

def centers_of_segments(array):
    if False:
        for i in range(10):
            print('nop')
    np = import_module('numpy')
    return np.mean(np.vstack((array[:-1], array[1:])), 0)

def centers_of_faces(array):
    if False:
        return 10
    np = import_module('numpy')
    return np.mean(np.dstack((array[:-1, :-1], array[1:, :-1], array[:-1, 1:], array[:-1, :-1])), 2)

def flat(x, y, z, eps=0.001):
    if False:
        while True:
            i = 10
    'Checks whether three points are almost collinear'
    np = import_module('numpy')
    vector_a = (x - y).astype(float)
    vector_b = (z - y).astype(float)
    dot_product = np.dot(vector_a, vector_b)
    vector_a_norm = np.linalg.norm(vector_a)
    vector_b_norm = np.linalg.norm(vector_b)
    cos_theta = dot_product / (vector_a_norm * vector_b_norm)
    return abs(cos_theta + 1) < eps

def _set_discretization_points(kwargs, pt):
    if False:
        i = 10
        return i + 15
    'Allow the use of the keyword arguments ``n, n1, n2`` to\n    specify the number of discretization points in one and two\n    directions, while keeping back-compatibility with older keyword arguments\n    like, ``nb_of_points, nb_of_points_*, points``.\n\n    Parameters\n    ==========\n\n    kwargs : dict\n        Dictionary of keyword arguments passed into a plotting function.\n    pt : type\n        The type of the series, which indicates the kind of plot we are\n        trying to create.\n    '
    replace_old_keywords = {'nb_of_points': 'n', 'nb_of_points_x': 'n1', 'nb_of_points_y': 'n2', 'nb_of_points_u': 'n1', 'nb_of_points_v': 'n2', 'points': 'n'}
    for (k, v) in replace_old_keywords.items():
        if k in kwargs.keys():
            kwargs[v] = kwargs.pop(k)
    if pt in [LineOver1DRangeSeries, Parametric2DLineSeries, Parametric3DLineSeries]:
        if 'n' in kwargs.keys():
            kwargs['n1'] = kwargs['n']
            if hasattr(kwargs['n'], '__iter__') and len(kwargs['n']) > 0:
                kwargs['n1'] = kwargs['n'][0]
    elif pt in [SurfaceOver2DRangeSeries, ContourSeries, ParametricSurfaceSeries, ImplicitSeries]:
        if 'n' in kwargs.keys():
            if hasattr(kwargs['n'], '__iter__') and len(kwargs['n']) > 1:
                kwargs['n1'] = kwargs['n'][0]
                kwargs['n2'] = kwargs['n'][1]
            else:
                kwargs['n1'] = kwargs['n2'] = kwargs['n']
    return kwargs