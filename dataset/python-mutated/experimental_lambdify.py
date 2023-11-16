""" rewrite of lambdify - This stuff is not stable at all.

It is for internal use in the new plotting module.
It may (will! see the Q'n'A in the source) be rewritten.

It's completely self contained. Especially it does not use lambdarepr.

It does not aim to replace the current lambdify. Most importantly it will never
ever support anything else than SymPy expressions (no Matrices, dictionaries
and so on).
"""
import re
from sympy.core.numbers import I, NumberSymbol, oo, zoo
from sympy.core.symbol import Symbol
from sympy.utilities.iterables import numbered_symbols
from sympy.external import import_module
import warnings

class vectorized_lambdify:
    """ Return a sufficiently smart, vectorized and lambdified function.

    Returns only reals.

    Explanation
    ===========

    This function uses experimental_lambdify to created a lambdified
    expression ready to be used with numpy. Many of the functions in SymPy
    are not implemented in numpy so in some cases we resort to Python cmath or
    even to evalf.

    The following translations are tried:
      only numpy complex
      - on errors raised by SymPy trying to work with ndarray:
          only Python cmath and then vectorize complex128

    When using Python cmath there is no need for evalf or float/complex
    because Python cmath calls those.

    This function never tries to mix numpy directly with evalf because numpy
    does not understand SymPy Float. If this is needed one can use the
    float_wrap_evalf/complex_wrap_evalf options of experimental_lambdify or
    better one can be explicit about the dtypes that numpy works with.
    Check numpy bug http://projects.scipy.org/numpy/ticket/1013 to know what
    types of errors to expect.
    """

    def __init__(self, args, expr):
        if False:
            print('Hello World!')
        self.args = args
        self.expr = expr
        self.np = import_module('numpy')
        self.lambda_func_1 = experimental_lambdify(args, expr, use_np=True)
        self.vector_func_1 = self.lambda_func_1
        self.lambda_func_2 = experimental_lambdify(args, expr, use_python_cmath=True)
        self.vector_func_2 = self.np.vectorize(self.lambda_func_2, otypes=[complex])
        self.vector_func = self.vector_func_1
        self.failure = False

    def __call__(self, *args):
        if False:
            return 10
        np = self.np
        try:
            temp_args = (np.array(a, dtype=complex) for a in args)
            results = self.vector_func(*temp_args)
            results = np.ma.masked_where(np.abs(results.imag) > 1e-07 * np.abs(results), results.real, copy=False)
            return results
        except ValueError:
            if self.failure:
                raise
            self.failure = True
            self.vector_func = self.vector_func_2
            warnings.warn('The evaluation of the expression is problematic. We are trying a failback method that may still work. Please report this as a bug.')
            return self.__call__(*args)

class lambdify:
    """Returns the lambdified function.

    Explanation
    ===========

    This function uses experimental_lambdify to create a lambdified
    expression. It uses cmath to lambdify the expression. If the function
    is not implemented in Python cmath, Python cmath calls evalf on those
    functions.
    """

    def __init__(self, args, expr):
        if False:
            print('Hello World!')
        self.args = args
        self.expr = expr
        self.lambda_func_1 = experimental_lambdify(args, expr, use_python_cmath=True, use_evalf=True)
        self.lambda_func_2 = experimental_lambdify(args, expr, use_python_math=True, use_evalf=True)
        self.lambda_func_3 = experimental_lambdify(args, expr, use_evalf=True, complex_wrap_evalf=True)
        self.lambda_func = self.lambda_func_1
        self.failure = False

    def __call__(self, args):
        if False:
            i = 10
            return i + 15
        try:
            result = complex(self.lambda_func(args))
            if abs(result.imag) > 1e-07 * abs(result):
                return None
            return result.real
        except (ZeroDivisionError, OverflowError):
            return None
        except TypeError as e:
            if self.failure:
                raise e
            if self.lambda_func == self.lambda_func_1:
                self.lambda_func = self.lambda_func_2
                return self.__call__(args)
            self.failure = True
            self.lambda_func = self.lambda_func_3
            warnings.warn('The evaluation of the expression is problematic. We are trying a failback method that may still work. Please report this as a bug.', stacklevel=2)
            return self.__call__(args)

def experimental_lambdify(*args, **kwargs):
    if False:
        return 10
    l = Lambdifier(*args, **kwargs)
    return l

class Lambdifier:

    def __init__(self, args, expr, print_lambda=False, use_evalf=False, float_wrap_evalf=False, complex_wrap_evalf=False, use_np=False, use_python_math=False, use_python_cmath=False, use_interval=False):
        if False:
            while True:
                i = 10
        self.print_lambda = print_lambda
        self.use_evalf = use_evalf
        self.float_wrap_evalf = float_wrap_evalf
        self.complex_wrap_evalf = complex_wrap_evalf
        self.use_np = use_np
        self.use_python_math = use_python_math
        self.use_python_cmath = use_python_cmath
        self.use_interval = use_interval
        if not all((isinstance(a, Symbol) for a in args)):
            raise ValueError('The arguments must be Symbols.')
        syms = numbered_symbols(exclude=expr.free_symbols)
        newargs = [next(syms) for _ in args]
        expr = expr.xreplace(dict(zip(args, newargs)))
        argstr = ', '.join([str(a) for a in newargs])
        del syms, newargs, args
        self.dict_str = self.get_dict_str()
        self.dict_fun = self.get_dict_fun()
        exprstr = str(expr)
        newexpr = self.tree2str_translate(self.str2tree(exprstr))
        namespace = {}
        namespace.update(self.sympy_atoms_namespace(expr))
        namespace.update(self.sympy_expression_namespace(expr))
        from sympy.functions.elementary.miscellaneous import sqrt
        namespace.update({'sqrt': sqrt})
        namespace.update({'Eq': lambda x, y: x == y})
        namespace.update({'Ne': lambda x, y: x != y})
        if use_python_math:
            namespace.update({'math': __import__('math')})
        if use_python_cmath:
            namespace.update({'cmath': __import__('cmath')})
        if use_np:
            try:
                namespace.update({'np': __import__('numpy')})
            except ImportError:
                raise ImportError('experimental_lambdify failed to import numpy.')
        if use_interval:
            namespace.update({'imath': __import__('sympy.plotting.intervalmath', fromlist=['intervalmath'])})
            namespace.update({'math': __import__('math')})
        if self.print_lambda:
            print(newexpr)
        eval_str = 'lambda %s : ( %s )' % (argstr, newexpr)
        self.eval_str = eval_str
        exec('MYNEWLAMBDA = %s' % eval_str, namespace)
        self.lambda_func = namespace['MYNEWLAMBDA']

    def __call__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return self.lambda_func(*args, **kwargs)
    builtin_functions_different = {'Min': 'min', 'Max': 'max', 'Abs': 'abs'}
    builtin_not_functions = {'I': '1j'}
    numpy_functions_same = ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'exp', 'log', 'sqrt', 'floor', 'conjugate', 'sign']
    numpy_functions_different = {'acos': 'arccos', 'acosh': 'arccosh', 'arg': 'angle', 'asin': 'arcsin', 'asinh': 'arcsinh', 'atan': 'arctan', 'atan2': 'arctan2', 'atanh': 'arctanh', 'ceiling': 'ceil', 'im': 'imag', 'ln': 'log', 'Max': 'amax', 'Min': 'amin', 're': 'real', 'Abs': 'abs'}
    numpy_not_functions = {'pi': 'np.pi', 'oo': 'np.inf', 'E': 'np.e'}
    math_functions_same = ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'exp', 'log', 'erf', 'sqrt', 'floor', 'factorial', 'gamma']
    math_functions_different = {'ceiling': 'ceil', 'ln': 'log', 'loggamma': 'lgamma'}
    math_not_functions = {'pi': 'math.pi', 'E': 'math.e'}
    cmath_functions_same = ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'exp', 'log', 'sqrt']
    cmath_functions_different = {'ln': 'log', 'arg': 'phase'}
    cmath_not_functions = {'pi': 'cmath.pi', 'E': 'cmath.e'}
    interval_not_functions = {'pi': 'math.pi', 'E': 'math.e'}
    interval_functions_same = ['sin', 'cos', 'exp', 'tan', 'atan', 'log', 'sqrt', 'cosh', 'sinh', 'tanh', 'floor', 'acos', 'asin', 'acosh', 'asinh', 'atanh', 'Abs', 'And', 'Or']
    interval_functions_different = {'Min': 'imin', 'Max': 'imax', 'ceiling': 'ceil'}

    def get_dict_str(self):
        if False:
            print('Hello World!')
        dict_str = dict(self.builtin_not_functions)
        if self.use_np:
            dict_str.update(self.numpy_not_functions)
        if self.use_python_math:
            dict_str.update(self.math_not_functions)
        if self.use_python_cmath:
            dict_str.update(self.cmath_not_functions)
        if self.use_interval:
            dict_str.update(self.interval_not_functions)
        return dict_str

    def get_dict_fun(self):
        if False:
            i = 10
            return i + 15
        dict_fun = dict(self.builtin_functions_different)
        if self.use_np:
            for s in self.numpy_functions_same:
                dict_fun[s] = 'np.' + s
            for (k, v) in self.numpy_functions_different.items():
                dict_fun[k] = 'np.' + v
        if self.use_python_math:
            for s in self.math_functions_same:
                dict_fun[s] = 'math.' + s
            for (k, v) in self.math_functions_different.items():
                dict_fun[k] = 'math.' + v
        if self.use_python_cmath:
            for s in self.cmath_functions_same:
                dict_fun[s] = 'cmath.' + s
            for (k, v) in self.cmath_functions_different.items():
                dict_fun[k] = 'cmath.' + v
        if self.use_interval:
            for s in self.interval_functions_same:
                dict_fun[s] = 'imath.' + s
            for (k, v) in self.interval_functions_different.items():
                dict_fun[k] = 'imath.' + v
        return dict_fun

    def str2tree(self, exprstr):
        if False:
            for i in range(10):
                print('nop')
        "Converts an expression string to a tree.\n\n        Explanation\n        ===========\n\n        Functions are represented by ('func_name(', tree_of_arguments).\n        Other expressions are (head_string, mid_tree, tail_str).\n        Expressions that do not contain functions are directly returned.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import x, y, z\n        >>> from sympy import Integral, sin\n        >>> from sympy.plotting.experimental_lambdify import Lambdifier\n        >>> str2tree = Lambdifier([x], x).str2tree\n\n        >>> str2tree(str(Integral(x, (x, 1, y))))\n        ('', ('Integral(', 'x, (x, 1, y)'), ')')\n        >>> str2tree(str(x+y))\n        'x + y'\n        >>> str2tree(str(x+y*sin(z)+1))\n        ('x + y*', ('sin(', 'z'), ') + 1')\n        >>> str2tree('sin(y*(y + 1.1) + (sin(y)))')\n        ('', ('sin(', ('y*(y + 1.1) + (', ('sin(', 'y'), '))')), ')')\n        "
        first_par = re.search('(\\w+\\()', exprstr)
        if first_par is None:
            return exprstr
        else:
            start = first_par.start()
            end = first_par.end()
            head = exprstr[:start]
            func = exprstr[start:end]
            tail = exprstr[end:]
            count = 0
            for (i, c) in enumerate(tail):
                if c == '(':
                    count += 1
                elif c == ')':
                    count -= 1
                if count == -1:
                    break
            func_tail = self.str2tree(tail[:i])
            tail = self.str2tree(tail[i:])
            return (head, (func, func_tail), tail)

    @classmethod
    def tree2str(cls, tree):
        if False:
            i = 10
            return i + 15
        "Converts a tree to string without translations.\n\n        Examples\n        ========\n\n        >>> from sympy.abc import x, y, z\n        >>> from sympy import sin\n        >>> from sympy.plotting.experimental_lambdify import Lambdifier\n        >>> str2tree = Lambdifier([x], x).str2tree\n        >>> tree2str = Lambdifier([x], x).tree2str\n\n        >>> tree2str(str2tree(str(x+y*sin(z)+1)))\n        'x + y*sin(z) + 1'\n        "
        if isinstance(tree, str):
            return tree
        else:
            return ''.join(map(cls.tree2str, tree))

    def tree2str_translate(self, tree):
        if False:
            return 10
        'Converts a tree to string with translations.\n\n        Explanation\n        ===========\n\n        Function names are translated by translate_func.\n        Other strings are translated by translate_str.\n        '
        if isinstance(tree, str):
            return self.translate_str(tree)
        elif isinstance(tree, tuple) and len(tree) == 2:
            return self.translate_func(tree[0][:-1], tree[1])
        else:
            return ''.join([self.tree2str_translate(t) for t in tree])

    def translate_str(self, estr):
        if False:
            i = 10
            return i + 15
        'Translate substrings of estr using in order the dictionaries in\n        dict_tuple_str.'
        for (pattern, repl) in self.dict_str.items():
            estr = re.sub(pattern, repl, estr)
        return estr

    def translate_func(self, func_name, argtree):
        if False:
            for i in range(10):
                print('nop')
        'Translate function names and the tree of arguments.\n\n        Explanation\n        ===========\n\n        If the function name is not in the dictionaries of dict_tuple_fun then the\n        function is surrounded by a float((...).evalf()).\n\n        The use of float is necessary as np.<function>(sympy.Float(..)) raises an\n        error.'
        if func_name in self.dict_fun:
            new_name = self.dict_fun[func_name]
            argstr = self.tree2str_translate(argtree)
            return new_name + '(' + argstr
        elif func_name in ['Eq', 'Ne']:
            op = {'Eq': '==', 'Ne': '!='}
            return '(lambda x, y: x {} y)({}'.format(op[func_name], self.tree2str_translate(argtree))
        else:
            template = '(%s(%s)).evalf(' if self.use_evalf else '%s(%s'
            if self.float_wrap_evalf:
                template = 'float(%s)' % template
            elif self.complex_wrap_evalf:
                template = 'complex(%s)' % template
            float_wrap_evalf = self.float_wrap_evalf
            complex_wrap_evalf = self.complex_wrap_evalf
            self.float_wrap_evalf = False
            self.complex_wrap_evalf = False
            ret = template % (func_name, self.tree2str_translate(argtree))
            self.float_wrap_evalf = float_wrap_evalf
            self.complex_wrap_evalf = complex_wrap_evalf
            return ret

    @classmethod
    def sympy_expression_namespace(cls, expr):
        if False:
            return 10
        'Traverses the (func, args) tree of an expression and creates a SymPy\n        namespace. All other modules are imported only as a module name. That way\n        the namespace is not polluted and rests quite small. It probably causes much\n        more variable lookups and so it takes more time, but there are no tests on\n        that for the moment.'
        if expr is None:
            return {}
        else:
            funcname = str(expr.func)
            regexlist = ["<class \\'sympy[\\w.]*?.([\\w]*)\\'>$", '<function ([\\w]*) at 0x[\\w]*>$']
            for r in regexlist:
                m = re.match(r, funcname)
                if m is not None:
                    funcname = m.groups()[0]
            args_dict = {}
            for a in expr.args:
                if isinstance(a, (Symbol, NumberSymbol)) or a in [I, zoo, oo]:
                    continue
                else:
                    args_dict.update(cls.sympy_expression_namespace(a))
            args_dict.update({funcname: expr.func})
            return args_dict

    @staticmethod
    def sympy_atoms_namespace(expr):
        if False:
            return 10
        'For no real reason this function is separated from\n        sympy_expression_namespace. It can be moved to it.'
        atoms = expr.atoms(Symbol, NumberSymbol, I, zoo, oo)
        d = {}
        for a in atoms:
            d[str(a)] = a
        return d