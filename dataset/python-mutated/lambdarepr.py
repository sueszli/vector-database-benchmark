from .pycode import PythonCodePrinter, MpmathPrinter
from .numpy import NumPyPrinter
from sympy.core.sorting import default_sort_key
__all__ = ['PythonCodePrinter', 'MpmathPrinter', 'NumPyPrinter', 'LambdaPrinter', 'NumPyPrinter', 'IntervalPrinter', 'lambdarepr']

class LambdaPrinter(PythonCodePrinter):
    """
    This printer converts expressions into strings that can be used by
    lambdify.
    """
    printmethod = '_lambdacode'

    def _print_And(self, expr):
        if False:
            return 10
        result = ['(']
        for arg in sorted(expr.args, key=default_sort_key):
            result.extend(['(', self._print(arg), ')'])
            result.append(' and ')
        result = result[:-1]
        result.append(')')
        return ''.join(result)

    def _print_Or(self, expr):
        if False:
            print('Hello World!')
        result = ['(']
        for arg in sorted(expr.args, key=default_sort_key):
            result.extend(['(', self._print(arg), ')'])
            result.append(' or ')
        result = result[:-1]
        result.append(')')
        return ''.join(result)

    def _print_Not(self, expr):
        if False:
            return 10
        result = ['(', 'not (', self._print(expr.args[0]), '))']
        return ''.join(result)

    def _print_BooleanTrue(self, expr):
        if False:
            return 10
        return 'True'

    def _print_BooleanFalse(self, expr):
        if False:
            while True:
                i = 10
        return 'False'

    def _print_ITE(self, expr):
        if False:
            for i in range(10):
                print('nop')
        result = ['((', self._print(expr.args[1]), ') if (', self._print(expr.args[0]), ') else (', self._print(expr.args[2]), '))']
        return ''.join(result)

    def _print_NumberSymbol(self, expr):
        if False:
            while True:
                i = 10
        return str(expr)

    def _print_Pow(self, expr, **kwargs):
        if False:
            return 10
        return super(PythonCodePrinter, self)._print_Pow(expr, **kwargs)

class NumExprPrinter(LambdaPrinter):
    printmethod = '_numexprcode'
    _numexpr_functions = {'sin': 'sin', 'cos': 'cos', 'tan': 'tan', 'asin': 'arcsin', 'acos': 'arccos', 'atan': 'arctan', 'atan2': 'arctan2', 'sinh': 'sinh', 'cosh': 'cosh', 'tanh': 'tanh', 'asinh': 'arcsinh', 'acosh': 'arccosh', 'atanh': 'arctanh', 'ln': 'log', 'log': 'log', 'exp': 'exp', 'sqrt': 'sqrt', 'Abs': 'abs', 'conjugate': 'conj', 'im': 'imag', 're': 'real', 'where': 'where', 'complex': 'complex', 'contains': 'contains'}
    module = 'numexpr'

    def _print_ImaginaryUnit(self, expr):
        if False:
            print('Hello World!')
        return '1j'

    def _print_seq(self, seq, delimiter=', '):
        if False:
            return 10
        s = [self._print(item) for item in seq]
        if s:
            return delimiter.join(s)
        else:
            return ''

    def _print_Function(self, e):
        if False:
            for i in range(10):
                print('nop')
        func_name = e.func.__name__
        nstr = self._numexpr_functions.get(func_name, None)
        if nstr is None:
            if hasattr(e, '_imp_'):
                return '(%s)' % self._print(e._imp_(*e.args))
            else:
                raise TypeError("numexpr does not support function '%s'" % func_name)
        return '%s(%s)' % (nstr, self._print_seq(e.args))

    def _print_Piecewise(self, expr):
        if False:
            i = 10
            return i + 15
        'Piecewise function printer'
        exprs = [self._print(arg.expr) for arg in expr.args]
        conds = [self._print(arg.cond) for arg in expr.args]
        ans = []
        parenthesis_count = 0
        is_last_cond_True = False
        for (cond, expr) in zip(conds, exprs):
            if cond == 'True':
                ans.append(expr)
                is_last_cond_True = True
                break
            else:
                ans.append('where(%s, %s, ' % (cond, expr))
                parenthesis_count += 1
        if not is_last_cond_True:
            ans.append('log(-1)')
        return ''.join(ans) + ')' * parenthesis_count

    def _print_ITE(self, expr):
        if False:
            while True:
                i = 10
        from sympy.functions.elementary.piecewise import Piecewise
        return self._print(expr.rewrite(Piecewise))

    def blacklisted(self, expr):
        if False:
            i = 10
            return i + 15
        raise TypeError('numexpr cannot be used with %s' % expr.__class__.__name__)
    _print_SparseRepMatrix = _print_MutableSparseMatrix = _print_ImmutableSparseMatrix = _print_Matrix = _print_DenseMatrix = _print_MutableDenseMatrix = _print_ImmutableMatrix = _print_ImmutableDenseMatrix = blacklisted
    _print_list = _print_tuple = _print_Tuple = _print_dict = _print_Dict = blacklisted

    def _print_NumExprEvaluate(self, expr):
        if False:
            for i in range(10):
                print('nop')
        evaluate = self._module_format(self.module + '.evaluate')
        return "%s('%s', truediv=True)" % (evaluate, self._print(expr.expr))

    def doprint(self, expr):
        if False:
            i = 10
            return i + 15
        from sympy.codegen.ast import CodegenAST
        from sympy.codegen.pynodes import NumExprEvaluate
        if not isinstance(expr, CodegenAST):
            expr = NumExprEvaluate(expr)
        return super().doprint(expr)

    def _print_Return(self, expr):
        if False:
            while True:
                i = 10
        from sympy.codegen.pynodes import NumExprEvaluate
        (r,) = expr.args
        if not isinstance(r, NumExprEvaluate):
            expr = expr.func(NumExprEvaluate(r))
        return super()._print_Return(expr)

    def _print_Assignment(self, expr):
        if False:
            for i in range(10):
                print('nop')
        from sympy.codegen.pynodes import NumExprEvaluate
        (lhs, rhs, *args) = expr.args
        if not isinstance(rhs, NumExprEvaluate):
            expr = expr.func(lhs, NumExprEvaluate(rhs), *args)
        return super()._print_Assignment(expr)

    def _print_CodeBlock(self, expr):
        if False:
            return 10
        from sympy.codegen.ast import CodegenAST
        from sympy.codegen.pynodes import NumExprEvaluate
        args = [arg if isinstance(arg, CodegenAST) else NumExprEvaluate(arg) for arg in expr.args]
        return super()._print_CodeBlock(self, expr.func(*args))

class IntervalPrinter(MpmathPrinter, LambdaPrinter):
    """Use ``lambda`` printer but print numbers as ``mpi`` intervals. """

    def _print_Integer(self, expr):
        if False:
            while True:
                i = 10
        return "mpi('%s')" % super(PythonCodePrinter, self)._print_Integer(expr)

    def _print_Rational(self, expr):
        if False:
            print('Hello World!')
        return "mpi('%s')" % super(PythonCodePrinter, self)._print_Rational(expr)

    def _print_Half(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return "mpi('%s')" % super(PythonCodePrinter, self)._print_Rational(expr)

    def _print_Pow(self, expr):
        if False:
            for i in range(10):
                print('nop')
        return super(MpmathPrinter, self)._print_Pow(expr, rational=True)
for k in NumExprPrinter._numexpr_functions:
    setattr(NumExprPrinter, '_print_%s' % k, NumExprPrinter._print_Function)

def lambdarepr(expr, **settings):
    if False:
        while True:
            i = 10
    '\n    Returns a string usable for lambdifying.\n    '
    return LambdaPrinter(settings).doprint(expr)