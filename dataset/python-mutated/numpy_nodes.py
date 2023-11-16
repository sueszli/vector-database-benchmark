from sympy.core.function import Add, ArgumentIndexError, Function
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.functions.elementary.exponential import exp, log

def _logaddexp(x1, x2, *, evaluate=True):
    if False:
        while True:
            i = 10
    return log(Add(exp(x1, evaluate=evaluate), exp(x2, evaluate=evaluate), evaluate=evaluate))
_two = S.One * 2
_ln2 = log(_two)

def _lb(x, *, evaluate=True):
    if False:
        for i in range(10):
            print('nop')
    return log(x, evaluate=evaluate) / _ln2

def _exp2(x, *, evaluate=True):
    if False:
        i = 10
        return i + 15
    return Pow(_two, x, evaluate=evaluate)

def _logaddexp2(x1, x2, *, evaluate=True):
    if False:
        return 10
    return _lb(Add(_exp2(x1, evaluate=evaluate), _exp2(x2, evaluate=evaluate), evaluate=evaluate))

class logaddexp(Function):
    """ Logarithm of the sum of exponentiations of the inputs.

    Helper class for use with e.g. numpy.logaddexp

    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html
    """
    nargs = 2

    def __new__(cls, *args):
        if False:
            while True:
                i = 10
        return Function.__new__(cls, *sorted(args, key=default_sort_key))

    def fdiff(self, argindex=1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the first derivative of this function.\n        '
        if argindex == 1:
            (wrt, other) = self.args
        elif argindex == 2:
            (other, wrt) = self.args
        else:
            raise ArgumentIndexError(self, argindex)
        return S.One / (S.One + exp(other - wrt))

    def _eval_rewrite_as_log(self, x1, x2, **kwargs):
        if False:
            while True:
                i = 10
        return _logaddexp(x1, x2)

    def _eval_evalf(self, *args, **kwargs):
        if False:
            return 10
        return self.rewrite(log).evalf(*args, **kwargs)

    def _eval_simplify(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        (a, b) = (x.simplify(**kwargs) for x in self.args)
        candidate = _logaddexp(a, b)
        if candidate != _logaddexp(a, b, evaluate=False):
            return candidate
        else:
            return logaddexp(a, b)

class logaddexp2(Function):
    """ Logarithm of the sum of exponentiations of the inputs in base-2.

    Helper class for use with e.g. numpy.logaddexp2

    See Also
    ========

    https://numpy.org/doc/stable/reference/generated/numpy.logaddexp2.html
    """
    nargs = 2

    def __new__(cls, *args):
        if False:
            i = 10
            return i + 15
        return Function.__new__(cls, *sorted(args, key=default_sort_key))

    def fdiff(self, argindex=1):
        if False:
            return 10
        '\n        Returns the first derivative of this function.\n        '
        if argindex == 1:
            (wrt, other) = self.args
        elif argindex == 2:
            (other, wrt) = self.args
        else:
            raise ArgumentIndexError(self, argindex)
        return S.One / (S.One + _exp2(other - wrt))

    def _eval_rewrite_as_log(self, x1, x2, **kwargs):
        if False:
            while True:
                i = 10
        return _logaddexp2(x1, x2)

    def _eval_evalf(self, *args, **kwargs):
        if False:
            return 10
        return self.rewrite(log).evalf(*args, **kwargs)

    def _eval_simplify(self, *args, **kwargs):
        if False:
            print('Hello World!')
        (a, b) = (x.simplify(**kwargs).factor() for x in self.args)
        candidate = _logaddexp2(a, b)
        if candidate != _logaddexp2(a, b, evaluate=False):
            return candidate
        else:
            return logaddexp2(a, b)