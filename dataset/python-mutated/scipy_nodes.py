from sympy.core.function import Add, ArgumentIndexError, Function
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import cos, sin

def _cosm1(x, *, evaluate=True):
    if False:
        return 10
    return Add(cos(x, evaluate=evaluate), -S.One, evaluate=evaluate)

class cosm1(Function):
    """ Minus one plus cosine of x, i.e. cos(x) - 1. For use when x is close to zero.

    Helper class for use with e.g. scipy.special.cosm1
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.cosm1.html
    """
    nargs = 1

    def fdiff(self, argindex=1):
        if False:
            while True:
                i = 10
        '\n        Returns the first derivative of this function.\n        '
        if argindex == 1:
            return -sin(*self.args)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_cos(self, x, **kwargs):
        if False:
            i = 10
            return i + 15
        return _cosm1(x)

    def _eval_evalf(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.rewrite(cos).evalf(*args, **kwargs)

    def _eval_simplify(self, **kwargs):
        if False:
            print('Hello World!')
        (x,) = self.args
        candidate = _cosm1(x.simplify(**kwargs))
        if candidate != _cosm1(x, evaluate=False):
            return candidate
        else:
            return cosm1(x)

def _powm1(x, y, *, evaluate=True):
    if False:
        i = 10
        return i + 15
    return Add(Pow(x, y, evaluate=evaluate), -S.One, evaluate=evaluate)

class powm1(Function):
    """ Minus one plus x to the power of y, i.e. x**y - 1. For use when x is close to one or y is close to zero.

    Helper class for use with e.g. scipy.special.powm1
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.powm1.html
    """
    nargs = 2

    def fdiff(self, argindex=1):
        if False:
            while True:
                i = 10
        '\n        Returns the first derivative of this function.\n        '
        if argindex == 1:
            return Pow(self.args[0], self.args[1]) * self.args[1] / self.args[0]
        elif argindex == 2:
            return log(self.args[0]) * Pow(*self.args)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Pow(self, x, y, **kwargs):
        if False:
            return 10
        return _powm1(x, y)

    def _eval_evalf(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.rewrite(Pow).evalf(*args, **kwargs)

    def _eval_simplify(self, **kwargs):
        if False:
            i = 10
            return i + 15
        (x, y) = self.args
        candidate = _powm1(x.simplify(**kwargs), y.simplify(**kwargs))
        if candidate != _powm1(x, y, evaluate=False):
            return candidate
        else:
            return powm1(x, y)