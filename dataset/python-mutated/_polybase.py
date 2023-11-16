"""
Abstract base class for the various polynomial Classes.

The ABCPolyBase class provides the methods needed to implement the common API
for the various polynomial classes. It operates as a mixin, but uses the
abc module from the stdlib, hence it is only available for Python >= 2.6.

"""
import os
import abc
import numbers
import numpy as np
from . import polyutils as pu
__all__ = ['ABCPolyBase']

class ABCPolyBase(abc.ABC):
    """An abstract base class for immutable series classes.

    ABCPolyBase provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' along with the
    methods listed below.

    .. versionadded:: 1.9.0

    Parameters
    ----------
    coef : array_like
        Series coefficients in order of increasing degree, i.e.,
        ``(1, 2, 3)`` gives ``1*P_0(x) + 2*P_1(x) + 3*P_2(x)``, where
        ``P_i`` is the basis polynomials of degree ``i``.
    domain : (2,) array_like, optional
        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
        to the interval ``[window[0], window[1]]`` by shifting and scaling.
        The default value is the derived class domain.
    window : (2,) array_like, optional
        Window, see domain for its use. The default value is the
        derived class window.
    symbol : str, optional
        Symbol used to represent the independent variable in string 
        representations of the polynomial expression, e.g. for printing.
        The symbol must be a valid Python identifier. Default value is 'x'.

        .. versionadded:: 1.24

    Attributes
    ----------
    coef : (N,) ndarray
        Series coefficients in order of increasing degree.
    domain : (2,) ndarray
        Domain that is mapped to window.
    window : (2,) ndarray
        Window that domain is mapped to.
    symbol : str
        Symbol representing the independent variable.

    Class Attributes
    ----------------
    maxpower : int
        Maximum power allowed, i.e., the largest number ``n`` such that
        ``p(x)**n`` is allowed. This is to limit runaway polynomial size.
    domain : (2,) ndarray
        Default domain of the class.
    window : (2,) ndarray
        Default window of the class.

    """
    __hash__ = None
    __array_ufunc__ = None
    maxpower = 100
    _superscript_mapping = str.maketrans({'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'})
    _subscript_mapping = str.maketrans({'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'})
    _use_unicode = not os.name == 'nt'

    @property
    def symbol(self):
        if False:
            return 10
        return self._symbol

    @property
    @abc.abstractmethod
    def domain(self):
        if False:
            return 10
        pass

    @property
    @abc.abstractmethod
    def window(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @property
    @abc.abstractmethod
    def basis_name(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @staticmethod
    @abc.abstractmethod
    def _add(c1, c2):
        if False:
            return 10
        pass

    @staticmethod
    @abc.abstractmethod
    def _sub(c1, c2):
        if False:
            i = 10
            return i + 15
        pass

    @staticmethod
    @abc.abstractmethod
    def _mul(c1, c2):
        if False:
            i = 10
            return i + 15
        pass

    @staticmethod
    @abc.abstractmethod
    def _div(c1, c2):
        if False:
            for i in range(10):
                print('nop')
        pass

    @staticmethod
    @abc.abstractmethod
    def _pow(c, pow, maxpower=None):
        if False:
            while True:
                i = 10
        pass

    @staticmethod
    @abc.abstractmethod
    def _val(x, c):
        if False:
            print('Hello World!')
        pass

    @staticmethod
    @abc.abstractmethod
    def _int(c, m, k, lbnd, scl):
        if False:
            return 10
        pass

    @staticmethod
    @abc.abstractmethod
    def _der(c, m, scl):
        if False:
            return 10
        pass

    @staticmethod
    @abc.abstractmethod
    def _fit(x, y, deg, rcond, full):
        if False:
            print('Hello World!')
        pass

    @staticmethod
    @abc.abstractmethod
    def _line(off, scl):
        if False:
            return 10
        pass

    @staticmethod
    @abc.abstractmethod
    def _roots(c):
        if False:
            while True:
                i = 10
        pass

    @staticmethod
    @abc.abstractmethod
    def _fromroots(r):
        if False:
            print('Hello World!')
        pass

    def has_samecoef(self, other):
        if False:
            i = 10
            return i + 15
        'Check if coefficients match.\n\n        .. versionadded:: 1.6.0\n\n        Parameters\n        ----------\n        other : class instance\n            The other class must have the ``coef`` attribute.\n\n        Returns\n        -------\n        bool : boolean\n            True if the coefficients are the same, False otherwise.\n\n        '
        if len(self.coef) != len(other.coef):
            return False
        elif not np.all(self.coef == other.coef):
            return False
        else:
            return True

    def has_samedomain(self, other):
        if False:
            return 10
        'Check if domains match.\n\n        .. versionadded:: 1.6.0\n\n        Parameters\n        ----------\n        other : class instance\n            The other class must have the ``domain`` attribute.\n\n        Returns\n        -------\n        bool : boolean\n            True if the domains are the same, False otherwise.\n\n        '
        return np.all(self.domain == other.domain)

    def has_samewindow(self, other):
        if False:
            print('Hello World!')
        'Check if windows match.\n\n        .. versionadded:: 1.6.0\n\n        Parameters\n        ----------\n        other : class instance\n            The other class must have the ``window`` attribute.\n\n        Returns\n        -------\n        bool : boolean\n            True if the windows are the same, False otherwise.\n\n        '
        return np.all(self.window == other.window)

    def has_sametype(self, other):
        if False:
            while True:
                i = 10
        'Check if types match.\n\n        .. versionadded:: 1.7.0\n\n        Parameters\n        ----------\n        other : object\n            Class instance.\n\n        Returns\n        -------\n        bool : boolean\n            True if other is same class as self\n\n        '
        return isinstance(other, self.__class__)

    def _get_coefficients(self, other):
        if False:
            print('Hello World!')
        'Interpret other as polynomial coefficients.\n\n        The `other` argument is checked to see if it is of the same\n        class as self with identical domain and window. If so,\n        return its coefficients, otherwise return `other`.\n\n        .. versionadded:: 1.9.0\n\n        Parameters\n        ----------\n        other : anything\n            Object to be checked.\n\n        Returns\n        -------\n        coef\n            The coefficients of`other` if it is a compatible instance,\n            of ABCPolyBase, otherwise `other`.\n\n        Raises\n        ------\n        TypeError\n            When `other` is an incompatible instance of ABCPolyBase.\n\n        '
        if isinstance(other, ABCPolyBase):
            if not isinstance(other, self.__class__):
                raise TypeError('Polynomial types differ')
            elif not np.all(self.domain == other.domain):
                raise TypeError('Domains differ')
            elif not np.all(self.window == other.window):
                raise TypeError('Windows differ')
            elif self.symbol != other.symbol:
                raise ValueError('Polynomial symbols differ')
            return other.coef
        return other

    def __init__(self, coef, domain=None, window=None, symbol='x'):
        if False:
            print('Hello World!')
        [coef] = pu.as_series([coef], trim=False)
        self.coef = coef
        if domain is not None:
            [domain] = pu.as_series([domain], trim=False)
            if len(domain) != 2:
                raise ValueError('Domain has wrong number of elements.')
            self.domain = domain
        if window is not None:
            [window] = pu.as_series([window], trim=False)
            if len(window) != 2:
                raise ValueError('Window has wrong number of elements.')
            self.window = window
        try:
            if not symbol.isidentifier():
                raise ValueError('Symbol string must be a valid Python identifier')
        except AttributeError:
            raise TypeError('Symbol must be a non-empty string')
        self._symbol = symbol

    def __repr__(self):
        if False:
            return 10
        coef = repr(self.coef)[6:-1]
        domain = repr(self.domain)[6:-1]
        window = repr(self.window)[6:-1]
        name = self.__class__.__name__
        return f"{name}({coef}, domain={domain}, window={window}, symbol='{self.symbol}')"

    def __format__(self, fmt_str):
        if False:
            for i in range(10):
                print('nop')
        if fmt_str == '':
            return self.__str__()
        if fmt_str not in ('ascii', 'unicode'):
            raise ValueError(f"Unsupported format string '{fmt_str}' passed to {self.__class__}.__format__. Valid options are 'ascii' and 'unicode'")
        if fmt_str == 'ascii':
            return self._generate_string(self._str_term_ascii)
        return self._generate_string(self._str_term_unicode)

    def __str__(self):
        if False:
            return 10
        if self._use_unicode:
            return self._generate_string(self._str_term_unicode)
        return self._generate_string(self._str_term_ascii)

    def _generate_string(self, term_method):
        if False:
            print('Hello World!')
        '\n        Generate the full string representation of the polynomial, using\n        ``term_method`` to generate each polynomial term.\n        '
        linewidth = np.get_printoptions().get('linewidth', 75)
        if linewidth < 1:
            linewidth = 1
        out = pu.format_float(self.coef[0])
        for (i, coef) in enumerate(self.coef[1:]):
            out += ' '
            power = str(i + 1)
            try:
                if coef >= 0:
                    next_term = f'+ ' + pu.format_float(coef, parens=True)
                else:
                    next_term = f'- ' + pu.format_float(-coef, parens=True)
            except TypeError:
                next_term = f'+ {coef}'
            next_term += term_method(power, self.symbol)
            line_len = len(out.split('\n')[-1]) + len(next_term)
            if i < len(self.coef[1:]) - 1:
                line_len += 2
            if line_len >= linewidth:
                next_term = next_term.replace(' ', '\n', 1)
            out += next_term
        return out

    @classmethod
    def _str_term_unicode(cls, i, arg_str):
        if False:
            for i in range(10):
                print('nop')
        '\n        String representation of single polynomial term using unicode\n        characters for superscripts and subscripts.\n        '
        if cls.basis_name is None:
            raise NotImplementedError('Subclasses must define either a basis_name, or override _str_term_unicode(cls, i, arg_str)')
        return f'·{cls.basis_name}{i.translate(cls._subscript_mapping)}({arg_str})'

    @classmethod
    def _str_term_ascii(cls, i, arg_str):
        if False:
            return 10
        '\n        String representation of a single polynomial term using ** and _ to\n        represent superscripts and subscripts, respectively.\n        '
        if cls.basis_name is None:
            raise NotImplementedError('Subclasses must define either a basis_name, or override _str_term_ascii(cls, i, arg_str)')
        return f' {cls.basis_name}_{i}({arg_str})'

    @classmethod
    def _repr_latex_term(cls, i, arg_str, needs_parens):
        if False:
            print('Hello World!')
        if cls.basis_name is None:
            raise NotImplementedError('Subclasses must define either a basis name, or override _repr_latex_term(i, arg_str, needs_parens)')
        return f'{{{cls.basis_name}}}_{{{i}}}({arg_str})'

    @staticmethod
    def _repr_latex_scalar(x, parens=False):
        if False:
            while True:
                i = 10
        return '\\text{{{}}}'.format(pu.format_float(x, parens=parens))

    def _repr_latex_(self):
        if False:
            for i in range(10):
                print('nop')
        (off, scale) = self.mapparms()
        if off == 0 and scale == 1:
            term = self.symbol
            needs_parens = False
        elif scale == 1:
            term = f'{self._repr_latex_scalar(off)} + {self.symbol}'
            needs_parens = True
        elif off == 0:
            term = f'{self._repr_latex_scalar(scale)}{self.symbol}'
            needs_parens = True
        else:
            term = f'{self._repr_latex_scalar(off)} + {self._repr_latex_scalar(scale)}{self.symbol}'
            needs_parens = True
        mute = '\\color{{LightGray}}{{{}}}'.format
        parts = []
        for (i, c) in enumerate(self.coef):
            if i == 0:
                coef_str = f'{self._repr_latex_scalar(c)}'
            elif not isinstance(c, numbers.Real):
                coef_str = f' + ({self._repr_latex_scalar(c)})'
            elif not np.signbit(c):
                coef_str = f' + {self._repr_latex_scalar(c, parens=True)}'
            else:
                coef_str = f' - {self._repr_latex_scalar(-c, parens=True)}'
            term_str = self._repr_latex_term(i, term, needs_parens)
            if term_str == '1':
                part = coef_str
            else:
                part = f'{coef_str}\\,{term_str}'
            if c == 0:
                part = mute(part)
            parts.append(part)
        if parts:
            body = ''.join(parts)
        else:
            body = '0'
        return f'${self.symbol} \\mapsto {body}$'

    def __getstate__(self):
        if False:
            return 10
        ret = self.__dict__.copy()
        ret['coef'] = self.coef.copy()
        ret['domain'] = self.domain.copy()
        ret['window'] = self.window.copy()
        ret['symbol'] = self.symbol
        return ret

    def __setstate__(self, dict):
        if False:
            return 10
        self.__dict__ = dict

    def __call__(self, arg):
        if False:
            while True:
                i = 10
        arg = pu.mapdomain(arg, self.domain, self.window)
        return self._val(arg, self.coef)

    def __iter__(self):
        if False:
            return 10
        return iter(self.coef)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.coef)

    def __neg__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__class__(-self.coef, self.domain, self.window, self.symbol)

    def __pos__(self):
        if False:
            while True:
                i = 10
        return self

    def __add__(self, other):
        if False:
            i = 10
            return i + 15
        othercoef = self._get_coefficients(other)
        try:
            coef = self._add(self.coef, othercoef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __sub__(self, other):
        if False:
            i = 10
            return i + 15
        othercoef = self._get_coefficients(other)
        try:
            coef = self._sub(self.coef, othercoef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __mul__(self, other):
        if False:
            return 10
        othercoef = self._get_coefficients(other)
        try:
            coef = self._mul(self.coef, othercoef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __truediv__(self, other):
        if False:
            print('Hello World!')
        if not isinstance(other, numbers.Number) or isinstance(other, bool):
            raise TypeError(f"unsupported types for true division: '{type(self)}', '{type(other)}'")
        return self.__floordiv__(other)

    def __floordiv__(self, other):
        if False:
            for i in range(10):
                print('nop')
        res = self.__divmod__(other)
        if res is NotImplemented:
            return res
        return res[0]

    def __mod__(self, other):
        if False:
            return 10
        res = self.__divmod__(other)
        if res is NotImplemented:
            return res
        return res[1]

    def __divmod__(self, other):
        if False:
            return 10
        othercoef = self._get_coefficients(other)
        try:
            (quo, rem) = self._div(self.coef, othercoef)
        except ZeroDivisionError:
            raise
        except Exception:
            return NotImplemented
        quo = self.__class__(quo, self.domain, self.window, self.symbol)
        rem = self.__class__(rem, self.domain, self.window, self.symbol)
        return (quo, rem)

    def __pow__(self, other):
        if False:
            print('Hello World!')
        coef = self._pow(self.coef, other, maxpower=self.maxpower)
        res = self.__class__(coef, self.domain, self.window, self.symbol)
        return res

    def __radd__(self, other):
        if False:
            i = 10
            return i + 15
        try:
            coef = self._add(other, self.coef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __rsub__(self, other):
        if False:
            return 10
        try:
            coef = self._sub(other, self.coef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __rmul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        try:
            coef = self._mul(other, self.coef)
        except Exception:
            return NotImplemented
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def __rdiv__(self, other):
        if False:
            while True:
                i = 10
        return self.__rfloordiv__(other)

    def __rtruediv__(self, other):
        if False:
            while True:
                i = 10
        return NotImplemented

    def __rfloordiv__(self, other):
        if False:
            return 10
        res = self.__rdivmod__(other)
        if res is NotImplemented:
            return res
        return res[0]

    def __rmod__(self, other):
        if False:
            while True:
                i = 10
        res = self.__rdivmod__(other)
        if res is NotImplemented:
            return res
        return res[1]

    def __rdivmod__(self, other):
        if False:
            i = 10
            return i + 15
        try:
            (quo, rem) = self._div(other, self.coef)
        except ZeroDivisionError:
            raise
        except Exception:
            return NotImplemented
        quo = self.__class__(quo, self.domain, self.window, self.symbol)
        rem = self.__class__(rem, self.domain, self.window, self.symbol)
        return (quo, rem)

    def __eq__(self, other):
        if False:
            return 10
        res = isinstance(other, self.__class__) and np.all(self.domain == other.domain) and np.all(self.window == other.window) and (self.coef.shape == other.coef.shape) and np.all(self.coef == other.coef) and (self.symbol == other.symbol)
        return res

    def __ne__(self, other):
        if False:
            i = 10
            return i + 15
        return not self.__eq__(other)

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a copy.\n\n        Returns\n        -------\n        new_series : series\n            Copy of self.\n\n        '
        return self.__class__(self.coef, self.domain, self.window, self.symbol)

    def degree(self):
        if False:
            for i in range(10):
                print('nop')
        'The degree of the series.\n\n        .. versionadded:: 1.5.0\n\n        Returns\n        -------\n        degree : int\n            Degree of the series, one less than the number of coefficients.\n\n        Examples\n        --------\n\n        Create a polynomial object for ``1 + 7*x + 4*x**2``:\n\n        >>> poly = np.polynomial.Polynomial([1, 7, 4])\n        >>> print(poly)\n        1.0 + 7.0·x + 4.0·x²\n        >>> poly.degree()\n        2\n\n        Note that this method does not check for non-zero coefficients.\n        You must trim the polynomial to remove any trailing zeroes:\n\n        >>> poly = np.polynomial.Polynomial([1, 7, 0])\n        >>> print(poly)\n        1.0 + 7.0·x + 0.0·x²\n        >>> poly.degree()\n        2\n        >>> poly.trim().degree()\n        1\n\n        '
        return len(self) - 1

    def cutdeg(self, deg):
        if False:
            i = 10
            return i + 15
        'Truncate series to the given degree.\n\n        Reduce the degree of the series to `deg` by discarding the\n        high order terms. If `deg` is greater than the current degree a\n        copy of the current series is returned. This can be useful in least\n        squares where the coefficients of the high degree terms may be very\n        small.\n\n        .. versionadded:: 1.5.0\n\n        Parameters\n        ----------\n        deg : non-negative int\n            The series is reduced to degree `deg` by discarding the high\n            order terms. The value of `deg` must be a non-negative integer.\n\n        Returns\n        -------\n        new_series : series\n            New instance of series with reduced degree.\n\n        '
        return self.truncate(deg + 1)

    def trim(self, tol=0):
        if False:
            while True:
                i = 10
        'Remove trailing coefficients\n\n        Remove trailing coefficients until a coefficient is reached whose\n        absolute value greater than `tol` or the beginning of the series is\n        reached. If all the coefficients would be removed the series is set\n        to ``[0]``. A new series instance is returned with the new\n        coefficients.  The current instance remains unchanged.\n\n        Parameters\n        ----------\n        tol : non-negative number.\n            All trailing coefficients less than `tol` will be removed.\n\n        Returns\n        -------\n        new_series : series\n            New instance of series with trimmed coefficients.\n\n        '
        coef = pu.trimcoef(self.coef, tol)
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def truncate(self, size):
        if False:
            return 10
        'Truncate series to length `size`.\n\n        Reduce the series to length `size` by discarding the high\n        degree terms. The value of `size` must be a positive integer. This\n        can be useful in least squares where the coefficients of the\n        high degree terms may be very small.\n\n        Parameters\n        ----------\n        size : positive int\n            The series is reduced to length `size` by discarding the high\n            degree terms. The value of `size` must be a positive integer.\n\n        Returns\n        -------\n        new_series : series\n            New instance of series with truncated coefficients.\n\n        '
        isize = int(size)
        if isize != size or isize < 1:
            raise ValueError('size must be a positive integer')
        if isize >= len(self.coef):
            coef = self.coef
        else:
            coef = self.coef[:isize]
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def convert(self, domain=None, kind=None, window=None):
        if False:
            return 10
        'Convert series to a different kind and/or domain and/or window.\n\n        Parameters\n        ----------\n        domain : array_like, optional\n            The domain of the converted series. If the value is None,\n            the default domain of `kind` is used.\n        kind : class, optional\n            The polynomial series type class to which the current instance\n            should be converted. If kind is None, then the class of the\n            current instance is used.\n        window : array_like, optional\n            The window of the converted series. If the value is None,\n            the default window of `kind` is used.\n\n        Returns\n        -------\n        new_series : series\n            The returned class can be of different type than the current\n            instance and/or have a different domain and/or different\n            window.\n\n        Notes\n        -----\n        Conversion between domains and class types can result in\n        numerically ill defined series.\n\n        '
        if kind is None:
            kind = self.__class__
        if domain is None:
            domain = kind.domain
        if window is None:
            window = kind.window
        return self(kind.identity(domain, window=window, symbol=self.symbol))

    def mapparms(self):
        if False:
            i = 10
            return i + 15
        'Return the mapping parameters.\n\n        The returned values define a linear map ``off + scl*x`` that is\n        applied to the input arguments before the series is evaluated. The\n        map depends on the ``domain`` and ``window``; if the current\n        ``domain`` is equal to the ``window`` the resulting map is the\n        identity.  If the coefficients of the series instance are to be\n        used by themselves outside this class, then the linear function\n        must be substituted for the ``x`` in the standard representation of\n        the base polynomials.\n\n        Returns\n        -------\n        off, scl : float or complex\n            The mapping function is defined by ``off + scl*x``.\n\n        Notes\n        -----\n        If the current domain is the interval ``[l1, r1]`` and the window\n        is ``[l2, r2]``, then the linear mapping function ``L`` is\n        defined by the equations::\n\n            L(l1) = l2\n            L(r1) = r2\n\n        '
        return pu.mapparms(self.domain, self.window)

    def integ(self, m=1, k=[], lbnd=None):
        if False:
            print('Hello World!')
        'Integrate.\n\n        Return a series instance that is the definite integral of the\n        current series.\n\n        Parameters\n        ----------\n        m : non-negative int\n            The number of integrations to perform.\n        k : array_like\n            Integration constants. The first constant is applied to the\n            first integration, the second to the second, and so on. The\n            list of values must less than or equal to `m` in length and any\n            missing values are set to zero.\n        lbnd : Scalar\n            The lower bound of the definite integral.\n\n        Returns\n        -------\n        new_series : series\n            A new series representing the integral. The domain is the same\n            as the domain of the integrated series.\n\n        '
        (off, scl) = self.mapparms()
        if lbnd is None:
            lbnd = 0
        else:
            lbnd = off + scl * lbnd
        coef = self._int(self.coef, m, k, lbnd, 1.0 / scl)
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def deriv(self, m=1):
        if False:
            i = 10
            return i + 15
        'Differentiate.\n\n        Return a series instance of that is the derivative of the current\n        series.\n\n        Parameters\n        ----------\n        m : non-negative int\n            Find the derivative of order `m`.\n\n        Returns\n        -------\n        new_series : series\n            A new series representing the derivative. The domain is the same\n            as the domain of the differentiated series.\n\n        '
        (off, scl) = self.mapparms()
        coef = self._der(self.coef, m, scl)
        return self.__class__(coef, self.domain, self.window, self.symbol)

    def roots(self):
        if False:
            while True:
                i = 10
        'Return the roots of the series polynomial.\n\n        Compute the roots for the series. Note that the accuracy of the\n        roots decreases the further outside the `domain` they lie.\n\n        Returns\n        -------\n        roots : ndarray\n            Array containing the roots of the series.\n\n        '
        roots = self._roots(self.coef)
        return pu.mapdomain(roots, self.window, self.domain)

    def linspace(self, n=100, domain=None):
        if False:
            print('Hello World!')
        'Return x, y values at equally spaced points in domain.\n\n        Returns the x, y values at `n` linearly spaced points across the\n        domain.  Here y is the value of the polynomial at the points x. By\n        default the domain is the same as that of the series instance.\n        This method is intended mostly as a plotting aid.\n\n        .. versionadded:: 1.5.0\n\n        Parameters\n        ----------\n        n : int, optional\n            Number of point pairs to return. The default value is 100.\n        domain : {None, array_like}, optional\n            If not None, the specified domain is used instead of that of\n            the calling instance. It should be of the form ``[beg,end]``.\n            The default is None which case the class domain is used.\n\n        Returns\n        -------\n        x, y : ndarray\n            x is equal to linspace(self.domain[0], self.domain[1], n) and\n            y is the series evaluated at element of x.\n\n        '
        if domain is None:
            domain = self.domain
        x = np.linspace(domain[0], domain[1], n)
        y = self(x)
        return (x, y)

    @classmethod
    def fit(cls, x, y, deg, domain=None, rcond=None, full=False, w=None, window=None, symbol='x'):
        if False:
            return 10
        "Least squares fit to data.\n\n        Return a series instance that is the least squares fit to the data\n        `y` sampled at `x`. The domain of the returned instance can be\n        specified and this will often result in a superior fit with less\n        chance of ill conditioning.\n\n        Parameters\n        ----------\n        x : array_like, shape (M,)\n            x-coordinates of the M sample points ``(x[i], y[i])``.\n        y : array_like, shape (M,)\n            y-coordinates of the M sample points ``(x[i], y[i])``.\n        deg : int or 1-D array_like\n            Degree(s) of the fitting polynomials. If `deg` is a single integer\n            all terms up to and including the `deg`'th term are included in the\n            fit. For NumPy versions >= 1.11.0 a list of integers specifying the\n            degrees of the terms to include may be used instead.\n        domain : {None, [beg, end], []}, optional\n            Domain to use for the returned series. If ``None``,\n            then a minimal domain that covers the points `x` is chosen.  If\n            ``[]`` the class domain is used. The default value was the\n            class domain in NumPy 1.4 and ``None`` in later versions.\n            The ``[]`` option was added in numpy 1.5.0.\n        rcond : float, optional\n            Relative condition number of the fit. Singular values smaller\n            than this relative to the largest singular value will be\n            ignored. The default value is len(x)*eps, where eps is the\n            relative precision of the float type, about 2e-16 in most\n            cases.\n        full : bool, optional\n            Switch determining nature of return value. When it is False\n            (the default) just the coefficients are returned, when True\n            diagnostic information from the singular value decomposition is\n            also returned.\n        w : array_like, shape (M,), optional\n            Weights. If not None, the weight ``w[i]`` applies to the unsquared\n            residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are\n            chosen so that the errors of the products ``w[i]*y[i]`` all have\n            the same variance.  When using inverse-variance weighting, use\n            ``w[i] = 1/sigma(y[i])``.  The default value is None.\n\n            .. versionadded:: 1.5.0\n        window : {[beg, end]}, optional\n            Window to use for the returned series. The default\n            value is the default class domain\n\n            .. versionadded:: 1.6.0\n        symbol : str, optional\n            Symbol representing the independent variable. Default is 'x'.\n\n        Returns\n        -------\n        new_series : series\n            A series that represents the least squares fit to the data and\n            has the domain and window specified in the call. If the\n            coefficients for the unscaled and unshifted basis polynomials are\n            of interest, do ``new_series.convert().coef``.\n\n        [resid, rank, sv, rcond] : list\n            These values are only returned if ``full == True``\n\n            - resid -- sum of squared residuals of the least squares fit\n            - rank -- the numerical rank of the scaled Vandermonde matrix\n            - sv -- singular values of the scaled Vandermonde matrix\n            - rcond -- value of `rcond`.\n\n            For more details, see `linalg.lstsq`.\n\n        "
        if domain is None:
            domain = pu.getdomain(x)
        elif type(domain) is list and len(domain) == 0:
            domain = cls.domain
        if window is None:
            window = cls.window
        xnew = pu.mapdomain(x, domain, window)
        res = cls._fit(xnew, y, deg, w=w, rcond=rcond, full=full)
        if full:
            [coef, status] = res
            return (cls(coef, domain=domain, window=window, symbol=symbol), status)
        else:
            coef = res
            return cls(coef, domain=domain, window=window, symbol=symbol)

    @classmethod
    def fromroots(cls, roots, domain=[], window=None, symbol='x'):
        if False:
            i = 10
            return i + 15
        "Return series instance that has the specified roots.\n\n        Returns a series representing the product\n        ``(x - r[0])*(x - r[1])*...*(x - r[n-1])``, where ``r`` is a\n        list of roots.\n\n        Parameters\n        ----------\n        roots : array_like\n            List of roots.\n        domain : {[], None, array_like}, optional\n            Domain for the resulting series. If None the domain is the\n            interval from the smallest root to the largest. If [] the\n            domain is the class domain. The default is [].\n        window : {None, array_like}, optional\n            Window for the returned series. If None the class window is\n            used. The default is None.\n        symbol : str, optional\n            Symbol representing the independent variable. Default is 'x'.\n\n        Returns\n        -------\n        new_series : series\n            Series with the specified roots.\n\n        "
        [roots] = pu.as_series([roots], trim=False)
        if domain is None:
            domain = pu.getdomain(roots)
        elif type(domain) is list and len(domain) == 0:
            domain = cls.domain
        if window is None:
            window = cls.window
        deg = len(roots)
        (off, scl) = pu.mapparms(domain, window)
        rnew = off + scl * roots
        coef = cls._fromroots(rnew) / scl ** deg
        return cls(coef, domain=domain, window=window, symbol=symbol)

    @classmethod
    def identity(cls, domain=None, window=None, symbol='x'):
        if False:
            return 10
        "Identity function.\n\n        If ``p`` is the returned series, then ``p(x) == x`` for all\n        values of x.\n\n        Parameters\n        ----------\n        domain : {None, array_like}, optional\n            If given, the array must be of the form ``[beg, end]``, where\n            ``beg`` and ``end`` are the endpoints of the domain. If None is\n            given then the class domain is used. The default is None.\n        window : {None, array_like}, optional\n            If given, the resulting array must be if the form\n            ``[beg, end]``, where ``beg`` and ``end`` are the endpoints of\n            the window. If None is given then the class window is used. The\n            default is None.\n        symbol : str, optional\n            Symbol representing the independent variable. Default is 'x'.\n\n        Returns\n        -------\n        new_series : series\n             Series of representing the identity.\n\n        "
        if domain is None:
            domain = cls.domain
        if window is None:
            window = cls.window
        (off, scl) = pu.mapparms(window, domain)
        coef = cls._line(off, scl)
        return cls(coef, domain, window, symbol)

    @classmethod
    def basis(cls, deg, domain=None, window=None, symbol='x'):
        if False:
            for i in range(10):
                print('nop')
        "Series basis polynomial of degree `deg`.\n\n        Returns the series representing the basis polynomial of degree `deg`.\n\n        .. versionadded:: 1.7.0\n\n        Parameters\n        ----------\n        deg : int\n            Degree of the basis polynomial for the series. Must be >= 0.\n        domain : {None, array_like}, optional\n            If given, the array must be of the form ``[beg, end]``, where\n            ``beg`` and ``end`` are the endpoints of the domain. If None is\n            given then the class domain is used. The default is None.\n        window : {None, array_like}, optional\n            If given, the resulting array must be if the form\n            ``[beg, end]``, where ``beg`` and ``end`` are the endpoints of\n            the window. If None is given then the class window is used. The\n            default is None.\n        symbol : str, optional\n            Symbol representing the independent variable. Default is 'x'.\n\n        Returns\n        -------\n        new_series : series\n            A series with the coefficient of the `deg` term set to one and\n            all others zero.\n\n        "
        if domain is None:
            domain = cls.domain
        if window is None:
            window = cls.window
        ideg = int(deg)
        if ideg != deg or ideg < 0:
            raise ValueError('deg must be non-negative integer')
        return cls([0] * ideg + [1], domain, window, symbol)

    @classmethod
    def cast(cls, series, domain=None, window=None):
        if False:
            return 10
        'Convert series to series of this class.\n\n        The `series` is expected to be an instance of some polynomial\n        series of one of the types supported by by the numpy.polynomial\n        module, but could be some other class that supports the convert\n        method.\n\n        .. versionadded:: 1.7.0\n\n        Parameters\n        ----------\n        series : series\n            The series instance to be converted.\n        domain : {None, array_like}, optional\n            If given, the array must be of the form ``[beg, end]``, where\n            ``beg`` and ``end`` are the endpoints of the domain. If None is\n            given then the class domain is used. The default is None.\n        window : {None, array_like}, optional\n            If given, the resulting array must be if the form\n            ``[beg, end]``, where ``beg`` and ``end`` are the endpoints of\n            the window. If None is given then the class window is used. The\n            default is None.\n\n        Returns\n        -------\n        new_series : series\n            A series of the same kind as the calling class and equal to\n            `series` when evaluated.\n\n        See Also\n        --------\n        convert : similar instance method\n\n        '
        if domain is None:
            domain = cls.domain
        if window is None:
            window = cls.window
        return series.convert(domain, cls, window)