from scipy._lib._util import getfullargspec_no_self as _getfullargspec
import sys
import keyword
import re
import types
import warnings
from itertools import zip_longest
from scipy._lib import doccer
from ._distr_params import distcont, distdiscrete
from scipy._lib._util import check_random_state
from scipy.special import comb, entr
from scipy import optimize
from scipy import integrate
from scipy._lib._finite_differences import _derivative
from scipy import stats
from numpy import arange, putmask, ones, shape, ndarray, zeros, floor, logical_and, log, sqrt, place, argmax, vectorize, asarray, nan, inf, isinf, empty
import numpy as np
from ._constants import _XMAX, _LOGXMAX
from ._censored_data import CensoredData
from scipy.stats._warnings_errors import FitError
docheaders = {'methods': '\nMethods\n-------\n', 'notes': '\nNotes\n-----\n', 'examples': '\nExamples\n--------\n'}
_doc_rvs = 'rvs(%(shapes)s, loc=0, scale=1, size=1, random_state=None)\n    Random variates.\n'
_doc_pdf = 'pdf(x, %(shapes)s, loc=0, scale=1)\n    Probability density function.\n'
_doc_logpdf = 'logpdf(x, %(shapes)s, loc=0, scale=1)\n    Log of the probability density function.\n'
_doc_pmf = 'pmf(k, %(shapes)s, loc=0, scale=1)\n    Probability mass function.\n'
_doc_logpmf = 'logpmf(k, %(shapes)s, loc=0, scale=1)\n    Log of the probability mass function.\n'
_doc_cdf = 'cdf(x, %(shapes)s, loc=0, scale=1)\n    Cumulative distribution function.\n'
_doc_logcdf = 'logcdf(x, %(shapes)s, loc=0, scale=1)\n    Log of the cumulative distribution function.\n'
_doc_sf = 'sf(x, %(shapes)s, loc=0, scale=1)\n    Survival function  (also defined as ``1 - cdf``, but `sf` is sometimes more accurate).\n'
_doc_logsf = 'logsf(x, %(shapes)s, loc=0, scale=1)\n    Log of the survival function.\n'
_doc_ppf = 'ppf(q, %(shapes)s, loc=0, scale=1)\n    Percent point function (inverse of ``cdf`` --- percentiles).\n'
_doc_isf = 'isf(q, %(shapes)s, loc=0, scale=1)\n    Inverse survival function (inverse of ``sf``).\n'
_doc_moment = 'moment(order, %(shapes)s, loc=0, scale=1)\n    Non-central moment of the specified order.\n'
_doc_stats = "stats(%(shapes)s, loc=0, scale=1, moments='mv')\n    Mean('m'), variance('v'), skew('s'), and/or kurtosis('k').\n"
_doc_entropy = 'entropy(%(shapes)s, loc=0, scale=1)\n    (Differential) entropy of the RV.\n'
_doc_fit = 'fit(data)\n    Parameter estimates for generic data.\n    See `scipy.stats.rv_continuous.fit <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html#scipy.stats.rv_continuous.fit>`__ for detailed documentation of the\n    keyword arguments.\n'
_doc_expect = 'expect(func, args=(%(shapes_)s), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds)\n    Expected value of a function (of one argument) with respect to the distribution.\n'
_doc_expect_discrete = 'expect(func, args=(%(shapes_)s), loc=0, lb=None, ub=None, conditional=False)\n    Expected value of a function (of one argument) with respect to the distribution.\n'
_doc_median = 'median(%(shapes)s, loc=0, scale=1)\n    Median of the distribution.\n'
_doc_mean = 'mean(%(shapes)s, loc=0, scale=1)\n    Mean of the distribution.\n'
_doc_var = 'var(%(shapes)s, loc=0, scale=1)\n    Variance of the distribution.\n'
_doc_std = 'std(%(shapes)s, loc=0, scale=1)\n    Standard deviation of the distribution.\n'
_doc_interval = 'interval(confidence, %(shapes)s, loc=0, scale=1)\n    Confidence interval with equal areas around the median.\n'
_doc_allmethods = ''.join([docheaders['methods'], _doc_rvs, _doc_pdf, _doc_logpdf, _doc_cdf, _doc_logcdf, _doc_sf, _doc_logsf, _doc_ppf, _doc_isf, _doc_moment, _doc_stats, _doc_entropy, _doc_fit, _doc_expect, _doc_median, _doc_mean, _doc_var, _doc_std, _doc_interval])
_doc_default_longsummary = 'As an instance of the `rv_continuous` class, `%(name)s` object inherits from it\na collection of generic methods (see below for the full list),\nand completes them with details specific for this particular distribution.\n'
_doc_default_frozen_note = '\nAlternatively, the object may be called (as a function) to fix the shape,\nlocation, and scale parameters returning a "frozen" continuous RV object:\n\nrv = %(name)s(%(shapes)s, loc=0, scale=1)\n    - Frozen RV object with the same methods but holding the given shape,\n      location, and scale fixed.\n'
_doc_default_example = 'Examples\n--------\n>>> import numpy as np\n>>> from scipy.stats import %(name)s\n>>> import matplotlib.pyplot as plt\n>>> fig, ax = plt.subplots(1, 1)\n\nCalculate the first four moments:\n\n%(set_vals_stmt)s\n>>> mean, var, skew, kurt = %(name)s.stats(%(shapes)s, moments=\'mvsk\')\n\nDisplay the probability density function (``pdf``):\n\n>>> x = np.linspace(%(name)s.ppf(0.01, %(shapes)s),\n...                 %(name)s.ppf(0.99, %(shapes)s), 100)\n>>> ax.plot(x, %(name)s.pdf(x, %(shapes)s),\n...        \'r-\', lw=5, alpha=0.6, label=\'%(name)s pdf\')\n\nAlternatively, the distribution object can be called (as a function)\nto fix the shape, location and scale parameters. This returns a "frozen"\nRV object holding the given parameters fixed.\n\nFreeze the distribution and display the frozen ``pdf``:\n\n>>> rv = %(name)s(%(shapes)s)\n>>> ax.plot(x, rv.pdf(x), \'k-\', lw=2, label=\'frozen pdf\')\n\nCheck accuracy of ``cdf`` and ``ppf``:\n\n>>> vals = %(name)s.ppf([0.001, 0.5, 0.999], %(shapes)s)\n>>> np.allclose([0.001, 0.5, 0.999], %(name)s.cdf(vals, %(shapes)s))\nTrue\n\nGenerate random numbers:\n\n>>> r = %(name)s.rvs(%(shapes)s, size=1000)\n\nAnd compare the histogram:\n\n>>> ax.hist(r, density=True, bins=\'auto\', histtype=\'stepfilled\', alpha=0.2)\n>>> ax.set_xlim([x[0], x[-1]])\n>>> ax.legend(loc=\'best\', frameon=False)\n>>> plt.show()\n\n'
_doc_default_locscale = 'The probability density above is defined in the "standardized" form. To shift\nand/or scale the distribution use the ``loc`` and ``scale`` parameters.\nSpecifically, ``%(name)s.pdf(x, %(shapes)s, loc, scale)`` is identically\nequivalent to ``%(name)s.pdf(y, %(shapes)s) / scale`` with\n``y = (x - loc) / scale``. Note that shifting the location of a distribution\ndoes not make it a "noncentral" distribution; noncentral generalizations of\nsome distributions are available in separate classes.\n'
_doc_default = ''.join([_doc_default_longsummary, _doc_allmethods, '\n', _doc_default_example])
_doc_default_before_notes = ''.join([_doc_default_longsummary, _doc_allmethods])
docdict = {'rvs': _doc_rvs, 'pdf': _doc_pdf, 'logpdf': _doc_logpdf, 'cdf': _doc_cdf, 'logcdf': _doc_logcdf, 'sf': _doc_sf, 'logsf': _doc_logsf, 'ppf': _doc_ppf, 'isf': _doc_isf, 'stats': _doc_stats, 'entropy': _doc_entropy, 'fit': _doc_fit, 'moment': _doc_moment, 'expect': _doc_expect, 'interval': _doc_interval, 'mean': _doc_mean, 'std': _doc_std, 'var': _doc_var, 'median': _doc_median, 'allmethods': _doc_allmethods, 'longsummary': _doc_default_longsummary, 'frozennote': _doc_default_frozen_note, 'example': _doc_default_example, 'default': _doc_default, 'before_notes': _doc_default_before_notes, 'after_notes': _doc_default_locscale}
docdict_discrete = docdict.copy()
docdict_discrete['pmf'] = _doc_pmf
docdict_discrete['logpmf'] = _doc_logpmf
docdict_discrete['expect'] = _doc_expect_discrete
_doc_disc_methods = ['rvs', 'pmf', 'logpmf', 'cdf', 'logcdf', 'sf', 'logsf', 'ppf', 'isf', 'stats', 'entropy', 'expect', 'median', 'mean', 'var', 'std', 'interval']
for obj in _doc_disc_methods:
    docdict_discrete[obj] = docdict_discrete[obj].replace(', scale=1', '')
_doc_disc_methods_err_varname = ['cdf', 'logcdf', 'sf', 'logsf']
for obj in _doc_disc_methods_err_varname:
    docdict_discrete[obj] = docdict_discrete[obj].replace('(x, ', '(k, ')
docdict_discrete.pop('pdf')
docdict_discrete.pop('logpdf')
_doc_allmethods = ''.join([docdict_discrete[obj] for obj in _doc_disc_methods])
docdict_discrete['allmethods'] = docheaders['methods'] + _doc_allmethods
docdict_discrete['longsummary'] = _doc_default_longsummary.replace('rv_continuous', 'rv_discrete')
_doc_default_frozen_note = '\nAlternatively, the object may be called (as a function) to fix the shape and\nlocation parameters returning a "frozen" discrete RV object:\n\nrv = %(name)s(%(shapes)s, loc=0)\n    - Frozen RV object with the same methods but holding the given shape and\n      location fixed.\n'
docdict_discrete['frozennote'] = _doc_default_frozen_note
_doc_default_discrete_example = 'Examples\n--------\n>>> import numpy as np\n>>> from scipy.stats import %(name)s\n>>> import matplotlib.pyplot as plt\n>>> fig, ax = plt.subplots(1, 1)\n\nCalculate the first four moments:\n\n%(set_vals_stmt)s\n>>> mean, var, skew, kurt = %(name)s.stats(%(shapes)s, moments=\'mvsk\')\n\nDisplay the probability mass function (``pmf``):\n\n>>> x = np.arange(%(name)s.ppf(0.01, %(shapes)s),\n...               %(name)s.ppf(0.99, %(shapes)s))\n>>> ax.plot(x, %(name)s.pmf(x, %(shapes)s), \'bo\', ms=8, label=\'%(name)s pmf\')\n>>> ax.vlines(x, 0, %(name)s.pmf(x, %(shapes)s), colors=\'b\', lw=5, alpha=0.5)\n\nAlternatively, the distribution object can be called (as a function)\nto fix the shape and location. This returns a "frozen" RV object holding\nthe given parameters fixed.\n\nFreeze the distribution and display the frozen ``pmf``:\n\n>>> rv = %(name)s(%(shapes)s)\n>>> ax.vlines(x, 0, rv.pmf(x), colors=\'k\', linestyles=\'-\', lw=1,\n...         label=\'frozen pmf\')\n>>> ax.legend(loc=\'best\', frameon=False)\n>>> plt.show()\n\nCheck accuracy of ``cdf`` and ``ppf``:\n\n>>> prob = %(name)s.cdf(x, %(shapes)s)\n>>> np.allclose(x, %(name)s.ppf(prob, %(shapes)s))\nTrue\n\nGenerate random numbers:\n\n>>> r = %(name)s.rvs(%(shapes)s, size=1000)\n'
_doc_default_discrete_locscale = 'The probability mass function above is defined in the "standardized" form.\nTo shift distribution use the ``loc`` parameter.\nSpecifically, ``%(name)s.pmf(k, %(shapes)s, loc)`` is identically\nequivalent to ``%(name)s.pmf(k - loc, %(shapes)s)``.\n'
docdict_discrete['example'] = _doc_default_discrete_example
docdict_discrete['after_notes'] = _doc_default_discrete_locscale
_doc_default_before_notes = ''.join([docdict_discrete['longsummary'], docdict_discrete['allmethods']])
docdict_discrete['before_notes'] = _doc_default_before_notes
_doc_default_disc = ''.join([docdict_discrete['longsummary'], docdict_discrete['allmethods'], docdict_discrete['frozennote'], docdict_discrete['example']])
docdict_discrete['default'] = _doc_default_disc
for obj in [s for s in dir() if s.startswith('_doc_')]:
    exec('del ' + obj)
del obj

def _moment(data, n, mu=None):
    if False:
        for i in range(10):
            print('nop')
    if mu is None:
        mu = data.mean()
    return ((data - mu) ** n).mean()

def _moment_from_stats(n, mu, mu2, g1, g2, moment_func, args):
    if False:
        print('Hello World!')
    if n == 0:
        return 1.0
    elif n == 1:
        if mu is None:
            val = moment_func(1, *args)
        else:
            val = mu
    elif n == 2:
        if mu2 is None or mu is None:
            val = moment_func(2, *args)
        else:
            val = mu2 + mu * mu
    elif n == 3:
        if g1 is None or mu2 is None or mu is None:
            val = moment_func(3, *args)
        else:
            mu3 = g1 * np.power(mu2, 1.5)
            val = mu3 + 3 * mu * mu2 + mu * mu * mu
    elif n == 4:
        if g1 is None or g2 is None or mu2 is None or (mu is None):
            val = moment_func(4, *args)
        else:
            mu4 = (g2 + 3.0) * mu2 ** 2.0
            mu3 = g1 * np.power(mu2, 1.5)
            val = mu4 + 4 * mu * mu3 + 6 * mu * mu * mu2 + mu * mu * mu * mu
    else:
        val = moment_func(n, *args)
    return val

def _skew(data):
    if False:
        for i in range(10):
            print('nop')
    '\n    skew is third central moment / variance**(1.5)\n    '
    data = np.ravel(data)
    mu = data.mean()
    m2 = ((data - mu) ** 2).mean()
    m3 = ((data - mu) ** 3).mean()
    return m3 / np.power(m2, 1.5)

def _kurtosis(data):
    if False:
        i = 10
        return i + 15
    'kurtosis is fourth central moment / variance**2 - 3.'
    data = np.ravel(data)
    mu = data.mean()
    m2 = ((data - mu) ** 2).mean()
    m4 = ((data - mu) ** 4).mean()
    return m4 / m2 ** 2 - 3

def _fit_determine_optimizer(optimizer):
    if False:
        print('Hello World!')
    if not callable(optimizer) and isinstance(optimizer, str):
        if not optimizer.startswith('fmin_'):
            optimizer = 'fmin_' + optimizer
        if optimizer == 'fmin_':
            optimizer = 'fmin'
        try:
            optimizer = getattr(optimize, optimizer)
        except AttributeError as e:
            raise ValueError('%s is not a valid optimizer' % optimizer) from e
    return optimizer

def _sum_finite(x):
    if False:
        i = 10
        return i + 15
    '\n    For a 1D array x, return a tuple containing the sum of the\n    finite values of x and the number of nonfinite values.\n\n    This is a utility function used when evaluating the negative\n    loglikelihood for a distribution and an array of samples.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.stats._distn_infrastructure import _sum_finite\n    >>> tot, nbad = _sum_finite(np.array([-2, -np.inf, 5, 1]))\n    >>> tot\n    4.0\n    >>> nbad\n    1\n    '
    finite_x = np.isfinite(x)
    bad_count = finite_x.size - np.count_nonzero(finite_x)
    return (np.sum(x[finite_x]), bad_count)

class rv_frozen:

    def __init__(self, dist, *args, **kwds):
        if False:
            return 10
        self.args = args
        self.kwds = kwds
        self.dist = dist.__class__(**dist._updated_ctor_param())
        (shapes, _, _) = self.dist._parse_args(*args, **kwds)
        (self.a, self.b) = self.dist._get_support(*shapes)

    @property
    def random_state(self):
        if False:
            i = 10
            return i + 15
        return self.dist._random_state

    @random_state.setter
    def random_state(self, seed):
        if False:
            while True:
                i = 10
        self.dist._random_state = check_random_state(seed)

    def cdf(self, x):
        if False:
            return 10
        return self.dist.cdf(x, *self.args, **self.kwds)

    def logcdf(self, x):
        if False:
            return 10
        return self.dist.logcdf(x, *self.args, **self.kwds)

    def ppf(self, q):
        if False:
            print('Hello World!')
        return self.dist.ppf(q, *self.args, **self.kwds)

    def isf(self, q):
        if False:
            return 10
        return self.dist.isf(q, *self.args, **self.kwds)

    def rvs(self, size=None, random_state=None):
        if False:
            while True:
                i = 10
        kwds = self.kwds.copy()
        kwds.update({'size': size, 'random_state': random_state})
        return self.dist.rvs(*self.args, **kwds)

    def sf(self, x):
        if False:
            print('Hello World!')
        return self.dist.sf(x, *self.args, **self.kwds)

    def logsf(self, x):
        if False:
            return 10
        return self.dist.logsf(x, *self.args, **self.kwds)

    def stats(self, moments='mv'):
        if False:
            print('Hello World!')
        kwds = self.kwds.copy()
        kwds.update({'moments': moments})
        return self.dist.stats(*self.args, **kwds)

    def median(self):
        if False:
            i = 10
            return i + 15
        return self.dist.median(*self.args, **self.kwds)

    def mean(self):
        if False:
            while True:
                i = 10
        return self.dist.mean(*self.args, **self.kwds)

    def var(self):
        if False:
            return 10
        return self.dist.var(*self.args, **self.kwds)

    def std(self):
        if False:
            i = 10
            return i + 15
        return self.dist.std(*self.args, **self.kwds)

    def moment(self, order=None):
        if False:
            print('Hello World!')
        return self.dist.moment(order, *self.args, **self.kwds)

    def entropy(self):
        if False:
            return 10
        return self.dist.entropy(*self.args, **self.kwds)

    def interval(self, confidence=None):
        if False:
            i = 10
            return i + 15
        return self.dist.interval(confidence, *self.args, **self.kwds)

    def expect(self, func=None, lb=None, ub=None, conditional=False, **kwds):
        if False:
            print('Hello World!')
        (a, loc, scale) = self.dist._parse_args(*self.args, **self.kwds)
        if isinstance(self.dist, rv_discrete):
            return self.dist.expect(func, a, loc, lb, ub, conditional, **kwds)
        else:
            return self.dist.expect(func, a, loc, scale, lb, ub, conditional, **kwds)

    def support(self):
        if False:
            while True:
                i = 10
        return self.dist.support(*self.args, **self.kwds)

class rv_discrete_frozen(rv_frozen):

    def pmf(self, k):
        if False:
            i = 10
            return i + 15
        return self.dist.pmf(k, *self.args, **self.kwds)

    def logpmf(self, k):
        if False:
            for i in range(10):
                print('nop')
        return self.dist.logpmf(k, *self.args, **self.kwds)

class rv_continuous_frozen(rv_frozen):

    def pdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        return self.dist.pdf(x, *self.args, **self.kwds)

    def logpdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        return self.dist.logpdf(x, *self.args, **self.kwds)

def argsreduce(cond, *args):
    if False:
        for i in range(10):
            print('nop')
    'Clean arguments to:\n\n    1. Ensure all arguments are iterable (arrays of dimension at least one\n    2. If cond != True and size > 1, ravel(args[i]) where ravel(condition) is\n       True, in 1D.\n\n    Return list of processed arguments.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.stats._distn_infrastructure import argsreduce\n    >>> rng = np.random.default_rng()\n    >>> A = rng.random((4, 5))\n    >>> B = 2\n    >>> C = rng.random((1, 5))\n    >>> cond = np.ones(A.shape)\n    >>> [A1, B1, C1] = argsreduce(cond, A, B, C)\n    >>> A1.shape\n    (4, 5)\n    >>> B1.shape\n    (1,)\n    >>> C1.shape\n    (1, 5)\n    >>> cond[2,:] = 0\n    >>> [A1, B1, C1] = argsreduce(cond, A, B, C)\n    >>> A1.shape\n    (15,)\n    >>> B1.shape\n    (1,)\n    >>> C1.shape\n    (15,)\n\n    '
    newargs = np.atleast_1d(*args)
    if not isinstance(newargs, list):
        newargs = [newargs]
    if np.all(cond):
        (*newargs, cond) = np.broadcast_arrays(*newargs, cond)
        return [arg.ravel() for arg in newargs]
    s = cond.shape
    return [arg if np.size(arg) == 1 else np.extract(cond, np.broadcast_to(arg, s)) for arg in newargs]
parse_arg_template = "\ndef _parse_args(self, %(shape_arg_str)s %(locscale_in)s):\n    return (%(shape_arg_str)s), %(locscale_out)s\n\ndef _parse_args_rvs(self, %(shape_arg_str)s %(locscale_in)s, size=None):\n    return self._argcheck_rvs(%(shape_arg_str)s %(locscale_out)s, size=size)\n\ndef _parse_args_stats(self, %(shape_arg_str)s %(locscale_in)s, moments='mv'):\n    return (%(shape_arg_str)s), %(locscale_out)s, moments\n"

class rv_generic:
    """Class which encapsulates common functionality between rv_discrete
    and rv_continuous.

    """

    def __init__(self, seed=None):
        if False:
            while True:
                i = 10
        super().__init__()
        sig = _getfullargspec(self._stats)
        self._stats_has_moments = sig.varkw is not None or 'moments' in sig.args or 'moments' in sig.kwonlyargs
        self._random_state = check_random_state(seed)

    @property
    def random_state(self):
        if False:
            while True:
                i = 10
        'Get or set the generator object for generating random variates.\n\n        If `random_state` is None (or `np.random`), the\n        `numpy.random.RandomState` singleton is used.\n        If `random_state` is an int, a new ``RandomState`` instance is used,\n        seeded with `random_state`.\n        If `random_state` is already a ``Generator`` or ``RandomState``\n        instance, that instance is used.\n\n        '
        return self._random_state

    @random_state.setter
    def random_state(self, seed):
        if False:
            for i in range(10):
                print('nop')
        self._random_state = check_random_state(seed)

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.__dict__.update(state)
            self._attach_methods()
        except ValueError:
            self._ctor_param = state[0]
            self._random_state = state[1]
            self.__init__()

    def _attach_methods(self):
        if False:
            i = 10
            return i + 15
        'Attaches dynamically created methods to the rv_* instance.\n\n        This method must be overridden by subclasses, and must itself call\n         _attach_argparser_methods. This method is called in __init__ in\n         subclasses, and in __setstate__\n        '
        raise NotImplementedError

    def _attach_argparser_methods(self):
        if False:
            return 10
        '\n        Generates the argument-parsing functions dynamically and attaches\n        them to the instance.\n\n        Should be called from `_attach_methods`, typically in __init__ and\n        during unpickling (__setstate__)\n        '
        ns = {}
        exec(self._parse_arg_template, ns)
        for name in ['_parse_args', '_parse_args_stats', '_parse_args_rvs']:
            setattr(self, name, types.MethodType(ns[name], self))

    def _construct_argparser(self, meths_to_inspect, locscale_in, locscale_out):
        if False:
            while True:
                i = 10
        'Construct the parser string for the shape arguments.\n\n        This method should be called in __init__ of a class for each\n        distribution. It creates the `_parse_arg_template` attribute that is\n        then used by `_attach_argparser_methods` to dynamically create and\n        attach the `_parse_args`, `_parse_args_stats`, `_parse_args_rvs`\n        methods to the instance.\n\n        If self.shapes is a non-empty string, interprets it as a\n        comma-separated list of shape parameters.\n\n        Otherwise inspects the call signatures of `meths_to_inspect`\n        and constructs the argument-parsing functions from these.\n        In this case also sets `shapes` and `numargs`.\n        '
        if self.shapes:
            if not isinstance(self.shapes, str):
                raise TypeError('shapes must be a string.')
            shapes = self.shapes.replace(',', ' ').split()
            for field in shapes:
                if keyword.iskeyword(field):
                    raise SyntaxError('keywords cannot be used as shapes.')
                if not re.match('^[_a-zA-Z][_a-zA-Z0-9]*$', field):
                    raise SyntaxError('shapes must be valid python identifiers')
        else:
            shapes_list = []
            for meth in meths_to_inspect:
                shapes_args = _getfullargspec(meth)
                args = shapes_args.args[1:]
                if args:
                    shapes_list.append(args)
                    if shapes_args.varargs is not None:
                        raise TypeError('*args are not allowed w/out explicit shapes')
                    if shapes_args.varkw is not None:
                        raise TypeError('**kwds are not allowed w/out explicit shapes')
                    if shapes_args.kwonlyargs:
                        raise TypeError('kwonly args are not allowed w/out explicit shapes')
                    if shapes_args.defaults is not None:
                        raise TypeError('defaults are not allowed for shapes')
            if shapes_list:
                shapes = shapes_list[0]
                for item in shapes_list:
                    if item != shapes:
                        raise TypeError('Shape arguments are inconsistent.')
            else:
                shapes = []
        shapes_str = ', '.join(shapes) + ', ' if shapes else ''
        dct = dict(shape_arg_str=shapes_str, locscale_in=locscale_in, locscale_out=locscale_out)
        self._parse_arg_template = parse_arg_template % dct
        self.shapes = ', '.join(shapes) if shapes else None
        if not hasattr(self, 'numargs'):
            self.numargs = len(shapes)

    def _construct_doc(self, docdict, shapes_vals=None):
        if False:
            i = 10
            return i + 15
        'Construct the instance docstring with string substitutions.'
        tempdict = docdict.copy()
        tempdict['name'] = self.name or 'distname'
        tempdict['shapes'] = self.shapes or ''
        if shapes_vals is None:
            shapes_vals = ()
        vals = ', '.join(('%.3g' % val for val in shapes_vals))
        tempdict['vals'] = vals
        tempdict['shapes_'] = self.shapes or ''
        if self.shapes and self.numargs == 1:
            tempdict['shapes_'] += ','
        if self.shapes:
            tempdict['set_vals_stmt'] = f'>>> {self.shapes} = {vals}'
        else:
            tempdict['set_vals_stmt'] = ''
        if self.shapes is None:
            for item in ['default', 'before_notes']:
                tempdict[item] = tempdict[item].replace('\n%(shapes)s : array_like\n    shape parameters', '')
        for i in range(2):
            if self.shapes is None:
                self.__doc__ = self.__doc__.replace('%(shapes)s, ', '')
            try:
                self.__doc__ = doccer.docformat(self.__doc__, tempdict)
            except TypeError as e:
                raise Exception('Unable to construct docstring for distribution "%s": %s' % (self.name, repr(e))) from e
        self.__doc__ = self.__doc__.replace('(, ', '(').replace(', )', ')')

    def _construct_default_doc(self, longname=None, docdict=None, discrete='continuous'):
        if False:
            print('Hello World!')
        'Construct instance docstring from the default template.'
        if longname is None:
            longname = 'A'
        self.__doc__ = ''.join([f'{longname} {discrete} random variable.', '\n\n%(before_notes)s\n', docheaders['notes'], '\n%(example)s'])
        self._construct_doc(docdict)

    def freeze(self, *args, **kwds):
        if False:
            i = 10
            return i + 15
        'Freeze the distribution for the given arguments.\n\n        Parameters\n        ----------\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution.  Should include all\n            the non-optional arguments, may include ``loc`` and ``scale``.\n\n        Returns\n        -------\n        rv_frozen : rv_frozen instance\n            The frozen distribution.\n\n        '
        if isinstance(self, rv_continuous):
            return rv_continuous_frozen(self, *args, **kwds)
        else:
            return rv_discrete_frozen(self, *args, **kwds)

    def __call__(self, *args, **kwds):
        if False:
            print('Hello World!')
        return self.freeze(*args, **kwds)
    __call__.__doc__ = freeze.__doc__

    def _stats(self, *args, **kwds):
        if False:
            print('Hello World!')
        return (None, None, None, None)

    def _munp(self, n, *args):
        if False:
            print('Hello World!')
        with np.errstate(all='ignore'):
            vals = self.generic_moment(n, *args)
        return vals

    def _argcheck_rvs(self, *args, **kwargs):
        if False:
            return 10
        size = kwargs.get('size', None)
        all_bcast = np.broadcast_arrays(*args)

        def squeeze_left(a):
            if False:
                return 10
            while a.ndim > 0 and a.shape[0] == 1:
                a = a[0]
            return a
        all_bcast = [squeeze_left(a) for a in all_bcast]
        bcast_shape = all_bcast[0].shape
        bcast_ndim = all_bcast[0].ndim
        if size is None:
            size_ = bcast_shape
        else:
            size_ = tuple(np.atleast_1d(size))
        ndiff = bcast_ndim - len(size_)
        if ndiff < 0:
            bcast_shape = (1,) * -ndiff + bcast_shape
        elif ndiff > 0:
            size_ = (1,) * ndiff + size_
        ok = all([bcdim == 1 or bcdim == szdim for (bcdim, szdim) in zip(bcast_shape, size_)])
        if not ok:
            raise ValueError(f'size does not match the broadcast shape of the parameters. {size}, {size_}, {bcast_shape}')
        param_bcast = all_bcast[:-2]
        loc_bcast = all_bcast[-2]
        scale_bcast = all_bcast[-1]
        return (param_bcast, loc_bcast, scale_bcast, size_)

    def _argcheck(self, *args):
        if False:
            while True:
                i = 10
        "Default check for correct values on args and keywords.\n\n        Returns condition array of 1's where arguments are correct and\n         0's where they are not.\n\n        "
        cond = 1
        for arg in args:
            cond = logical_and(cond, asarray(arg) > 0)
        return cond

    def _get_support(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Return the support of the (unscaled, unshifted) distribution.\n\n        *Must* be overridden by distributions which have support dependent\n        upon the shape parameters of the distribution.  Any such override\n        *must not* set or change any of the class members, as these members\n        are shared amongst all instances of the distribution.\n\n        Parameters\n        ----------\n        arg1, arg2, ... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information).\n\n        Returns\n        -------\n        a, b : numeric (float, or int or +/-np.inf)\n            end-points of the distribution's support for the specified\n            shape parameters.\n        "
        return (self.a, self.b)

    def _support_mask(self, x, *args):
        if False:
            return 10
        (a, b) = self._get_support(*args)
        with np.errstate(invalid='ignore'):
            return (a <= x) & (x <= b)

    def _open_support_mask(self, x, *args):
        if False:
            for i in range(10):
                print('nop')
        (a, b) = self._get_support(*args)
        with np.errstate(invalid='ignore'):
            return (a < x) & (x < b)

    def _rvs(self, *args, size=None, random_state=None):
        if False:
            while True:
                i = 10
        U = random_state.uniform(size=size)
        Y = self._ppf(U, *args)
        return Y

    def _logcdf(self, x, *args):
        if False:
            print('Hello World!')
        with np.errstate(divide='ignore'):
            return log(self._cdf(x, *args))

    def _sf(self, x, *args):
        if False:
            return 10
        return 1.0 - self._cdf(x, *args)

    def _logsf(self, x, *args):
        if False:
            print('Hello World!')
        with np.errstate(divide='ignore'):
            return log(self._sf(x, *args))

    def _ppf(self, q, *args):
        if False:
            i = 10
            return i + 15
        return self._ppfvec(q, *args)

    def _isf(self, q, *args):
        if False:
            print('Hello World!')
        return self._ppf(1.0 - q, *args)

    def rvs(self, *args, **kwds):
        if False:
            i = 10
            return i + 15
        'Random variates of given type.\n\n        Parameters\n        ----------\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information).\n        loc : array_like, optional\n            Location parameter (default=0).\n        scale : array_like, optional\n            Scale parameter (default=1).\n        size : int or tuple of ints, optional\n            Defining number of random variates (default is 1).\n        random_state : {None, int, `numpy.random.Generator`,\n                        `numpy.random.RandomState`}, optional\n\n            If `random_state` is None (or `np.random`), the\n            `numpy.random.RandomState` singleton is used.\n            If `random_state` is an int, a new ``RandomState`` instance is\n            used, seeded with `random_state`.\n            If `random_state` is already a ``Generator`` or ``RandomState``\n            instance, that instance is used.\n\n        Returns\n        -------\n        rvs : ndarray or scalar\n            Random variates of given `size`.\n\n        '
        discrete = kwds.pop('discrete', None)
        rndm = kwds.pop('random_state', None)
        (args, loc, scale, size) = self._parse_args_rvs(*args, **kwds)
        cond = logical_and(self._argcheck(*args), scale >= 0)
        if not np.all(cond):
            message = f'Domain error in arguments. The `scale` parameter must be positive for all distributions, and many distributions have restrictions on shape parameters. Please see the `scipy.stats.{self.name}` documentation for details.'
            raise ValueError(message)
        if np.all(scale == 0):
            return loc * ones(size, 'd')
        if rndm is not None:
            random_state_saved = self._random_state
            random_state = check_random_state(rndm)
        else:
            random_state = self._random_state
        vals = self._rvs(*args, size=size, random_state=random_state)
        vals = vals * scale + loc
        if rndm is not None:
            self._random_state = random_state_saved
        if discrete and (not isinstance(self, rv_sample)):
            if size == ():
                vals = int(vals)
            else:
                vals = vals.astype(np.int64)
        return vals

    def stats(self, *args, **kwds):
        if False:
            return 10
        "Some statistics of the given RV.\n\n        Parameters\n        ----------\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information)\n        loc : array_like, optional\n            location parameter (default=0)\n        scale : array_like, optional (continuous RVs only)\n            scale parameter (default=1)\n        moments : str, optional\n            composed of letters ['mvsk'] defining which moments to compute:\n            'm' = mean,\n            'v' = variance,\n            's' = (Fisher's) skew,\n            'k' = (Fisher's) kurtosis.\n            (default is 'mv')\n\n        Returns\n        -------\n        stats : sequence\n            of requested moments.\n\n        "
        (args, loc, scale, moments) = self._parse_args_stats(*args, **kwds)
        (loc, scale) = map(asarray, (loc, scale))
        args = tuple(map(asarray, args))
        cond = self._argcheck(*args) & (scale > 0) & (loc == loc)
        output = []
        default = np.full(shape(cond), fill_value=self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *args + (scale, loc))
            (scale, loc, goodargs) = (goodargs[-2], goodargs[-1], goodargs[:-2])
            if self._stats_has_moments:
                (mu, mu2, g1, g2) = self._stats(*goodargs, **{'moments': moments})
            else:
                (mu, mu2, g1, g2) = self._stats(*goodargs)
            if 'm' in moments:
                if mu is None:
                    mu = self._munp(1, *goodargs)
                out0 = default.copy()
                place(out0, cond, mu * scale + loc)
                output.append(out0)
            if 'v' in moments:
                if mu2 is None:
                    mu2p = self._munp(2, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    with np.errstate(invalid='ignore'):
                        mu2 = np.where(~np.isinf(mu), mu2p - mu ** 2, np.inf)
                out0 = default.copy()
                place(out0, cond, mu2 * scale * scale)
                output.append(out0)
            if 's' in moments:
                if g1 is None:
                    mu3p = self._munp(3, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    if mu2 is None:
                        mu2p = self._munp(2, *goodargs)
                        mu2 = mu2p - mu * mu
                    with np.errstate(invalid='ignore'):
                        mu3 = (-mu * mu - 3 * mu2) * mu + mu3p
                        g1 = mu3 / np.power(mu2, 1.5)
                out0 = default.copy()
                place(out0, cond, g1)
                output.append(out0)
            if 'k' in moments:
                if g2 is None:
                    mu4p = self._munp(4, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    if mu2 is None:
                        mu2p = self._munp(2, *goodargs)
                        mu2 = mu2p - mu * mu
                    if g1 is None:
                        mu3 = None
                    else:
                        mu3 = g1 * np.power(mu2, 1.5)
                    if mu3 is None:
                        mu3p = self._munp(3, *goodargs)
                        with np.errstate(invalid='ignore'):
                            mu3 = (-mu * mu - 3 * mu2) * mu + mu3p
                    with np.errstate(invalid='ignore'):
                        mu4 = ((-mu ** 2 - 6 * mu2) * mu - 4 * mu3) * mu + mu4p
                        g2 = mu4 / mu2 ** 2.0 - 3.0
                out0 = default.copy()
                place(out0, cond, g2)
                output.append(out0)
        else:
            output = [default.copy() for _ in moments]
        output = [out[()] for out in output]
        if len(output) == 1:
            return output[0]
        else:
            return tuple(output)

    def entropy(self, *args, **kwds):
        if False:
            i = 10
            return i + 15
        'Differential entropy of the RV.\n\n        Parameters\n        ----------\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information).\n        loc : array_like, optional\n            Location parameter (default=0).\n        scale : array_like, optional  (continuous distributions only).\n            Scale parameter (default=1).\n\n        Notes\n        -----\n        Entropy is defined base `e`:\n\n        >>> import numpy as np\n        >>> from scipy.stats._distn_infrastructure import rv_discrete\n        >>> drv = rv_discrete(values=((0, 1), (0.5, 0.5)))\n        >>> np.allclose(drv.entropy(), np.log(2.0))\n        True\n\n        '
        (args, loc, scale) = self._parse_args(*args, **kwds)
        (loc, scale) = map(asarray, (loc, scale))
        args = tuple(map(asarray, args))
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        output = zeros(shape(cond0), 'd')
        place(output, 1 - cond0, self.badvalue)
        goodargs = argsreduce(cond0, scale, *args)
        goodscale = goodargs[0]
        goodargs = goodargs[1:]
        place(output, cond0, self.vecentropy(*goodargs) + log(goodscale))
        return output[()]

    def moment(self, order, *args, **kwds):
        if False:
            return 10
        'non-central moment of distribution of specified order.\n\n        Parameters\n        ----------\n        order : int, order >= 1\n            Order of moment.\n        arg1, arg2, arg3,... : float\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information).\n        loc : array_like, optional\n            location parameter (default=0)\n        scale : array_like, optional\n            scale parameter (default=1)\n\n        '
        n = order
        (shapes, loc, scale) = self._parse_args(*args, **kwds)
        args = np.broadcast_arrays(*(*shapes, loc, scale))
        (*shapes, loc, scale) = args
        i0 = np.logical_and(self._argcheck(*shapes), scale > 0)
        i1 = np.logical_and(i0, loc == 0)
        i2 = np.logical_and(i0, loc != 0)
        args = argsreduce(i0, *shapes, loc, scale)
        (*shapes, loc, scale) = args
        if floor(n) != n:
            raise ValueError('Moment must be an integer.')
        if n < 0:
            raise ValueError('Moment must be positive.')
        (mu, mu2, g1, g2) = (None, None, None, None)
        if n > 0 and n < 5:
            if self._stats_has_moments:
                mdict = {'moments': {1: 'm', 2: 'v', 3: 'vs', 4: 'mvsk'}[n]}
            else:
                mdict = {}
            (mu, mu2, g1, g2) = self._stats(*shapes, **mdict)
        val = np.empty(loc.shape)
        val[...] = _moment_from_stats(n, mu, mu2, g1, g2, self._munp, shapes)
        result = zeros(i0.shape)
        place(result, ~i0, self.badvalue)
        if i1.any():
            res1 = scale[loc == 0] ** n * val[loc == 0]
            place(result, i1, res1)
        if i2.any():
            mom = [mu, mu2, g1, g2]
            arrs = [i for i in mom if i is not None]
            idx = [i for i in range(4) if mom[i] is not None]
            if any(idx):
                arrs = argsreduce(loc != 0, *arrs)
                j = 0
                for i in idx:
                    mom[i] = arrs[j]
                    j += 1
            (mu, mu2, g1, g2) = mom
            args = argsreduce(loc != 0, *shapes, loc, scale, val)
            (*shapes, loc, scale, val) = args
            res2 = zeros(loc.shape, dtype='d')
            fac = scale / loc
            for k in range(n):
                valk = _moment_from_stats(k, mu, mu2, g1, g2, self._munp, shapes)
                res2 += comb(n, k, exact=True) * fac ** k * valk
            res2 += fac ** n * val
            res2 *= loc ** n
            place(result, i2, res2)
        return result[()]

    def median(self, *args, **kwds):
        if False:
            while True:
                i = 10
        'Median of the distribution.\n\n        Parameters\n        ----------\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information)\n        loc : array_like, optional\n            Location parameter, Default is 0.\n        scale : array_like, optional\n            Scale parameter, Default is 1.\n\n        Returns\n        -------\n        median : float\n            The median of the distribution.\n\n        See Also\n        --------\n        rv_discrete.ppf\n            Inverse of the CDF\n\n        '
        return self.ppf(0.5, *args, **kwds)

    def mean(self, *args, **kwds):
        if False:
            i = 10
            return i + 15
        'Mean of the distribution.\n\n        Parameters\n        ----------\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information)\n        loc : array_like, optional\n            location parameter (default=0)\n        scale : array_like, optional\n            scale parameter (default=1)\n\n        Returns\n        -------\n        mean : float\n            the mean of the distribution\n\n        '
        kwds['moments'] = 'm'
        res = self.stats(*args, **kwds)
        if isinstance(res, ndarray) and res.ndim == 0:
            return res[()]
        return res

    def var(self, *args, **kwds):
        if False:
            while True:
                i = 10
        'Variance of the distribution.\n\n        Parameters\n        ----------\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information)\n        loc : array_like, optional\n            location parameter (default=0)\n        scale : array_like, optional\n            scale parameter (default=1)\n\n        Returns\n        -------\n        var : float\n            the variance of the distribution\n\n        '
        kwds['moments'] = 'v'
        res = self.stats(*args, **kwds)
        if isinstance(res, ndarray) and res.ndim == 0:
            return res[()]
        return res

    def std(self, *args, **kwds):
        if False:
            while True:
                i = 10
        'Standard deviation of the distribution.\n\n        Parameters\n        ----------\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information)\n        loc : array_like, optional\n            location parameter (default=0)\n        scale : array_like, optional\n            scale parameter (default=1)\n\n        Returns\n        -------\n        std : float\n            standard deviation of the distribution\n\n        '
        kwds['moments'] = 'v'
        res = sqrt(self.stats(*args, **kwds))
        return res

    def interval(self, confidence, *args, **kwds):
        if False:
            print('Hello World!')
        "Confidence interval with equal areas around the median.\n\n        Parameters\n        ----------\n        confidence : array_like of float\n            Probability that an rv will be drawn from the returned range.\n            Each value should be in the range [0, 1].\n        arg1, arg2, ... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information).\n        loc : array_like, optional\n            location parameter, Default is 0.\n        scale : array_like, optional\n            scale parameter, Default is 1.\n\n        Returns\n        -------\n        a, b : ndarray of float\n            end-points of range that contain ``100 * alpha %`` of the rv's\n            possible values.\n\n        Notes\n        -----\n        This is implemented as ``ppf([p_tail, 1-p_tail])``, where\n        ``ppf`` is the inverse cumulative distribution function and\n        ``p_tail = (1-confidence)/2``. Suppose ``[c, d]`` is the support of a\n        discrete distribution; then ``ppf([0, 1]) == (c-1, d)``. Therefore,\n        when ``confidence=1`` and the distribution is discrete, the left end\n        of the interval will be beyond the support of the distribution.\n        For discrete distributions, the interval will limit the probability\n        in each tail to be less than or equal to ``p_tail`` (usually\n        strictly less).\n\n        "
        alpha = confidence
        alpha = asarray(alpha)
        if np.any((alpha > 1) | (alpha < 0)):
            raise ValueError('alpha must be between 0 and 1 inclusive')
        q1 = (1.0 - alpha) / 2
        q2 = (1.0 + alpha) / 2
        a = self.ppf(q1, *args, **kwds)
        b = self.ppf(q2, *args, **kwds)
        return (a, b)

    def support(self, *args, **kwargs):
        if False:
            print('Hello World!')
        "Support of the distribution.\n\n        Parameters\n        ----------\n        arg1, arg2, ... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information).\n        loc : array_like, optional\n            location parameter, Default is 0.\n        scale : array_like, optional\n            scale parameter, Default is 1.\n\n        Returns\n        -------\n        a, b : array_like\n            end-points of the distribution's support.\n\n        "
        (args, loc, scale) = self._parse_args(*args, **kwargs)
        arrs = np.broadcast_arrays(*args, loc, scale)
        (args, loc, scale) = (arrs[:-2], arrs[-2], arrs[-1])
        cond = self._argcheck(*args) & (scale > 0)
        (_a, _b) = self._get_support(*args)
        if cond.all():
            return (_a * scale + loc, _b * scale + loc)
        elif cond.ndim == 0:
            return (self.badvalue, self.badvalue)
        (_a, _b) = (np.asarray(_a).astype('d'), np.asarray(_b).astype('d'))
        (out_a, out_b) = (_a * scale + loc, _b * scale + loc)
        place(out_a, 1 - cond, self.badvalue)
        place(out_b, 1 - cond, self.badvalue)
        return (out_a, out_b)

    def nnlf(self, theta, x):
        if False:
            i = 10
            return i + 15
        'Negative loglikelihood function.\n        Notes\n        -----\n        This is ``-sum(log pdf(x, theta), axis=0)`` where `theta` are the\n        parameters (including loc and scale).\n        '
        (loc, scale, args) = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        x = (asarray(x) - loc) / scale
        n_log_scale = len(x) * log(scale)
        if np.any(~self._support_mask(x, *args)):
            return inf
        return self._nnlf(x, *args) + n_log_scale

    def _nnlf(self, x, *args):
        if False:
            while True:
                i = 10
        return -np.sum(self._logpxf(x, *args), axis=0)

    def _nlff_and_penalty(self, x, args, log_fitfun):
        if False:
            for i in range(10):
                print('nop')
        cond0 = ~self._support_mask(x, *args)
        n_bad = np.count_nonzero(cond0, axis=0)
        if n_bad > 0:
            x = argsreduce(~cond0, x)[0]
        logff = log_fitfun(x, *args)
        finite_logff = np.isfinite(logff)
        n_bad += np.sum(~finite_logff, axis=0)
        if n_bad > 0:
            penalty = n_bad * log(_XMAX) * 100
            return -np.sum(logff[finite_logff], axis=0) + penalty
        return -np.sum(logff, axis=0)

    def _penalized_nnlf(self, theta, x):
        if False:
            return 10
        'Penalized negative loglikelihood function.\n        i.e., - sum (log pdf(x, theta), axis=0) + penalty\n        where theta are the parameters (including loc and scale)\n        '
        (loc, scale, args) = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        x = asarray((x - loc) / scale)
        n_log_scale = len(x) * log(scale)
        return self._nlff_and_penalty(x, args, self._logpxf) + n_log_scale

    def _penalized_nlpsf(self, theta, x):
        if False:
            i = 10
            return i + 15
        'Penalized negative log product spacing function.\n        i.e., - sum (log (diff (cdf (x, theta))), axis=0) + penalty\n        where theta are the parameters (including loc and scale)\n        Follows reference [1] of scipy.stats.fit\n        '
        (loc, scale, args) = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        x = (np.sort(x) - loc) / scale

        def log_psf(x, *args):
            if False:
                return 10
            (x, lj) = np.unique(x, return_counts=True)
            cdf_data = self._cdf(x, *args) if x.size else []
            if not (x.size and 1 - cdf_data[-1] <= 0):
                cdf = np.concatenate(([0], cdf_data, [1]))
                lj = np.concatenate((lj, [1]))
            else:
                cdf = np.concatenate(([0], cdf_data))
            return lj * np.log(np.diff(cdf) / lj)
        return self._nlff_and_penalty(x, args, log_psf)

class _ShapeInfo:

    def __init__(self, name, integrality=False, domain=(-np.inf, np.inf), inclusive=(True, True)):
        if False:
            return 10
        self.name = name
        self.integrality = integrality
        domain = list(domain)
        if np.isfinite(domain[0]) and (not inclusive[0]):
            domain[0] = np.nextafter(domain[0], np.inf)
        if np.isfinite(domain[1]) and (not inclusive[1]):
            domain[1] = np.nextafter(domain[1], -np.inf)
        self.domain = domain

def _get_fixed_fit_value(kwds, names):
    if False:
        print('Hello World!')
    "\n    Given names such as `['f0', 'fa', 'fix_a']`, check that there is\n    at most one non-None value in `kwds` associaed with those names.\n    Return that value, or None if none of the names occur in `kwds`.\n    As a side effect, all occurrences of those names in `kwds` are\n    removed.\n    "
    vals = [(name, kwds.pop(name)) for name in names if name in kwds]
    if len(vals) > 1:
        repeated = [name for (name, val) in vals]
        raise ValueError('fit method got multiple keyword arguments to specify the same fixed parameter: ' + ', '.join(repeated))
    return vals[0][1] if vals else None

class rv_continuous(rv_generic):
    """A generic continuous random variable class meant for subclassing.

    `rv_continuous` is a base class to construct specific distribution classes
    and instances for continuous random variables. It cannot be used
    directly as a distribution.

    Parameters
    ----------
    momtype : int, optional
        The type of generic moment calculation to use: 0 for pdf, 1 (default)
        for ppf.
    a : float, optional
        Lower bound of the support of the distribution, default is minus
        infinity.
    b : float, optional
        Upper bound of the support of the distribution, default is plus
        infinity.
    xtol : float, optional
        The tolerance for fixed point calculation for generic ppf.
    badvalue : float, optional
        The value in a result arrays that indicates a value that for which
        some argument restriction is violated, default is np.nan.
    name : str, optional
        The name of the instance. This string is used to construct the default
        example for distributions.
    longname : str, optional
        This string is used as part of the first line of the docstring returned
        when a subclass has no docstring of its own. Note: `longname` exists
        for backwards compatibility, do not use for new subclasses.
    shapes : str, optional
        The shape of the distribution. For example ``"m, n"`` for a
        distribution that takes two integers as the two shape arguments for all
        its methods. If not provided, shape parameters will be inferred from
        the signature of the private methods, ``_pdf`` and ``_cdf`` of the
        instance.
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Methods
    -------
    rvs
    pdf
    logpdf
    cdf
    logcdf
    sf
    logsf
    ppf
    isf
    moment
    stats
    entropy
    expect
    median
    mean
    std
    var
    interval
    __call__
    fit
    fit_loc_scale
    nnlf
    support

    Notes
    -----
    Public methods of an instance of a distribution class (e.g., ``pdf``,
    ``cdf``) check their arguments and pass valid arguments to private,
    computational methods (``_pdf``, ``_cdf``). For ``pdf(x)``, ``x`` is valid
    if it is within the support of the distribution.
    Whether a shape parameter is valid is decided by an ``_argcheck`` method
    (which defaults to checking that its arguments are strictly positive.)

    **Subclassing**

    New random variables can be defined by subclassing the `rv_continuous` class
    and re-defining at least the ``_pdf`` or the ``_cdf`` method (normalized
    to location 0 and scale 1).

    If positive argument checking is not correct for your RV
    then you will also need to re-define the ``_argcheck`` method.

    For most of the scipy.stats distributions, the support interval doesn't
    depend on the shape parameters. ``x`` being in the support interval is
    equivalent to ``self.a <= x <= self.b``.  If either of the endpoints of
    the support do depend on the shape parameters, then
    i) the distribution must implement the ``_get_support`` method; and
    ii) those dependent endpoints must be omitted from the distribution's
    call to the ``rv_continuous`` initializer.

    Correct, but potentially slow defaults exist for the remaining
    methods but for speed and/or accuracy you can over-ride::

      _logpdf, _cdf, _logcdf, _ppf, _rvs, _isf, _sf, _logsf

    The default method ``_rvs`` relies on the inverse of the cdf, ``_ppf``,
    applied to a uniform random variate. In order to generate random variates
    efficiently, either the default ``_ppf`` needs to be overwritten (e.g.
    if the inverse cdf can expressed in an explicit form) or a sampling
    method needs to be implemented in a custom ``_rvs`` method.

    If possible, you should override ``_isf``, ``_sf`` or ``_logsf``.
    The main reason would be to improve numerical accuracy: for example,
    the survival function ``_sf`` is computed as ``1 - _cdf`` which can
    result in loss of precision if ``_cdf(x)`` is close to one.

    **Methods that can be overwritten by subclasses**
    ::

      _rvs
      _pdf
      _cdf
      _sf
      _ppf
      _isf
      _stats
      _munp
      _entropy
      _argcheck
      _get_support

    There are additional (internal and private) generic methods that can
    be useful for cross-checking and for debugging, but might work in all
    cases when directly called.

    A note on ``shapes``: subclasses need not specify them explicitly. In this
    case, `shapes` will be automatically deduced from the signatures of the
    overridden methods (`pdf`, `cdf` etc).
    If, for some reason, you prefer to avoid relying on introspection, you can
    specify ``shapes`` explicitly as an argument to the instance constructor.


    **Frozen Distributions**

    Normally, you must provide shape parameters (and, optionally, location and
    scale parameters to each call of a method of a distribution.

    Alternatively, the object may be called (as a function) to fix the shape,
    location, and scale parameters returning a "frozen" continuous RV object:

    rv = generic(<shape(s)>, loc=0, scale=1)
        `rv_frozen` object with the same methods but holding the given shape,
        location, and scale fixed

    **Statistics**

    Statistics are computed using numerical integration by default.
    For speed you can redefine this using ``_stats``:

     - take shape parameters and return mu, mu2, g1, g2
     - If you can't compute one of these, return it as None
     - Can also be defined with a keyword argument ``moments``, which is a
       string composed of "m", "v", "s", and/or "k".
       Only the components appearing in string should be computed and
       returned in the order "m", "v", "s", or "k"  with missing values
       returned as None.

    Alternatively, you can override ``_munp``, which takes ``n`` and shape
    parameters and returns the n-th non-central moment of the distribution.

    **Deepcopying / Pickling**

    If a distribution or frozen distribution is deepcopied (pickled/unpickled,
    etc.), any underlying random number generator is deepcopied with it. An
    implication is that if a distribution relies on the singleton RandomState
    before copying, it will rely on a copy of that random state after copying,
    and ``np.random.seed`` will no longer control the state.

    Examples
    --------
    To create a new Gaussian distribution, we would do the following:

    >>> from scipy.stats import rv_continuous
    >>> class gaussian_gen(rv_continuous):
    ...     "Gaussian distribution"
    ...     def _pdf(self, x):
    ...         return np.exp(-x**2 / 2.) / np.sqrt(2.0 * np.pi)
    >>> gaussian = gaussian_gen(name='gaussian')

    ``scipy.stats`` distributions are *instances*, so here we subclass
    `rv_continuous` and create an instance. With this, we now have
    a fully functional distribution with all relevant methods automagically
    generated by the framework.

    Note that above we defined a standard normal distribution, with zero mean
    and unit variance. Shifting and scaling of the distribution can be done
    by using ``loc`` and ``scale`` parameters: ``gaussian.pdf(x, loc, scale)``
    essentially computes ``y = (x - loc) / scale`` and
    ``gaussian._pdf(y) / scale``.

    """

    def __init__(self, momtype=1, a=None, b=None, xtol=1e-14, badvalue=None, name=None, longname=None, shapes=None, seed=None):
        if False:
            i = 10
            return i + 15
        super().__init__(seed)
        self._ctor_param = dict(momtype=momtype, a=a, b=b, xtol=xtol, badvalue=badvalue, name=name, longname=longname, shapes=shapes, seed=seed)
        if badvalue is None:
            badvalue = nan
        if name is None:
            name = 'Distribution'
        self.badvalue = badvalue
        self.name = name
        self.a = a
        self.b = b
        if a is None:
            self.a = -inf
        if b is None:
            self.b = inf
        self.xtol = xtol
        self.moment_type = momtype
        self.shapes = shapes
        self._construct_argparser(meths_to_inspect=[self._pdf, self._cdf], locscale_in='loc=0, scale=1', locscale_out='loc, scale')
        self._attach_methods()
        if longname is None:
            if name[0] in ['aeiouAEIOU']:
                hstr = 'An '
            else:
                hstr = 'A '
            longname = hstr + name
        if sys.flags.optimize < 2:
            if self.__doc__ is None:
                self._construct_default_doc(longname=longname, docdict=docdict, discrete='continuous')
            else:
                dct = dict(distcont)
                self._construct_doc(docdict, dct.get(self.name))

    def __getstate__(self):
        if False:
            print('Hello World!')
        dct = self.__dict__.copy()
        attrs = ['_parse_args', '_parse_args_stats', '_parse_args_rvs', '_cdfvec', '_ppfvec', 'vecentropy', 'generic_moment']
        [dct.pop(attr, None) for attr in attrs]
        return dct

    def _attach_methods(self):
        if False:
            i = 10
            return i + 15
        '\n        Attaches dynamically created methods to the rv_continuous instance.\n        '
        self._attach_argparser_methods()
        self._ppfvec = vectorize(self._ppf_single, otypes='d')
        self._ppfvec.nin = self.numargs + 1
        self.vecentropy = vectorize(self._entropy, otypes='d')
        self._cdfvec = vectorize(self._cdf_single, otypes='d')
        self._cdfvec.nin = self.numargs + 1
        if self.moment_type == 0:
            self.generic_moment = vectorize(self._mom0_sc, otypes='d')
        else:
            self.generic_moment = vectorize(self._mom1_sc, otypes='d')
        self.generic_moment.nin = self.numargs + 1

    def _updated_ctor_param(self):
        if False:
            i = 10
            return i + 15
        'Return the current version of _ctor_param, possibly updated by user.\n\n        Used by freezing.\n        Keep this in sync with the signature of __init__.\n        '
        dct = self._ctor_param.copy()
        dct['a'] = self.a
        dct['b'] = self.b
        dct['xtol'] = self.xtol
        dct['badvalue'] = self.badvalue
        dct['name'] = self.name
        dct['shapes'] = self.shapes
        return dct

    def _ppf_to_solve(self, x, q, *args):
        if False:
            i = 10
            return i + 15
        return self.cdf(*(x,) + args) - q

    def _ppf_single(self, q, *args):
        if False:
            print('Hello World!')
        factor = 10.0
        (left, right) = self._get_support(*args)
        if np.isinf(left):
            left = min(-factor, right)
            while self._ppf_to_solve(left, q, *args) > 0.0:
                (left, right) = (left * factor, left)
        if np.isinf(right):
            right = max(factor, left)
            while self._ppf_to_solve(right, q, *args) < 0.0:
                (left, right) = (right, right * factor)
        return optimize.brentq(self._ppf_to_solve, left, right, args=(q,) + args, xtol=self.xtol)

    def _mom_integ0(self, x, m, *args):
        if False:
            for i in range(10):
                print('nop')
        return x ** m * self.pdf(x, *args)

    def _mom0_sc(self, m, *args):
        if False:
            for i in range(10):
                print('nop')
        (_a, _b) = self._get_support(*args)
        return integrate.quad(self._mom_integ0, _a, _b, args=(m,) + args)[0]

    def _mom_integ1(self, q, m, *args):
        if False:
            print('Hello World!')
        return self.ppf(q, *args) ** m

    def _mom1_sc(self, m, *args):
        if False:
            i = 10
            return i + 15
        return integrate.quad(self._mom_integ1, 0, 1, args=(m,) + args)[0]

    def _pdf(self, x, *args):
        if False:
            for i in range(10):
                print('nop')
        return _derivative(self._cdf, x, dx=1e-05, args=args, order=5)

    def _logpdf(self, x, *args):
        if False:
            for i in range(10):
                print('nop')
        p = self._pdf(x, *args)
        with np.errstate(divide='ignore'):
            return log(p)

    def _logpxf(self, x, *args):
        if False:
            i = 10
            return i + 15
        return self._logpdf(x, *args)

    def _cdf_single(self, x, *args):
        if False:
            return 10
        (_a, _b) = self._get_support(*args)
        return integrate.quad(self._pdf, _a, x, args=args)[0]

    def _cdf(self, x, *args):
        if False:
            while True:
                i = 10
        return self._cdfvec(x, *args)

    def pdf(self, x, *args, **kwds):
        if False:
            print('Hello World!')
        'Probability density function at x of the given RV.\n\n        Parameters\n        ----------\n        x : array_like\n            quantiles\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information)\n        loc : array_like, optional\n            location parameter (default=0)\n        scale : array_like, optional\n            scale parameter (default=1)\n\n        Returns\n        -------\n        pdf : ndarray\n            Probability density function evaluated at x\n\n        '
        (args, loc, scale) = self._parse_args(*args, **kwds)
        (x, loc, scale) = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc) / scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._support_mask(x, *args) & (scale > 0)
        cond = cond0 & cond1
        output = zeros(shape(cond), dtyp)
        putmask(output, 1 - cond0 + np.isnan(x), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *(x,) + args + (scale,))
            (scale, goodargs) = (goodargs[-1], goodargs[:-1])
            place(output, cond, self._pdf(*goodargs) / scale)
        if output.ndim == 0:
            return output[()]
        return output

    def logpdf(self, x, *args, **kwds):
        if False:
            i = 10
            return i + 15
        'Log of the probability density function at x of the given RV.\n\n        This uses a more numerically accurate calculation if available.\n\n        Parameters\n        ----------\n        x : array_like\n            quantiles\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information)\n        loc : array_like, optional\n            location parameter (default=0)\n        scale : array_like, optional\n            scale parameter (default=1)\n\n        Returns\n        -------\n        logpdf : array_like\n            Log of the probability density function evaluated at x\n\n        '
        (args, loc, scale) = self._parse_args(*args, **kwds)
        (x, loc, scale) = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc) / scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._support_mask(x, *args) & (scale > 0)
        cond = cond0 & cond1
        output = empty(shape(cond), dtyp)
        output.fill(-inf)
        putmask(output, 1 - cond0 + np.isnan(x), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *(x,) + args + (scale,))
            (scale, goodargs) = (goodargs[-1], goodargs[:-1])
            place(output, cond, self._logpdf(*goodargs) - log(scale))
        if output.ndim == 0:
            return output[()]
        return output

    def cdf(self, x, *args, **kwds):
        if False:
            for i in range(10):
                print('nop')
        '\n        Cumulative distribution function of the given RV.\n\n        Parameters\n        ----------\n        x : array_like\n            quantiles\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information)\n        loc : array_like, optional\n            location parameter (default=0)\n        scale : array_like, optional\n            scale parameter (default=1)\n\n        Returns\n        -------\n        cdf : ndarray\n            Cumulative distribution function evaluated at `x`\n\n        '
        (args, loc, scale) = self._parse_args(*args, **kwds)
        (x, loc, scale) = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        (_a, _b) = self._get_support(*args)
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc) / scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = (x >= np.asarray(_b)) & cond0
        cond = cond0 & cond1
        output = zeros(shape(cond), dtyp)
        place(output, 1 - cond0 + np.isnan(x), self.badvalue)
        place(output, cond2, 1.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *(x,) + args)
            place(output, cond, self._cdf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def logcdf(self, x, *args, **kwds):
        if False:
            i = 10
            return i + 15
        'Log of the cumulative distribution function at x of the given RV.\n\n        Parameters\n        ----------\n        x : array_like\n            quantiles\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information)\n        loc : array_like, optional\n            location parameter (default=0)\n        scale : array_like, optional\n            scale parameter (default=1)\n\n        Returns\n        -------\n        logcdf : array_like\n            Log of the cumulative distribution function evaluated at x\n\n        '
        (args, loc, scale) = self._parse_args(*args, **kwds)
        (x, loc, scale) = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        (_a, _b) = self._get_support(*args)
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc) / scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = (x >= _b) & cond0
        cond = cond0 & cond1
        output = empty(shape(cond), dtyp)
        output.fill(-inf)
        place(output, (1 - cond0) * (cond1 == cond1) + np.isnan(x), self.badvalue)
        place(output, cond2, 0.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *(x,) + args)
            place(output, cond, self._logcdf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def sf(self, x, *args, **kwds):
        if False:
            print('Hello World!')
        'Survival function (1 - `cdf`) at x of the given RV.\n\n        Parameters\n        ----------\n        x : array_like\n            quantiles\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information)\n        loc : array_like, optional\n            location parameter (default=0)\n        scale : array_like, optional\n            scale parameter (default=1)\n\n        Returns\n        -------\n        sf : array_like\n            Survival function evaluated at x\n\n        '
        (args, loc, scale) = self._parse_args(*args, **kwds)
        (x, loc, scale) = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        (_a, _b) = self._get_support(*args)
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc) / scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = cond0 & (x <= _a)
        cond = cond0 & cond1
        output = zeros(shape(cond), dtyp)
        place(output, 1 - cond0 + np.isnan(x), self.badvalue)
        place(output, cond2, 1.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *(x,) + args)
            place(output, cond, self._sf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def logsf(self, x, *args, **kwds):
        if False:
            while True:
                i = 10
        'Log of the survival function of the given RV.\n\n        Returns the log of the "survival function," defined as (1 - `cdf`),\n        evaluated at `x`.\n\n        Parameters\n        ----------\n        x : array_like\n            quantiles\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information)\n        loc : array_like, optional\n            location parameter (default=0)\n        scale : array_like, optional\n            scale parameter (default=1)\n\n        Returns\n        -------\n        logsf : ndarray\n            Log of the survival function evaluated at `x`.\n\n        '
        (args, loc, scale) = self._parse_args(*args, **kwds)
        (x, loc, scale) = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        (_a, _b) = self._get_support(*args)
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc) / scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = cond0 & (x <= _a)
        cond = cond0 & cond1
        output = empty(shape(cond), dtyp)
        output.fill(-inf)
        place(output, 1 - cond0 + np.isnan(x), self.badvalue)
        place(output, cond2, 0.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *(x,) + args)
            place(output, cond, self._logsf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def ppf(self, q, *args, **kwds):
        if False:
            while True:
                i = 10
        'Percent point function (inverse of `cdf`) at q of the given RV.\n\n        Parameters\n        ----------\n        q : array_like\n            lower tail probability\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information)\n        loc : array_like, optional\n            location parameter (default=0)\n        scale : array_like, optional\n            scale parameter (default=1)\n\n        Returns\n        -------\n        x : array_like\n            quantile corresponding to the lower tail probability q.\n\n        '
        (args, loc, scale) = self._parse_args(*args, **kwds)
        (q, loc, scale) = map(asarray, (q, loc, scale))
        args = tuple(map(asarray, args))
        (_a, _b) = self._get_support(*args)
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        cond1 = (0 < q) & (q < 1)
        cond2 = cond0 & (q == 0)
        cond3 = cond0 & (q == 1)
        cond = cond0 & cond1
        output = np.full(shape(cond), fill_value=self.badvalue)
        lower_bound = _a * scale + loc
        upper_bound = _b * scale + loc
        place(output, cond2, argsreduce(cond2, lower_bound)[0])
        place(output, cond3, argsreduce(cond3, upper_bound)[0])
        if np.any(cond):
            goodargs = argsreduce(cond, *(q,) + args + (scale, loc))
            (scale, loc, goodargs) = (goodargs[-2], goodargs[-1], goodargs[:-2])
            place(output, cond, self._ppf(*goodargs) * scale + loc)
        if output.ndim == 0:
            return output[()]
        return output

    def isf(self, q, *args, **kwds):
        if False:
            print('Hello World!')
        'Inverse survival function (inverse of `sf`) at q of the given RV.\n\n        Parameters\n        ----------\n        q : array_like\n            upper tail probability\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information)\n        loc : array_like, optional\n            location parameter (default=0)\n        scale : array_like, optional\n            scale parameter (default=1)\n\n        Returns\n        -------\n        x : ndarray or scalar\n            Quantile corresponding to the upper tail probability q.\n\n        '
        (args, loc, scale) = self._parse_args(*args, **kwds)
        (q, loc, scale) = map(asarray, (q, loc, scale))
        args = tuple(map(asarray, args))
        (_a, _b) = self._get_support(*args)
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        cond1 = (0 < q) & (q < 1)
        cond2 = cond0 & (q == 1)
        cond3 = cond0 & (q == 0)
        cond = cond0 & cond1
        output = np.full(shape(cond), fill_value=self.badvalue)
        lower_bound = _a * scale + loc
        upper_bound = _b * scale + loc
        place(output, cond2, argsreduce(cond2, lower_bound)[0])
        place(output, cond3, argsreduce(cond3, upper_bound)[0])
        if np.any(cond):
            goodargs = argsreduce(cond, *(q,) + args + (scale, loc))
            (scale, loc, goodargs) = (goodargs[-2], goodargs[-1], goodargs[:-2])
            place(output, cond, self._isf(*goodargs) * scale + loc)
        if output.ndim == 0:
            return output[()]
        return output

    def _unpack_loc_scale(self, theta):
        if False:
            for i in range(10):
                print('nop')
        try:
            loc = theta[-2]
            scale = theta[-1]
            args = tuple(theta[:-2])
        except IndexError as e:
            raise ValueError('Not enough input arguments.') from e
        return (loc, scale, args)

    def _nnlf_and_penalty(self, x, args):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute the penalized negative log-likelihood for the\n        "standardized" data (i.e. already shifted by loc and\n        scaled by scale) for the shape parameters in `args`.\n\n        `x` can be a 1D numpy array or a CensoredData instance.\n        '
        if isinstance(x, CensoredData):
            xs = x._supported(*self._get_support(*args))
            n_bad = len(x) - len(xs)
            (i1, i2) = xs._interval.T
            terms = [self._logpdf(xs._uncensored, *args), self._logcdf(xs._left, *args), self._logsf(xs._right, *args), np.log(self._delta_cdf(i1, i2, *args))]
        else:
            cond0 = ~self._support_mask(x, *args)
            n_bad = np.count_nonzero(cond0)
            if n_bad > 0:
                x = argsreduce(~cond0, x)[0]
            terms = [self._logpdf(x, *args)]
        (totals, bad_counts) = zip(*[_sum_finite(term) for term in terms])
        total = sum(totals)
        n_bad += sum(bad_counts)
        return -total + n_bad * _LOGXMAX * 100

    def _penalized_nnlf(self, theta, x):
        if False:
            i = 10
            return i + 15
        'Penalized negative loglikelihood function.\n\n        i.e., - sum (log pdf(x, theta), axis=0) + penalty\n        where theta are the parameters (including loc and scale)\n        '
        (loc, scale, args) = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        if isinstance(x, CensoredData):
            x = (x - loc) / scale
            n_log_scale = (len(x) - x.num_censored()) * log(scale)
        else:
            x = (x - loc) / scale
            n_log_scale = len(x) * log(scale)
        return self._nnlf_and_penalty(x, args) + n_log_scale

    def _fitstart(self, data, args=None):
        if False:
            print('Hello World!')
        'Starting point for fit (shape arguments + loc + scale).'
        if args is None:
            args = (1.0,) * self.numargs
        (loc, scale) = self._fit_loc_scale_support(data, *args)
        return args + (loc, scale)

    def _reduce_func(self, args, kwds, data=None):
        if False:
            print('Hello World!')
        '\n        Return the (possibly reduced) function to optimize in order to find MLE\n        estimates for the .fit method.\n        '
        shapes = []
        if self.shapes:
            shapes = self.shapes.replace(',', ' ').split()
            for (j, s) in enumerate(shapes):
                key = 'f' + str(j)
                names = [key, 'f' + s, 'fix_' + s]
                val = _get_fixed_fit_value(kwds, names)
                if val is not None:
                    kwds[key] = val
        args = list(args)
        Nargs = len(args)
        fixedn = []
        names = ['f%d' % n for n in range(Nargs - 2)] + ['floc', 'fscale']
        x0 = []
        for (n, key) in enumerate(names):
            if key in kwds:
                fixedn.append(n)
                args[n] = kwds.pop(key)
            else:
                x0.append(args[n])
        methods = {'mle', 'mm'}
        method = kwds.pop('method', 'mle').lower()
        if method == 'mm':
            n_params = len(shapes) + 2 - len(fixedn)
            exponents = np.arange(1, n_params + 1)[:, np.newaxis]
            data_moments = np.sum(data[None, :] ** exponents / len(data), axis=1)

            def objective(theta, x):
                if False:
                    i = 10
                    return i + 15
                return self._moment_error(theta, x, data_moments)
        elif method == 'mle':
            objective = self._penalized_nnlf
        else:
            raise ValueError("Method '{}' not available; must be one of {}".format(method, methods))
        if len(fixedn) == 0:
            func = objective
            restore = None
        else:
            if len(fixedn) == Nargs:
                raise ValueError('All parameters fixed. There is nothing to optimize.')

            def restore(args, theta):
                if False:
                    print('Hello World!')
                i = 0
                for n in range(Nargs):
                    if n not in fixedn:
                        args[n] = theta[i]
                        i += 1
                return args

            def func(theta, x):
                if False:
                    while True:
                        i = 10
                newtheta = restore(args[:], theta)
                return objective(newtheta, x)
        return (x0, func, restore, args)

    def _moment_error(self, theta, x, data_moments):
        if False:
            i = 10
            return i + 15
        (loc, scale, args) = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        dist_moments = np.array([self.moment(i + 1, *args, loc=loc, scale=scale) for i in range(len(data_moments))])
        if np.any(np.isnan(dist_moments)):
            raise ValueError("Method of moments encountered a non-finite distribution moment and cannot continue. Consider trying method='MLE'.")
        return (((data_moments - dist_moments) / np.maximum(np.abs(data_moments), 1e-08)) ** 2).sum()

    def fit(self, data, *args, **kwds):
        if False:
            print('Hello World!')
        '\n        Return estimates of shape (if applicable), location, and scale\n        parameters from data. The default estimation method is Maximum\n        Likelihood Estimation (MLE), but Method of Moments (MM)\n        is also available.\n\n        Starting estimates for the fit are given by input arguments;\n        for any arguments not provided with starting estimates,\n        ``self._fitstart(data)`` is called to generate such.\n\n        One can hold some parameters fixed to specific values by passing in\n        keyword arguments ``f0``, ``f1``, ..., ``fn`` (for shape parameters)\n        and ``floc`` and ``fscale`` (for location and scale parameters,\n        respectively).\n\n        Parameters\n        ----------\n        data : array_like or `CensoredData` instance\n            Data to use in estimating the distribution parameters.\n        arg1, arg2, arg3,... : floats, optional\n            Starting value(s) for any shape-characterizing arguments (those not\n            provided will be determined by a call to ``_fitstart(data)``).\n            No default value.\n        **kwds : floats, optional\n            - `loc`: initial guess of the distribution\'s location parameter.\n            - `scale`: initial guess of the distribution\'s scale parameter.\n\n            Special keyword arguments are recognized as holding certain\n            parameters fixed:\n\n            - f0...fn : hold respective shape parameters fixed.\n              Alternatively, shape parameters to fix can be specified by name.\n              For example, if ``self.shapes == "a, b"``, ``fa`` and ``fix_a``\n              are equivalent to ``f0``, and ``fb`` and ``fix_b`` are\n              equivalent to ``f1``.\n\n            - floc : hold location parameter fixed to specified value.\n\n            - fscale : hold scale parameter fixed to specified value.\n\n            - optimizer : The optimizer to use.  The optimizer must take\n              ``func`` and starting position as the first two arguments,\n              plus ``args`` (for extra arguments to pass to the\n              function to be optimized) and ``disp=0`` to suppress\n              output as keyword arguments.\n\n            - method : The method to use. The default is "MLE" (Maximum\n              Likelihood Estimate); "MM" (Method of Moments)\n              is also available.\n\n        Raises\n        ------\n        TypeError, ValueError\n            If an input is invalid\n        `~scipy.stats.FitError`\n            If fitting fails or the fit produced would be invalid\n\n        Returns\n        -------\n        parameter_tuple : tuple of floats\n            Estimates for any shape parameters (if applicable), followed by\n            those for location and scale. For most random variables, shape\n            statistics will be returned, but there are exceptions (e.g.\n            ``norm``).\n\n        Notes\n        -----\n        With ``method="MLE"`` (default), the fit is computed by minimizing\n        the negative log-likelihood function. A large, finite penalty\n        (rather than infinite negative log-likelihood) is applied for\n        observations beyond the support of the distribution.\n\n        With ``method="MM"``, the fit is computed by minimizing the L2 norm\n        of the relative errors between the first *k* raw (about zero) data\n        moments and the corresponding distribution moments, where *k* is the\n        number of non-fixed parameters.\n        More precisely, the objective function is::\n\n            (((data_moments - dist_moments)\n              / np.maximum(np.abs(data_moments), 1e-8))**2).sum()\n\n        where the constant ``1e-8`` avoids division by zero in case of\n        vanishing data moments. Typically, this error norm can be reduced to\n        zero.\n        Note that the standard method of moments can produce parameters for\n        which some data are outside the support of the fitted distribution;\n        this implementation does nothing to prevent this.\n\n        For either method,\n        the returned answer is not guaranteed to be globally optimal; it\n        may only be locally optimal, or the optimization may fail altogether.\n        If the data contain any of ``np.nan``, ``np.inf``, or ``-np.inf``,\n        the `fit` method will raise a ``RuntimeError``.\n\n        Examples\n        --------\n\n        Generate some data to fit: draw random variates from the `beta`\n        distribution\n\n        >>> from scipy.stats import beta\n        >>> a, b = 1., 2.\n        >>> x = beta.rvs(a, b, size=1000)\n\n        Now we can fit all four parameters (``a``, ``b``, ``loc`` and\n        ``scale``):\n\n        >>> a1, b1, loc1, scale1 = beta.fit(x)\n\n        We can also use some prior knowledge about the dataset: let\'s keep\n        ``loc`` and ``scale`` fixed:\n\n        >>> a1, b1, loc1, scale1 = beta.fit(x, floc=0, fscale=1)\n        >>> loc1, scale1\n        (0, 1)\n\n        We can also keep shape parameters fixed by using ``f``-keywords. To\n        keep the zero-th shape parameter ``a`` equal 1, use ``f0=1`` or,\n        equivalently, ``fa=1``:\n\n        >>> a1, b1, loc1, scale1 = beta.fit(x, fa=1, floc=0, fscale=1)\n        >>> a1\n        1\n\n        Not all distributions return estimates for the shape parameters.\n        ``norm`` for example just returns estimates for location and scale:\n\n        >>> from scipy.stats import norm\n        >>> x = norm.rvs(a, b, size=1000, random_state=123)\n        >>> loc1, scale1 = norm.fit(x)\n        >>> loc1, scale1\n        (0.92087172783841631, 2.0015750750324668)\n        '
        method = kwds.get('method', 'mle').lower()
        censored = isinstance(data, CensoredData)
        if censored:
            if method != 'mle':
                raise ValueError('For censored data, the method must be "MLE".')
            if data.num_censored() == 0:
                data = data._uncensored
                censored = False
        Narg = len(args)
        if Narg > self.numargs:
            raise TypeError('Too many input arguments.')
        if not censored:
            data = np.asarray(data).ravel()
            if not np.isfinite(data).all():
                raise ValueError('The data contains non-finite values.')
        start = [None] * 2
        if Narg < self.numargs or not ('loc' in kwds and 'scale' in kwds):
            start = self._fitstart(data)
            args += start[Narg:-2]
        loc = kwds.pop('loc', start[-2])
        scale = kwds.pop('scale', start[-1])
        args += (loc, scale)
        (x0, func, restore, args) = self._reduce_func(args, kwds, data=data)
        optimizer = kwds.pop('optimizer', optimize.fmin)
        optimizer = _fit_determine_optimizer(optimizer)
        if kwds:
            raise TypeError('Unknown arguments: %s.' % kwds)
        vals = optimizer(func, x0, args=(data,), disp=0)
        obj = func(vals, data)
        if restore is not None:
            vals = restore(args, vals)
        vals = tuple(vals)
        (loc, scale, shapes) = self._unpack_loc_scale(vals)
        if not (np.all(self._argcheck(*shapes)) and scale > 0):
            raise FitError('Optimization converged to parameters that are outside the range allowed by the distribution.')
        if method == 'mm':
            if not np.isfinite(obj):
                raise FitError('Optimization failed: either a data moment or fitted distribution moment is non-finite.')
        return vals

    def _fit_loc_scale_support(self, data, *args):
        if False:
            while True:
                i = 10
        'Estimate loc and scale parameters from data accounting for support.\n\n        Parameters\n        ----------\n        data : array_like\n            Data to fit.\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information).\n\n        Returns\n        -------\n        Lhat : float\n            Estimated location parameter for the data.\n        Shat : float\n            Estimated scale parameter for the data.\n\n        '
        if isinstance(data, CensoredData):
            data = data._uncensor()
        else:
            data = np.asarray(data)
        (loc_hat, scale_hat) = self.fit_loc_scale(data, *args)
        self._argcheck(*args)
        (_a, _b) = self._get_support(*args)
        (a, b) = (_a, _b)
        support_width = b - a
        if support_width <= 0:
            return (loc_hat, scale_hat)
        a_hat = loc_hat + a * scale_hat
        b_hat = loc_hat + b * scale_hat
        data_a = np.min(data)
        data_b = np.max(data)
        if a_hat < data_a and data_b < b_hat:
            return (loc_hat, scale_hat)
        data_width = data_b - data_a
        rel_margin = 0.1
        margin = data_width * rel_margin
        if support_width < np.inf:
            loc_hat = data_a - a - margin
            scale_hat = (data_width + 2 * margin) / support_width
            return (loc_hat, scale_hat)
        if a > -np.inf:
            return (data_a - a - margin, 1)
        elif b < np.inf:
            return (data_b - b + margin, 1)
        else:
            raise RuntimeError

    def fit_loc_scale(self, data, *args):
        if False:
            return 10
        '\n        Estimate loc and scale parameters from data using 1st and 2nd moments.\n\n        Parameters\n        ----------\n        data : array_like\n            Data to fit.\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information).\n\n        Returns\n        -------\n        Lhat : float\n            Estimated location parameter for the data.\n        Shat : float\n            Estimated scale parameter for the data.\n\n        '
        (mu, mu2) = self.stats(*args, **{'moments': 'mv'})
        tmp = asarray(data)
        muhat = tmp.mean()
        mu2hat = tmp.var()
        Shat = sqrt(mu2hat / mu2)
        with np.errstate(invalid='ignore'):
            Lhat = muhat - Shat * mu
        if not np.isfinite(Lhat):
            Lhat = 0
        if not (np.isfinite(Shat) and 0 < Shat):
            Shat = 1
        return (Lhat, Shat)

    def _entropy(self, *args):
        if False:
            for i in range(10):
                print('nop')

        def integ(x):
            if False:
                i = 10
                return i + 15
            val = self._pdf(x, *args)
            return entr(val)
        (_a, _b) = self._get_support(*args)
        with np.errstate(over='ignore'):
            h = integrate.quad(integ, _a, _b)[0]
        if not np.isnan(h):
            return h
        else:
            (low, upp) = self.ppf([1e-10, 1.0 - 1e-10], *args)
            if np.isinf(_b):
                upper = upp
            else:
                upper = _b
            if np.isinf(_a):
                lower = low
            else:
                lower = _a
            return integrate.quad(integ, lower, upper)[0]

    def expect(self, func=None, args=(), loc=0, scale=1, lb=None, ub=None, conditional=False, **kwds):
        if False:
            for i in range(10):
                print('nop')
        'Calculate expected value of a function with respect to the\n        distribution by numerical integration.\n\n        The expected value of a function ``f(x)`` with respect to a\n        distribution ``dist`` is defined as::\n\n                    ub\n            E[f(x)] = Integral(f(x) * dist.pdf(x)),\n                    lb\n\n        where ``ub`` and ``lb`` are arguments and ``x`` has the ``dist.pdf(x)``\n        distribution. If the bounds ``lb`` and ``ub`` correspond to the\n        support of the distribution, e.g. ``[-inf, inf]`` in the default\n        case, then the integral is the unrestricted expectation of ``f(x)``.\n        Also, the function ``f(x)`` may be defined such that ``f(x)`` is ``0``\n        outside a finite interval in which case the expectation is\n        calculated within the finite range ``[lb, ub]``.\n\n        Parameters\n        ----------\n        func : callable, optional\n            Function for which integral is calculated. Takes only one argument.\n            The default is the identity mapping f(x) = x.\n        args : tuple, optional\n            Shape parameters of the distribution.\n        loc : float, optional\n            Location parameter (default=0).\n        scale : float, optional\n            Scale parameter (default=1).\n        lb, ub : scalar, optional\n            Lower and upper bound for integration. Default is set to the\n            support of the distribution.\n        conditional : bool, optional\n            If True, the integral is corrected by the conditional probability\n            of the integration interval.  The return value is the expectation\n            of the function, conditional on being in the given interval.\n            Default is False.\n\n        Additional keyword arguments are passed to the integration routine.\n\n        Returns\n        -------\n        expect : float\n            The calculated expected value.\n\n        Notes\n        -----\n        The integration behavior of this function is inherited from\n        `scipy.integrate.quad`. Neither this function nor\n        `scipy.integrate.quad` can verify whether the integral exists or is\n        finite. For example ``cauchy(0).mean()`` returns ``np.nan`` and\n        ``cauchy(0).expect()`` returns ``0.0``.\n\n        Likewise, the accuracy of results is not verified by the function.\n        `scipy.integrate.quad` is typically reliable for integrals that are\n        numerically favorable, but it is not guaranteed to converge\n        to a correct value for all possible intervals and integrands. This\n        function is provided for convenience; for critical applications,\n        check results against other integration methods.\n\n        The function is not vectorized.\n\n        Examples\n        --------\n\n        To understand the effect of the bounds of integration consider\n\n        >>> from scipy.stats import expon\n        >>> expon(1).expect(lambda x: 1, lb=0.0, ub=2.0)\n        0.6321205588285578\n\n        This is close to\n\n        >>> expon(1).cdf(2.0) - expon(1).cdf(0.0)\n        0.6321205588285577\n\n        If ``conditional=True``\n\n        >>> expon(1).expect(lambda x: 1, lb=0.0, ub=2.0, conditional=True)\n        1.0000000000000002\n\n        The slight deviation from 1 is due to numerical integration.\n\n        The integrand can be treated as a complex-valued function\n        by passing ``complex_func=True`` to `scipy.integrate.quad` .\n\n        >>> import numpy as np\n        >>> from scipy.stats import vonmises\n        >>> res = vonmises(loc=2, kappa=1).expect(lambda x: np.exp(1j*x),\n        ...                                       complex_func=True)\n        >>> res\n        (-0.18576377217422957+0.40590124735052263j)\n\n        >>> np.angle(res)  # location of the (circular) distribution\n        2.0\n\n        '
        lockwds = {'loc': loc, 'scale': scale}
        self._argcheck(*args)
        (_a, _b) = self._get_support(*args)
        if func is None:

            def fun(x, *args):
                if False:
                    i = 10
                    return i + 15
                return x * self.pdf(x, *args, **lockwds)
        else:

            def fun(x, *args):
                if False:
                    i = 10
                    return i + 15
                return func(x) * self.pdf(x, *args, **lockwds)
        if lb is None:
            lb = loc + _a * scale
        if ub is None:
            ub = loc + _b * scale
        cdf_bounds = self.cdf([lb, ub], *args, **lockwds)
        invfac = cdf_bounds[1] - cdf_bounds[0]
        kwds['args'] = args
        alpha = 0.05
        inner_bounds = np.array([alpha, 1 - alpha])
        cdf_inner_bounds = cdf_bounds[0] + invfac * inner_bounds
        (c, d) = loc + self._ppf(cdf_inner_bounds, *args) * scale
        lbc = integrate.quad(fun, lb, c, **kwds)[0]
        cd = integrate.quad(fun, c, d, **kwds)[0]
        dub = integrate.quad(fun, d, ub, **kwds)[0]
        vals = lbc + cd + dub
        if conditional:
            vals /= invfac
        return np.array(vals)[()]

    def _param_info(self):
        if False:
            return 10
        shape_info = self._shape_info()
        loc_info = _ShapeInfo('loc', False, (-np.inf, np.inf), (False, False))
        scale_info = _ShapeInfo('scale', False, (0, np.inf), (False, False))
        param_info = shape_info + [loc_info, scale_info]
        return param_info

    def _delta_cdf(self, x1, x2, *args, loc=0, scale=1):
        if False:
            print('Hello World!')
        '\n        Compute CDF(x2) - CDF(x1).\n\n        Where x1 is greater than the median, compute SF(x1) - SF(x2),\n        otherwise compute CDF(x2) - CDF(x1).\n\n        This function is only useful if `dist.sf(x, ...)` has an implementation\n        that is numerically more accurate than `1 - dist.cdf(x, ...)`.\n        '
        cdf1 = self.cdf(x1, *args, loc=loc, scale=scale)
        result = np.where(cdf1 > 0.5, self.sf(x1, *args, loc=loc, scale=scale) - self.sf(x2, *args, loc=loc, scale=scale), self.cdf(x2, *args, loc=loc, scale=scale) - cdf1)
        if result.ndim == 0:
            result = result[()]
        return result

def _drv2_moment(self, n, *args):
    if False:
        return 10
    'Non-central moment of discrete distribution.'

    def fun(x):
        if False:
            print('Hello World!')
        return np.power(x, n) * self._pmf(x, *args)
    (_a, _b) = self._get_support(*args)
    return _expect(fun, _a, _b, self.ppf(0.5, *args), self.inc)

def _drv2_ppfsingle(self, q, *args):
    if False:
        i = 10
        return i + 15
    (_a, _b) = self._get_support(*args)
    b = _b
    a = _a
    if isinf(b):
        b = int(max(100 * q, 10))
        while 1:
            if b >= _b:
                qb = 1.0
                break
            qb = self._cdf(b, *args)
            if qb < q:
                b += 10
            else:
                break
    else:
        qb = 1.0
    if isinf(a):
        a = int(min(-100 * q, -10))
        while 1:
            if a <= _a:
                qb = 0.0
                break
            qa = self._cdf(a, *args)
            if qa > q:
                a -= 10
            else:
                break
    else:
        qa = self._cdf(a, *args)
    while 1:
        if qa == q:
            return a
        if qb == q:
            return b
        if b <= a + 1:
            if qa > q:
                return a
            else:
                return b
        c = int((a + b) / 2.0)
        qc = self._cdf(c, *args)
        if qc < q:
            if a != c:
                a = c
            else:
                raise RuntimeError('updating stopped, endless loop')
            qa = qc
        elif qc > q:
            if b != c:
                b = c
            else:
                raise RuntimeError('updating stopped, endless loop')
            qb = qc
        else:
            return c

class rv_discrete(rv_generic):
    """A generic discrete random variable class meant for subclassing.

    `rv_discrete` is a base class to construct specific distribution classes
    and instances for discrete random variables. It can also be used
    to construct an arbitrary distribution defined by a list of support
    points and corresponding probabilities.

    Parameters
    ----------
    a : float, optional
        Lower bound of the support of the distribution, default: 0
    b : float, optional
        Upper bound of the support of the distribution, default: plus infinity
    moment_tol : float, optional
        The tolerance for the generic calculation of moments.
    values : tuple of two array_like, optional
        ``(xk, pk)`` where ``xk`` are integers and ``pk`` are the non-zero
        probabilities between 0 and 1 with ``sum(pk) = 1``. ``xk``
        and ``pk`` must have the same shape, and ``xk`` must be unique.
    inc : integer, optional
        Increment for the support of the distribution.
        Default is 1. (other values have not been tested)
    badvalue : float, optional
        The value in a result arrays that indicates a value that for which
        some argument restriction is violated, default is np.nan.
    name : str, optional
        The name of the instance. This string is used to construct the default
        example for distributions.
    longname : str, optional
        This string is used as part of the first line of the docstring returned
        when a subclass has no docstring of its own. Note: `longname` exists
        for backwards compatibility, do not use for new subclasses.
    shapes : str, optional
        The shape of the distribution. For example "m, n" for a distribution
        that takes two integers as the two shape arguments for all its methods
        If not provided, shape parameters will be inferred from
        the signatures of the private methods, ``_pmf`` and ``_cdf`` of
        the instance.
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

    Methods
    -------
    rvs
    pmf
    logpmf
    cdf
    logcdf
    sf
    logsf
    ppf
    isf
    moment
    stats
    entropy
    expect
    median
    mean
    std
    var
    interval
    __call__
    support

    Notes
    -----
    This class is similar to `rv_continuous`. Whether a shape parameter is
    valid is decided by an ``_argcheck`` method (which defaults to checking
    that its arguments are strictly positive.)
    The main differences are as follows.

    - The support of the distribution is a set of integers.
    - Instead of the probability density function, ``pdf`` (and the
      corresponding private ``_pdf``), this class defines the
      *probability mass function*, `pmf` (and the corresponding
      private ``_pmf``.)
    - There is no ``scale`` parameter.
    - The default implementations of methods (e.g. ``_cdf``) are not designed
      for distributions with support that is unbounded below (i.e.
      ``a=-np.inf``), so they must be overridden.

    To create a new discrete distribution, we would do the following:

    >>> from scipy.stats import rv_discrete
    >>> class poisson_gen(rv_discrete):
    ...     "Poisson distribution"
    ...     def _pmf(self, k, mu):
    ...         return exp(-mu) * mu**k / factorial(k)

    and create an instance::

    >>> poisson = poisson_gen(name="poisson")

    Note that above we defined the Poisson distribution in the standard form.
    Shifting the distribution can be done by providing the ``loc`` parameter
    to the methods of the instance. For example, ``poisson.pmf(x, mu, loc)``
    delegates the work to ``poisson._pmf(x-loc, mu)``.

    **Discrete distributions from a list of probabilities**

    Alternatively, you can construct an arbitrary discrete rv defined
    on a finite set of values ``xk`` with ``Prob{X=xk} = pk`` by using the
    ``values`` keyword argument to the `rv_discrete` constructor.

    **Deepcopying / Pickling**

    If a distribution or frozen distribution is deepcopied (pickled/unpickled,
    etc.), any underlying random number generator is deepcopied with it. An
    implication is that if a distribution relies on the singleton RandomState
    before copying, it will rely on a copy of that random state after copying,
    and ``np.random.seed`` will no longer control the state.

    Examples
    --------
    Custom made discrete distribution:

    >>> import numpy as np
    >>> from scipy import stats
    >>> xk = np.arange(7)
    >>> pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2)
    >>> custm = stats.rv_discrete(name='custm', values=(xk, pk))
    >>>
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(1, 1)
    >>> ax.plot(xk, custm.pmf(xk), 'ro', ms=12, mec='r')
    >>> ax.vlines(xk, 0, custm.pmf(xk), colors='r', lw=4)
    >>> plt.show()

    Random number generation:

    >>> R = custm.rvs(size=100)

    """

    def __new__(cls, a=0, b=inf, name=None, badvalue=None, moment_tol=1e-08, values=None, inc=1, longname=None, shapes=None, seed=None):
        if False:
            return 10
        if values is not None:
            return super().__new__(rv_sample)
        else:
            return super().__new__(cls)

    def __init__(self, a=0, b=inf, name=None, badvalue=None, moment_tol=1e-08, values=None, inc=1, longname=None, shapes=None, seed=None):
        if False:
            i = 10
            return i + 15
        super().__init__(seed)
        self._ctor_param = dict(a=a, b=b, name=name, badvalue=badvalue, moment_tol=moment_tol, values=values, inc=inc, longname=longname, shapes=shapes, seed=seed)
        if badvalue is None:
            badvalue = nan
        self.badvalue = badvalue
        self.a = a
        self.b = b
        self.moment_tol = moment_tol
        self.inc = inc
        self.shapes = shapes
        if values is not None:
            raise ValueError('rv_discrete.__init__(..., values != None, ...)')
        self._construct_argparser(meths_to_inspect=[self._pmf, self._cdf], locscale_in='loc=0', locscale_out='loc, 1')
        self._attach_methods()
        self._construct_docstrings(name, longname)

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        dct = self.__dict__.copy()
        attrs = ['_parse_args', '_parse_args_stats', '_parse_args_rvs', '_cdfvec', '_ppfvec', 'generic_moment']
        [dct.pop(attr, None) for attr in attrs]
        return dct

    def _attach_methods(self):
        if False:
            i = 10
            return i + 15
        'Attaches dynamically created methods to the rv_discrete instance.'
        self._cdfvec = vectorize(self._cdf_single, otypes='d')
        self.vecentropy = vectorize(self._entropy)
        self._attach_argparser_methods()
        _vec_generic_moment = vectorize(_drv2_moment, otypes='d')
        _vec_generic_moment.nin = self.numargs + 2
        self.generic_moment = types.MethodType(_vec_generic_moment, self)
        _vppf = vectorize(_drv2_ppfsingle, otypes='d')
        _vppf.nin = self.numargs + 2
        self._ppfvec = types.MethodType(_vppf, self)
        self._cdfvec.nin = self.numargs + 1

    def _construct_docstrings(self, name, longname):
        if False:
            print('Hello World!')
        if name is None:
            name = 'Distribution'
        self.name = name
        if longname is None:
            if name[0] in ['aeiouAEIOU']:
                hstr = 'An '
            else:
                hstr = 'A '
            longname = hstr + name
        if sys.flags.optimize < 2:
            if self.__doc__ is None:
                self._construct_default_doc(longname=longname, docdict=docdict_discrete, discrete='discrete')
            else:
                dct = dict(distdiscrete)
                self._construct_doc(docdict_discrete, dct.get(self.name))
            self.__doc__ = self.__doc__.replace('\n    scale : array_like, optional\n        scale parameter (default=1)', '')

    def _updated_ctor_param(self):
        if False:
            print('Hello World!')
        'Return the current version of _ctor_param, possibly updated by user.\n\n        Used by freezing.\n        Keep this in sync with the signature of __init__.\n        '
        dct = self._ctor_param.copy()
        dct['a'] = self.a
        dct['b'] = self.b
        dct['badvalue'] = self.badvalue
        dct['moment_tol'] = self.moment_tol
        dct['inc'] = self.inc
        dct['name'] = self.name
        dct['shapes'] = self.shapes
        return dct

    def _nonzero(self, k, *args):
        if False:
            i = 10
            return i + 15
        return floor(k) == k

    def _pmf(self, k, *args):
        if False:
            print('Hello World!')
        return self._cdf(k, *args) - self._cdf(k - 1, *args)

    def _logpmf(self, k, *args):
        if False:
            i = 10
            return i + 15
        return log(self._pmf(k, *args))

    def _logpxf(self, k, *args):
        if False:
            while True:
                i = 10
        return self._logpmf(k, *args)

    def _unpack_loc_scale(self, theta):
        if False:
            while True:
                i = 10
        try:
            loc = theta[-1]
            scale = 1
            args = tuple(theta[:-1])
        except IndexError as e:
            raise ValueError('Not enough input arguments.') from e
        return (loc, scale, args)

    def _cdf_single(self, k, *args):
        if False:
            return 10
        (_a, _b) = self._get_support(*args)
        m = arange(int(_a), k + 1)
        return np.sum(self._pmf(m, *args), axis=0)

    def _cdf(self, x, *args):
        if False:
            i = 10
            return i + 15
        k = floor(x)
        return self._cdfvec(k, *args)

    def rvs(self, *args, **kwargs):
        if False:
            return 10
        'Random variates of given type.\n\n        Parameters\n        ----------\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information).\n        loc : array_like, optional\n            Location parameter (default=0).\n        size : int or tuple of ints, optional\n            Defining number of random variates (Default is 1). Note that `size`\n            has to be given as keyword, not as positional argument.\n        random_state : {None, int, `numpy.random.Generator`,\n                        `numpy.random.RandomState`}, optional\n\n            If `random_state` is None (or `np.random`), the\n            `numpy.random.RandomState` singleton is used.\n            If `random_state` is an int, a new ``RandomState`` instance is\n            used, seeded with `random_state`.\n            If `random_state` is already a ``Generator`` or ``RandomState``\n            instance, that instance is used.\n\n        Returns\n        -------\n        rvs : ndarray or scalar\n            Random variates of given `size`.\n\n        '
        kwargs['discrete'] = True
        return super().rvs(*args, **kwargs)

    def pmf(self, k, *args, **kwds):
        if False:
            for i in range(10):
                print('nop')
        'Probability mass function at k of the given RV.\n\n        Parameters\n        ----------\n        k : array_like\n            Quantiles.\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information)\n        loc : array_like, optional\n            Location parameter (default=0).\n\n        Returns\n        -------\n        pmf : array_like\n            Probability mass function evaluated at k\n\n        '
        (args, loc, _) = self._parse_args(*args, **kwds)
        (k, loc) = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        (_a, _b) = self._get_support(*args)
        k = asarray(k - loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k <= _b)
        if not isinstance(self, rv_sample):
            cond1 = cond1 & self._nonzero(k, *args)
        cond = cond0 & cond1
        output = zeros(shape(cond), 'd')
        place(output, 1 - cond0 + np.isnan(k), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *(k,) + args)
            place(output, cond, np.clip(self._pmf(*goodargs), 0, 1))
        if output.ndim == 0:
            return output[()]
        return output

    def logpmf(self, k, *args, **kwds):
        if False:
            for i in range(10):
                print('nop')
        'Log of the probability mass function at k of the given RV.\n\n        Parameters\n        ----------\n        k : array_like\n            Quantiles.\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information).\n        loc : array_like, optional\n            Location parameter. Default is 0.\n\n        Returns\n        -------\n        logpmf : array_like\n            Log of the probability mass function evaluated at k.\n\n        '
        (args, loc, _) = self._parse_args(*args, **kwds)
        (k, loc) = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        (_a, _b) = self._get_support(*args)
        k = asarray(k - loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k <= _b)
        if not isinstance(self, rv_sample):
            cond1 = cond1 & self._nonzero(k, *args)
        cond = cond0 & cond1
        output = empty(shape(cond), 'd')
        output.fill(-inf)
        place(output, 1 - cond0 + np.isnan(k), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *(k,) + args)
            place(output, cond, self._logpmf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def cdf(self, k, *args, **kwds):
        if False:
            print('Hello World!')
        'Cumulative distribution function of the given RV.\n\n        Parameters\n        ----------\n        k : array_like, int\n            Quantiles.\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information).\n        loc : array_like, optional\n            Location parameter (default=0).\n\n        Returns\n        -------\n        cdf : ndarray\n            Cumulative distribution function evaluated at `k`.\n\n        '
        (args, loc, _) = self._parse_args(*args, **kwds)
        (k, loc) = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        (_a, _b) = self._get_support(*args)
        k = asarray(k - loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = k >= _b
        cond3 = np.isneginf(k)
        cond = cond0 & cond1 & np.isfinite(k)
        output = zeros(shape(cond), 'd')
        place(output, cond2 * (cond0 == cond0), 1.0)
        place(output, cond3 * (cond0 == cond0), 0.0)
        place(output, 1 - cond0 + np.isnan(k), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *(k,) + args)
            place(output, cond, np.clip(self._cdf(*goodargs), 0, 1))
        if output.ndim == 0:
            return output[()]
        return output

    def logcdf(self, k, *args, **kwds):
        if False:
            while True:
                i = 10
        'Log of the cumulative distribution function at k of the given RV.\n\n        Parameters\n        ----------\n        k : array_like, int\n            Quantiles.\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information).\n        loc : array_like, optional\n            Location parameter (default=0).\n\n        Returns\n        -------\n        logcdf : array_like\n            Log of the cumulative distribution function evaluated at k.\n\n        '
        (args, loc, _) = self._parse_args(*args, **kwds)
        (k, loc) = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        (_a, _b) = self._get_support(*args)
        k = asarray(k - loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = k >= _b
        cond = cond0 & cond1
        output = empty(shape(cond), 'd')
        output.fill(-inf)
        place(output, 1 - cond0 + np.isnan(k), self.badvalue)
        place(output, cond2 * (cond0 == cond0), 0.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *(k,) + args)
            place(output, cond, self._logcdf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def sf(self, k, *args, **kwds):
        if False:
            print('Hello World!')
        'Survival function (1 - `cdf`) at k of the given RV.\n\n        Parameters\n        ----------\n        k : array_like\n            Quantiles.\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information).\n        loc : array_like, optional\n            Location parameter (default=0).\n\n        Returns\n        -------\n        sf : array_like\n            Survival function evaluated at k.\n\n        '
        (args, loc, _) = self._parse_args(*args, **kwds)
        (k, loc) = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        (_a, _b) = self._get_support(*args)
        k = asarray(k - loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = ((k < _a) | np.isneginf(k)) & cond0
        cond = cond0 & cond1 & np.isfinite(k)
        output = zeros(shape(cond), 'd')
        place(output, 1 - cond0 + np.isnan(k), self.badvalue)
        place(output, cond2, 1.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *(k,) + args)
            place(output, cond, np.clip(self._sf(*goodargs), 0, 1))
        if output.ndim == 0:
            return output[()]
        return output

    def logsf(self, k, *args, **kwds):
        if False:
            return 10
        'Log of the survival function of the given RV.\n\n        Returns the log of the "survival function," defined as 1 - `cdf`,\n        evaluated at `k`.\n\n        Parameters\n        ----------\n        k : array_like\n            Quantiles.\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information).\n        loc : array_like, optional\n            Location parameter (default=0).\n\n        Returns\n        -------\n        logsf : ndarray\n            Log of the survival function evaluated at `k`.\n\n        '
        (args, loc, _) = self._parse_args(*args, **kwds)
        (k, loc) = map(asarray, (k, loc))
        args = tuple(map(asarray, args))
        (_a, _b) = self._get_support(*args)
        k = asarray(k - loc)
        cond0 = self._argcheck(*args)
        cond1 = (k >= _a) & (k < _b)
        cond2 = (k < _a) & cond0
        cond = cond0 & cond1
        output = empty(shape(cond), 'd')
        output.fill(-inf)
        place(output, 1 - cond0 + np.isnan(k), self.badvalue)
        place(output, cond2, 0.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *(k,) + args)
            place(output, cond, self._logsf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def ppf(self, q, *args, **kwds):
        if False:
            print('Hello World!')
        'Percent point function (inverse of `cdf`) at q of the given RV.\n\n        Parameters\n        ----------\n        q : array_like\n            Lower tail probability.\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information).\n        loc : array_like, optional\n            Location parameter (default=0).\n\n        Returns\n        -------\n        k : array_like\n            Quantile corresponding to the lower tail probability, q.\n\n        '
        (args, loc, _) = self._parse_args(*args, **kwds)
        (q, loc) = map(asarray, (q, loc))
        args = tuple(map(asarray, args))
        (_a, _b) = self._get_support(*args)
        cond0 = self._argcheck(*args) & (loc == loc)
        cond1 = (q > 0) & (q < 1)
        cond2 = (q == 1) & cond0
        cond = cond0 & cond1
        output = np.full(shape(cond), fill_value=self.badvalue, dtype='d')
        place(output, (q == 0) * (cond == cond), _a - 1 + loc)
        place(output, cond2, _b + loc)
        if np.any(cond):
            goodargs = argsreduce(cond, *(q,) + args + (loc,))
            (loc, goodargs) = (goodargs[-1], goodargs[:-1])
            place(output, cond, self._ppf(*goodargs) + loc)
        if output.ndim == 0:
            return output[()]
        return output

    def isf(self, q, *args, **kwds):
        if False:
            return 10
        'Inverse survival function (inverse of `sf`) at q of the given RV.\n\n        Parameters\n        ----------\n        q : array_like\n            Upper tail probability.\n        arg1, arg2, arg3,... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information).\n        loc : array_like, optional\n            Location parameter (default=0).\n\n        Returns\n        -------\n        k : ndarray or scalar\n            Quantile corresponding to the upper tail probability, q.\n\n        '
        (args, loc, _) = self._parse_args(*args, **kwds)
        (q, loc) = map(asarray, (q, loc))
        args = tuple(map(asarray, args))
        (_a, _b) = self._get_support(*args)
        cond0 = self._argcheck(*args) & (loc == loc)
        cond1 = (q > 0) & (q < 1)
        cond2 = (q == 1) & cond0
        cond3 = (q == 0) & cond0
        cond = cond0 & cond1
        output = np.full(shape(cond), fill_value=self.badvalue, dtype='d')
        lower_bound = _a - 1 + loc
        upper_bound = _b + loc
        place(output, cond2 * (cond == cond), lower_bound)
        place(output, cond3 * (cond == cond), upper_bound)
        if np.any(cond):
            goodargs = argsreduce(cond, *(q,) + args + (loc,))
            (loc, goodargs) = (goodargs[-1], goodargs[:-1])
            place(output, cond, self._isf(*goodargs) + loc)
        if output.ndim == 0:
            return output[()]
        return output

    def _entropy(self, *args):
        if False:
            print('Hello World!')
        if hasattr(self, 'pk'):
            return stats.entropy(self.pk)
        else:
            (_a, _b) = self._get_support(*args)
            return _expect(lambda x: entr(self.pmf(x, *args)), _a, _b, self.ppf(0.5, *args), self.inc)

    def expect(self, func=None, args=(), loc=0, lb=None, ub=None, conditional=False, maxcount=1000, tolerance=1e-10, chunksize=32):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculate expected value of a function with respect to the distribution\n        for discrete distribution by numerical summation.\n\n        Parameters\n        ----------\n        func : callable, optional\n            Function for which the expectation value is calculated.\n            Takes only one argument.\n            The default is the identity mapping f(k) = k.\n        args : tuple, optional\n            Shape parameters of the distribution.\n        loc : float, optional\n            Location parameter.\n            Default is 0.\n        lb, ub : int, optional\n            Lower and upper bound for the summation, default is set to the\n            support of the distribution, inclusive (``lb <= k <= ub``).\n        conditional : bool, optional\n            If true then the expectation is corrected by the conditional\n            probability of the summation interval. The return value is the\n            expectation of the function, `func`, conditional on being in\n            the given interval (k such that ``lb <= k <= ub``).\n            Default is False.\n        maxcount : int, optional\n            Maximal number of terms to evaluate (to avoid an endless loop for\n            an infinite sum). Default is 1000.\n        tolerance : float, optional\n            Absolute tolerance for the summation. Default is 1e-10.\n        chunksize : int, optional\n            Iterate over the support of a distributions in chunks of this size.\n            Default is 32.\n\n        Returns\n        -------\n        expect : float\n            Expected value.\n\n        Notes\n        -----\n        For heavy-tailed distributions, the expected value may or\n        may not exist,\n        depending on the function, `func`. If it does exist, but the\n        sum converges\n        slowly, the accuracy of the result may be rather low. For instance, for\n        ``zipf(4)``, accuracy for mean, variance in example is only 1e-5.\n        increasing `maxcount` and/or `chunksize` may improve the result,\n        but may also make zipf very slow.\n\n        The function is not vectorized.\n\n        '
        if func is None:

            def fun(x):
                if False:
                    while True:
                        i = 10
                return (x + loc) * self._pmf(x, *args)
        else:

            def fun(x):
                if False:
                    while True:
                        i = 10
                return func(x + loc) * self._pmf(x, *args)
        (_a, _b) = self._get_support(*args)
        if lb is None:
            lb = _a
        else:
            lb = lb - loc
        if ub is None:
            ub = _b
        else:
            ub = ub - loc
        if conditional:
            invfac = self.sf(lb - 1, *args) - self.sf(ub, *args)
        else:
            invfac = 1.0
        if isinstance(self, rv_sample):
            res = self._expect(fun, lb, ub)
            return res / invfac
        x0 = self.ppf(0.5, *args)
        res = _expect(fun, lb, ub, x0, self.inc, maxcount, tolerance, chunksize)
        return res / invfac

    def _param_info(self):
        if False:
            print('Hello World!')
        shape_info = self._shape_info()
        loc_info = _ShapeInfo('loc', True, (-np.inf, np.inf), (False, False))
        param_info = shape_info + [loc_info]
        return param_info

def _expect(fun, lb, ub, x0, inc, maxcount=1000, tolerance=1e-10, chunksize=32):
    if False:
        return 10
    'Helper for computing the expectation value of `fun`.'
    if ub - lb <= chunksize:
        supp = np.arange(lb, ub + 1, inc)
        vals = fun(supp)
        return np.sum(vals)
    if x0 < lb:
        x0 = lb
    if x0 > ub:
        x0 = ub
    (count, tot) = (0, 0.0)
    for x in _iter_chunked(x0, ub + 1, chunksize=chunksize, inc=inc):
        count += x.size
        delta = np.sum(fun(x))
        tot += delta
        if abs(delta) < tolerance * x.size:
            break
        if count > maxcount:
            warnings.warn('expect(): sum did not converge', RuntimeWarning)
            return tot
    for x in _iter_chunked(x0 - 1, lb - 1, chunksize=chunksize, inc=-inc):
        count += x.size
        delta = np.sum(fun(x))
        tot += delta
        if abs(delta) < tolerance * x.size:
            break
        if count > maxcount:
            warnings.warn('expect(): sum did not converge', RuntimeWarning)
            break
    return tot

def _iter_chunked(x0, x1, chunksize=4, inc=1):
    if False:
        for i in range(10):
            print('nop')
    'Iterate from x0 to x1 in chunks of chunksize and steps inc.\n\n    x0 must be finite, x1 need not be. In the latter case, the iterator is\n    infinite.\n    Handles both x0 < x1 and x0 > x1. In the latter case, iterates downwards\n    (make sure to set inc < 0.)\n\n    >>> from scipy.stats._distn_infrastructure import _iter_chunked\n    >>> [x for x in _iter_chunked(2, 5, inc=2)]\n    [array([2, 4])]\n    >>> [x for x in _iter_chunked(2, 11, inc=2)]\n    [array([2, 4, 6, 8]), array([10])]\n    >>> [x for x in _iter_chunked(2, -5, inc=-2)]\n    [array([ 2,  0, -2, -4])]\n    >>> [x for x in _iter_chunked(2, -9, inc=-2)]\n    [array([ 2,  0, -2, -4]), array([-6, -8])]\n\n    '
    if inc == 0:
        raise ValueError('Cannot increment by zero.')
    if chunksize <= 0:
        raise ValueError('Chunk size must be positive; got %s.' % chunksize)
    s = 1 if inc > 0 else -1
    stepsize = abs(chunksize * inc)
    x = x0
    while (x - x1) * inc < 0:
        delta = min(stepsize, abs(x - x1))
        step = delta * s
        supp = np.arange(x, x + step, inc)
        x += step
        yield supp

class rv_sample(rv_discrete):
    """A 'sample' discrete distribution defined by the support and values.

    The ctor ignores most of the arguments, only needs the `values` argument.
    """

    def __init__(self, a=0, b=inf, name=None, badvalue=None, moment_tol=1e-08, values=None, inc=1, longname=None, shapes=None, seed=None):
        if False:
            for i in range(10):
                print('nop')
        super(rv_discrete, self).__init__(seed)
        if values is None:
            raise ValueError('rv_sample.__init__(..., values=None,...)')
        self._ctor_param = dict(a=a, b=b, name=name, badvalue=badvalue, moment_tol=moment_tol, values=values, inc=inc, longname=longname, shapes=shapes, seed=seed)
        if badvalue is None:
            badvalue = nan
        self.badvalue = badvalue
        self.moment_tol = moment_tol
        self.inc = inc
        self.shapes = shapes
        self.vecentropy = self._entropy
        (xk, pk) = values
        if np.shape(xk) != np.shape(pk):
            raise ValueError('xk and pk must have the same shape.')
        if np.less(pk, 0.0).any():
            raise ValueError('All elements of pk must be non-negative.')
        if not np.allclose(np.sum(pk), 1):
            raise ValueError('The sum of provided pk is not 1.')
        if not len(set(np.ravel(xk))) == np.size(xk):
            raise ValueError('xk may not contain duplicate values.')
        indx = np.argsort(np.ravel(xk))
        self.xk = np.take(np.ravel(xk), indx, 0)
        self.pk = np.take(np.ravel(pk), indx, 0)
        self.a = self.xk[0]
        self.b = self.xk[-1]
        self.qvals = np.cumsum(self.pk, axis=0)
        self.shapes = ' '
        self._construct_argparser(meths_to_inspect=[self._pmf], locscale_in='loc=0', locscale_out='loc, 1')
        self._attach_methods()
        self._construct_docstrings(name, longname)

    def __getstate__(self):
        if False:
            return 10
        dct = self.__dict__.copy()
        attrs = ['_parse_args', '_parse_args_stats', '_parse_args_rvs']
        [dct.pop(attr, None) for attr in attrs]
        return dct

    def _attach_methods(self):
        if False:
            i = 10
            return i + 15
        'Attaches dynamically created argparser methods.'
        self._attach_argparser_methods()

    def _get_support(self, *args):
        if False:
            i = 10
            return i + 15
        "Return the support of the (unscaled, unshifted) distribution.\n\n        Parameters\n        ----------\n        arg1, arg2, ... : array_like\n            The shape parameter(s) for the distribution (see docstring of the\n            instance object for more information).\n\n        Returns\n        -------\n        a, b : numeric (float, or int or +/-np.inf)\n            end-points of the distribution's support.\n        "
        return (self.a, self.b)

    def _pmf(self, x):
        if False:
            print('Hello World!')
        return np.select([x == k for k in self.xk], [np.broadcast_arrays(p, x)[0] for p in self.pk], 0)

    def _cdf(self, x):
        if False:
            for i in range(10):
                print('nop')
        (xx, xxk) = np.broadcast_arrays(x[:, None], self.xk)
        indx = np.argmax(xxk > xx, axis=-1) - 1
        return self.qvals[indx]

    def _ppf(self, q):
        if False:
            print('Hello World!')
        (qq, sqq) = np.broadcast_arrays(q[..., None], self.qvals)
        indx = argmax(sqq >= qq, axis=-1)
        return self.xk[indx]

    def _rvs(self, size=None, random_state=None):
        if False:
            i = 10
            return i + 15
        U = random_state.uniform(size=size)
        if size is None:
            U = np.array(U, ndmin=1)
            Y = self._ppf(U)[0]
        else:
            Y = self._ppf(U)
        return Y

    def _entropy(self):
        if False:
            i = 10
            return i + 15
        return stats.entropy(self.pk)

    def generic_moment(self, n):
        if False:
            i = 10
            return i + 15
        n = asarray(n)
        return np.sum(self.xk ** n[np.newaxis, ...] * self.pk, axis=0)

    def _expect(self, fun, lb, ub, *args, **kwds):
        if False:
            print('Hello World!')
        supp = self.xk[(lb <= self.xk) & (self.xk <= ub)]
        vals = fun(supp)
        return np.sum(vals)

def _check_shape(argshape, size):
    if False:
        while True:
            i = 10
    '\n    This is a utility function used by `_rvs()` in the class geninvgauss_gen.\n    It compares the tuple argshape to the tuple size.\n\n    Parameters\n    ----------\n    argshape : tuple of integers\n        Shape of the arguments.\n    size : tuple of integers or integer\n        Size argument of rvs().\n\n    Returns\n    -------\n    The function returns two tuples, scalar_shape and bc.\n\n    scalar_shape : tuple\n        Shape to which the 1-d array of random variates returned by\n        _rvs_scalar() is converted when it is copied into the\n        output array of _rvs().\n\n    bc : tuple of booleans\n        bc is an tuple the same length as size. bc[j] is True if the data\n        associated with that index is generated in one call of _rvs_scalar().\n\n    '
    scalar_shape = []
    bc = []
    for (argdim, sizedim) in zip_longest(argshape[::-1], size[::-1], fillvalue=1):
        if sizedim > argdim or argdim == sizedim == 1:
            scalar_shape.append(sizedim)
            bc.append(True)
        else:
            bc.append(False)
    return (tuple(scalar_shape[::-1]), tuple(bc[::-1]))

def get_distribution_names(namespace_pairs, rv_base_class):
    if False:
        print('Hello World!')
    'Collect names of statistical distributions and their generators.\n\n    Parameters\n    ----------\n    namespace_pairs : sequence\n        A snapshot of (name, value) pairs in the namespace of a module.\n    rv_base_class : class\n        The base class of random variable generator classes in a module.\n\n    Returns\n    -------\n    distn_names : list of strings\n        Names of the statistical distributions.\n    distn_gen_names : list of strings\n        Names of the generators of the statistical distributions.\n        Note that these are not simply the names of the statistical\n        distributions, with a _gen suffix added.\n\n    '
    distn_names = []
    distn_gen_names = []
    for (name, value) in namespace_pairs:
        if name.startswith('_'):
            continue
        if name.endswith('_gen') and issubclass(value, rv_base_class):
            distn_gen_names.append(name)
        if isinstance(value, rv_base_class):
            distn_names.append(name)
    return (distn_names, distn_gen_names)