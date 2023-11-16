"""Some simple financial calculations

patterned after spreadsheet computations.

There is some complexity in each function
so that the functions behave like ufuncs with
broadcasting and being able to be called with scalars
or arrays (or other sequences).

Functions support the :class:`decimal.Decimal` type unless
otherwise stated.
"""
from __future__ import division, absolute_import, print_function
from decimal import Decimal
import functools
import numpy as np
from numpy.core import overrides
array_function_dispatch = functools.partial(overrides.array_function_dispatch, module='numpy')
__all__ = ['fv', 'pmt', 'nper', 'ipmt', 'ppmt', 'pv', 'rate', 'irr', 'npv', 'mirr']
_when_to_num = {'end': 0, 'begin': 1, 'e': 0, 'b': 1, 0: 0, 1: 1, 'beginning': 1, 'start': 1, 'finish': 0}

def _convert_when(when):
    if False:
        i = 10
        return i + 15
    if isinstance(when, np.ndarray):
        return when
    try:
        return _when_to_num[when]
    except (KeyError, TypeError):
        return [_when_to_num[x] for x in when]

def _fv_dispatcher(rate, nper, pmt, pv, when=None):
    if False:
        i = 10
        return i + 15
    return (rate, nper, pmt, pv)

@array_function_dispatch(_fv_dispatcher)
def fv(rate, nper, pmt, pv, when='end'):
    if False:
        while True:
            i = 10
    "\n    Compute the future value.\n\n    Given:\n     * a present value, `pv`\n     * an interest `rate` compounded once per period, of which\n       there are\n     * `nper` total\n     * a (fixed) payment, `pmt`, paid either\n     * at the beginning (`when` = {'begin', 1}) or the end\n       (`when` = {'end', 0}) of each period\n\n    Return:\n       the value at the end of the `nper` periods\n\n    Parameters\n    ----------\n    rate : scalar or array_like of shape(M, )\n        Rate of interest as decimal (not per cent) per period\n    nper : scalar or array_like of shape(M, )\n        Number of compounding periods\n    pmt : scalar or array_like of shape(M, )\n        Payment\n    pv : scalar or array_like of shape(M, )\n        Present value\n    when : {{'begin', 1}, {'end', 0}}, {string, int}, optional\n        When payments are due ('begin' (1) or 'end' (0)).\n        Defaults to {'end', 0}.\n\n    Returns\n    -------\n    out : ndarray\n        Future values.  If all input is scalar, returns a scalar float.  If\n        any input is array_like, returns future values for each input element.\n        If multiple inputs are array_like, they all must have the same shape.\n\n    Notes\n    -----\n    The future value is computed by solving the equation::\n\n     fv +\n     pv*(1+rate)**nper +\n     pmt*(1 + rate*when)/rate*((1 + rate)**nper - 1) == 0\n\n    or, when ``rate == 0``::\n\n     fv + pv + pmt * nper == 0\n\n    References\n    ----------\n    .. [WRW] Wheeler, D. A., E. Rathke, and R. Weir (Eds.) (2009, May).\n       Open Document Format for Office Applications (OpenDocument)v1.2,\n       Part 2: Recalculated Formula (OpenFormula) Format - Annotated Version,\n       Pre-Draft 12. Organization for the Advancement of Structured Information\n       Standards (OASIS). Billerica, MA, USA. [ODT Document].\n       Available:\n       http://www.oasis-open.org/committees/documents.php?wg_abbrev=office-formula\n       OpenDocument-formula-20090508.odt\n\n    Examples\n    --------\n    What is the future value after 10 years of saving $100 now, with\n    an additional monthly savings of $100.  Assume the interest rate is\n    5% (annually) compounded monthly?\n\n    >>> np.fv(0.05/12, 10*12, -100, -100)\n    15692.928894335748\n\n    By convention, the negative sign represents cash flow out (i.e. money not\n    available today).  Thus, saving $100 a month at 5% annual interest leads\n    to $15,692.93 available to spend in 10 years.\n\n    If any input is array_like, returns an array of equal shape.  Let's\n    compare different interest rates from the example above.\n\n    >>> a = np.array((0.05, 0.06, 0.07))/12\n    >>> np.fv(a, 10*12, -100, -100)\n    array([ 15692.92889434,  16569.87435405,  17509.44688102])\n\n    "
    when = _convert_when(when)
    (rate, nper, pmt, pv, when) = map(np.asarray, [rate, nper, pmt, pv, when])
    temp = (1 + rate) ** nper
    fact = np.where(rate == 0, nper, (1 + rate * when) * (temp - 1) / rate)
    return -(pv * temp + pmt * fact)

def _pmt_dispatcher(rate, nper, pv, fv=None, when=None):
    if False:
        for i in range(10):
            print('nop')
    return (rate, nper, pv, fv)

@array_function_dispatch(_pmt_dispatcher)
def pmt(rate, nper, pv, fv=0, when='end'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Compute the payment against loan principal plus interest.\n\n    Given:\n     * a present value, `pv` (e.g., an amount borrowed)\n     * a future value, `fv` (e.g., 0)\n     * an interest `rate` compounded once per period, of which\n       there are\n     * `nper` total\n     * and (optional) specification of whether payment is made\n       at the beginning (`when` = {'begin', 1}) or the end\n       (`when` = {'end', 0}) of each period\n\n    Return:\n       the (fixed) periodic payment.\n\n    Parameters\n    ----------\n    rate : array_like\n        Rate of interest (per period)\n    nper : array_like\n        Number of compounding periods\n    pv : array_like\n        Present value\n    fv : array_like,  optional\n        Future value (default = 0)\n    when : {{'begin', 1}, {'end', 0}}, {string, int}\n        When payments are due ('begin' (1) or 'end' (0))\n\n    Returns\n    -------\n    out : ndarray\n        Payment against loan plus interest.  If all input is scalar, returns a\n        scalar float.  If any input is array_like, returns payment for each\n        input element. If multiple inputs are array_like, they all must have\n        the same shape.\n\n    Notes\n    -----\n    The payment is computed by solving the equation::\n\n     fv +\n     pv*(1 + rate)**nper +\n     pmt*(1 + rate*when)/rate*((1 + rate)**nper - 1) == 0\n\n    or, when ``rate == 0``::\n\n      fv + pv + pmt * nper == 0\n\n    for ``pmt``.\n\n    Note that computing a monthly mortgage payment is only\n    one use for this function.  For example, pmt returns the\n    periodic deposit one must make to achieve a specified\n    future balance given an initial deposit, a fixed,\n    periodically compounded interest rate, and the total\n    number of periods.\n\n    References\n    ----------\n    .. [WRW] Wheeler, D. A., E. Rathke, and R. Weir (Eds.) (2009, May).\n       Open Document Format for Office Applications (OpenDocument)v1.2,\n       Part 2: Recalculated Formula (OpenFormula) Format - Annotated Version,\n       Pre-Draft 12. Organization for the Advancement of Structured Information\n       Standards (OASIS). Billerica, MA, USA. [ODT Document].\n       Available:\n       http://www.oasis-open.org/committees/documents.php\n       ?wg_abbrev=office-formulaOpenDocument-formula-20090508.odt\n\n    Examples\n    --------\n    What is the monthly payment needed to pay off a $200,000 loan in 15\n    years at an annual interest rate of 7.5%?\n\n    >>> np.pmt(0.075/12, 12*15, 200000)\n    -1854.0247200054619\n\n    In order to pay-off (i.e., have a future-value of 0) the $200,000 obtained\n    today, a monthly payment of $1,854.02 would be required.  Note that this\n    example illustrates usage of `fv` having a default value of 0.\n\n    "
    when = _convert_when(when)
    (rate, nper, pv, fv, when) = map(np.array, [rate, nper, pv, fv, when])
    temp = (1 + rate) ** nper
    mask = rate == 0
    masked_rate = np.where(mask, 1, rate)
    fact = np.where(mask != 0, nper, (1 + masked_rate * when) * (temp - 1) / masked_rate)
    return -(fv + pv * temp) / fact

def _nper_dispatcher(rate, pmt, pv, fv=None, when=None):
    if False:
        for i in range(10):
            print('nop')
    return (rate, pmt, pv, fv)

@array_function_dispatch(_nper_dispatcher)
def nper(rate, pmt, pv, fv=0, when='end'):
    if False:
        i = 10
        return i + 15
    "\n    Compute the number of periodic payments.\n\n    :class:`decimal.Decimal` type is not supported.\n\n    Parameters\n    ----------\n    rate : array_like\n        Rate of interest (per period)\n    pmt : array_like\n        Payment\n    pv : array_like\n        Present value\n    fv : array_like, optional\n        Future value\n    when : {{'begin', 1}, {'end', 0}}, {string, int}, optional\n        When payments are due ('begin' (1) or 'end' (0))\n\n    Notes\n    -----\n    The number of periods ``nper`` is computed by solving the equation::\n\n     fv + pv*(1+rate)**nper + pmt*(1+rate*when)/rate*((1+rate)**nper-1) = 0\n\n    but if ``rate = 0`` then::\n\n     fv + pv + pmt*nper = 0\n\n    Examples\n    --------\n    If you only had $150/month to pay towards the loan, how long would it take\n    to pay-off a loan of $8,000 at 7% annual interest?\n\n    >>> print(round(np.nper(0.07/12, -150, 8000), 5))\n    64.07335\n\n    So, over 64 months would be required to pay off the loan.\n\n    The same analysis could be done with several different interest rates\n    and/or payments and/or total amounts to produce an entire table.\n\n    >>> np.nper(*(np.ogrid[0.07/12: 0.08/12: 0.01/12,\n    ...                    -150   : -99     : 50    ,\n    ...                    8000   : 9001    : 1000]))\n    array([[[  64.07334877,   74.06368256],\n            [ 108.07548412,  127.99022654]],\n           [[  66.12443902,   76.87897353],\n            [ 114.70165583,  137.90124779]]])\n\n    "
    when = _convert_when(when)
    (rate, pmt, pv, fv, when) = map(np.asarray, [rate, pmt, pv, fv, when])
    use_zero_rate = False
    with np.errstate(divide='raise'):
        try:
            z = pmt * (1 + rate * when) / rate
        except FloatingPointError:
            use_zero_rate = True
    if use_zero_rate:
        return (-fv + pv) / pmt
    else:
        A = -(fv + pv) / (pmt + 0)
        B = np.log((-fv + z) / (pv + z)) / np.log(1 + rate)
        return np.where(rate == 0, A, B)

def _ipmt_dispatcher(rate, per, nper, pv, fv=None, when=None):
    if False:
        return 10
    return (rate, per, nper, pv, fv)

@array_function_dispatch(_ipmt_dispatcher)
def ipmt(rate, per, nper, pv, fv=0, when='end'):
    if False:
        print('Hello World!')
    "\n    Compute the interest portion of a payment.\n\n    Parameters\n    ----------\n    rate : scalar or array_like of shape(M, )\n        Rate of interest as decimal (not per cent) per period\n    per : scalar or array_like of shape(M, )\n        Interest paid against the loan changes during the life or the loan.\n        The `per` is the payment period to calculate the interest amount.\n    nper : scalar or array_like of shape(M, )\n        Number of compounding periods\n    pv : scalar or array_like of shape(M, )\n        Present value\n    fv : scalar or array_like of shape(M, ), optional\n        Future value\n    when : {{'begin', 1}, {'end', 0}}, {string, int}, optional\n        When payments are due ('begin' (1) or 'end' (0)).\n        Defaults to {'end', 0}.\n\n    Returns\n    -------\n    out : ndarray\n        Interest portion of payment.  If all input is scalar, returns a scalar\n        float.  If any input is array_like, returns interest payment for each\n        input element. If multiple inputs are array_like, they all must have\n        the same shape.\n\n    See Also\n    --------\n    ppmt, pmt, pv\n\n    Notes\n    -----\n    The total payment is made up of payment against principal plus interest.\n\n    ``pmt = ppmt + ipmt``\n\n    Examples\n    --------\n    What is the amortization schedule for a 1 year loan of $2500 at\n    8.24% interest per year compounded monthly?\n\n    >>> principal = 2500.00\n\n    The 'per' variable represents the periods of the loan.  Remember that\n    financial equations start the period count at 1!\n\n    >>> per = np.arange(1*12) + 1\n    >>> ipmt = np.ipmt(0.0824/12, per, 1*12, principal)\n    >>> ppmt = np.ppmt(0.0824/12, per, 1*12, principal)\n\n    Each element of the sum of the 'ipmt' and 'ppmt' arrays should equal\n    'pmt'.\n\n    >>> pmt = np.pmt(0.0824/12, 1*12, principal)\n    >>> np.allclose(ipmt + ppmt, pmt)\n    True\n\n    >>> fmt = '{0:2d} {1:8.2f} {2:8.2f} {3:8.2f}'\n    >>> for payment in per:\n    ...     index = payment - 1\n    ...     principal = principal + ppmt[index]\n    ...     print(fmt.format(payment, ppmt[index], ipmt[index], principal))\n     1  -200.58   -17.17  2299.42\n     2  -201.96   -15.79  2097.46\n     3  -203.35   -14.40  1894.11\n     4  -204.74   -13.01  1689.37\n     5  -206.15   -11.60  1483.22\n     6  -207.56   -10.18  1275.66\n     7  -208.99    -8.76  1066.67\n     8  -210.42    -7.32   856.25\n     9  -211.87    -5.88   644.38\n    10  -213.32    -4.42   431.05\n    11  -214.79    -2.96   216.26\n    12  -216.26    -1.49    -0.00\n\n    >>> interestpd = np.sum(ipmt)\n    >>> np.round(interestpd, 2)\n    -112.98\n\n    "
    when = _convert_when(when)
    (rate, per, nper, pv, fv, when) = np.broadcast_arrays(rate, per, nper, pv, fv, when)
    total_pmt = pmt(rate, nper, pv, fv, when)
    ipmt = _rbl(rate, per, total_pmt, pv, when) * rate
    try:
        ipmt = np.where(when == 1, ipmt / (1 + rate), ipmt)
        ipmt = np.where(np.logical_and(when == 1, per == 1), 0, ipmt)
    except IndexError:
        pass
    return ipmt

def _rbl(rate, per, pmt, pv, when):
    if False:
        i = 10
        return i + 15
    "\n    This function is here to simply have a different name for the 'fv'\n    function to not interfere with the 'fv' keyword argument within the 'ipmt'\n    function.  It is the 'remaining balance on loan' which might be useful as\n    it's own function, but is easily calculated with the 'fv' function.\n    "
    return fv(rate, per - 1, pmt, pv, when)

def _ppmt_dispatcher(rate, per, nper, pv, fv=None, when=None):
    if False:
        print('Hello World!')
    return (rate, per, nper, pv, fv)

@array_function_dispatch(_ppmt_dispatcher)
def ppmt(rate, per, nper, pv, fv=0, when='end'):
    if False:
        print('Hello World!')
    "\n    Compute the payment against loan principal.\n\n    Parameters\n    ----------\n    rate : array_like\n        Rate of interest (per period)\n    per : array_like, int\n        Amount paid against the loan changes.  The `per` is the period of\n        interest.\n    nper : array_like\n        Number of compounding periods\n    pv : array_like\n        Present value\n    fv : array_like, optional\n        Future value\n    when : {{'begin', 1}, {'end', 0}}, {string, int}\n        When payments are due ('begin' (1) or 'end' (0))\n\n    See Also\n    --------\n    pmt, pv, ipmt\n\n    "
    total = pmt(rate, nper, pv, fv, when)
    return total - ipmt(rate, per, nper, pv, fv, when)

def _pv_dispatcher(rate, nper, pmt, fv=None, when=None):
    if False:
        return 10
    return (rate, nper, nper, pv, fv)

@array_function_dispatch(_pv_dispatcher)
def pv(rate, nper, pmt, fv=0, when='end'):
    if False:
        while True:
            i = 10
    '\n    Compute the present value.\n\n    Given:\n     * a future value, `fv`\n     * an interest `rate` compounded once per period, of which\n       there are\n     * `nper` total\n     * a (fixed) payment, `pmt`, paid either\n     * at the beginning (`when` = {\'begin\', 1}) or the end\n       (`when` = {\'end\', 0}) of each period\n\n    Return:\n       the value now\n\n    Parameters\n    ----------\n    rate : array_like\n        Rate of interest (per period)\n    nper : array_like\n        Number of compounding periods\n    pmt : array_like\n        Payment\n    fv : array_like, optional\n        Future value\n    when : {{\'begin\', 1}, {\'end\', 0}}, {string, int}, optional\n        When payments are due (\'begin\' (1) or \'end\' (0))\n\n    Returns\n    -------\n    out : ndarray, float\n        Present value of a series of payments or investments.\n\n    Notes\n    -----\n    The present value is computed by solving the equation::\n\n     fv +\n     pv*(1 + rate)**nper +\n     pmt*(1 + rate*when)/rate*((1 + rate)**nper - 1) = 0\n\n    or, when ``rate = 0``::\n\n     fv + pv + pmt * nper = 0\n\n    for `pv`, which is then returned.\n\n    References\n    ----------\n    .. [WRW] Wheeler, D. A., E. Rathke, and R. Weir (Eds.) (2009, May).\n       Open Document Format for Office Applications (OpenDocument)v1.2,\n       Part 2: Recalculated Formula (OpenFormula) Format - Annotated Version,\n       Pre-Draft 12. Organization for the Advancement of Structured Information\n       Standards (OASIS). Billerica, MA, USA. [ODT Document].\n       Available:\n       http://www.oasis-open.org/committees/documents.php?wg_abbrev=office-formula\n       OpenDocument-formula-20090508.odt\n\n    Examples\n    --------\n    What is the present value (e.g., the initial investment)\n    of an investment that needs to total $15692.93\n    after 10 years of saving $100 every month?  Assume the\n    interest rate is 5% (annually) compounded monthly.\n\n    >>> np.pv(0.05/12, 10*12, -100, 15692.93)\n    -100.00067131625819\n\n    By convention, the negative sign represents cash flow out\n    (i.e., money not available today).  Thus, to end up with\n    $15,692.93 in 10 years saving $100 a month at 5% annual\n    interest, one\'s initial deposit should also be $100.\n\n    If any input is array_like, ``pv`` returns an array of equal shape.\n    Let\'s compare different interest rates in the example above:\n\n    >>> a = np.array((0.05, 0.04, 0.03))/12\n    >>> np.pv(a, 10*12, -100, 15692.93)\n    array([ -100.00067132,  -649.26771385, -1273.78633713])\n\n    So, to end up with the same $15692.93 under the same $100 per month\n    "savings plan," for annual interest rates of 4% and 3%, one would\n    need initial investments of $649.27 and $1273.79, respectively.\n\n    '
    when = _convert_when(when)
    (rate, nper, pmt, fv, when) = map(np.asarray, [rate, nper, pmt, fv, when])
    temp = (1 + rate) ** nper
    fact = np.where(rate == 0, nper, (1 + rate * when) * (temp - 1) / rate)
    return -(fv + pmt * fact) / temp

def _g_div_gp(r, n, p, x, y, w):
    if False:
        return 10
    t1 = (r + 1) ** n
    t2 = (r + 1) ** (n - 1)
    return (y + t1 * x + p * (t1 - 1) * (r * w + 1) / r) / (n * t2 * x - p * (t1 - 1) * (r * w + 1) / r ** 2 + n * p * t2 * (r * w + 1) / r + p * (t1 - 1) * w / r)

def _rate_dispatcher(nper, pmt, pv, fv, when=None, guess=None, tol=None, maxiter=None):
    if False:
        while True:
            i = 10
    return (nper, pmt, pv, fv)

@array_function_dispatch(_rate_dispatcher)
def rate(nper, pmt, pv, fv, when='end', guess=None, tol=None, maxiter=100):
    if False:
        i = 10
        return i + 15
    "\n    Compute the rate of interest per period.\n\n    Parameters\n    ----------\n    nper : array_like\n        Number of compounding periods\n    pmt : array_like\n        Payment\n    pv : array_like\n        Present value\n    fv : array_like\n        Future value\n    when : {{'begin', 1}, {'end', 0}}, {string, int}, optional\n        When payments are due ('begin' (1) or 'end' (0))\n    guess : Number, optional\n        Starting guess for solving the rate of interest, default 0.1\n    tol : Number, optional\n        Required tolerance for the solution, default 1e-6\n    maxiter : int, optional\n        Maximum iterations in finding the solution\n\n    Notes\n    -----\n    The rate of interest is computed by iteratively solving the\n    (non-linear) equation::\n\n     fv + pv*(1+rate)**nper + pmt*(1+rate*when)/rate * ((1+rate)**nper - 1) = 0\n\n    for ``rate``.\n\n    References\n    ----------\n    Wheeler, D. A., E. Rathke, and R. Weir (Eds.) (2009, May). Open Document\n    Format for Office Applications (OpenDocument)v1.2, Part 2: Recalculated\n    Formula (OpenFormula) Format - Annotated Version, Pre-Draft 12.\n    Organization for the Advancement of Structured Information Standards\n    (OASIS). Billerica, MA, USA. [ODT Document]. Available:\n    http://www.oasis-open.org/committees/documents.php?wg_abbrev=office-formula\n    OpenDocument-formula-20090508.odt\n\n    "
    when = _convert_when(when)
    default_type = Decimal if isinstance(pmt, Decimal) else float
    if guess is None:
        guess = default_type('0.1')
    if tol is None:
        tol = default_type('1e-6')
    (nper, pmt, pv, fv, when) = map(np.asarray, [nper, pmt, pv, fv, when])
    rn = guess
    iterator = 0
    close = False
    while iterator < maxiter and (not close):
        rnp1 = rn - _g_div_gp(rn, nper, pmt, pv, fv, when)
        diff = abs(rnp1 - rn)
        close = np.all(diff < tol)
        iterator += 1
        rn = rnp1
    if not close:
        return np.nan + rn
    else:
        return rn

def _irr_dispatcher(values):
    if False:
        return 10
    return (values,)

@array_function_dispatch(_irr_dispatcher)
def irr(values):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the Internal Rate of Return (IRR).\n\n    This is the "average" periodically compounded rate of return\n    that gives a net present value of 0.0; for a more complete explanation,\n    see Notes below.\n\n    :class:`decimal.Decimal` type is not supported.\n\n    Parameters\n    ----------\n    values : array_like, shape(N,)\n        Input cash flows per time period.  By convention, net "deposits"\n        are negative and net "withdrawals" are positive.  Thus, for\n        example, at least the first element of `values`, which represents\n        the initial investment, will typically be negative.\n\n    Returns\n    -------\n    out : float\n        Internal Rate of Return for periodic input values.\n\n    Notes\n    -----\n    The IRR is perhaps best understood through an example (illustrated\n    using np.irr in the Examples section below).  Suppose one invests 100\n    units and then makes the following withdrawals at regular (fixed)\n    intervals: 39, 59, 55, 20.  Assuming the ending value is 0, one\'s 100\n    unit investment yields 173 units; however, due to the combination of\n    compounding and the periodic withdrawals, the "average" rate of return\n    is neither simply 0.73/4 nor (1.73)^0.25-1.  Rather, it is the solution\n    (for :math:`r`) of the equation:\n\n    .. math:: -100 + \\frac{39}{1+r} + \\frac{59}{(1+r)^2}\n     + \\frac{55}{(1+r)^3} + \\frac{20}{(1+r)^4} = 0\n\n    In general, for `values` :math:`= [v_0, v_1, ... v_M]`,\n    irr is the solution of the equation: [G]_\n\n    .. math:: \\sum_{t=0}^M{\\frac{v_t}{(1+irr)^{t}}} = 0\n\n    References\n    ----------\n    .. [G] L. J. Gitman, "Principles of Managerial Finance, Brief," 3rd ed.,\n       Addison-Wesley, 2003, pg. 348.\n\n    Examples\n    --------\n    >>> round(irr([-100, 39, 59, 55, 20]), 5)\n    0.28095\n    >>> round(irr([-100, 0, 0, 74]), 5)\n    -0.0955\n    >>> round(irr([-100, 100, 0, -7]), 5)\n    -0.0833\n    >>> round(irr([-100, 100, 0, 7]), 5)\n    0.06206\n    >>> round(irr([-5, 10.5, 1, -8, 1]), 5)\n    0.0886\n\n    (Compare with the Example given for numpy.lib.financial.npv)\n\n    '
    res = np.roots(values[::-1])
    mask = (res.imag == 0) & (res.real > 0)
    if not mask.any():
        return np.nan
    res = res[mask].real
    rate = 1 / res - 1
    rate = rate.item(np.argmin(np.abs(rate)))
    return rate

def _npv_dispatcher(rate, values):
    if False:
        return 10
    return (values,)

@array_function_dispatch(_npv_dispatcher)
def npv(rate, values):
    if False:
        while True:
            i = 10
    '\n    Returns the NPV (Net Present Value) of a cash flow series.\n\n    Parameters\n    ----------\n    rate : scalar\n        The discount rate.\n    values : array_like, shape(M, )\n        The values of the time series of cash flows.  The (fixed) time\n        interval between cash flow "events" must be the same as that for\n        which `rate` is given (i.e., if `rate` is per year, then precisely\n        a year is understood to elapse between each cash flow event).  By\n        convention, investments or "deposits" are negative, income or\n        "withdrawals" are positive; `values` must begin with the initial\n        investment, thus `values[0]` will typically be negative.\n\n    Returns\n    -------\n    out : float\n        The NPV of the input cash flow series `values` at the discount\n        `rate`.\n\n    Notes\n    -----\n    Returns the result of: [G]_\n\n    .. math :: \\sum_{t=0}^{M-1}{\\frac{values_t}{(1+rate)^{t}}}\n\n    References\n    ----------\n    .. [G] L. J. Gitman, "Principles of Managerial Finance, Brief," 3rd ed.,\n       Addison-Wesley, 2003, pg. 346.\n\n    Examples\n    --------\n    >>> np.npv(0.281,[-100, 39, 59, 55, 20])\n    -0.0084785916384548798\n\n    (Compare with the Example given for numpy.lib.financial.irr)\n\n    '
    values = np.asarray(values)
    return (values / (1 + rate) ** np.arange(0, len(values))).sum(axis=0)

def _mirr_dispatcher(values, finance_rate, reinvest_rate):
    if False:
        return 10
    return (values,)

@array_function_dispatch(_mirr_dispatcher)
def mirr(values, finance_rate, reinvest_rate):
    if False:
        print('Hello World!')
    '\n    Modified internal rate of return.\n\n    Parameters\n    ----------\n    values : array_like\n        Cash flows (must contain at least one positive and one negative\n        value) or nan is returned.  The first value is considered a sunk\n        cost at time zero.\n    finance_rate : scalar\n        Interest rate paid on the cash flows\n    reinvest_rate : scalar\n        Interest rate received on the cash flows upon reinvestment\n\n    Returns\n    -------\n    out : float\n        Modified internal rate of return\n\n    '
    values = np.asarray(values)
    n = values.size
    if isinstance(finance_rate, Decimal):
        n = Decimal(n)
    pos = values > 0
    neg = values < 0
    if not (pos.any() and neg.any()):
        return np.nan
    numer = np.abs(npv(reinvest_rate, values * pos))
    denom = np.abs(npv(finance_rate, values * neg))
    return (numer / denom) ** (1 / (n - 1)) * (1 + reinvest_rate) - 1