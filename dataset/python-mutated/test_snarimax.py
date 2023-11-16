from __future__ import annotations
import calendar
import math
import random
import pytest
import sympy
from river import compose, datasets, metrics, time_series
from river.time_series.snarimax import Differencer

class Yt(sympy.IndexedBase):
    t = sympy.symbols('t', cls=sympy.Idx)

    def __getitem__(self, idx):
        if False:
            while True:
                i = 10
        return super().__getitem__(self.t - idx)

def test_diff_formula():
    if False:
        while True:
            i = 10
    "\n\n    >>> import sympy\n    >>> from river.time_series.snarimax import Differencer\n\n    >>> Y = Yt('y')\n    >>> Y\n    y\n\n    >>> p = sympy.symbols('p')\n    >>> p\n    p\n\n    >>> D = Differencer\n\n    1\n    >>> D(0).diff(p, Y)\n    p\n\n    (1 - B)\n    >>> D(1).diff(p, Y)\n    p - y[t]\n\n    (1 - B)^2\n    >>> D(2).diff(p, Y)\n    p + y[t - 1] - 2*y[t]\n\n    (1 - B^m)\n    >>> m = sympy.symbols('m', cls=sympy.Idx)\n    >>> D(1, m).diff(p, Y)\n    p - y[-m + t + 1]\n\n    (1 - B)(1 - B^m)\n    >>> (D(1) * D(1, m)).diff(p, Y)\n    p - y[-m + t + 1] + y[-m + t] - y[t]\n\n    (1 - B)(1 - B^12)\n    >>> (D(1) * D(1, 12)).diff(p, Y)\n    p - y[t - 11] + y[t - 12] - y[t]\n\n    "

def test_diff_example():
    if False:
        while True:
            i = 10
    "https://people.duke.edu/~rnau/411sdif.htm\n\n    >>> import pandas as pd\n    >>> from river.time_series.snarimax import Differencer\n\n    >>> sales = pd.DataFrame([\n    ...     {'date': 'Jan-70', 'autosale': 4.79, 'cpi': 0.297},\n    ...     {'date': 'Feb-70', 'autosale': 4.96, 'cpi': 0.298},\n    ...     {'date': 'Mar-70', 'autosale': 5.64, 'cpi': 0.300},\n    ...     {'date': 'Apr-70', 'autosale': 5.98, 'cpi': 0.302},\n    ...     {'date': 'May-70', 'autosale': 6.08, 'cpi': 0.303},\n    ...     {'date': 'Jun-70', 'autosale': 6.55, 'cpi': 0.305},\n    ...     {'date': 'Jul-70', 'autosale': 6.11, 'cpi': 0.306},\n    ...     {'date': 'Aug-70', 'autosale': 5.37, 'cpi': 0.306},\n    ...     {'date': 'Sep-70', 'autosale': 5.17, 'cpi': 0.308},\n    ...     {'date': 'Oct-70', 'autosale': 5.48, 'cpi': 0.309},\n    ...     {'date': 'Nov-70', 'autosale': 4.49, 'cpi': 0.311},\n    ...     {'date': 'Dec-70', 'autosale': 4.65, 'cpi': 0.312},\n    ...     {'date': 'Jan-71', 'autosale': 5.17, 'cpi': 0.312},\n    ...     {'date': 'Feb-71', 'autosale': 5.57, 'cpi': 0.313},\n    ...     {'date': 'Mar-71', 'autosale': 6.92, 'cpi': 0.314},\n    ...     {'date': 'Apr-71', 'autosale': 7.10, 'cpi': 0.315},\n    ...     {'date': 'May-71', 'autosale': 7.02, 'cpi': 0.316},\n    ...     {'date': 'Jun-71', 'autosale': 7.58, 'cpi': 0.319},\n    ...     {'date': 'Jul-71', 'autosale': 6.93, 'cpi': 0.319},\n    ... ])\n\n    >>> sales['autosale/cpi'] = sales.eval('autosale / cpi').round(2)\n    >>> Y = sales['autosale/cpi'].to_list()\n\n    >>> diff = Differencer(1)\n    >>> sales['(1 - B)'] = [\n    ...     diff.diff(p, Y[:i][::-1])\n    ...     if i else ''\n    ...     for i, p in enumerate(Y)\n    ... ]\n\n    >>> sdiff = Differencer(1, 12)\n    >>> sales['(1 - B^12)'] = [\n    ...     sdiff.diff(p, Y[:i][::-1])\n    ...     if i >= 12 else ''\n    ...     for i, p in enumerate(Y)\n    ... ]\n\n    >>> sales['(1 - B)(1 - B^12)'] = [\n    ...     (diff * sdiff).diff(p, Y[:i][::-1])\n    ...     if i >= 13 else ''\n    ...     for i, p in enumerate(Y)\n    ... ]\n\n    >>> sales\n          date  autosale    cpi  autosale/cpi (1 - B) (1 - B^12) (1 - B)(1 - B^12)\n    0   Jan-70      4.79  0.297         16.13\n    1   Feb-70      4.96  0.298         16.64    0.51\n    2   Mar-70      5.64  0.300         18.80    2.16\n    3   Apr-70      5.98  0.302         19.80     1.0\n    4   May-70      6.08  0.303         20.07    0.27\n    5   Jun-70      6.55  0.305         21.48    1.41\n    6   Jul-70      6.11  0.306         19.97   -1.51\n    7   Aug-70      5.37  0.306         17.55   -2.42\n    8   Sep-70      5.17  0.308         16.79   -0.76\n    9   Oct-70      5.48  0.309         17.73    0.94\n    10  Nov-70      4.49  0.311         14.44   -3.29\n    11  Dec-70      4.65  0.312         14.90    0.46\n    12  Jan-71      5.17  0.312         16.57    1.67       0.44\n    13  Feb-71      5.57  0.313         17.80    1.23       1.16              0.72\n    14  Mar-71      6.92  0.314         22.04    4.24       3.24              2.08\n    15  Apr-71      7.10  0.315         22.54     0.5       2.74              -0.5\n    16  May-71      7.02  0.316         22.22   -0.32       2.15             -0.59\n    17  Jun-71      7.58  0.319         23.76    1.54       2.28              0.13\n    18  Jul-71      6.93  0.319         21.72   -2.04       1.75             -0.53\n\n    "

@pytest.mark.parametrize('differencer', [Differencer(1), Differencer(2), Differencer(1, 2), Differencer(2, 2), Differencer(1, 10), Differencer(2, 10), Differencer(1) * Differencer(1), Differencer(2) * Differencer(1), Differencer(1) * Differencer(2), Differencer(1) * Differencer(1, 2), Differencer(2) * Differencer(1, 2), Differencer(1, 2) * Differencer(1, 10), Differencer(1, 2) * Differencer(2, 10), Differencer(2, 2) * Differencer(1, 10), Differencer(2, 2) * Differencer(2, 10)])
def test_undiff(differencer):
    if False:
        for i in range(10):
            print('nop')
    Y = [random.random() for _ in range(max(differencer.coeffs))]
    p = random.random()
    diffed = differencer.diff(p, Y)
    undiffed = differencer.undiff(diffed, Y)
    assert math.isclose(undiffed, p)

@pytest.mark.parametrize('snarimax, Y, errors, expected', [(time_series.SNARIMAX(p=3, d=0, q=3), [1, 2, 3], [-4, -5, -6], {'e-1': -4, 'e-2': -5, 'e-3': -6, 'y-1': 1, 'y-2': 2, 'y-3': 3}), (time_series.SNARIMAX(p=2, d=0, q=3), [1, 2, 3], [-4, -5, -6], {'e-1': -4, 'e-2': -5, 'e-3': -6, 'y-1': 1, 'y-2': 2}), (time_series.SNARIMAX(p=3, d=0, q=2), [1, 2, 3], [-4, -5, -6], {'e-1': -4, 'e-2': -5, 'y-1': 1, 'y-2': 2, 'y-3': 3}), (time_series.SNARIMAX(p=2, d=0, q=2), [1, 2, 3], [-4, -5, -6], {'e-1': -4, 'e-2': -5, 'y-1': 1, 'y-2': 2}), (time_series.SNARIMAX(p=3, d=0, q=3), [1, 2], [-4, -5], {'e-1': -4, 'e-2': -5, 'y-1': 1, 'y-2': 2}), (time_series.SNARIMAX(p=2, d=0, q=2, m=3, sp=2), [i for i in range(12)], [i for i in range(12)], {'e-1': 0, 'e-2': 1, 'sy-3': 2, 'sy-6': 5, 'y-1': 0, 'y-2': 1}), (time_series.SNARIMAX(p=2, d=0, q=2, m=2, sp=2), [i for i in range(12)], [i for i in range(12)], {'e-1': 0, 'e-2': 1, 'sy-2': 1, 'sy-4': 3, 'y-1': 0, 'y-2': 1}), (time_series.SNARIMAX(p=2, d=0, q=2, m=2, sp=3), [i for i in range(12)], [i for i in range(12)], {'e-1': 0, 'e-2': 1, 'sy-2': 1, 'sy-4': 3, 'sy-6': 5, 'y-1': 0, 'y-2': 1}), (time_series.SNARIMAX(p=2, d=0, q=2, m=3, sq=2), [i for i in range(12)], [i for i in range(12)], {'e-1': 0, 'e-2': 1, 'se-3': 2, 'se-6': 5, 'y-1': 0, 'y-2': 1}), (time_series.SNARIMAX(p=2, d=0, q=2, m=3, sq=4), [i for i in range(12)], [i for i in range(12)], {'e-1': 0, 'e-2': 1, 'se-3': 2, 'se-6': 5, 'se-9': 8, 'se-12': 11, 'y-1': 0, 'y-2': 1}), (time_series.SNARIMAX(p=1, d=0, q=1, m=2, sq=4), [i for i in range(12)], [i for i in range(12)], {'e-1': 0, 'se-2': 1, 'se-4': 3, 'se-6': 5, 'se-8': 7, 'y-1': 0})])
def test_add_lag_features(snarimax, Y, errors, expected):
    if False:
        print('Hello World!')
    features = snarimax._add_lag_features(x=None, Y=Y, errors=errors)
    assert features == expected

@pytest.mark.parametrize('snarimax', [time_series.SNARIMAX(p=1, d=1, q=0, m=12, sp=0, sd=1, sq=0), time_series.SNARIMAX(p=0, d=1, q=0, m=12, sp=1, sd=1, sq=0), time_series.SNARIMAX(p=1, d=2, q=0, m=12, sp=0, sd=0, sq=0), time_series.SNARIMAX(p=1, d=0, q=0, m=12, sp=0, sd=2, sq=0)])
def test_no_overflow(snarimax):
    if False:
        for i in range(10):
            print('nop')

    def get_month_distances(x):
        if False:
            i = 10
            return i + 15
        return {calendar.month_name[month]: math.exp(-(x['month'].month - month) ** 2) for month in range(1, 13)}

    def get_ordinal_date(x):
        if False:
            for i in range(10):
                print('nop')
        return {'ordinal_date': x['month'].toordinal()}
    extract_features = compose.TransformerUnion(get_ordinal_date, get_month_distances)
    model = extract_features | snarimax
    time_series.evaluate(dataset=datasets.AirlinePassengers(), model=model, metric=metrics.MAE(), horizon=12)