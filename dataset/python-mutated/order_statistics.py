import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back, handle_numpy_out

def _cpercentile(N, percent, key=lambda x: x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Find the percentile   of a list of values.\n\n    @parameter N - is a list of values. Note N MUST BE already sorted.\n    @parameter percent - a float value from 0.0 to 1.0.\n    @parameter key - optional key function to compute value from each element of N.\n\n    @return - the percentile  of the values\n    '
    N.sort()
    k = (len(N) - 1) * percent
    f = ivy.math.floor(k)
    c = ivy.math.ceil(k)
    if f == c:
        return key(N[int(k)])
    d0 = key(N[int(f)]) * (c - k)
    d1 = key(N[int(c)]) * (k - f)
    return d0 + d1

def _quantile_is_valid(q):
    if False:
        i = 10
        return i + 15
    if q.ndim == 1 and q.size < 10:
        for i in range(q.size):
            if not 0.0 <= q[i] <= 1.0:
                return False
    elif not (ivy.all(q >= 0) and ivy.all(q <= 1)):
        return False
    return True

def nanpercentile(a, /, *, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False, interpolation=None):
    if False:
        print('Hello World!')
    a = ivy.array(a)
    q = ivy.divide(q, 100.0)
    q = ivy.array(q)
    if not _quantile_is_valid(q):
        ivy.logging.warning('percentile s must be in the range [0, 100]')
        return []
    if axis is None:
        resultarray = []
        nanlessarray = []
        for x in a:
            for i in x:
                if not ivy.isnan(i):
                    nanlessarray.append(i)
        for i in q:
            resultarray.append(_cpercentile(nanlessarray, i))
        return resultarray
    elif axis == 1:
        resultarray = []
        nanlessarrayofarrays = []
        for i in a:
            nanlessarray = []
            for t in i:
                if not ivy.isnan(t):
                    nanlessarray.append(t)
            nanlessarrayofarrays.append(nanlessarray)
        for i in q:
            arrayofpercentiles = []
            for ii in nanlessarrayofarrays:
                arrayofpercentiles.append(_cpercentile(ii, i))
            resultarray.append(arrayofpercentiles)
        return resultarray
    elif axis == 0:
        resultarray = []
        try:
            a = ivy.swapaxes(a, 0, 1)
        except ivy.utils.exceptions.IvyError:
            ivy.logging.warning("axis is 0 but couldn't swap")
        finally:
            nanlessarrayofarrays = []
            for i in a:
                nanlessarray = []
                for t in i:
                    if not ivy.isnan(t):
                        nanlessarray.append(t)
                nanlessarrayofarrays.append(nanlessarray)
            for i in q:
                arrayofpercentiles = []
                for ii in nanlessarrayofarrays:
                    arrayofpercentiles.append(_cpercentile(ii, i))
                resultarray.append(arrayofpercentiles)
        return resultarray

@to_ivy_arrays_and_back
@handle_numpy_out
def ptp(a, axis=None, out=None, keepdims=False):
    if False:
        while True:
            i = 10
    x = ivy.max(a, axis=axis, keepdims=keepdims)
    y = ivy.min(a, axis=axis, keepdims=keepdims)
    ret = ivy.subtract(x, y)
    return ret.astype(a.dtype, copy=False)