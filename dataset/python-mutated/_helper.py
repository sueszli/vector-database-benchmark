from typing import Dict, List, Tuple
import math
_next_fast_len_cache: Dict[Tuple[int, List[int]], int] = {}

def _next_fast_len_impl(n, primes):
    if False:
        return 10
    if len(primes) == 0:
        return math.inf
    result = _next_fast_len_cache.get((n, primes), None)
    if result is None:
        if n == 1:
            result = 1
        else:
            p = primes[0]
            result = min(_next_fast_len_impl((n + p - 1) // p, primes) * p, _next_fast_len_impl(n, primes[1:]))
        _next_fast_len_cache[n, primes] = result
    return result

def next_fast_len(target, real=False):
    if False:
        while True:
            i = 10
    "Find the next fast size to ``fft``.\n\n    Args:\n        target (int): The size of input array.\n        real (bool): ``True`` if the FFT involves real input or output.\n            This parameter is of no use, and only for compatibility to\n            SciPy's interface.\n\n    Returns:\n        int: The smallest fast length greater than or equal to the input value.\n\n    .. seealso:: :func:`scipy.fft.next_fast_len`\n\n    .. note::\n        It may return a different value to :func:`scipy.fft.next_fast_len`\n        as pocketfft's prime factors are different from cuFFT's factors.\n        For details, see the `cuFFT documentation`_.\n\n    .. _cuFFT documentation:\n        https://docs.nvidia.com/cuda/cufft/index.html#accuracy-and-performance\n    "
    if target == 0:
        return 0
    primes = (2, 3, 5, 7)
    return _next_fast_len_impl(target, primes)