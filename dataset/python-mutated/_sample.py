from cupy import _core
from cupy._creation import basic
from cupy.random import _distributions
from cupy.random import _generator

def rand(*size, **kwarg):
    if False:
        for i in range(10):
            print('nop')
    'Returns an array of uniform random values over the interval ``[0, 1)``.\n\n    Each element of the array is uniformly distributed on the half-open\n    interval ``[0, 1)``. All elements are identically and independently\n    distributed (i.i.d.).\n\n    Args:\n        size (ints): The shape of the array.\n        dtype: Data type specifier. Only :class:`numpy.float32` and\n            :class:`numpy.float64` types are allowed. The default is\n            :class:`numpy.float64`.\n\n    Returns:\n        cupy.ndarray: A random array.\n\n    .. seealso:: :meth:`numpy.random.rand`\n\n    .. admonition:: Example\n\n       .. code-block:: python\n\n          >>> cupy.random.rand(3, 2)\n          array([[0.86476479, 0.05633727],   # random\n                 [0.27283185, 0.38255354],   # random\n                 [0.16592278, 0.75150313]])  # random\n\n          >>> cupy.random.rand(3, 2, dtype=cupy.float32)\n          array([[0.9672306 , 0.9590486 ],                  # random\n                 [0.6851264 , 0.70457625],                  # random\n                 [0.22382522, 0.36055237]], dtype=float32)  # random\n\n    '
    dtype = kwarg.pop('dtype', float)
    if kwarg:
        raise TypeError('rand() got unexpected keyword arguments %s' % ', '.join(kwarg.keys()))
    return random_sample(size=size, dtype=dtype)

def randn(*size, **kwarg):
    if False:
        while True:
            i = 10
    'Returns an array of standard normal random values.\n\n    Each element of the array is normally distributed with zero mean and unit\n    variance. All elements are identically and independently distributed\n    (i.i.d.).\n\n    Args:\n        size (ints): The shape of the array.\n        dtype: Data type specifier. Only :class:`numpy.float32` and\n            :class:`numpy.float64` types are allowed.\n            The default is :class:`numpy.float64`.\n\n    Returns:\n        cupy.ndarray: An array of standard normal random values.\n\n    .. seealso:: :meth:`numpy.random.randn`\n\n    .. admonition:: Example\n\n       .. code-block:: python\n\n          >>> cupy.random.randn(3, 2)\n          array([[0.41193321, 1.59579542],   # random\n                 [0.47904589, 0.18566376],   # random\n                 [0.59748424, 2.32602829]])  # random\n\n          >>> cupy.random.randn(3, 2, dtype=cupy.float32)\n          array([[ 0.1373886 ,  2.403238  ],                  # random\n                 [ 0.84020025,  1.5089266 ],                  # random\n                 [-1.2268474 , -0.48219103]], dtype=float32)  # random\n\n    '
    dtype = kwarg.pop('dtype', float)
    if kwarg:
        raise TypeError('randn() got unexpected keyword arguments %s' % ', '.join(kwarg.keys()))
    return _distributions.normal(size=size, dtype=dtype)

def randint(low, high=None, size=None, dtype='l'):
    if False:
        while True:
            i = 10
    'Returns a scalar or an array of integer values over ``[low, high)``.\n\n    Each element of returned values are independently sampled from\n    uniform distribution over left-close and right-open interval\n    ``[low, high)``.\n\n    Args:\n        low (int): If ``high`` is not ``None``,\n            it is the lower bound of the interval.\n            Otherwise, it is the **upper** bound of the interval\n            and lower bound of the interval is set to ``0``.\n        high (int): Upper bound of the interval.\n        size (None or int or tuple of ints): The shape of returned value.\n        dtype: Data type specifier.\n\n    Returns:\n        int or cupy.ndarray of ints: If size is ``None``,\n        it is single integer sampled.\n        If size is integer, it is the 1D-array of length ``size`` element.\n        Otherwise, it is the array whose shape specified by ``size``.\n    '
    rs = _generator.get_random_state()
    return rs.randint(low, high, size, dtype)

def random_integers(low, high=None, size=None):
    if False:
        i = 10
        return i + 15
    'Return a scalar or an array of integer values over ``[low, high]``\n\n    Each element of returned values are independently sampled from\n    uniform distribution over closed interval ``[low, high]``.\n\n    Args:\n        low (int): If ``high`` is not ``None``,\n            it is the lower bound of the interval.\n            Otherwise, it is the **upper** bound of the interval\n            and the lower bound is set to ``1``.\n        high (int): Upper bound of the interval.\n        size (None or int or tuple of ints): The shape of returned value.\n\n    Returns:\n        int or cupy.ndarray of ints: If size is ``None``,\n        it is single integer sampled.\n        If size is integer, it is the 1D-array of length ``size`` element.\n        Otherwise, it is the array whose shape specified by ``size``.\n    '
    if high is None:
        high = low
        low = 1
    return randint(low, high + 1, size)

def random_sample(size=None, dtype=float):
    if False:
        return 10
    'Returns an array of random values over the interval ``[0, 1)``.\n\n    This is a variant of :func:`cupy.random.rand`.\n\n    Args:\n        size (int or tuple of ints): The shape of the array.\n        dtype: Data type specifier. Only :class:`numpy.float32` and\n            :class:`numpy.float64` types are allowed.\n\n    Returns:\n        cupy.ndarray: An array of uniformly distributed random values.\n\n    .. seealso:: :meth:`numpy.random.random_sample`\n\n    '
    rs = _generator.get_random_state()
    return rs.random_sample(size=size, dtype=dtype)

def choice(a, size=None, replace=True, p=None):
    if False:
        while True:
            i = 10
    'Returns an array of random values from a given 1-D array.\n\n    Each element of the returned array is independently sampled\n    from ``a`` according to ``p`` or uniformly.\n\n    .. note::\n\n       Currently ``p`` is not supported when ``replace=False``.\n\n    Args:\n        a (1-D array-like or int):\n            If an array-like,\n            a random sample is generated from its elements.\n            If an int, the random sample is generated as if ``a`` was\n            ``cupy.arange(n)``\n        size (int or tuple of ints): The shape of the array.\n        replace (boolean): Whether the sample is with or without replacement.\n        p (1-D array-like):\n            The probabilities associated with each entry in ``a``.\n            If not given the sample assumes a uniform distribution over all\n            entries in ``a``.\n\n    Returns:\n        cupy.ndarray: An array of ``a`` values distributed according to\n        ``p`` or uniformly.\n\n    .. seealso:: :meth:`numpy.random.choice`\n\n    '
    rs = _generator.get_random_state()
    return rs.choice(a, size, replace, p)
_multinominal_kernel = _core.ElementwiseKernel('int64 x, int32 p, int32 n', 'raw U ys', 'atomicAdd(&ys[i / n * p + x], U(1))', 'cupy_random_multinomial')

def multinomial(n, pvals, size=None):
    if False:
        while True:
            i = 10
    'Returns an array from multinomial distribution.\n\n    Args:\n        n (int): Number of trials.\n        pvals (cupy.ndarray): Probabilities of each of the ``p`` different\n            outcomes. The sum of these values must be 1.\n        size (int or tuple of ints or None): Shape of a sample in each trial.\n            For example when ``size`` is ``(a, b)``, shape of returned value is\n            ``(a, b, p)`` where ``p`` is ``len(pvals)``.\n            If ``size`` is ``None``, it is treated as ``()``. So, shape of\n            returned value is ``(p,)``.\n\n    Returns:\n        cupy.ndarray: An array drawn from multinomial distribution.\n\n    .. note::\n       It does not support ``sum(pvals) < 1`` case.\n\n    .. seealso:: :meth:`numpy.random.multinomial`\n    '
    if size is None:
        m = 1
        size = ()
    elif isinstance(size, int):
        m = size
        size = (size,)
    else:
        size = tuple(size)
        m = 1
        for x in size:
            m *= x
    p = len(pvals)
    shape = size + (p,)
    ys = basic.zeros(shape, 'l')
    if ys.size > 0:
        xs = choice(p, p=pvals, size=n * m)
        _multinominal_kernel(xs, p, n, ys)
    return ys