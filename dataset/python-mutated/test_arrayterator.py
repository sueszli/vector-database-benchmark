from operator import mul
from functools import reduce
import numpy as np
from numpy.random import randint
from numpy.lib import Arrayterator
from numpy.testing import assert_

def test():
    if False:
        while True:
            i = 10
    np.random.seed(np.arange(10))
    ndims = randint(5) + 1
    shape = tuple((randint(10) + 1 for dim in range(ndims)))
    els = reduce(mul, shape)
    a = np.arange(els)
    a.shape = shape
    buf_size = randint(2 * els)
    b = Arrayterator(a, buf_size)
    for block in b:
        assert_(len(block.flat) <= (buf_size or els))
    assert_(list(b.flat) == list(a.flat))
    start = [randint(dim) for dim in shape]
    stop = [randint(dim) + 1 for dim in shape]
    step = [randint(dim) + 1 for dim in shape]
    slice_ = tuple((slice(*t) for t in zip(start, stop, step)))
    c = b[slice_]
    d = a[slice_]
    for block in c:
        assert_(len(block.flat) <= (buf_size or els))
    assert_(np.all(c.__array__() == d))
    assert_(list(c.flat) == list(d.flat))