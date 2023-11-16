import numpy as np
import pytest
from neon import NervanaObject, logger

@pytest.mark.unsupported
@pytest.mark.skip(reason='not implemented for backend_cpu')
def test_sr_cpu(backend_cpu):
    if False:
        i = 10
        return i + 15
    '\n    Performs stochastic rounding with 1 bit mantissa for an addition operation\n    and checks that the resulting array is rounded correctly\n    '
    be = NervanaObject.be
    n = 10
    A = be.ones((n, n), dtype=np.float16)
    B = be.ones((n, n), dtype=np.float16)
    be.multiply(B, 0.1, out=B)
    C = be.ones((n, n), dtype=np.float16)
    C.rounding = 1
    C[:] = A + B
    C_host = C.get()
    logger.display('Rounded Buf: {}'.format(C_host))
    assert sum([C_host.flatten()[i] in [1.0, 1.5] for i in range(n ** 2)]) == n ** 2
    assert sum([C_host.flatten()[i] in [1.5] for i in range(n ** 2)]) > 0.1 * n ** 2
    assert sum([C_host.flatten()[i] in [1.0] for i in range(n ** 2)]) > 0.7 * n ** 2

@pytest.mark.unsupported
@pytest.mark.skip(reason='float16 not supported for backend_mkl')
def test_sr_mkl(backend_mkl):
    if False:
        for i in range(10):
            print('nop')
    '\n    Performs stochastic rounding with 1 bit mantissa for an addition operation\n    and checks that the resulting array is rounded correctly\n    '
    be = NervanaObject.be
    n = 10
    A = be.ones((n, n), dtype=np.float16)
    B = be.ones((n, n), dtype=np.float16)
    be.multiply(B, 0.1, out=B)
    C = be.ones((n, n), dtype=np.float16)
    C.rounding = 1
    C[:] = A + B
    C_host = C.get()
    logger.display('Rounded Buf: {}'.format(C_host))
    assert sum([C_host.flatten()[i] in [1.0, 1.5] for i in range(n ** 2)]) == n ** 2
    assert sum([C_host.flatten()[i] in [1.5] for i in range(n ** 2)]) > 0.1 * n ** 2
    assert sum([C_host.flatten()[i] in [1.0] for i in range(n ** 2)]) > 0.7 * n ** 2

@pytest.mark.hasgpu
def test_sr(backend_gpu):
    if False:
        print('Hello World!')
    '\n    Performs stochastic rounding with 1 bit mantissa for an addition operation\n    and checks that the resulting array is rounded correctly\n    '
    be = NervanaObject.be
    n = 10
    A = be.ones((n, n), dtype=np.float16)
    B = be.ones((n, n), dtype=np.float16)
    be.multiply(B, 0.1, out=B)
    C = be.ones((n, n), dtype=np.float16)
    C.rounding = 1
    C[:] = A + B
    C_host = C.get()
    logger.display('Rounded Buf: {}'.format(C_host))
    assert sum([C_host.flatten()[i] in [1.0, 1.5] for i in range(n ** 2)]) == n ** 2
    assert sum([C_host.flatten()[i] in [1.5] for i in range(n ** 2)]) > 0.1 * n ** 2
    assert sum([C_host.flatten()[i] in [1.0] for i in range(n ** 2)]) > 0.7 * n ** 2
if __name__ == '__main__':
    test_sr()