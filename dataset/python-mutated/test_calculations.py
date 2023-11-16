import numpy as np
from numpy.testing import assert_allclose
from vispy.testing import assert_raises
from vispy.geometry import resize

def test_resize():
    if False:
        return 10
    'Test image resizing algorithms'
    assert_raises(ValueError, resize, np.zeros(3), (3, 3))
    assert_raises(ValueError, resize, np.zeros((3, 3)), (3,))
    assert_raises(ValueError, resize, np.zeros((3, 3)), (4, 4), kind='foo')
    for (kind, tol) in (('nearest', 1e-05), ('linear', 0.2)):
        shape = np.array((10, 11, 3))
        data = np.random.RandomState(0).rand(*shape)
        assert_allclose(data, resize(data, shape[:2], kind=kind), rtol=1e-05, atol=1e-05)
        assert_allclose(data, resize(resize(data, 2 * shape[:2], kind=kind), shape[:2], kind=kind), atol=tol, rtol=tol)