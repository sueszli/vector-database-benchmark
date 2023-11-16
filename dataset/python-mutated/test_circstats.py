import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from astropy import units as u
from astropy.stats.circstats import _length, circcorrcoef, circmean, circmoment, circvar, rayleightest, vonmisesmle, vtest
from astropy.utils.compat.optional_deps import HAS_SCIPY

def test__length():
    if False:
        while True:
            i = 10
    weights = np.array([12, 1, 6, 1, 2, 1, 1])
    answer = 0.766282
    data = np.array([0, 3.6, 36, 72, 108, 169.2, 324]) * u.deg
    assert_allclose(answer, _length(data, weights=weights), atol=0.0001)

def test_circmean():
    if False:
        return 10
    data = np.array([51, 67, 40, 109, 31, 358]) * u.deg
    answer = 48.63 * u.deg
    assert_equal(answer, np.around(circmean(data), 2))

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
def test_circmean_against_scipy():
    if False:
        i = 10
        return i + 15
    import scipy.stats
    data = np.array([0.89011792, 1.1693706, 0.6981317, 1.90240888, 0.54105207, 6.24827872])
    answer = scipy.stats.circmean(data)
    assert_equal(np.around(answer, 2), np.around(circmean(data), 2))

def test_circvar():
    if False:
        print('Hello World!')
    data = np.array([51, 67, 40, 109, 31, 358]) * u.deg
    answer = 0.1635635
    assert_allclose(answer, circvar(data), atol=0.0001)

def test_circmoment():
    if False:
        while True:
            i = 10
    data = np.array([51, 67, 40, 109, 31, 358]) * u.deg
    answer = np.array([1.588121, 1.963919, 2.685556])
    answer = np.around(np.rad2deg(answer) * u.deg, 4)
    result = (np.around(circmoment(data, p=2)[0], 4), np.around(circmoment(data, p=3)[0], 4), np.around(circmoment(data, p=4)[0], 4))
    assert_equal(answer[0], result[0])
    assert_equal(answer[1], result[1])
    assert_equal(answer[2], result[2])
    answer = np.array([0.4800428, 0.236541, 0.2255761])
    assert_allclose(answer, (circmoment(data, p=2)[1], circmoment(data, p=3)[1], circmoment(data, p=4)[1]), atol=0.0001)

def test_circcorrcoef():
    if False:
        for i in range(10):
            print('nop')
    alpha = np.array([356, 97, 211, 232, 343, 292, 157, 302, 335, 302, 324, 85, 324, 340, 157, 238, 254, 146, 232, 122, 329]) * u.deg
    beta = np.array([119, 162, 221, 259, 270, 29, 97, 292, 40, 313, 94, 45, 47, 108, 221, 270, 119, 248, 270, 45, 23]) * u.deg
    answer = 0.2704648
    assert_allclose(answer, circcorrcoef(alpha, beta), atol=0.0001)

def test_rayleightest():
    if False:
        print('Hello World!')
    data = np.array([190.18, 175.48, 155.95, 217.83, 156.36]) * u.deg
    answer = (0.00640418, 0.9202565)
    result = (rayleightest(data), _length(data))
    assert_allclose(answer[0], result[0], atol=0.0001)
    assert_allclose(answer[1], result[1], atol=0.0001)

@pytest.mark.skipif(not HAS_SCIPY, reason='requires scipy')
def test_vtest():
    if False:
        while True:
            i = 10
    data = np.array([190.18, 175.48, 155.95, 217.83, 156.36]) * u.deg
    answer = 0.9994725
    assert_allclose(answer, vtest(data), atol=1e-05)

def test_vonmisesmle():
    if False:
        return 10
    data = np.array([3.3699057, 4.041163, 0.5014477, 2.6223103, 3.7336524, 1.8136389, 4.1566039, 2.7806317, 2.4672173, 2.8493644])
    answer = (3.006514, 1.474132)
    assert_allclose(answer[0], vonmisesmle(data)[0], atol=1e-05)
    assert_allclose(answer[1], vonmisesmle(data)[1], atol=1e-05)
    data = np.rad2deg(data) * u.deg
    answer = np.rad2deg(3.006514) * u.deg
    assert_equal(np.around(answer, 3), np.around(vonmisesmle(data)[0], 3))
    data = np.array([np.pi / 2, np.pi, np.pi / 2])
    answer = vonmisesmle(data)
    data_to_weigh = np.array([np.pi / 2, np.pi])
    weights = [2, 1]
    assert_allclose(answer[0], vonmisesmle(data_to_weigh, weights=weights)[0], atol=1e-05)
    assert_allclose(answer[1], vonmisesmle(data_to_weigh, weights=weights)[1], atol=1e-05)
    data = np.array([[3.3699057, 4.041163, 0.5014477, 2.6223103, 3.7336524, 1.8136389, 4.1566039, 2.7806317, 2.4672173, 2.8493644], [3.3699057, 4.041163, 0.5014477, 2.6223103, 3.7336524, 1.8136389, 4.1566039, 2.7806317, 2.4672173, 2.8493644]])
    answer = (np.array([3.006514, 3.006514]), np.array([1.474132, 1.474132]))
    assert_allclose(answer[0], vonmisesmle(data, axis=1)[0], atol=1e-05)
    assert_allclose(answer[1], vonmisesmle(data, axis=1)[1], atol=1e-05)
    data = np.rad2deg(data) * u.deg
    answer = (np.rad2deg(answer[0]) * u.deg, answer[1])
    assert_allclose(answer[0], vonmisesmle(data, axis=1)[0], atol=1e-05)
    assert_allclose(answer[1], vonmisesmle(data, axis=1)[1], atol=1e-05)