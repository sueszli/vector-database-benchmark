import math
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
import scipy.fft as fftmodule
from skimage._shared.utils import _supported_float_type
from skimage.data import astronaut, coins
from skimage.filters import butterworth
from skimage.filters._fft_based import _get_nd_butterworth_filter

def _fft_centered(x):
    if False:
        i = 10
        return i + 15
    return fftmodule.fftshift(fftmodule.fftn(fftmodule.fftshift(x)))

@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.uint8, np.int32])
@pytest.mark.parametrize('squared_butterworth', [False, True])
def test_butterworth_2D_zeros_dtypes(dtype, squared_butterworth):
    if False:
        print('Hello World!')
    im = np.zeros((4, 4), dtype=dtype)
    filtered = butterworth(im, squared_butterworth=squared_butterworth)
    assert filtered.shape == im.shape
    assert filtered.dtype == _supported_float_type(dtype)
    assert_array_equal(im, filtered)

@pytest.mark.parametrize('squared_butterworth', [False, True])
@pytest.mark.parametrize('high_pass', [False, True])
@pytest.mark.parametrize('order', [6, 10])
@pytest.mark.parametrize('cutoff', [0.2, 0.3])
def test_butterworth_cutoff(cutoff, order, high_pass, squared_butterworth):
    if False:
        i = 10
        return i + 15
    wfilt = _get_nd_butterworth_filter(shape=(512, 512), factor=cutoff, order=order, high_pass=high_pass, real=False, squared_butterworth=squared_butterworth)
    wfilt_profile = np.abs(wfilt[0])
    tol = 0.3 / order
    if high_pass:
        assert abs(wfilt_profile[wfilt_profile.size // 2] - 1.0) < tol
    else:
        assert abs(wfilt_profile[0] - 1.0) < tol
    f_cutoff = int(cutoff * wfilt.shape[0])
    if squared_butterworth:
        assert abs(wfilt_profile[f_cutoff] - 0.5) < tol
    else:
        assert abs(wfilt_profile[f_cutoff] - 1 / math.sqrt(2)) < tol

@pytest.mark.parametrize('cutoff', [-0.01, 0.51])
def test_butterworth_invalid_cutoff(cutoff):
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError):
        butterworth(np.ones((4, 4)), cutoff_frequency_ratio=cutoff)

@pytest.mark.parametrize('high_pass', [True, False])
@pytest.mark.parametrize('squared_butterworth', [False, True])
def test_butterworth_2D(high_pass, squared_butterworth):
    if False:
        print('Hello World!')
    order = 3 if squared_butterworth else 6
    im = np.random.randn(64, 128)
    filtered = butterworth(im, cutoff_frequency_ratio=0.2, order=order, high_pass=high_pass, squared_butterworth=squared_butterworth)
    im_fft = _fft_centered(im)
    im_fft = np.real(im_fft * np.conj(im_fft))
    filtered_fft = _fft_centered(filtered)
    filtered_fft = np.real(filtered_fft * np.conj(filtered_fft))
    outer_mask = np.ones(im.shape, dtype=bool)
    outer_mask[4:-4, 4:-4] = 0
    abs_filt_outer = filtered_fft[outer_mask].mean()
    abs_im_outer = im_fft[outer_mask].mean()
    inner_sl = tuple((slice(s // 2 - 4, s // 2 + 4) for s in im.shape))
    abs_filt_inner = filtered_fft[inner_sl].mean()
    abs_im_inner = im_fft[inner_sl].mean()
    if high_pass:
        assert abs_filt_outer > 0.9 * abs_im_outer
        assert abs_filt_inner < 0.1 * abs_im_inner
    else:
        assert abs_filt_outer < 0.1 * abs_im_outer
        assert abs_filt_inner > 0.9 * abs_im_inner

@pytest.mark.parametrize('high_pass', [True, False])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('squared_butterworth', [False, True])
def test_butterworth_2D_realfft(high_pass, dtype, squared_butterworth):
    if False:
        i = 10
        return i + 15
    'Filtering a real-valued array is equivalent to filtering a\n    complex-valued array where the imaginary part is zero.\n    '
    im = np.random.randn(32, 64).astype(dtype)
    kwargs = dict(cutoff_frequency_ratio=0.2, high_pass=high_pass, squared_butterworth=squared_butterworth)
    expected_dtype = _supported_float_type(im.dtype)
    filtered_real = butterworth(im, **kwargs)
    assert filtered_real.dtype == expected_dtype
    cplx_dtype = np.promote_types(im.dtype, np.complex64)
    filtered_cplx = butterworth(im.astype(cplx_dtype), **kwargs)
    assert filtered_cplx.real.dtype == expected_dtype
    if expected_dtype == np.float64:
        rtol = atol = 1e-13
    else:
        rtol = atol = 1e-05
    assert_allclose(filtered_real, filtered_cplx.real, rtol=rtol, atol=atol)

def test_butterworth_3D_zeros():
    if False:
        i = 10
        return i + 15
    im = np.zeros((3, 4, 5))
    filtered = butterworth(im)
    assert filtered.shape == im.shape
    assert_array_equal(im, filtered)

def test_butterworth_4D_zeros():
    if False:
        while True:
            i = 10
    im = np.zeros((3, 4, 5, 6))
    filtered = butterworth(im)
    assert filtered.shape == im.shape
    assert_array_equal(im, filtered)

@pytest.mark.parametrize('chan, dtype', [(0, np.float64), (1, np.complex128), (2, np.uint8), (3, np.int64)])
def test_butterworth_4D_channel(chan, dtype):
    if False:
        for i in range(10):
            print('nop')
    im = np.zeros((3, 4, 5, 6), dtype=dtype)
    filtered = butterworth(im, channel_axis=chan)
    assert filtered.shape == im.shape

def test_butterworth_correctness_bw():
    if False:
        while True:
            i = 10
    small = coins()[180:190, 260:270]
    filtered = butterworth(small, cutoff_frequency_ratio=0.2)
    correct = np.array([[28.63019362, -17.69023786, 26.95346957, 20.57423019, -15.1933463, -28.05828136, -35.25135674, -25.70376951, -43.37121955, -16.87688457], [4.62077869, 36.5726672, 28.41926375, -22.86436829, -25.32375274, -19.94182623, -2.9666164, 6.62250413, 3.55910886, -33.15358921], [25.00377084, 34.2948942, -15.13862785, -15.34354183, -12.68722526, 12.82729905, 5.21622357, 11.41087761, 16.33690526, -50.39790969], [72.62331496, -14.7924709, -22.14868895, -7.47854864, 9.66784721, 24.37625693, 12.5479457, -1.38194367, 2.40079497, -26.61141413], [21.85962078, -56.73932031, -14.82425429, 4.10524297, -19.16561768, -48.19021687, 5.0258744, 28.82432166, 0.66992097, 9.8378842], [-54.93417679, -5.12037233, 19.2956981, 38.56431593, 27.95408908, -3.53103389, 23.75329532, -6.92615359, -8.50823024, 7.05743093], [-30.51016624, -9.99691211, -7.1080672, 23.67643315, 1.61919726, 12.94103905, -29.08141699, -11.56937511, 22.70988847, 32.04063285], [-7.51780937, -30.27899181, -2.57782655, -1.58947887, -2.13564576, -11.34039302, 1.59165041, 14.39173421, -14.15148821, -2.21664717], [14.81167298, -3.75274782, 18.41459894, 15.80334075, -19.7477109, -3.68619619, -2.9513036, -10.17325791, 18.32438702, 18.68003971], [-50.53430811, 12.14152989, 17.69341877, 9.1858496, 12.1470914, 1.45865179, 61.08961357, 29.76775029, -11.04603619, 24.18621404]])
    assert_allclose(filtered, correct)

def test_butterworth_correctness_rgb():
    if False:
        while True:
            i = 10
    small = astronaut()[135:145, 205:215]
    filtered = butterworth(small, cutoff_frequency_ratio=0.3, high_pass=True, channel_axis=-1)
    correct = np.array([[[-0.530292781, 2.17985072, 2.86622486], [6.3936074, 9.30643715, 8.6122666], [5.47978436, 10.3641402, 10.2940329], [8.88312002, 7.29681652, 8.16021235], [4.67693778, 0.633135145, -2.51296407], [3.21039522, -0.914893931, -2.11543661], [0.0461985125, -6.10682453, -1.7283765], [-4.59492989, -7.35887525, -10.3532871], [-3.67859542, -4.36371621, -5.67371459], [-4.3826408, -6.0836228, -9.20394882]], [[-2.4619139, -2.1099696, -1.41287606], [0.487042304, 0.47068676, 2.90817746], [0.933095004, -0.211867564, 3.10917925], [-2.35660768, -1.35043153, -2.67062162], [-1.22363424, 0.111155488, 1.25392954], [-1.0566768, 0.158195605, 0.611873557], [-4.1212891, -3.55994486, -8.75303054], [2.4717179, 2.70762582, 5.69543552], [0.697042504, -2.24173305, 0.326477871], [0.500195333, -2.66024743, -1.87479563]], [[-4.4013626, -4.02254309, -4.89246563], [-4.64563864, -6.21442755, -9.31399553], [-2.11532959, -2.58844609, -4.20629743], [-3.40862389, -3.29511853, -4.78220207], [-0.806768327, -4.01651211, -2.84783939], [1.72379068, 1.00930709, 2.57575911], [-2.13771052, -1.75564076, -2.88676819], [-0.272015191, -0.161610409, 2.15906305], [-3.80356741, -0.730201675, -3.79800352], [0.143534281, -2.95121861, -2.67070135]], [[1.03480271, 6.34545011, 3.53610283], [7.44740677, 9.97350707, 12.5152734], [3.10493189, 5.15553793, 6.4835494], [-0.789260096, 0.304573015, -1.4382981], [-1.46298411, 1.23095495, -1.33983509], [2.82986807, 2.80546223, 6.39492794], [-0.915293187, 2.88688464, -0.96941748], [4.50217964, 2.90410068, 5.39107589], [-0.571608069, 1.78198962, -0.372062011], [7.43870617, 8.78780364, 9.91142612]], [[-10.1243616, -14.6725955, -16.3691866], [10.6512445, 7.84699418, 10.5428678], [4.75343829, 6.11329861, 2.81633365], [7.78936796, 9.63684277, 12.1495065], [5.19238043, 5.38165743, 8.03025884], [1.67424214, 2.25530135, 0.24416139], [-0.318012002, 1.99405335, -0.433960644], [-1.21700957, -2.659739, -0.631515766], [-4.87805104, -5.55289609, -8.50052504], [14.3493808, 17.7252074, 19.2810954]], [[-42.1712178, -44.7409535, -41.6375264], [1.18585381, 1.33732681, -1.45927283], [-4.83275742, -7.14344851, -7.59812923], [-7.13716513, -11.0025632, -11.6111397], [-5.00275373, -4.20410732, -4.93030043], [1.98421851, 2.68393141, 3.14898078], [1.97471502, -2.11937555, 2.0467415], [3.42916035, 4.98808524, 6.74436447], [-0.3290359, -0.988239773, 0.0238909382], [25.864694, 24.0294324, 26.4535438]], [[-27.2408341, -25.1637965, -29.5566971], [4.83667855, 5.63749968, 6.97366639], [6.18182884, 2.89519333, 6.15697112], [-1.0332654, -4.04702873, -5.50872246], [-6.92401355, -8.99374166, -9.43201766], [-7.24967366, -11.6225741, -12.6385982], [-7.7947022, -9.36143025, -7.13686778], [-12.2393834, -12.4094588, -17.0498337], [-4.6503486, -3.93071471, -4.65788605], [24.7461715, 24.638956, 31.1843915]], [[1.22510987, -3.30423292, -10.2785428], [-24.1285934, -16.0883211, -19.7064909], [9.7724242, 18.0847757, 20.1088714], [12.1335913, 11.6812821, 12.0531919], [-1.64052514, -4.45404672, -3.83103216], [0.997519545, -6.32678881, -0.48984318], [10.7464325, 26.2880708, 17.4691665], [2.31869805, 1.91935135, -0.965612833], [-9.23026361, -17.5138706, -13.9243019], [4.60784836, 5.60845273, 5.28255564]], [[9.68795318, 2.78851276, 5.45620353], [-10.2076906, -14.7926224, -14.3257049], [-2.17025353, 6.23446752, 5.21771748], [15.7074742, 21.7163634, 25.5600809], [7.03884578, 12.9273058, 7.50960315], [-6.69896692, -18.3433042, -16.0702492], [7.44877725, 12.8971365, 11.0234666], [5.25663607, 9.80648891, 12.2955858], [-7.44903684, -19.2670342, -16.8232131], [12.560922, 20.9808909, 22.1425299]], [[2.87825597, 3.37945227, 3.0577736], [1.18858884, -5.27430874, -6.96009863], [-7.55910235, -21.2068126, -20.692579], [-14.7217788, -14.5626702, -15.6493571], [-5.60886203, 0.881908697, 5.47367282], [-10.0478644, -8.01471176, -7.45670458], [3.61521638, 8.99194959, 4.93826323], [7.87025438, 13.4804191, 19.6899695], [-5.50012037, -6.40490471, -11.7265188], [6.17010624, 15.6199152, 17.9889524]]])
    assert_allclose(filtered, correct)