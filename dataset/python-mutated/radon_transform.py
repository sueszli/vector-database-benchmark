import numpy as np
from scipy.interpolate import interp1d
from scipy.constants import golden_ratio
from scipy.fft import fft, ifft, fftfreq, fftshift
from ._warps import warp
from ._radon_transform import sart_projection_update
from .._shared.utils import convert_to_float
from warnings import warn
from functools import partial
__all__ = ['radon', 'order_angles_golden_ratio', 'iradon', 'iradon_sart']

def radon(image, theta=None, circle=True, *, preserve_range=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculates the radon transform of an image given specified\n    projection angles.\n\n    Parameters\n    ----------\n    image : ndarray\n        Input image. The rotation axis will be located in the pixel with\n        indices ``(image.shape[0] // 2, image.shape[1] // 2)``.\n    theta : array, optional\n        Projection angles (in degrees). If `None`, the value is set to\n        np.arange(180).\n    circle : boolean, optional\n        Assume image is zero outside the inscribed circle, making the\n        width of each projection (the first dimension of the sinogram)\n        equal to ``min(image.shape)``.\n    preserve_range : bool, optional\n        Whether to keep the original range of values. Otherwise, the input\n        image is converted according to the conventions of `img_as_float`.\n        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html\n\n    Returns\n    -------\n    radon_image : ndarray\n        Radon transform (sinogram).  The tomography rotation axis will lie\n        at the pixel index ``radon_image.shape[0] // 2`` along the 0th\n        dimension of ``radon_image``.\n\n    References\n    ----------\n    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic\n           Imaging", IEEE Press 1988.\n    .. [2] B.R. Ramesh, N. Srinivasa, K. Rajgopal, "An Algorithm for Computing\n           the Discrete Radon Transform With Some Applications", Proceedings of\n           the Fourth IEEE Region 10 International Conference, TENCON \'89, 1989\n\n    Notes\n    -----\n    Based on code of Justin K. Romberg\n    (https://www.clear.rice.edu/elec431/projects96/DSP/bpanalysis.html)\n\n    '
    if image.ndim != 2:
        raise ValueError('The input image must be 2-D')
    if theta is None:
        theta = np.arange(180)
    image = convert_to_float(image, preserve_range)
    if circle:
        shape_min = min(image.shape)
        radius = shape_min // 2
        img_shape = np.array(image.shape)
        coords = np.array(np.ogrid[:image.shape[0], :image.shape[1]], dtype=object)
        dist = ((coords - img_shape // 2) ** 2).sum(0)
        outside_reconstruction_circle = dist > radius ** 2
        if np.any(image[outside_reconstruction_circle]):
            warn('Radon transform: image must be zero outside the reconstruction circle')
        slices = tuple((slice(int(np.ceil(excess / 2)), int(np.ceil(excess / 2) + shape_min)) if excess > 0 else slice(None) for excess in img_shape - shape_min))
        padded_image = image[slices]
    else:
        diagonal = np.sqrt(2) * max(image.shape)
        pad = [int(np.ceil(diagonal - s)) for s in image.shape]
        new_center = [(s + p) // 2 for (s, p) in zip(image.shape, pad)]
        old_center = [s // 2 for s in image.shape]
        pad_before = [nc - oc for (oc, nc) in zip(old_center, new_center)]
        pad_width = [(pb, p - pb) for (pb, p) in zip(pad_before, pad)]
        padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
    if padded_image.shape[0] != padded_image.shape[1]:
        raise ValueError('padded_image must be a square')
    center = padded_image.shape[0] // 2
    radon_image = np.zeros((padded_image.shape[0], len(theta)), dtype=image.dtype)
    for (i, angle) in enumerate(np.deg2rad(theta)):
        (cos_a, sin_a) = (np.cos(angle), np.sin(angle))
        R = np.array([[cos_a, sin_a, -center * (cos_a + sin_a - 1)], [-sin_a, cos_a, -center * (cos_a - sin_a - 1)], [0, 0, 1]])
        rotated = warp(padded_image, R, clip=False)
        radon_image[:, i] = rotated.sum(0)
    return radon_image

def _sinogram_circle_to_square(sinogram):
    if False:
        return 10
    diagonal = int(np.ceil(np.sqrt(2) * sinogram.shape[0]))
    pad = diagonal - sinogram.shape[0]
    old_center = sinogram.shape[0] // 2
    new_center = diagonal // 2
    pad_before = new_center - old_center
    pad_width = ((pad_before, pad - pad_before), (0, 0))
    return np.pad(sinogram, pad_width, mode='constant', constant_values=0)

def _get_fourier_filter(size, filter_name):
    if False:
        i = 10
        return i + 15
    'Construct the Fourier filter.\n\n    This computation lessens artifacts and removes a small bias as\n    explained in [1], Chap 3. Equation 61.\n\n    Parameters\n    ----------\n    size : int\n        filter size. Must be even.\n    filter_name : str\n        Filter used in frequency domain filtering. Filters available:\n        ramp, shepp-logan, cosine, hamming, hann. Assign None to use\n        no filter.\n\n    Returns\n    -------\n    fourier_filter: ndarray\n        The computed Fourier filter.\n\n    References\n    ----------\n    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic\n           Imaging", IEEE Press 1988.\n\n    '
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int), np.arange(size / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    fourier_filter = 2 * np.real(fft(f))
    if filter_name == 'ramp':
        pass
    elif filter_name == 'shepp-logan':
        omega = np.pi * fftfreq(size)[1:]
        fourier_filter[1:] *= np.sin(omega) / omega
    elif filter_name == 'cosine':
        freq = np.linspace(0, np.pi, size, endpoint=False)
        cosine_filter = fftshift(np.sin(freq))
        fourier_filter *= cosine_filter
    elif filter_name == 'hamming':
        fourier_filter *= fftshift(np.hamming(size))
    elif filter_name == 'hann':
        fourier_filter *= fftshift(np.hanning(size))
    elif filter_name is None:
        fourier_filter[:] = 1
    return fourier_filter[:, np.newaxis]

def iradon(radon_image, theta=None, output_size=None, filter_name='ramp', interpolation='linear', circle=True, preserve_range=True):
    if False:
        i = 10
        return i + 15
    'Inverse radon transform.\n\n    Reconstruct an image from the radon transform, using the filtered\n    back projection algorithm.\n\n    Parameters\n    ----------\n    radon_image : ndarray\n        Image containing radon transform (sinogram). Each column of\n        the image corresponds to a projection along a different\n        angle. The tomography rotation axis should lie at the pixel\n        index ``radon_image.shape[0] // 2`` along the 0th dimension of\n        ``radon_image``.\n    theta : array, optional\n        Reconstruction angles (in degrees). Default: m angles evenly spaced\n        between 0 and 180 (if the shape of `radon_image` is (N, M)).\n    output_size : int, optional\n        Number of rows and columns in the reconstruction.\n    filter_name : str, optional\n        Filter used in frequency domain filtering. Ramp filter used by default.\n        Filters available: ramp, shepp-logan, cosine, hamming, hann.\n        Assign None to use no filter.\n    interpolation : str, optional\n        Interpolation method used in reconstruction. Methods available:\n        \'linear\', \'nearest\', and \'cubic\' (\'cubic\' is slow).\n    circle : boolean, optional\n        Assume the reconstructed image is zero outside the inscribed circle.\n        Also changes the default output_size to match the behaviour of\n        ``radon`` called with ``circle=True``.\n    preserve_range : bool, optional\n        Whether to keep the original range of values. Otherwise, the input\n        image is converted according to the conventions of `img_as_float`.\n        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html\n\n    Returns\n    -------\n    reconstructed : ndarray\n        Reconstructed image. The rotation axis will be located in the pixel\n        with indices\n        ``(reconstructed.shape[0] // 2, reconstructed.shape[1] // 2)``.\n\n    .. versionchanged:: 0.19\n        In ``iradon``, ``filter`` argument is deprecated in favor of\n        ``filter_name``.\n\n    References\n    ----------\n    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic\n           Imaging", IEEE Press 1988.\n    .. [2] B.R. Ramesh, N. Srinivasa, K. Rajgopal, "An Algorithm for Computing\n           the Discrete Radon Transform With Some Applications", Proceedings of\n           the Fourth IEEE Region 10 International Conference, TENCON \'89, 1989\n\n    Notes\n    -----\n    It applies the Fourier slice theorem to reconstruct an image by\n    multiplying the frequency domain of the filter with the FFT of the\n    projection data. This algorithm is called filtered back projection.\n\n    '
    if radon_image.ndim != 2:
        raise ValueError('The input image must be 2-D')
    if theta is None:
        theta = np.linspace(0, 180, radon_image.shape[1], endpoint=False)
    angles_count = len(theta)
    if angles_count != radon_image.shape[1]:
        raise ValueError('The given ``theta`` does not match the number of projections in ``radon_image``.')
    interpolation_types = ('linear', 'nearest', 'cubic')
    if interpolation not in interpolation_types:
        raise ValueError(f'Unknown interpolation: {interpolation}')
    filter_types = ('ramp', 'shepp-logan', 'cosine', 'hamming', 'hann', None)
    if filter_name not in filter_types:
        raise ValueError(f'Unknown filter: {filter_name}')
    radon_image = convert_to_float(radon_image, preserve_range)
    dtype = radon_image.dtype
    img_shape = radon_image.shape[0]
    if output_size is None:
        if circle:
            output_size = img_shape
        else:
            output_size = int(np.floor(np.sqrt(img_shape ** 2 / 2.0)))
    if circle:
        radon_image = _sinogram_circle_to_square(radon_image)
        img_shape = radon_image.shape[0]
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * img_shape))))
    pad_width = ((0, projection_size_padded - img_shape), (0, 0))
    img = np.pad(radon_image, pad_width, mode='constant', constant_values=0)
    fourier_filter = _get_fourier_filter(projection_size_padded, filter_name)
    projection = fft(img, axis=0) * fourier_filter
    radon_filtered = np.real(ifft(projection, axis=0)[:img_shape, :])
    reconstructed = np.zeros((output_size, output_size), dtype=dtype)
    radius = output_size // 2
    (xpr, ypr) = np.mgrid[:output_size, :output_size] - radius
    x = np.arange(img_shape) - img_shape // 2
    for (col, angle) in zip(radon_filtered.T, np.deg2rad(theta)):
        t = ypr * np.cos(angle) - xpr * np.sin(angle)
        if interpolation == 'linear':
            interpolant = partial(np.interp, xp=x, fp=col, left=0, right=0)
        else:
            interpolant = interp1d(x, col, kind=interpolation, bounds_error=False, fill_value=0)
        reconstructed += interpolant(t)
    if circle:
        out_reconstruction_circle = xpr ** 2 + ypr ** 2 > radius ** 2
        reconstructed[out_reconstruction_circle] = 0.0
    return reconstructed * np.pi / (2 * angles_count)

def order_angles_golden_ratio(theta):
    if False:
        i = 10
        return i + 15
    'Order angles to reduce the amount of correlated information in\n    subsequent projections.\n\n    Parameters\n    ----------\n    theta : array of floats, shape (M,)\n        Projection angles in degrees. Duplicate angles are not allowed.\n\n    Returns\n    -------\n    indices_generator : generator yielding unsigned integers\n        The returned generator yields indices into ``theta`` such that\n        ``theta[indices]`` gives the approximate golden ratio ordering\n        of the projections. In total, ``len(theta)`` indices are yielded.\n        All non-negative integers < ``len(theta)`` are yielded exactly once.\n\n    Notes\n    -----\n    The method used here is that of the golden ratio introduced\n    by T. Kohler.\n\n    References\n    ----------\n    .. [1] Kohler, T. "A projection access scheme for iterative\n           reconstruction based on the golden section." Nuclear Science\n           Symposium Conference Record, 2004 IEEE. Vol. 6. IEEE, 2004.\n    .. [2] Winkelmann, Stefanie, et al. "An optimal radial profile order\n           based on the Golden Ratio for time-resolved MRI."\n           Medical Imaging, IEEE Transactions on 26.1 (2007): 68-76.\n\n    '
    interval = 180
    remaining_indices = list(np.argsort(theta))
    angle = theta[remaining_indices[0]]
    yield remaining_indices.pop(0)
    angle_increment = interval / golden_ratio ** 2
    while remaining_indices:
        remaining_angles = theta[remaining_indices]
        angle = (angle + angle_increment) % interval
        index_above = np.searchsorted(remaining_angles, angle)
        index_below = index_above - 1
        index_above %= len(remaining_indices)
        diff_below = abs(angle - remaining_angles[index_below])
        distance_below = min(diff_below % interval, diff_below % -interval)
        diff_above = abs(angle - remaining_angles[index_above])
        distance_above = min(diff_above % interval, diff_above % -interval)
        if distance_below < distance_above:
            yield remaining_indices.pop(index_below)
        else:
            yield remaining_indices.pop(index_above)

def iradon_sart(radon_image, theta=None, image=None, projection_shifts=None, clip=None, relaxation=0.15, dtype=None):
    if False:
        while True:
            i = 10
    'Inverse radon transform.\n\n    Reconstruct an image from the radon transform, using a single iteration of\n    the Simultaneous Algebraic Reconstruction Technique (SART) algorithm.\n\n    Parameters\n    ----------\n    radon_image : ndarray, shape (M, N)\n        Image containing radon transform (sinogram). Each column of\n        the image corresponds to a projection along a different angle. The\n        tomography rotation axis should lie at the pixel index\n        ``radon_image.shape[0] // 2`` along the 0th dimension of\n        ``radon_image``.\n    theta : array, shape (N,), optional\n        Reconstruction angles (in degrees). Default: m angles evenly spaced\n        between 0 and 180 (if the shape of `radon_image` is (N, M)).\n    image : ndarray, shape (M, M), optional\n        Image containing an initial reconstruction estimate. Default is an array of zeros.\n    projection_shifts : array, shape (N,), optional\n        Shift the projections contained in ``radon_image`` (the sinogram) by\n        this many pixels before reconstructing the image. The i\'th value\n        defines the shift of the i\'th column of ``radon_image``.\n    clip : length-2 sequence of floats, optional\n        Force all values in the reconstructed tomogram to lie in the range\n        ``[clip[0], clip[1]]``\n    relaxation : float, optional\n        Relaxation parameter for the update step. A higher value can\n        improve the convergence rate, but one runs the risk of instabilities.\n        Values close to or higher than 1 are not recommended.\n    dtype : dtype, optional\n        Output data type, must be floating point. By default, if input\n        data type is not float, input is cast to double, otherwise\n        dtype is set to input data type.\n\n    Returns\n    -------\n    reconstructed : ndarray\n        Reconstructed image. The rotation axis will be located in the pixel\n        with indices\n        ``(reconstructed.shape[0] // 2, reconstructed.shape[1] // 2)``.\n\n    Notes\n    -----\n    Algebraic Reconstruction Techniques are based on formulating the tomography\n    reconstruction problem as a set of linear equations. Along each ray,\n    the projected value is the sum of all the values of the cross section along\n    the ray. A typical feature of SART (and a few other variants of algebraic\n    techniques) is that it samples the cross section at equidistant points\n    along the ray, using linear interpolation between the pixel values of the\n    cross section. The resulting set of linear equations are then solved using\n    a slightly modified Kaczmarz method.\n\n    When using SART, a single iteration is usually sufficient to obtain a good\n    reconstruction. Further iterations will tend to enhance high-frequency\n    information, but will also often increase the noise.\n\n    References\n    ----------\n    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic\n           Imaging", IEEE Press 1988.\n    .. [2] AH Andersen, AC Kak, "Simultaneous algebraic reconstruction\n           technique (SART): a superior implementation of the ART algorithm",\n           Ultrasonic Imaging 6 pp 81--94 (1984)\n    .. [3] S Kaczmarz, "Angenäherte auflösung von systemen linearer\n           gleichungen", Bulletin International de l’Academie Polonaise des\n           Sciences et des Lettres 35 pp 355--357 (1937)\n    .. [4] Kohler, T. "A projection access scheme for iterative\n           reconstruction based on the golden section." Nuclear Science\n           Symposium Conference Record, 2004 IEEE. Vol. 6. IEEE, 2004.\n    .. [5] Kaczmarz\' method, Wikipedia,\n           https://en.wikipedia.org/wiki/Kaczmarz_method\n\n    '
    if radon_image.ndim != 2:
        raise ValueError('radon_image must be two dimensional')
    if dtype is None:
        if radon_image.dtype.char in 'fd':
            dtype = radon_image.dtype
        else:
            warn('Only floating point data type are valid for SART inverse radon transform. Input data is cast to float. To disable this warning, please cast image_radon to float.')
            dtype = np.dtype(float)
    elif np.dtype(dtype).char not in 'fd':
        raise ValueError('Only floating point data type are valid for inverse radon transform.')
    dtype = np.dtype(dtype)
    radon_image = radon_image.astype(dtype, copy=False)
    reconstructed_shape = (radon_image.shape[0], radon_image.shape[0])
    if theta is None:
        theta = np.linspace(0, 180, radon_image.shape[1], endpoint=False, dtype=dtype)
    elif len(theta) != radon_image.shape[1]:
        raise ValueError(f'Shape of theta ({len(theta)}) does not match the number of projections ({radon_image.shape[1]})')
    else:
        theta = np.asarray(theta, dtype=dtype)
    if image is None:
        image = np.zeros(reconstructed_shape, dtype=dtype)
    elif image.shape != reconstructed_shape:
        raise ValueError(f'Shape of image ({image.shape}) does not match first dimension of radon_image ({reconstructed_shape})')
    elif image.dtype != dtype:
        warn(f'image dtype does not match output dtype: image is cast to {dtype}')
    image = np.asarray(image, dtype=dtype)
    if projection_shifts is None:
        projection_shifts = np.zeros((radon_image.shape[1],), dtype=dtype)
    elif len(projection_shifts) != radon_image.shape[1]:
        raise ValueError(f'Shape of projection_shifts ({len(projection_shifts)}) does not match the number of projections ({radon_image.shape[1]})')
    else:
        projection_shifts = np.asarray(projection_shifts, dtype=dtype)
    if clip is not None:
        if len(clip) != 2:
            raise ValueError('clip must be a length-2 sequence')
        clip = np.asarray(clip, dtype=dtype)
    for angle_index in order_angles_golden_ratio(theta):
        image_update = sart_projection_update(image, theta[angle_index], radon_image[:, angle_index], projection_shifts[angle_index])
        image += relaxation * image_update
        if clip is not None:
            image = np.clip(image, clip[0], clip[1])
    return image