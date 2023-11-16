import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from .._shared.utils import _supported_float_type

def _cv_calculate_variation(image, phi, mu, lambda1, lambda2, dt):
    if False:
        i = 10
        return i + 15
    "Returns the variation of level set 'phi' based on algorithm parameters.\n\n    This corresponds to equation (22) of the paper by Pascal Getreuer,\n    which computes the next iteration of the level set based on a current\n    level set.\n\n    A full explanation regarding all the terms is beyond the scope of the\n    present description, but there is one difference of particular import.\n    In the original algorithm, convergence is accelerated, and required\n    memory is reduced, by using a single array. This array, therefore, is a\n    combination of non-updated and updated values. If this were to be\n    implemented in python, this would require a double loop, where the\n    benefits of having fewer iterations would be outweided by massively\n    increasing the time required to perform each individual iteration. A\n    similar approach is used by Rami Cohen, and it is from there that the\n    C1-4 notation is taken.\n    "
    eta = 1e-16
    P = np.pad(phi, 1, mode='edge')
    phixp = P[1:-1, 2:] - P[1:-1, 1:-1]
    phixn = P[1:-1, 1:-1] - P[1:-1, :-2]
    phix0 = (P[1:-1, 2:] - P[1:-1, :-2]) / 2.0
    phiyp = P[2:, 1:-1] - P[1:-1, 1:-1]
    phiyn = P[1:-1, 1:-1] - P[:-2, 1:-1]
    phiy0 = (P[2:, 1:-1] - P[:-2, 1:-1]) / 2.0
    C1 = 1.0 / np.sqrt(eta + phixp ** 2 + phiy0 ** 2)
    C2 = 1.0 / np.sqrt(eta + phixn ** 2 + phiy0 ** 2)
    C3 = 1.0 / np.sqrt(eta + phix0 ** 2 + phiyp ** 2)
    C4 = 1.0 / np.sqrt(eta + phix0 ** 2 + phiyn ** 2)
    K = P[1:-1, 2:] * C1 + P[1:-1, :-2] * C2 + P[2:, 1:-1] * C3 + P[:-2, 1:-1] * C4
    Hphi = 1 * (phi > 0)
    (c1, c2) = _cv_calculate_averages(image, Hphi)
    difference_from_average_term = -lambda1 * (image - c1) ** 2 + lambda2 * (image - c2) ** 2
    new_phi = phi + dt * _cv_delta(phi) * (mu * K + difference_from_average_term)
    return new_phi / (1 + mu * dt * _cv_delta(phi) * (C1 + C2 + C3 + C4))

def _cv_heavyside(x, eps=1.0):
    if False:
        for i in range(10):
            print('nop')
    'Returns the result of a regularised heavyside function of the\n    input value(s).\n    '
    return 0.5 * (1.0 + 2.0 / np.pi * np.arctan(x / eps))

def _cv_delta(x, eps=1.0):
    if False:
        print('Hello World!')
    'Returns the result of a regularised dirac function of the\n    input value(s).\n    '
    return eps / (eps ** 2 + x ** 2)

def _cv_calculate_averages(image, Hphi):
    if False:
        for i in range(10):
            print('nop')
    "Returns the average values 'inside' and 'outside'."
    H = Hphi
    Hinv = 1.0 - H
    Hsum = np.sum(H)
    Hinvsum = np.sum(Hinv)
    avg_inside = np.sum(image * H)
    avg_oustide = np.sum(image * Hinv)
    if Hsum != 0:
        avg_inside /= Hsum
    if Hinvsum != 0:
        avg_oustide /= Hinvsum
    return (avg_inside, avg_oustide)

def _cv_difference_from_average_term(image, Hphi, lambda_pos, lambda_neg):
    if False:
        print('Hello World!')
    "Returns the 'energy' contribution due to the difference from\n    the average value within a region at each point.\n    "
    (c1, c2) = _cv_calculate_averages(image, Hphi)
    Hinv = 1.0 - Hphi
    return lambda_pos * (image - c1) ** 2 * Hphi + lambda_neg * (image - c2) ** 2 * Hinv

def _cv_edge_length_term(phi, mu):
    if False:
        for i in range(10):
            print('nop')
    "Returns the 'energy' contribution due to the length of the\n    edge between regions at each point, multiplied by a factor 'mu'.\n    "
    P = np.pad(phi, 1, mode='edge')
    fy = (P[2:, 1:-1] - P[:-2, 1:-1]) / 2.0
    fx = (P[1:-1, 2:] - P[1:-1, :-2]) / 2.0
    return mu * _cv_delta(phi) * np.sqrt(fx ** 2 + fy ** 2)

def _cv_energy(image, phi, mu, lambda1, lambda2):
    if False:
        while True:
            i = 10
    'Returns the total \'energy\' of the current level set function.\n\n    This corresponds to equation (7) of the paper by Pascal Getreuer,\n    which is the weighted sum of the following:\n    (A) the length of the contour produced by the zero values of the\n    level set,\n    (B) the area of the "foreground" (area of the image where the\n    level set is positive),\n    (C) the variance of the image inside the foreground,\n    (D) the variance of the image outside of the foreground\n\n    Each value is computed for each pixel, and then summed. The weight\n    of (B) is set to 0 in this implementation.\n    '
    H = _cv_heavyside(phi)
    avgenergy = _cv_difference_from_average_term(image, H, lambda1, lambda2)
    lenenergy = _cv_edge_length_term(phi, mu)
    return np.sum(avgenergy) + np.sum(lenenergy)

def _cv_reset_level_set(phi):
    if False:
        print('Hello World!')
    'This is a placeholder function as resetting the level set is not\n    strictly necessary, and has not been done for this implementation.\n    '
    return phi

def _cv_checkerboard(image_size, square_size, dtype=np.float64):
    if False:
        while True:
            i = 10
    'Generates a checkerboard level set function.\n\n    According to Pascal Getreuer, such a level set function has fast\n    convergence.\n    '
    yv = np.arange(image_size[0], dtype=dtype).reshape(image_size[0], 1)
    xv = np.arange(image_size[1], dtype=dtype)
    sf = np.pi / square_size
    xv *= sf
    yv *= sf
    return np.sin(yv) * np.sin(xv)

def _cv_large_disk(image_size):
    if False:
        for i in range(10):
            print('nop')
    'Generates a disk level set function.\n\n    The disk covers the whole image along its smallest dimension.\n    '
    res = np.ones(image_size)
    centerY = int((image_size[0] - 1) / 2)
    centerX = int((image_size[1] - 1) / 2)
    res[centerY, centerX] = 0.0
    radius = float(min(centerX, centerY))
    return (radius - distance(res)) / radius

def _cv_small_disk(image_size):
    if False:
        while True:
            i = 10
    'Generates a disk level set function.\n\n    The disk covers half of the image along its smallest dimension.\n    '
    res = np.ones(image_size)
    centerY = int((image_size[0] - 1) / 2)
    centerX = int((image_size[1] - 1) / 2)
    res[centerY, centerX] = 0.0
    radius = float(min(centerX, centerY)) / 2.0
    return (radius - distance(res)) / (radius * 3)

def _cv_init_level_set(init_level_set, image_shape, dtype=np.float64):
    if False:
        return 10
    'Generates an initial level set function conditional on input arguments.'
    if isinstance(init_level_set, str):
        if init_level_set == 'checkerboard':
            res = _cv_checkerboard(image_shape, 5, dtype)
        elif init_level_set == 'disk':
            res = _cv_large_disk(image_shape)
        elif init_level_set == 'small disk':
            res = _cv_small_disk(image_shape)
        else:
            raise ValueError('Incorrect name for starting level set preset.')
    else:
        res = init_level_set
    return res.astype(dtype, copy=False)

def chan_vese(image, mu=0.25, lambda1=1.0, lambda2=1.0, tol=0.001, max_num_iter=500, dt=0.5, init_level_set='checkerboard', extended_output=False):
    if False:
        i = 10
        return i + 15
    'Chan-Vese segmentation algorithm.\n\n    Active contour model by evolving a level set. Can be used to\n    segment objects without clearly defined boundaries.\n\n    Parameters\n    ----------\n    image : (M, N) ndarray\n        Grayscale image to be segmented.\n    mu : float, optional\n        \'edge length\' weight parameter. Higher `mu` values will\n        produce a \'round\' edge, while values closer to zero will\n        detect smaller objects.\n    lambda1 : float, optional\n        \'difference from average\' weight parameter for the output\n        region with value \'True\'. If it is lower than `lambda2`, this\n        region will have a larger range of values than the other.\n    lambda2 : float, optional\n        \'difference from average\' weight parameter for the output\n        region with value \'False\'. If it is lower than `lambda1`, this\n        region will have a larger range of values than the other.\n    tol : float, positive, optional\n        Level set variation tolerance between iterations. If the\n        L2 norm difference between the level sets of successive\n        iterations normalized by the area of the image is below this\n        value, the algorithm will assume that the solution was\n        reached.\n    max_num_iter : uint, optional\n        Maximum number of iterations allowed before the algorithm\n        interrupts itself.\n    dt : float, optional\n        A multiplication factor applied at calculations for each step,\n        serves to accelerate the algorithm. While higher values may\n        speed up the algorithm, they may also lead to convergence\n        problems.\n    init_level_set : str or (M, N) ndarray, optional\n        Defines the starting level set used by the algorithm.\n        If a string is inputted, a level set that matches the image\n        size will automatically be generated. Alternatively, it is\n        possible to define a custom level set, which should be an\n        array of float values, with the same shape as \'image\'.\n        Accepted string values are as follows.\n\n        \'checkerboard\'\n            the starting level set is defined as\n            sin(x/5*pi)*sin(y/5*pi), where x and y are pixel\n            coordinates. This level set has fast convergence, but may\n            fail to detect implicit edges.\n        \'disk\'\n            the starting level set is defined as the opposite\n            of the distance from the center of the image minus half of\n            the minimum value between image width and image height.\n            This is somewhat slower, but is more likely to properly\n            detect implicit edges.\n        \'small disk\'\n            the starting level set is defined as the\n            opposite of the distance from the center of the image\n            minus a quarter of the minimum value between image width\n            and image height.\n    extended_output : bool, optional\n        If set to True, the return value will be a tuple containing\n        the three return values (see below). If set to False which\n        is the default value, only the \'segmentation\' array will be\n        returned.\n\n    Returns\n    -------\n    segmentation : (M, N) ndarray, bool\n        Segmentation produced by the algorithm.\n    phi : (M, N) ndarray of floats\n        Final level set computed by the algorithm.\n    energies : list of floats\n        Shows the evolution of the \'energy\' for each step of the\n        algorithm. This should allow to check whether the algorithm\n        converged.\n\n    Notes\n    -----\n    The Chan-Vese Algorithm is designed to segment objects without\n    clearly defined boundaries. This algorithm is based on level sets\n    that are evolved iteratively to minimize an energy, which is\n    defined by weighted values corresponding to the sum of differences\n    intensity from the average value outside the segmented region, the\n    sum of differences from the average value inside the segmented\n    region, and a term which is dependent on the length of the\n    boundary of the segmented region.\n\n    This algorithm was first proposed by Tony Chan and Luminita Vese,\n    in a publication entitled "An Active Contour Model Without Edges"\n    [1]_.\n\n    This implementation of the algorithm is somewhat simplified in the\n    sense that the area factor \'nu\' described in the original paper is\n    not implemented, and is only suitable for grayscale images.\n\n    Typical values for `lambda1` and `lambda2` are 1. If the\n    \'background\' is very different from the segmented object in terms\n    of distribution (for example, a uniform black image with figures\n    of varying intensity), then these values should be different from\n    each other.\n\n    Typical values for mu are between 0 and 1, though higher values\n    can be used when dealing with shapes with very ill-defined\n    contours.\n\n    The \'energy\' which this algorithm tries to minimize is defined\n    as the sum of the differences from the average within the region\n    squared and weighed by the \'lambda\' factors to which is added the\n    length of the contour multiplied by the \'mu\' factor.\n\n    Supports 2D grayscale images only, and does not implement the area\n    term described in the original article.\n\n    References\n    ----------\n    .. [1] An Active Contour Model without Edges, Tony Chan and\n           Luminita Vese, Scale-Space Theories in Computer Vision,\n           1999, :DOI:`10.1007/3-540-48236-9_13`\n    .. [2] Chan-Vese Segmentation, Pascal Getreuer Image Processing On\n           Line, 2 (2012), pp. 214-224,\n           :DOI:`10.5201/ipol.2012.g-cv`\n    .. [3] The Chan-Vese Algorithm - Project Report, Rami Cohen, 2011\n           :arXiv:`1107.2782`\n    '
    if len(image.shape) != 2:
        raise ValueError('Input image should be a 2D array.')
    float_dtype = _supported_float_type(image.dtype)
    phi = _cv_init_level_set(init_level_set, image.shape, dtype=float_dtype)
    if type(phi) != np.ndarray or phi.shape != image.shape:
        raise ValueError('The dimensions of initial level set do not match the dimensions of image.')
    image = image.astype(float_dtype, copy=False)
    image = image - np.min(image)
    if np.max(image) != 0:
        image = image / np.max(image)
    i = 0
    old_energy = _cv_energy(image, phi, mu, lambda1, lambda2)
    energies = []
    phivar = tol + 1
    segmentation = phi > 0
    while phivar > tol and i < max_num_iter:
        oldphi = phi
        phi = _cv_calculate_variation(image, phi, mu, lambda1, lambda2, dt)
        phi = _cv_reset_level_set(phi)
        phivar = np.sqrt(((phi - oldphi) ** 2).mean())
        segmentation = phi > 0
        new_energy = _cv_energy(image, phi, mu, lambda1, lambda2)
        energies.append(old_energy)
        old_energy = new_energy
        i += 1
    if extended_output:
        return (segmentation, phi, energies)
    else:
        return segmentation