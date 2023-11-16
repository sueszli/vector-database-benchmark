from math import sqrt
from numbers import Real
import numpy as np
from scipy import ndimage as ndi
STREL_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
STREL_8 = np.ones((3, 3), dtype=np.uint8)
EULER_COEFS2D_4 = [0, 1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0]
EULER_COEFS2D_8 = [0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, -1, 0]
EULER_COEFS3D_26 = np.array([0, 1, 1, 0, 1, 0, -2, -1, 1, -2, 0, -1, 0, -1, -1, 0, 1, 0, -2, -1, -2, -1, -1, -2, -6, -3, -3, -2, -3, -2, 0, -1, 1, -2, 0, -1, -6, -3, -3, -2, -2, -1, -1, -2, -3, 0, -2, -1, 0, -1, -1, 0, -3, -2, 0, -1, -3, 0, -2, -1, 0, 1, 1, 0, 1, -2, -6, -3, 0, -1, -3, -2, -2, -1, -3, 0, -1, -2, -2, -1, 0, -1, -3, -2, -1, 0, 0, -1, -3, 0, 0, 1, -2, -1, 1, 0, -2, -1, -3, 0, -3, 0, 0, 1, -1, 4, 0, 3, 0, 3, 1, 2, -1, -2, -2, -1, -2, -1, 1, 0, 0, 3, 1, 2, 1, 2, 2, 1, 1, -6, -2, -3, -2, -3, -1, 0, 0, -3, -1, -2, -1, -2, -2, -1, -2, -3, -1, 0, -1, 0, 4, 3, -3, 0, 0, 1, 0, 1, 3, 2, 0, -3, -1, -2, -3, 0, 0, 1, -1, 0, 0, -1, -2, 1, -1, 0, -1, -2, -2, -1, 0, 1, 3, 2, -2, 1, -1, 0, 1, 2, 2, 1, 0, -3, -3, 0, -1, -2, 0, 1, -1, 0, -2, 1, 0, -1, -1, 0, -1, -2, 0, 1, -2, -1, 3, 2, -2, 1, 1, 2, -1, 0, 2, 1, -1, 0, -2, 1, -2, 1, 1, 2, -2, 3, -1, 2, -1, 2, 0, 1, 0, -1, -1, 0, -1, 0, 2, 1, -1, 2, 0, 1, 0, 1, 1, 0])

def euler_number(image, connectivity=None):
    if False:
        return 10
    'Calculate the Euler characteristic in binary image.\n\n    For 2D objects, the Euler number is the number of objects minus the number\n    of holes. For 3D objects, the Euler number is obtained as the number of\n    objects plus the number of holes, minus the number of tunnels, or loops.\n\n    Parameters\n    ----------\n    image: (M, N[, P]) ndarray\n        Input image. If image is not binary, all values greater than zero\n        are considered as the object.\n    connectivity : int, optional\n        Maximum number of orthogonal hops to consider a pixel/voxel\n        as a neighbor.\n        Accepted values are ranging from  1 to input.ndim. If ``None``, a full\n        connectivity of ``input.ndim`` is used.\n        4 or 8 neighborhoods are defined for 2D images (connectivity 1 and 2,\n        respectively).\n        6 or 26 neighborhoods are defined for 3D images, (connectivity 1 and 3,\n        respectively). Connectivity 2 is not defined.\n\n    Returns\n    -------\n    euler_number : int\n        Euler characteristic of the set of all objects in the image.\n\n    Notes\n    -----\n    The Euler characteristic is an integer number that describes the\n    topology of the set of all objects in the input image. If object is\n    4-connected, then background is 8-connected, and conversely.\n\n    The computation of the Euler characteristic is based on an integral\n    geometry formula in discretized space. In practice, a neighborhood\n    configuration is constructed, and a LUT is applied for each\n    configuration. The coefficients used are the ones of Ohser et al.\n\n    It can be useful to compute the Euler characteristic for several\n    connectivities. A large relative difference between results\n    for different connectivities suggests that the image resolution\n    (with respect to the size of objects and holes) is too low.\n\n    References\n    ----------\n    .. [1] S. Rivollier. Analyse d’image geometrique et morphometrique par\n           diagrammes de forme et voisinages adaptatifs generaux. PhD thesis,\n           2010. Ecole Nationale Superieure des Mines de Saint-Etienne.\n           https://tel.archives-ouvertes.fr/tel-00560838\n    .. [2] Ohser J., Nagel W., Schladitz K. (2002) The Euler Number of\n           Discretized Sets - On the Choice of Adjacency in Homogeneous\n           Lattices. In: Mecke K., Stoyan D. (eds) Morphology of Condensed\n           Matter. Lecture Notes in Physics, vol 600. Springer, Berlin,\n           Heidelberg.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> SAMPLE = np.zeros((100,100,100));\n    >>> SAMPLE[40:60, 40:60, 40:60]=1\n    >>> euler_number(SAMPLE) # doctest: +ELLIPSIS\n    1...\n    >>> SAMPLE[45:55,45:55,45:55] = 0;\n    >>> euler_number(SAMPLE) # doctest: +ELLIPSIS\n    2...\n    >>> SAMPLE = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],\n    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],\n    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],\n    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],\n    ...                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],\n    ...                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],\n    ...                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],\n    ...                    [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],\n    ...                    [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1],\n    ...                    [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])\n    >>> euler_number(SAMPLE)  # doctest:\n    0\n    >>> euler_number(SAMPLE, connectivity=1)  # doctest:\n    2\n    '
    image = (image > 0).astype(int)
    image = np.pad(image, pad_width=1, mode='constant')
    if connectivity is None:
        connectivity = image.ndim
    if image.ndim == 2:
        config = np.array([[0, 0, 0], [0, 1, 4], [0, 2, 8]])
        if connectivity == 1:
            coefs = EULER_COEFS2D_4
        else:
            coefs = EULER_COEFS2D_8
        bins = 16
    else:
        if connectivity == 2:
            raise NotImplementedError('For 3D images, Euler number is implemented for connectivities 1 and 3 only')
        config = np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 4], [0, 2, 8]], [[0, 0, 0], [0, 16, 64], [0, 32, 128]]])
        if connectivity == 1:
            coefs = EULER_COEFS3D_26[::-1]
        else:
            coefs = EULER_COEFS3D_26
        bins = 256
    XF = ndi.convolve(image, config, mode='constant', cval=0)
    h = np.bincount(XF.ravel(), minlength=bins)
    if image.ndim == 2:
        return coefs @ h
    else:
        return int(0.125 * coefs @ h)

def perimeter(image, neighborhood=4):
    if False:
        return 10
    "Calculate total perimeter of all objects in binary image.\n\n    Parameters\n    ----------\n    image : (M, N) ndarray\n        Binary input image.\n    neighborhood : 4 or 8, optional\n        Neighborhood connectivity for border pixel determination. It is used to\n        compute the contour. A higher neighborhood widens the border on which\n        the perimeter is computed.\n\n    Returns\n    -------\n    perimeter : float\n        Total perimeter of all objects in binary image.\n\n    References\n    ----------\n    .. [1] K. Benkrid, D. Crookes. Design and FPGA Implementation of\n           a Perimeter Estimator. The Queen's University of Belfast.\n           http://www.cs.qub.ac.uk/~d.crookes/webpubs/papers/perimeter.doc\n\n    Examples\n    --------\n    >>> from skimage import data, util\n    >>> from skimage.measure import label\n    >>> # coins image (binary)\n    >>> img_coins = data.coins() > 110\n    >>> # total perimeter of all objects in the image\n    >>> perimeter(img_coins, neighborhood=4)  # doctest: +ELLIPSIS\n    7796.867...\n    >>> perimeter(img_coins, neighborhood=8)  # doctest: +ELLIPSIS\n    8806.268...\n\n    "
    if image.ndim != 2:
        raise NotImplementedError('`perimeter` supports 2D images only')
    if neighborhood == 4:
        strel = STREL_4
    else:
        strel = STREL_8
    image = image.astype(np.uint8)
    eroded_image = ndi.binary_erosion(image, strel, border_value=0)
    border_image = image - eroded_image
    perimeter_weights = np.zeros(50, dtype=np.float64)
    perimeter_weights[[5, 7, 15, 17, 25, 27]] = 1
    perimeter_weights[[21, 33]] = sqrt(2)
    perimeter_weights[[13, 23]] = (1 + sqrt(2)) / 2
    perimeter_image = ndi.convolve(border_image, np.array([[10, 2, 10], [2, 1, 2], [10, 2, 10]]), mode='constant', cval=0)
    perimeter_histogram = np.bincount(perimeter_image.ravel(), minlength=50)
    total_perimeter = perimeter_histogram @ perimeter_weights
    return total_perimeter

def perimeter_crofton(image, directions=4):
    if False:
        return 10
    'Calculate total Crofton perimeter of all objects in binary image.\n\n    Parameters\n    ----------\n    image : (M, N) ndarray\n        Input image. If image is not binary, all values greater than zero\n        are considered as the object.\n    directions : 2 or 4, optional\n        Number of directions used to approximate the Crofton perimeter. By\n        default, 4 is used: it should be more accurate than 2.\n        Computation time is the same in both cases.\n\n    Returns\n    -------\n    perimeter : float\n        Total perimeter of all objects in binary image.\n\n    Notes\n    -----\n    This measure is based on Crofton formula [1], which is a measure from\n    integral geometry. It is defined for general curve length evaluation via\n    a double integral along all directions. In a discrete\n    space, 2 or 4 directions give a quite good approximation, 4 being more\n    accurate than 2 for more complex shapes.\n\n    Similar to :func:`~.measure.perimeter`, this function returns an\n    approximation of the perimeter in continuous space.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Crofton_formula\n    .. [2] S. Rivollier. Analyse d’image geometrique et morphometrique par\n           diagrammes de forme et voisinages adaptatifs generaux. PhD thesis,\n           2010.\n           Ecole Nationale Superieure des Mines de Saint-Etienne.\n           https://tel.archives-ouvertes.fr/tel-00560838\n\n    Examples\n    --------\n    >>> from skimage import data, util\n    >>> from skimage.measure import label\n    >>> # coins image (binary)\n    >>> img_coins = data.coins() > 110\n    >>> # total perimeter of all objects in the image\n    >>> perimeter_crofton(img_coins, directions=2)  # doctest: +ELLIPSIS\n    8144.578...\n    >>> perimeter_crofton(img_coins, directions=4)  # doctest: +ELLIPSIS\n    7837.077...\n    '
    if image.ndim != 2:
        raise NotImplementedError('`perimeter_crofton` supports 2D images only')
    image = (image > 0).astype(np.uint8)
    image = np.pad(image, pad_width=1, mode='constant')
    XF = ndi.convolve(image, np.array([[0, 0, 0], [0, 1, 4], [0, 2, 8]]), mode='constant', cval=0)
    h = np.bincount(XF.ravel(), minlength=16)
    if directions == 2:
        coefs = [0, np.pi / 2, 0, 0, 0, np.pi / 2, 0, 0, np.pi / 2, np.pi, 0, 0, np.pi / 2, np.pi, 0, 0]
    else:
        coefs = [0, np.pi / 4 * (1 + 1 / np.sqrt(2)), np.pi / (4 * np.sqrt(2)), np.pi / (2 * np.sqrt(2)), 0, np.pi / 4 * (1 + 1 / np.sqrt(2)), 0, np.pi / (4 * np.sqrt(2)), np.pi / 4, np.pi / 2, np.pi / (4 * np.sqrt(2)), np.pi / (4 * np.sqrt(2)), np.pi / 4, np.pi / 2, 0, 0]
    total_perimeter = coefs @ h
    return total_perimeter

def _normalize_spacing(spacing, ndims):
    if False:
        for i in range(10):
            print('nop')
    'Normalize spacing parameter.\n\n    The `spacing` parameter should be a sequence of numbers matching\n    the image dimensions. If `spacing` is a scalar, assume equal\n    spacing along all dimensions.\n\n    Parameters\n    ---------\n    spacing : Any\n        User-provided `spacing` keyword.\n    ndims : int\n        Number of image dimensions.\n\n    Returns\n    -------\n    spacing : array\n        Corrected spacing.\n\n    Raises\n    ------\n    ValueError\n        If `spacing` is invalid.\n\n    '
    spacing = np.array(spacing)
    if spacing.shape == ():
        spacing = np.broadcast_to(spacing, shape=(ndims,))
    elif spacing.shape != (ndims,):
        raise ValueError(f"spacing isn't a scalar nor a sequence of shape {(ndims,)}, got {spacing}.")
    if not all((isinstance(s, Real) for s in spacing)):
        raise TypeError(f"Element of spacing isn't float or integer type, got {spacing}.")
    if not all(np.isfinite(spacing)):
        raise ValueError(f'Invalid spacing parameter. All elements must be finite, got {spacing}.')
    return spacing