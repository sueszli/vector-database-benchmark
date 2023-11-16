from itertools import chain
from operator import add
import numpy as np
from ._haar import haar_like_feature_coord_wrapper
from ._haar import haar_like_feature_wrapper
from .._shared.utils import deprecate_kwarg
from ..color import gray2rgb
from ..draw import rectangle
from ..util import img_as_float
FEATURE_TYPE = ('type-2-x', 'type-2-y', 'type-3-x', 'type-3-y', 'type-4')

def _validate_feature_type(feature_type):
    if False:
        print('Hello World!')
    'Transform feature type to an iterable and check that it exists.'
    if feature_type is None:
        feature_type_ = FEATURE_TYPE
    else:
        if isinstance(feature_type, str):
            feature_type_ = [feature_type]
        else:
            feature_type_ = feature_type
        for feat_t in feature_type_:
            if feat_t not in FEATURE_TYPE:
                raise ValueError(f'The given feature type is unknown. Got {feat_t} instead of one of {FEATURE_TYPE}.')
    return feature_type_

def haar_like_feature_coord(width, height, feature_type=None):
    if False:
        return 10
    "Compute the coordinates of Haar-like features.\n\n    Parameters\n    ----------\n    width : int\n        Width of the detection window.\n    height : int\n        Height of the detection window.\n    feature_type : str or list of str or None, optional\n        The type of feature to consider:\n\n        - 'type-2-x': 2 rectangles varying along the x axis;\n        - 'type-2-y': 2 rectangles varying along the y axis;\n        - 'type-3-x': 3 rectangles varying along the x axis;\n        - 'type-3-y': 3 rectangles varying along the y axis;\n        - 'type-4': 4 rectangles varying along x and y axis.\n\n        By default all features are extracted.\n\n    Returns\n    -------\n    feature_coord : (n_features, n_rectangles, 2, 2), ndarray of list of tuple coord\n        Coordinates of the rectangles for each feature.\n    feature_type : (n_features,), ndarray of str\n        The corresponding type for each feature.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from skimage.transform import integral_image\n    >>> from skimage.feature import haar_like_feature_coord\n    >>> feat_coord, feat_type = haar_like_feature_coord(2, 2, 'type-4')\n    >>> feat_coord # doctest: +SKIP\n    array([ list([[(0, 0), (0, 0)], [(0, 1), (0, 1)],\n                  [(1, 1), (1, 1)], [(1, 0), (1, 0)]])], dtype=object)\n    >>> feat_type\n    array(['type-4'], dtype=object)\n\n    "
    feature_type_ = _validate_feature_type(feature_type)
    (feat_coord, feat_type) = zip(*[haar_like_feature_coord_wrapper(width, height, feat_t) for feat_t in feature_type_])
    return (np.concatenate(feat_coord), np.hstack(feat_type))

def haar_like_feature(int_image, r, c, width, height, feature_type=None, feature_coord=None):
    if False:
        print('Hello World!')
    'Compute the Haar-like features for a region of interest (ROI) of an\n    integral image.\n\n    Haar-like features have been successfully used for image classification and\n    object detection [1]_. It has been used for real-time face detection\n    algorithm proposed in [2]_.\n\n    Parameters\n    ----------\n    int_image : (M, N) ndarray\n        Integral image for which the features need to be computed.\n    r : int\n        Row-coordinate of top left corner of the detection window.\n    c : int\n        Column-coordinate of top left corner of the detection window.\n    width : int\n        Width of the detection window.\n    height : int\n        Height of the detection window.\n    feature_type : str or list of str or None, optional\n        The type of feature to consider:\n\n        - \'type-2-x\': 2 rectangles varying along the x axis;\n        - \'type-2-y\': 2 rectangles varying along the y axis;\n        - \'type-3-x\': 3 rectangles varying along the x axis;\n        - \'type-3-y\': 3 rectangles varying along the y axis;\n        - \'type-4\': 4 rectangles varying along x and y axis.\n\n        By default all features are extracted.\n\n        If using with `feature_coord`, it should correspond to the feature\n        type of each associated coordinate feature.\n    feature_coord : ndarray of list of tuples or None, optional\n        The array of coordinates to be extracted. This is useful when you want\n        to recompute only a subset of features. In this case `feature_type`\n        needs to be an array containing the type of each feature, as returned\n        by :func:`haar_like_feature_coord`. By default, all coordinates are\n        computed.\n\n    Returns\n    -------\n    haar_features : (n_features,) ndarray of int or float\n        Resulting Haar-like features. Each value is equal to the subtraction of\n        sums of the positive and negative rectangles. The data type depends of\n        the data type of `int_image`: `int` when the data type of `int_image`\n        is `uint` or `int` and `float` when the data type of `int_image` is\n        `float`.\n\n    Notes\n    -----\n    When extracting those features in parallel, be aware that the choice of the\n    backend (i.e. multiprocessing vs threading) will have an impact on the\n    performance. The rule of thumb is as follows: use multiprocessing when\n    extracting features for all possible ROI in an image; use threading when\n    extracting the feature at specific location for a limited number of ROIs.\n    Refer to the example\n    :ref:`sphx_glr_auto_examples_applications_plot_haar_extraction_selection_classification.py`\n    for more insights.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from skimage.transform import integral_image\n    >>> from skimage.feature import haar_like_feature\n    >>> img = np.ones((5, 5), dtype=np.uint8)\n    >>> img_ii = integral_image(img)\n    >>> feature = haar_like_feature(img_ii, 0, 0, 5, 5, \'type-3-x\')\n    >>> feature\n    array([-1, -2, -3, -4, -5, -1, -2, -3, -4, -5, -1, -2, -3, -4, -5, -1, -2,\n           -3, -4, -1, -2, -3, -4, -1, -2, -3, -4, -1, -2, -3, -1, -2, -3, -1,\n           -2, -3, -1, -2, -1, -2, -1, -2, -1, -1, -1])\n\n    You can compute the feature for some pre-computed coordinates.\n\n    >>> from skimage.feature import haar_like_feature_coord\n    >>> feature_coord, feature_type = zip(\n    ...     *[haar_like_feature_coord(5, 5, feat_t)\n    ...       for feat_t in (\'type-2-x\', \'type-3-x\')])\n    >>> # only select one feature over two\n    >>> feature_coord = np.concatenate([x[::2] for x in feature_coord])\n    >>> feature_type = np.concatenate([x[::2] for x in feature_type])\n    >>> feature = haar_like_feature(img_ii, 0, 0, 5, 5,\n    ...                             feature_type=feature_type,\n    ...                             feature_coord=feature_coord)\n    >>> feature\n    array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,\n            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,\n            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, -3, -5, -2, -4, -1,\n           -3, -5, -2, -4, -2, -4, -2, -4, -2, -1, -3, -2, -1, -1, -1, -1, -1])\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Haar-like_feature\n    .. [2] Oren, M., Papageorgiou, C., Sinha, P., Osuna, E., & Poggio, T.\n           (1997, June). Pedestrian detection using wavelet templates.\n           In Computer Vision and Pattern Recognition, 1997. Proceedings.,\n           1997 IEEE Computer Society Conference on (pp. 193-199). IEEE.\n           http://tinyurl.com/y6ulxfta\n           :DOI:`10.1109/CVPR.1997.609319`\n    .. [3] Viola, Paul, and Michael J. Jones. "Robust real-time face\n           detection." International journal of computer vision 57.2\n           (2004): 137-154.\n           https://www.merl.com/publications/docs/TR2004-043.pdf\n           :DOI:`10.1109/CVPR.2001.990517`\n\n    '
    if feature_coord is None:
        feature_type_ = _validate_feature_type(feature_type)
        return np.hstack(list(chain.from_iterable((haar_like_feature_wrapper(int_image, r, c, width, height, feat_t, feature_coord) for feat_t in feature_type_))))
    else:
        if feature_coord.shape[0] != feature_type.shape[0]:
            raise ValueError('Inconsistent size between feature coordinatesand feature types.')
        mask_feature = [feature_type == feat_t for feat_t in FEATURE_TYPE]
        (haar_feature_idx, haar_feature) = zip(*[(np.flatnonzero(mask), haar_like_feature_wrapper(int_image, r, c, width, height, feat_t, feature_coord[mask])) for (mask, feat_t) in zip(mask_feature, FEATURE_TYPE) if np.count_nonzero(mask)])
        haar_feature_idx = np.concatenate(haar_feature_idx)
        haar_feature = np.concatenate(haar_feature)
        haar_feature[haar_feature_idx] = haar_feature.copy()
        return haar_feature

@deprecate_kwarg({'random_state': 'rng'}, deprecated_version='0.21', removed_version='0.23')
def draw_haar_like_feature(image, r, c, width, height, feature_coord, color_positive_block=(1.0, 0.0, 0.0), color_negative_block=(0.0, 1.0, 0.0), alpha=0.5, max_n_features=None, rng=None):
    if False:
        i = 10
        return i + 15
    "Visualization of Haar-like features.\n\n    Parameters\n    ----------\n    image : (M, N) ndarray\n        The region of an integral image for which the features need to be\n        computed.\n    r : int\n        Row-coordinate of top left corner of the detection window.\n    c : int\n        Column-coordinate of top left corner of the detection window.\n    width : int\n        Width of the detection window.\n    height : int\n        Height of the detection window.\n    feature_coord : ndarray of list of tuples or None, optional\n        The array of coordinates to be extracted. This is useful when you want\n        to recompute only a subset of features. In this case `feature_type`\n        needs to be an array containing the type of each feature, as returned\n        by :func:`haar_like_feature_coord`. By default, all coordinates are\n        computed.\n    color_positive_block : tuple of 3 floats\n        Floats specifying the color for the positive block. Corresponding\n        values define (R, G, B) values. Default value is red (1, 0, 0).\n    color_negative_block : tuple of 3 floats\n        Floats specifying the color for the negative block Corresponding values\n        define (R, G, B) values. Default value is blue (0, 1, 0).\n    alpha : float\n        Value in the range [0, 1] that specifies opacity of visualization. 1 -\n        fully transparent, 0 - opaque.\n    max_n_features : int, default=None\n        The maximum number of features to be returned.\n        By default, all features are returned.\n    rng : {`numpy.random.Generator`, int}, optional\n        Pseudo-random number generator.\n        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).\n        If `rng` is an int, it is used to seed the generator.\n\n        The rng is used when generating a set of features smaller than\n        the total number of available features.\n\n    Returns\n    -------\n    features : (M, N), ndarray\n        An image in which the different features will be added.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from skimage.feature import haar_like_feature_coord\n    >>> from skimage.feature import draw_haar_like_feature\n    >>> feature_coord, _ = haar_like_feature_coord(2, 2, 'type-4')\n    >>> image = draw_haar_like_feature(np.zeros((2, 2)),\n    ...                                0, 0, 2, 2,\n    ...                                feature_coord,\n    ...                                max_n_features=1)\n    >>> image\n    array([[[0. , 0.5, 0. ],\n            [0.5, 0. , 0. ]],\n    <BLANKLINE>\n           [[0.5, 0. , 0. ],\n            [0. , 0.5, 0. ]]])\n\n    "
    rng = np.random.default_rng(rng)
    color_positive_block = np.asarray(color_positive_block, dtype=np.float64)
    color_negative_block = np.asarray(color_negative_block, dtype=np.float64)
    if max_n_features is None:
        feature_coord_ = feature_coord
    else:
        feature_coord_ = rng.choice(feature_coord, size=max_n_features, replace=False)
    output = np.copy(image)
    if len(image.shape) < 3:
        output = gray2rgb(image)
    output = img_as_float(output)
    for coord in feature_coord_:
        for (idx_rect, rect) in enumerate(coord):
            (coord_start, coord_end) = rect
            coord_start = tuple(map(add, coord_start, [r, c]))
            coord_end = tuple(map(add, coord_end, [r, c]))
            (rr, cc) = rectangle(coord_start, coord_end)
            if (idx_rect + 1) % 2 == 0:
                new_value = (1 - alpha) * output[rr, cc] + alpha * color_positive_block
            else:
                new_value = (1 - alpha) * output[rr, cc] + alpha * color_negative_block
            output[rr, cc] = new_value
    return output