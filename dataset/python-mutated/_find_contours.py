import numpy as np
from ._find_contours_cy import _get_contour_segments
from collections import deque
_param_options = ('high', 'low')

def find_contours(image, level=None, fully_connected='low', positive_orientation='low', *, mask=None):
    if False:
        for i in range(10):
            print('nop')
    'Find iso-valued contours in a 2D array for a given level value.\n\n    Uses the "marching squares" method to compute a the iso-valued contours of\n    the input 2D array for a particular level value. Array values are linearly\n    interpolated to provide better precision for the output contours.\n\n    Parameters\n    ----------\n    image : (M, N) ndarray of double\n        Input image in which to find contours.\n    level : float, optional\n        Value along which to find contours in the array. By default, the level\n        is set to (max(image) + min(image)) / 2\n\n        .. versionchanged:: 0.18\n            This parameter is now optional.\n    fully_connected : str, {\'low\', \'high\'}\n         Indicates whether array elements below the given level value are to be\n         considered fully-connected (and hence elements above the value will\n         only be face connected), or vice-versa. (See notes below for details.)\n    positive_orientation : str, {\'low\', \'high\'}\n         Indicates whether the output contours will produce positively-oriented\n         polygons around islands of low- or high-valued elements. If \'low\' then\n         contours will wind counter- clockwise around elements below the\n         iso-value. Alternately, this means that low-valued elements are always\n         on the left of the contour. (See below for details.)\n    mask : (M, N) ndarray of bool or None\n        A boolean mask, True where we want to draw contours.\n        Note that NaN values are always excluded from the considered region\n        (``mask`` is set to ``False`` wherever ``array`` is ``NaN``).\n\n    Returns\n    -------\n    contours : list of (K, 2) ndarrays\n        Each contour is a ndarray of ``(row, column)`` coordinates along the contour.\n\n    See Also\n    --------\n    skimage.measure.marching_cubes\n\n    Notes\n    -----\n    The marching squares algorithm is a special case of the marching cubes\n    algorithm [1]_.  A simple explanation is available here:\n\n    http://users.polytech.unice.fr/~lingrand/MarchingCubes/algo.html\n\n    There is a single ambiguous case in the marching squares algorithm: when\n    a given ``2 x 2``-element square has two high-valued and two low-valued\n    elements, each pair diagonally adjacent. (Where high- and low-valued is\n    with respect to the contour value sought.) In this case, either the\n    high-valued elements can be \'connected together\' via a thin isthmus that\n    separates the low-valued elements, or vice-versa. When elements are\n    connected together across a diagonal, they are considered \'fully\n    connected\' (also known as \'face+vertex-connected\' or \'8-connected\'). Only\n    high-valued or low-valued elements can be fully-connected, the other set\n    will be considered as \'face-connected\' or \'4-connected\'. By default,\n    low-valued elements are considered fully-connected; this can be altered\n    with the \'fully_connected\' parameter.\n\n    Output contours are not guaranteed to be closed: contours which intersect\n    the array edge or a masked-off region (either where mask is False or where\n    array is NaN) will be left open. All other contours will be closed. (The\n    closed-ness of a contours can be tested by checking whether the beginning\n    point is the same as the end point.)\n\n    Contours are oriented. By default, array values lower than the contour\n    value are to the left of the contour and values greater than the contour\n    value are to the right. This means that contours will wind\n    counter-clockwise (i.e. in \'positive orientation\') around islands of\n    low-valued pixels. This behavior can be altered with the\n    \'positive_orientation\' parameter.\n\n    The order of the contours in the output list is determined by the position\n    of the smallest ``x,y`` (in lexicographical order) coordinate in the\n    contour.  This is a side-effect of how the input array is traversed, but\n    can be relied upon.\n\n    .. warning::\n\n       Array coordinates/values are assumed to refer to the *center* of the\n       array element. Take a simple example input: ``[0, 1]``. The interpolated\n       position of 0.5 in this array is midway between the 0-element (at\n       ``x=0``) and the 1-element (at ``x=1``), and thus would fall at\n       ``x=0.5``.\n\n    This means that to find reasonable contours, it is best to find contours\n    midway between the expected "light" and "dark" values. In particular,\n    given a binarized array, *do not* choose to find contours at the low or\n    high value of the array. This will often yield degenerate contours,\n    especially around structures that are a single array element wide. Instead\n    choose a middle value, as above.\n\n    References\n    ----------\n    .. [1] Lorensen, William and Harvey E. Cline. Marching Cubes: A High\n           Resolution 3D Surface Construction Algorithm. Computer Graphics\n           (SIGGRAPH 87 Proceedings) 21(4) July 1987, p. 163-170).\n           :DOI:`10.1145/37401.37422`\n\n    Examples\n    --------\n    >>> a = np.zeros((3, 3))\n    >>> a[0, 0] = 1\n    >>> a\n    array([[1., 0., 0.],\n           [0., 0., 0.],\n           [0., 0., 0.]])\n    >>> find_contours(a, 0.5)\n    [array([[0. , 0.5],\n           [0.5, 0. ]])]\n    '
    if fully_connected not in _param_options:
        raise ValueError('Parameters "fully_connected" must be either "high" or "low".')
    if positive_orientation not in _param_options:
        raise ValueError('Parameters "positive_orientation" must be either "high" or "low".')
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError('Input array must be at least 2x2.')
    if image.ndim != 2:
        raise ValueError('Only 2D arrays are supported.')
    if mask is not None:
        if mask.shape != image.shape:
            raise ValueError('Parameters "array" and "mask" must have same shape.')
        if not np.can_cast(mask.dtype, bool, casting='safe'):
            raise TypeError('Parameter "mask" must be a binary array.')
        mask = mask.astype(np.uint8, copy=False)
    if level is None:
        level = (np.nanmin(image) + np.nanmax(image)) / 2.0
    segments = _get_contour_segments(image.astype(np.float64), float(level), fully_connected == 'high', mask=mask)
    contours = _assemble_contours(segments)
    if positive_orientation == 'high':
        contours = [c[::-1] for c in contours]
    return contours

def _assemble_contours(segments):
    if False:
        while True:
            i = 10
    current_index = 0
    contours = {}
    starts = {}
    ends = {}
    for (from_point, to_point) in segments:
        if from_point == to_point:
            continue
        (tail, tail_num) = starts.pop(to_point, (None, None))
        (head, head_num) = ends.pop(from_point, (None, None))
        if tail is not None and head is not None:
            if tail is head:
                head.append(to_point)
            elif tail_num > head_num:
                head.extend(tail)
                contours.pop(tail_num, None)
                starts[head[0]] = (head, head_num)
                ends[head[-1]] = (head, head_num)
            else:
                tail.extendleft(reversed(head))
                starts.pop(head[0], None)
                contours.pop(head_num, None)
                starts[tail[0]] = (tail, tail_num)
                ends[tail[-1]] = (tail, tail_num)
        elif tail is None and head is None:
            new_contour = deque((from_point, to_point))
            contours[current_index] = new_contour
            starts[from_point] = (new_contour, current_index)
            ends[to_point] = (new_contour, current_index)
            current_index += 1
        elif head is None:
            tail.appendleft(from_point)
            starts[from_point] = (tail, tail_num)
        else:
            head.append(to_point)
            ends[to_point] = (head, head_num)
    return [np.array(contour) for (_, contour) in sorted(contours.items())]