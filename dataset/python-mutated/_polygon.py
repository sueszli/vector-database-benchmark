import numpy as np
from scipy import signal

def approximate_polygon(coords, tolerance):
    if False:
        return 10
    'Approximate a polygonal chain with the specified tolerance.\n\n    It is based on the Douglas-Peucker algorithm.\n\n    Note that the approximated polygon is always within the convex hull of the\n    original polygon.\n\n    Parameters\n    ----------\n    coords : (K, 2) array\n        Coordinate array.\n    tolerance : float\n        Maximum distance from original points of polygon to approximated\n        polygonal chain. If tolerance is 0, the original coordinate array\n        is returned.\n\n    Returns\n    -------\n    coords : (L, 2) array\n        Approximated polygonal chain where L <= K.\n\n    References\n    ----------\n    .. [1] https://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm\n    '
    if tolerance <= 0:
        return coords
    chain = np.zeros(coords.shape[0], 'bool')
    dists = np.zeros(coords.shape[0])
    chain[0] = True
    chain[-1] = True
    pos_stack = [(0, chain.shape[0] - 1)]
    end_of_chain = False
    while not end_of_chain:
        (start, end) = pos_stack.pop()
        (r0, c0) = coords[start, :]
        (r1, c1) = coords[end, :]
        dr = r1 - r0
        dc = c1 - c0
        segment_angle = -np.arctan2(dr, dc)
        segment_dist = c0 * np.sin(segment_angle) + r0 * np.cos(segment_angle)
        segment_coords = coords[start + 1:end, :]
        segment_dists = dists[start + 1:end]
        dr0 = segment_coords[:, 0] - r0
        dc0 = segment_coords[:, 1] - c0
        dr1 = segment_coords[:, 0] - r1
        dc1 = segment_coords[:, 1] - c1
        projected_lengths0 = dr0 * dr + dc0 * dc
        projected_lengths1 = -dr1 * dr - dc1 * dc
        perp = np.logical_and(projected_lengths0 > 0, projected_lengths1 > 0)
        eucl = np.logical_not(perp)
        segment_dists[perp] = np.abs(segment_coords[perp, 0] * np.cos(segment_angle) + segment_coords[perp, 1] * np.sin(segment_angle) - segment_dist)
        segment_dists[eucl] = np.minimum(np.sqrt(dc0[eucl] ** 2 + dr0[eucl] ** 2), np.sqrt(dc1[eucl] ** 2 + dr1[eucl] ** 2))
        if np.any(segment_dists > tolerance):
            new_end = start + np.argmax(segment_dists) + 1
            pos_stack.append((new_end, end))
            pos_stack.append((start, new_end))
            chain[new_end] = True
        if len(pos_stack) == 0:
            end_of_chain = True
    return coords[chain, :]
_SUBDIVISION_MASKS = {1: ([1, 1], [1, 1]), 2: ([3, 1], [1, 3]), 3: ([1, 6, 1], [0, 4, 4]), 4: ([5, 10, 1], [1, 10, 5]), 5: ([1, 15, 15, 1], [0, 6, 20, 6]), 6: ([7, 35, 21, 1], [1, 21, 35, 7]), 7: ([1, 28, 70, 28, 1], [0, 8, 56, 56, 8])}

def subdivide_polygon(coords, degree=2, preserve_ends=False):
    if False:
        i = 10
        return i + 15
    'Subdivision of polygonal curves using B-Splines.\n\n    Note that the resulting curve is always within the convex hull of the\n    original polygon. Circular polygons stay closed after subdivision.\n\n    Parameters\n    ----------\n    coords : (K, 2) array\n        Coordinate array.\n    degree : {1, 2, 3, 4, 5, 6, 7}, optional\n        Degree of B-Spline. Default is 2.\n    preserve_ends : bool, optional\n        Preserve first and last coordinate of non-circular polygon. Default is\n        False.\n\n    Returns\n    -------\n    coords : (L, 2) array\n        Subdivided coordinate array.\n\n    References\n    ----------\n    .. [1] http://mrl.nyu.edu/publications/subdiv-course2000/coursenotes00.pdf\n    '
    if degree not in _SUBDIVISION_MASKS:
        raise ValueError('Invalid B-Spline degree. Only degree 1 - 7 is supported.')
    circular = np.all(coords[0, :] == coords[-1, :])
    method = 'valid'
    if circular:
        coords = coords[:-1, :]
        method = 'same'
    (mask_even, mask_odd) = _SUBDIVISION_MASKS[degree]
    mask_even = np.array(mask_even, float) / 2 ** degree
    mask_odd = np.array(mask_odd, float) / 2 ** degree
    even = signal.convolve2d(coords.T, np.atleast_2d(mask_even), mode=method, boundary='wrap')
    odd = signal.convolve2d(coords.T, np.atleast_2d(mask_odd), mode=method, boundary='wrap')
    out = np.zeros((even.shape[1] + odd.shape[1], 2))
    out[1::2] = even.T
    out[::2] = odd.T
    if circular:
        out = np.vstack([out, out[0, :]])
    if preserve_ends and (not circular):
        out = np.vstack([coords[0, :], out, coords[-1, :]])
    return out