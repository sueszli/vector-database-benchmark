import numpy as np
from . import _spath

def shortest_path(arr, reach=1, axis=-1, output_indexlist=False):
    if False:
        return 10
    'Find the shortest path through an n-d array from one side to another.\n\n    Parameters\n    ----------\n    arr : ndarray of float64\n    reach : int, optional\n        By default (``reach = 1``), the shortest path can only move\n        one row up or down for every step it moves forward (i.e.,\n        the path gradient is limited to 1). `reach` defines the\n        number of elements that can be skipped along each non-axis\n        dimension at each step.\n    axis : int, optional\n        The axis along which the path must always move forward (default -1)\n    output_indexlist : bool, optional\n        See return value `p` for explanation.\n\n    Returns\n    -------\n    p : iterable of int\n        For each step along `axis`, the coordinate of the shortest path.\n        If `output_indexlist` is True, then the path is returned as a list of\n        n-d tuples that index into `arr`. If False, then the path is returned\n        as an array listing the coordinates of the path along the non-axis\n        dimensions for each step along the axis dimension. That is,\n        `p.shape == (arr.shape[axis], arr.ndim-1)` except that p is squeezed\n        before returning so if `arr.ndim == 2`, then\n        `p.shape == (arr.shape[axis],)`\n    cost : float\n        Cost of path.  This is the absolute sum of all the\n        differences along the path.\n\n    '
    if axis < 0:
        axis += arr.ndim
    offset_ind_shape = (2 * reach + 1,) * (arr.ndim - 1)
    offset_indices = np.indices(offset_ind_shape) - reach
    offset_indices = np.insert(offset_indices, axis, np.ones(offset_ind_shape), axis=0)
    offset_size = np.multiply.reduce(offset_ind_shape)
    offsets = np.reshape(offset_indices, (arr.ndim, offset_size), order='F').T
    non_axis_shape = arr.shape[:axis] + arr.shape[axis + 1:]
    non_axis_indices = np.indices(non_axis_shape)
    non_axis_size = np.multiply.reduce(non_axis_shape)
    start_indices = np.insert(non_axis_indices, axis, np.zeros(non_axis_shape), axis=0)
    starts = np.reshape(start_indices, (arr.ndim, non_axis_size), order='F').T
    end_indices = np.insert(non_axis_indices, axis, np.full(non_axis_shape, -1, dtype=non_axis_indices.dtype), axis=0)
    ends = np.reshape(end_indices, (arr.ndim, non_axis_size), order='F').T
    m = _spath.MCP_Diff(arr, offsets=offsets)
    (costs, traceback) = m.find_costs(starts, ends, find_all_ends=False)
    for end in ends:
        cost = costs[tuple(end)]
        if cost != np.inf:
            break
    traceback = m.traceback(end)
    if not output_indexlist:
        traceback = np.array(traceback)
        traceback = np.concatenate([traceback[:, :axis], traceback[:, axis + 1:]], axis=1)
        traceback = np.squeeze(traceback)
    return (traceback, cost)