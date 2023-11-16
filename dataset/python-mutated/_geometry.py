__all__ = ['polygon_clip', 'polygon_area']
import numpy as np
from .version_requirements import require

@require('matplotlib', '>=3.3')
def polygon_clip(rp, cp, r0, c0, r1, c1):
    if False:
        print('Hello World!')
    'Clip a polygon to the given bounding box.\n\n    Parameters\n    ----------\n    rp, cp : (K,) ndarray of double\n        Row and column coordinates of the polygon.\n    (r0, c0), (r1, c1) : double\n        Top-left and bottom-right coordinates of the bounding box.\n\n    Returns\n    -------\n    r_clipped, c_clipped : (L,) ndarray of double\n        Coordinates of clipped polygon.\n\n    Notes\n    -----\n    This makes use of Sutherland-Hodgman clipping as implemented in\n    AGG 2.4 and exposed in Matplotlib.\n\n    '
    from matplotlib import path, transforms
    poly = path.Path(np.vstack((rp, cp)).T, closed=True)
    clip_rect = transforms.Bbox([[r0, c0], [r1, c1]])
    poly_clipped = poly.clip_to_bbox(clip_rect).to_polygons()[0]
    return (poly_clipped[:, 0], poly_clipped[:, 1])

def polygon_area(pr, pc):
    if False:
        print('Hello World!')
    'Compute the area of a polygon.\n\n    Parameters\n    ----------\n    pr, pc : (K,) array of float\n        Polygon row and column coordinates.\n\n    Returns\n    -------\n    a : float\n        Area of the polygon.\n    '
    pr = np.asarray(pr)
    pc = np.asarray(pc)
    return 0.5 * np.abs(np.sum(pc[:-1] * pr[1:] - pc[1:] * pr[:-1]))