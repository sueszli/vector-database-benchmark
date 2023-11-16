""" Functions useful for dealing with hexagonal tilings.

For more information on the concepts employed here, see this informative page

    https://www.redblobgames.com/grids/hexagons/

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import Any
import numpy as np
from .dependencies import import_required
__all__ = ('axial_to_cartesian', 'cartesian_to_axial', 'hexbin')

def axial_to_cartesian(q: Any, r: Any, size: float, orientation: str, aspect_scale: float=1) -> tuple[Any, Any]:
    if False:
        return 10
    ' Map axial *(q,r)* coordinates to cartesian *(x,y)* coordinates of\n    tiles centers.\n\n    This function can be useful for positioning other Bokeh glyphs with\n    cartesian coordinates in relation to a hex tiling.\n\n    This function was adapted from:\n\n    https://www.redblobgames.com/grids/hexagons/#hex-to-pixel\n\n    Args:\n        q (array[float]) :\n            A NumPy array of q-coordinates for binning\n\n        r (array[float]) :\n            A NumPy array of r-coordinates for binning\n\n        size (float) :\n            The size of the hexagonal tiling.\n\n            The size is defined as the distance from the center of a hexagon\n            to the top corner for "pointytop" orientation, or from the center\n            to a side corner for "flattop" orientation.\n\n        orientation (str) :\n            Whether the hex tile orientation should be "pointytop" or\n            "flattop".\n\n        aspect_scale (float, optional) :\n            Scale the hexagons in the "cross" dimension.\n\n            For "pointytop" orientations, hexagons are scaled in the horizontal\n            direction. For "flattop", they are scaled in vertical direction.\n\n            When working with a plot with ``aspect_scale != 1``, it may be\n            useful to set this value to match the plot.\n\n    Returns:\n        (array[int], array[int])\n\n    '
    if orientation == 'pointytop':
        x = size * np.sqrt(3) * (q + r / 2.0) / aspect_scale
        y = -size * 3 / 2.0 * r
    else:
        x = size * 3 / 2.0 * q
        y = -size * np.sqrt(3) * (r + q / 2.0) * aspect_scale
    return (x, y)

def cartesian_to_axial(x: Any, y: Any, size: float, orientation: str, aspect_scale: float=1) -> tuple[Any, Any]:
    if False:
        while True:
            i = 10
    ' Map Cartesion *(x,y)* points to axial *(q,r)* coordinates of enclosing\n    tiles.\n\n    This function was adapted from:\n\n    https://www.redblobgames.com/grids/hexagons/#pixel-to-hex\n\n    Args:\n        x (array[float]) :\n            A NumPy array of x-coordinates to convert\n\n        y (array[float]) :\n            A NumPy array of y-coordinates to convert\n\n        size (float) :\n            The size of the hexagonal tiling.\n\n            The size is defined as the distance from the center of a hexagon\n            to the top corner for "pointytop" orientation, or from the center\n            to a side corner for "flattop" orientation.\n\n        orientation (str) :\n            Whether the hex tile orientation should be "pointytop" or\n            "flattop".\n\n        aspect_scale (float, optional) :\n            Scale the hexagons in the "cross" dimension.\n\n            For "pointytop" orientations, hexagons are scaled in the horizontal\n            direction. For "flattop", they are scaled in vertical direction.\n\n            When working with a plot with ``aspect_scale != 1``, it may be\n            useful to set this value to match the plot.\n\n    Returns:\n        (array[int], array[int])\n\n    '
    HEX_FLAT = [2.0 / 3.0, 0.0, -1.0 / 3.0, np.sqrt(3.0) / 3.0]
    HEX_POINTY = [np.sqrt(3.0) / 3.0, -1.0 / 3.0, 0.0, 2.0 / 3.0]
    coords = HEX_FLAT if orientation == 'flattop' else HEX_POINTY
    x = x / size * (aspect_scale if orientation == 'pointytop' else 1)
    y = -y / size / (aspect_scale if orientation == 'flattop' else 1)
    q = coords[0] * x + coords[1] * y
    r = coords[2] * x + coords[3] * y
    return _round_hex(q, r)

def hexbin(x: Any, y: Any, size: float, orientation: str='pointytop', aspect_scale: float=1) -> Any:
    if False:
        for i in range(10):
            print('nop')
    ' Perform an equal-weight binning of data points into hexagonal tiles.\n\n    For more sophisticated use cases, e.g. weighted binning or scaling\n    individual tiles proportional to some other quantity, consider using\n    HoloViews.\n\n    Args:\n        x (array[float]) :\n            A NumPy array of x-coordinates for binning\n\n        y (array[float]) :\n            A NumPy array of y-coordinates for binning\n\n        size (float) :\n            The size of the hexagonal tiling.\n\n            The size is defined as the distance from the center of a hexagon\n            to the top corner for "pointytop" orientation, or from the center\n            to a side corner for "flattop" orientation.\n\n        orientation (str, optional) :\n            Whether the hex tile orientation should be "pointytop" or\n            "flattop". (default: "pointytop")\n\n        aspect_scale (float, optional) :\n            Match a plot\'s aspect ratio scaling.\n\n            When working with a plot with ``aspect_scale != 1``, this\n            parameter can be set to match the plot, in order to draw\n            regular hexagons (instead of "stretched" ones).\n\n            This is roughly equivalent to binning in "screen space", and\n            it may be better to use axis-aligned rectangular bins when\n            plot aspect scales are not one.\n\n    Returns:\n        DataFrame\n\n        The resulting DataFrame will have columns *q* and *r* that specify\n        hexagon tile locations in axial coordinates, and a column *counts* that\n        provides the count for each tile.\n\n    .. warning::\n        Hex binning only functions on linear scales, i.e. not on log plots.\n\n    '
    pd: Any = import_required('pandas', 'hexbin requires pandas to be installed')
    (q, r) = cartesian_to_axial(x, y, size, orientation, aspect_scale=aspect_scale)
    df = pd.DataFrame(dict(r=r, q=q))
    return df.groupby(['q', 'r']).size().reset_index(name='counts')

def _round_hex(q: Any, r: Any) -> tuple[Any, Any]:
    if False:
        print('Hello World!')
    ' Round floating point axial hex coordinates to integer *(q,r)*\n    coordinates.\n\n    This code was adapted from:\n\n        https://www.redblobgames.com/grids/hexagons/#rounding\n\n    Args:\n        q (array[float]) :\n            NumPy array of Floating point axial *q* coordinates to round\n\n        r (array[float]) :\n            NumPy array of Floating point axial *q* coordinates to round\n\n    Returns:\n        (array[int], array[int])\n\n    '
    x = q
    z = r
    y = -x - z
    rx = np.round(x)
    ry = np.round(y)
    rz = np.round(z)
    dx = np.abs(rx - x)
    dy = np.abs(ry - y)
    dz = np.abs(rz - z)
    cond = (dx > dy) & (dx > dz)
    q = np.where(cond, -(ry + rz), rx)
    r = np.where(~cond & ~(dy > dz), -(rx + ry), rz)
    return (q.astype(int), r.astype(int))