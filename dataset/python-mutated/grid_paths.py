import numpy as np
from matplotlib.lines import Path
from astropy.coordinates import angular_separation
ROUND_TRIP_RTOL = 1.0
DISCONT_FACTOR = 10.0

def get_lon_lat_path(lon_lat, pixel, lon_lat_check):
    if False:
        return 10
    '\n    Draw a curve, taking into account discontinuities.\n\n    Parameters\n    ----------\n    lon_lat : ndarray\n        The longitude and latitude values along the curve, given as a (n,2)\n        array.\n    pixel : ndarray\n        The pixel coordinates corresponding to ``lon_lat``\n    lon_lat_check : ndarray\n        The world coordinates derived from converting from ``pixel``, which is\n        used to ensure round-tripping.\n    '
    sep = angular_separation(np.radians(lon_lat[:, 0]), np.radians(lon_lat[:, 1]), np.radians(lon_lat_check[:, 0]), np.radians(lon_lat_check[:, 1]))
    scale_size = angular_separation(*np.radians(lon_lat[0, :]), *np.radians(lon_lat[1, :]))
    with np.errstate(invalid='ignore'):
        sep[sep > np.pi] -= 2.0 * np.pi
        mask = np.abs(sep > ROUND_TRIP_RTOL * scale_size)
    mask = mask | np.isnan(pixel[:, 0]) | np.isnan(pixel[:, 1])
    codes = np.zeros(lon_lat.shape[0], dtype=np.uint8)
    codes[:] = Path.LINETO
    codes[0] = Path.MOVETO
    codes[mask] = Path.MOVETO
    codes[1:][mask[:-1]] = Path.MOVETO
    step = np.sqrt((pixel[1:, 0] - pixel[:-1, 0]) ** 2 + (pixel[1:, 1] - pixel[:-1, 1]) ** 2)
    discontinuous = step[1:] > DISCONT_FACTOR * step[:-1]
    codes[2:][discontinuous] = Path.MOVETO
    if step[0] > DISCONT_FACTOR * step[1]:
        codes[1] = Path.MOVETO
    path = Path(pixel, codes=codes)
    return path

def get_gridline_path(world, pixel):
    if False:
        return 10
    '\n    Draw a grid line.\n\n    Parameters\n    ----------\n    world : ndarray\n        The longitude and latitude values along the curve, given as a (n,2)\n        array.\n    pixel : ndarray\n        The pixel coordinates corresponding to ``lon_lat``\n    '
    mask = np.isnan(pixel[:, 0]) | np.isnan(pixel[:, 1])
    codes = np.zeros(world.shape[0], dtype=np.uint8)
    codes[:] = Path.LINETO
    codes[0] = Path.MOVETO
    codes[mask] = Path.MOVETO
    codes[1:][mask[:-1]] = Path.MOVETO
    path = Path(pixel, codes=codes)
    return path