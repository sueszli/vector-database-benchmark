import warnings
import numpy as np
from astropy import units as u
from astropy.utils.exceptions import AstropyDeprecationWarning
LONLAT = {'longitude', 'latitude'}

def wrap_180(values):
    if False:
        while True:
            i = 10
    values_new = values % 360.0
    with np.errstate(invalid='ignore'):
        values_new[values_new > 180.0] -= 360
    return values_new

def find_coordinate_range(transform, extent, coord_types, coord_units, coord_wraps):
    if False:
        print('Hello World!')
    "\n    Find the range of coordinates to use for ticks/grids.\n\n    Parameters\n    ----------\n    transform : func\n        Function to transform pixel to world coordinates. Should take two\n        values (the pixel coordinates) and return two values (the world\n        coordinates).\n    extent : iterable\n        The range of the image viewport in pixel coordinates, given as [xmin,\n        xmax, ymin, ymax].\n    coord_types : list of str\n        Whether each coordinate is a ``'longitude'``, ``'latitude'``, or\n        ``'scalar'`` value.\n    coord_units : list of `astropy.units.Unit`\n        The units for each coordinate.\n    coord_wraps : list of `astropy.units.Quantity`\n        The wrap angles for longitudes.\n    "
    from . import conf
    if len(extent) == 4:
        nx = ny = conf.coordinate_range_samples
        x = np.linspace(extent[0], extent[1], nx + 1)
        y = np.linspace(extent[2], extent[3], ny + 1)
        (xp, yp) = np.meshgrid(x, y)
        with np.errstate(invalid='ignore'):
            world = transform.transform(np.vstack([xp.ravel(), yp.ravel()]).transpose())
    else:
        nx = conf.coordinate_range_samples
        xp = np.linspace(extent[0], extent[1], nx + 1)[None]
        with np.errstate(invalid='ignore'):
            world = transform.transform(xp.T)
    ranges = []
    for (coord_index, coord_type) in enumerate(coord_types):
        xw = world[:, coord_index].reshape(xp.shape)
        if coord_type in LONLAT:
            unit = coord_units[coord_index]
            xw = xw * unit.to(u.deg)
            wjump = xw[0, 1:] - xw[0, :-1]
            with np.errstate(invalid='ignore'):
                reset = np.abs(wjump) > 180.0
            if np.any(reset):
                wjump = wjump + np.sign(wjump) * 180.0
                wjump = 360.0 * np.trunc(wjump / 360.0)
                xw[0, 1:][reset] -= wjump[reset]
            wjump = xw[1:] - xw[:1]
            with np.errstate(invalid='ignore'):
                reset = np.abs(wjump) > 180.0
            if np.any(reset):
                wjump = wjump + np.sign(wjump) * 180.0
                wjump = 360.0 * np.trunc(wjump / 360.0)
                xw[1:][reset] -= wjump[reset]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            xw_min = np.nanmin(xw)
            xw_max = np.nanmax(xw)
        if coord_type in LONLAT:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                xw_min_check = np.nanmin(xw % 360.0)
                xw_max_check = np.nanmax(xw % 360.0)
            if xw_max_check - xw_min_check <= xw_max - xw_min < 360.0:
                xw_min = xw_min_check
                xw_max = xw_max_check
        if coord_type in LONLAT:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                xw_min_check = np.nanmin(wrap_180(xw))
                xw_max_check = np.nanmax(wrap_180(xw))
            if xw_max_check - xw_min_check < 360.0 and xw_max - xw_min >= xw_max_check - xw_min_check:
                xw_min = xw_min_check
                xw_max = xw_max_check
        x_range = xw_max - xw_min
        if coord_type == 'longitude':
            if x_range > 300.0:
                coord_wrap = coord_wraps[coord_index]
                if not isinstance(coord_wrap, u.Quantity):
                    warnings.warn("Passing 'coord_wraps' as numbers is deprecated. Use a Quantity with units convertible to angular degrees instead.", AstropyDeprecationWarning)
                    coord_wrap = coord_wrap * u.deg
                xw_min = coord_wrap.to_value(u.deg) - 360
                xw_max = coord_wrap.to_value(u.deg) - np.spacing(360.0)
            elif xw_min < 0.0:
                xw_min = max(-180.0, xw_min - 0.1 * x_range)
                xw_max = min(+180.0, xw_max + 0.1 * x_range)
            else:
                xw_min = max(0.0, xw_min - 0.1 * x_range)
                xw_max = min(360.0, xw_max + 0.1 * x_range)
        elif coord_type == 'latitude':
            xw_min = max(-90.0, xw_min - 0.1 * x_range)
            xw_max = min(+90.0, xw_max + 0.1 * x_range)
        if coord_type in LONLAT:
            xw_min *= u.deg.to(unit)
            xw_max *= u.deg.to(unit)
        ranges.append((xw_min, xw_max))
    return ranges