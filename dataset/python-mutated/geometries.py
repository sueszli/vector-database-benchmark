"""
Do not use: deprecated.

The `geometries` module has been renamed the `features` module. The
`geometries` module is deprecated and will be removed in a future release.
"""
from warnings import warn
from . import features
DEP_MSG = 'The `geometries` module and `geometries_from_X` functions have been renamed the `features` module and `features_from_X` functions. Use these instead. The `geometries` module and function names are deprecated and will be removed in a future release.'

def geometries_from_bbox(north, south, east, west, tags):
    if False:
        print('Hello World!')
    '\n    Do not use: deprecated.\n\n    The `geometries` module and `geometries_from_X` functions have been\n    renamed the `features` module and `features_from_X` functions. Use these\n    instead. The `geometries` module and functions are deprecated and will be\n    removed in a future release.\n\n    Parameters\n    ----------\n    north : float\n        Do not use: deprecated.\n    south : float\n        Do not use: deprecated.\n    east : float\n        Do not use: deprecated.\n    west : float\n        Do not use: deprecated.\n    tags : dict\n        Do not use: deprecated.\n\n    Returns\n    -------\n    gdf : geopandas.GeoDataFrame\n    '
    warn(DEP_MSG, stacklevel=2)
    return features.features_from_bbox(north, south, east, west, tags)

def geometries_from_point(center_point, tags, dist=1000):
    if False:
        i = 10
        return i + 15
    '\n    Do not use: deprecated.\n\n    The `geometries` module and `geometries_from_X` functions have been\n    renamed the `features` module and `features_from_X` functions. Use these\n    instead. The `geometries` module and functions are deprecated and will be\n    removed in a future release.\n\n    Parameters\n    ----------\n    center_point : tuple\n        Do not use: deprecated.\n    tags : dict\n        Do not use: deprecated.\n    dist : numeric\n        Do not use: deprecated.\n\n    Returns\n    -------\n    gdf : geopandas.GeoDataFrame\n    '
    warn(DEP_MSG, stacklevel=2)
    return features.features_from_point(center_point, tags, dist)

def geometries_from_address(address, tags, dist=1000):
    if False:
        for i in range(10):
            print('nop')
    '\n    Do not use: deprecated.\n\n    The `geometries` module and `geometries_from_X` functions have been\n    renamed the `features` module and `features_from_X` functions. Use these\n    instead. The `geometries` module and functions are deprecated and will be\n    removed in a future release.\n\n    Parameters\n    ----------\n    address : string\n        Do not use: deprecated.\n    tags : dict\n        Do not use: deprecated.\n    dist : numeric\n        Do not use: deprecated.\n\n    Returns\n    -------\n    gdf : geopandas.GeoDataFrame\n    '
    warn(DEP_MSG, stacklevel=2)
    return features.features_from_address(address, tags, dist)

def geometries_from_place(query, tags, which_result=None, buffer_dist=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Do not use: deprecated.\n\n    The `geometries` module and `geometries_from_X` functions have been\n    renamed the `features` module and `features_from_X` functions. Use these\n    instead. The `geometries` module and functions are deprecated and will be\n    removed in a future release.\n\n    Parameters\n    ----------\n    query : string or dict or list\n        Do not use: deprecated.\n    tags : dict\n        Do not use: deprecated.\n    which_result : int\n        Do not use: deprecated.\n    buffer_dist : float\n        Do not use: deprecated.\n\n    Returns\n    -------\n    gdf : geopandas.GeoDataFrame\n    '
    warn(DEP_MSG, stacklevel=2)
    return features.features_from_place(query, tags, which_result, buffer_dist)

def geometries_from_polygon(polygon, tags):
    if False:
        print('Hello World!')
    '\n    Do not use: deprecated.\n\n    The `geometries` module and `geometries_from_X` functions have been\n    renamed the `features` module and `features_from_X` functions. Use these\n    instead. The `geometries` module and functions are deprecated and will be\n    removed in a future release.\n\n    Parameters\n    ----------\n    polygon : shapely.geometry.Polygon or shapely.geometry.MultiPolygon\n        Do not use: deprecated.\n    tags : dict\n        Do not use: deprecated.\n\n    Returns\n    -------\n    gdf : geopandas.GeoDataFrame\n    '
    warn(DEP_MSG, stacklevel=2)
    return features.features_from_polygon(polygon, tags)

def geometries_from_xml(filepath, polygon=None, tags=None):
    if False:
        i = 10
        return i + 15
    '\n    Do not use: deprecated.\n\n    The `geometries` module and `geometries_from_X` functions have been\n    renamed the `features` module and `features_from_X` functions. Use these\n    instead. The `geometries` module and functions are deprecated and will be\n    removed in a future release.\n\n    Parameters\n    ----------\n    filepath : string or pathlib.Path\n        Do not use: deprecated.\n    polygon : shapely.geometry.Polygon\n        Do not use: deprecated.\n    tags : dict\n        Do not use: deprecated.\n\n    Returns\n    -------\n    gdf : geopandas.GeoDataFrame\n    '
    warn(DEP_MSG, stacklevel=2)
    return features.features_from_xml(filepath, polygon, tags)