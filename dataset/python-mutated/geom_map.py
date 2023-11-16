from __future__ import annotations
import typing
import numpy as np
import pandas as pd
from ..doctools import document
from ..exceptions import PlotnineError
from ..utils import SIZE_FACTOR, to_rgba
from .geom import geom
from .geom_point import geom_point
from .geom_polygon import geom_polygon
if typing.TYPE_CHECKING:
    from typing import Any
    import numpy.typing as npt
    from shapely.geometry.polygon import LinearRing, Polygon
    from plotnine.iapi import panel_view
    from plotnine.typing import Aes, Axes, Coord, DataLike, DrawingArea, Layer, PathPatch

@document
class geom_map(geom):
    """
    Draw map feature

    The map feature are drawn without any special projections.

    {usage}

    Parameters
    ----------
    {common_parameters}

    Notes
    -----
    This geom is best suited for plotting a shapefile read into
    geopandas dataframe. The dataframe should have a ``geometry``
    column.
    """
    DEFAULT_AES = {'alpha': 1, 'color': '#111111', 'fill': '#333333', 'linetype': 'solid', 'shape': 'o', 'size': 0.5, 'stroke': 0.5}
    DEFAULT_PARAMS = {'stat': 'identity', 'position': 'identity', 'na_rm': False}
    REQUIRED_AES = {'geometry'}

    def __init__(self, mapping: Aes | None=None, data: DataLike | None=None, **kwargs: Any):
        if False:
            i = 10
            return i + 15
        geom.__init__(self, mapping, data, **kwargs)
        if 'geometry' not in self.mapping:
            self.mapping['geometry'] = 'geometry'

    def setup_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if False:
            while True:
                i = 10
        if not len(data):
            return data
        bool_idx = np.array([g is not None for g in data['geometry']])
        if not np.all(bool_idx):
            data = data.loc[bool_idx]
        try:
            bounds = data['geometry'].bounds
        except AttributeError:
            bounds = pd.DataFrame(np.array([x.bounds for x in data['geometry']]), columns=['xmin', 'ymin', 'xmax', 'ymax'], index=data.index)
        else:
            bounds.rename(columns={'minx': 'xmin', 'maxx': 'xmax', 'miny': 'ymin', 'maxy': 'ymax'}, inplace=True)
        data = pd.concat([data, bounds], axis=1)
        return data

    def draw_panel(self, data: pd.DataFrame, panel_params: panel_view, coord: Coord, ax: Axes, **params: Any):
        if False:
            for i in range(10):
                print('nop')
        if not len(data):
            return
        data.loc[data['color'].isna(), 'color'] = 'none'
        data.loc[data['fill'].isna(), 'fill'] = 'none'
        data['fill'] = to_rgba(data['fill'], data['alpha'])
        geom_type = data.geometry.iloc[0].geom_type
        if geom_type in ('Polygon', 'MultiPolygon'):
            from matplotlib.collections import PatchCollection
            data['size'] *= SIZE_FACTOR
            patches = [PolygonPatch(g) for g in data['geometry']]
            coll = PatchCollection(patches, edgecolor=data['color'], facecolor=data['fill'], linestyle=data['linetype'], linewidth=data['size'], zorder=params['zorder'], rasterized=params['raster'])
            ax.add_collection(coll)
        elif geom_type == 'Point':
            arr = np.array([list(g.coords)[0] for g in data['geometry']])
            data['x'] = arr[:, 0]
            data['y'] = arr[:, 1]
            for (_, gdata) in data.groupby('group'):
                gdata.reset_index(inplace=True, drop=True)
                gdata.is_copy = None
                geom_point.draw_group(gdata, panel_params, coord, ax, **params)
        elif geom_type == 'MultiPoint':
            data['points'] = [[p.coords[0] for p in mp.geoms] for mp in data['geometry']]
            data = data.explode('points', ignore_index=True)
            data['x'] = [p[0] for p in data['points']]
            data['y'] = [p[1] for p in data['points']]
            geom_point.draw_group(data, panel_params, coord, ax, **params)
        elif geom_type in ('LineString', 'MultiLineString'):
            from matplotlib.collections import LineCollection
            data['size'] *= SIZE_FACTOR
            data['color'] = to_rgba(data['color'], data['alpha'])
            segments = []
            for g in data['geometry']:
                if g.geom_type == 'LineString':
                    segments.append(g.coords)
                else:
                    segments.extend((_g.coords for _g in g.geoms))
            coll = LineCollection(segments, edgecolor=data['color'], linewidth=data['size'], linestyle=data['linetype'], zorder=params['zorder'], rasterized=params['raster'])
            ax.add_collection(coll)
        else:
            raise TypeError(f"Could not plot geometry of type '{geom_type}'")

    @staticmethod
    def draw_legend(data: pd.Series[Any], da: DrawingArea, lyr: Layer) -> DrawingArea:
        if False:
            for i in range(10):
                print('nop')
        '\n        Draw a rectangle in the box\n\n        Parameters\n        ----------\n        data : Series\n            Data Row\n        da : DrawingArea\n            Canvas\n        lyr : layer\n            Layer\n\n        Returns\n        -------\n        out : DrawingArea\n        '
        data['size'] = data['stroke']
        del data['stroke']
        return geom_polygon.draw_legend(data, da, lyr)

def PolygonPatch(obj: Polygon) -> PathPatch:
    if False:
        while True:
            i = 10
    '\n    Return a Matplotlib patch from a Polygon/MultiPolygon Geometry\n\n    Parameters\n    ----------\n    obj : shapley.geometry.Polygon | shapley.geometry.MultiPolygon\n        A Polygon or MultiPolygon to create a patch for description\n\n    Returns\n    -------\n    result : matplotlib.patches.PathPatch\n        A patch representing the shapely geometry\n\n    Notes\n    -----\n    This functionality was originally provided by the descartes package\n    by Sean Gillies (BSD license, https://pypi.org/project/descartes)\n    which is nolonger being maintained.\n    '
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    def cw_coords(ring: LinearRing) -> npt.NDArray[Any]:
        if False:
            print('Hello World!')
        '\n        Return Clockwise array coordinates\n\n        Parameters\n        ----------\n        ring: shapely.geometry.polygon.LinearRing\n            LinearRing\n\n        Returns\n        -------\n        out: ndarray\n            (n x 2) array of coordinate points.\n        '
        if ring.is_ccw:
            return np.asarray(ring.coords)[:, :2][::-1]
        return np.asarray(ring.coords)[:, :2]

    def ccw_coords(ring: LinearRing) -> npt.NDArray[Any]:
        if False:
            i = 10
            return i + 15
        '\n        Return Counter Clockwise array coordinates\n\n        Parameters\n        ----------\n        ring: shapely.geometry.polygon.LinearRing\n            LinearRing\n\n        Returns\n        -------\n        out: ndarray\n            (n x 2) array of coordinate points.\n        '
        if ring.is_ccw:
            return np.asarray(ring.coords)[:, :2]
        return np.asarray(ring.coords)[:, :2][::-1]
    if obj.geom_type == 'Polygon':
        _exterior = [Path(cw_coords(obj.exterior))]
        _interior = [Path(ccw_coords(ring)) for ring in obj.interiors]
    else:
        _exterior = []
        _interior = []
        for p in obj.geoms:
            _exterior.append(Path(cw_coords(p.exterior)))
            _interior.extend([Path(ccw_coords(ring)) for ring in p.interiors])
    path = Path.make_compound_path(*_exterior, *_interior)
    return PathPatch(path)

def check_geopandas():
    if False:
        for i in range(10):
            print('nop')
    try:
        import geopandas
    except ImportError:
        raise PlotnineError('geom_map requires geopandas. Please install geopandas.')