from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING, Sequence, Union
import numpy as np
from ..core.property_mixins import FillProps, HatchProps, LineProps
from ..models.glyphs import MultiLine, MultiPolygons
from ..models.renderers import ContourRenderer, GlyphRenderer
from ..models.sources import ColumnDataSource
from ..palettes import interp_palette
from ..plotting._renderer import _process_sequence_literals
from ..util.dataclasses import dataclass, entries
if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from typing_extensions import TypeAlias
    from ..palettes import Palette, PaletteCollection
    from ..transform import ColorLike
    ContourColor: TypeAlias = Union[ColorLike, Sequence[ColorLike]]
    ContourColorOrPalette: TypeAlias = Union[ContourColor, Palette, PaletteCollection, ContourColor]
__all__ = ('contour_data', 'from_contour')

@dataclass(frozen=True)
class FillCoords:
    """ Coordinates for all filled polygons over a whole sequence of contour levels.
    """
    xs: list[list[list[np.ndarray]]]
    ys: list[list[list[np.ndarray]]]

@dataclass(frozen=True)
class LineCoords:
    """ Coordinates for all contour lines over a whole sequence of contour levels.
    """
    xs: list[np.ndarray]
    ys: list[np.ndarray]

@dataclass(frozen=True)
class ContourCoords:
    """ Combined filled and line contours over a whole sequence of contour levels.
    """
    fill_coords: FillCoords | None
    line_coords: LineCoords | None

@dataclass(frozen=True)
class FillData(FillCoords):
    """ Complete geometry data for filled polygons over a whole sequence of contour levels.
    """
    lower_levels: ArrayLike
    upper_levels: ArrayLike

    def asdict(self):
        if False:
            i = 10
            return i + 15
        return dict(entries(self))

@dataclass(frozen=True)
class LineData(LineCoords):
    """ Complete geometry data for contour lines over a whole sequence of contour levels.
    """
    levels: ArrayLike

    def asdict(self):
        if False:
            i = 10
            return i + 15
        return dict(entries(self))

@dataclass(frozen=True)
class ContourData:
    """ Complete geometry data for filled polygons and/or contour lines over a
    whole sequence of contour levels.

    :func:`~bokeh.plotting.contour.contour_data` returns an object of
    this class that can then be passed to :func:`bokeh.models.ContourRenderer.set_data`.
    """
    fill_data: FillData | None
    line_data: LineData | None

def contour_data(x: ArrayLike | None=None, y: ArrayLike | None=None, z: ArrayLike | np.ma.MaskedArray | None=None, levels: ArrayLike | None=None, *, want_fill: bool=True, want_line: bool=True) -> ContourData:
    if False:
        print('Hello World!')
    ' Return the contour data of filled and/or line contours that can be\n    passed to :func:`bokeh.models.ContourRenderer.set_data`\n    '
    levels = _validate_levels(levels)
    if len(levels) < 2:
        want_fill = False
    if not want_fill and (not want_line):
        raise ValueError('Neither fill nor line requested in contour_data')
    coords = _contour_coords(x, y, z, levels, want_fill, want_line)
    fill_data = None
    if coords.fill_coords:
        fill_coords = coords.fill_coords
        fill_data = FillData(xs=fill_coords.xs, ys=fill_coords.ys, lower_levels=levels[:-1], upper_levels=levels[1:])
    line_data = None
    if coords.line_coords:
        line_coords = coords.line_coords
        line_data = LineData(xs=line_coords.xs, ys=line_coords.ys, levels=levels)
    return ContourData(fill_data, line_data)

def from_contour(x: ArrayLike | None=None, y: ArrayLike | None=None, z: ArrayLike | np.ma.MaskedArray | None=None, levels: ArrayLike | None=None, **visuals) -> ContourRenderer:
    if False:
        while True:
            i = 10
    ' Creates a :class:`bokeh.models.ContourRenderer` containing filled\n    polygons and/or contour lines.\n\n    Usually it is preferable to call :func:`~bokeh.plotting.figure.contour`\n    instead of this function.\n\n    Filled contour polygons are calculated if ``fill_color`` is set,\n    contour lines if ``line_color`` is set.\n\n    Args:\n        x (array-like[float] of shape (ny, nx) or (nx,), optional) :\n            The x-coordinates of the ``z`` values. May be 2D with the same\n            shape as ``z.shape``, or 1D with length ``nx = z.shape[1]``.\n            If not specified are assumed to be ``np.arange(nx)``. Must be\n            ordered monotonically.\n\n        y (array-like[float] of shape (ny, nx) or (ny,), optional) :\n            The y-coordinates of the ``z`` values. May be 2D with the same\n            shape as ``z.shape``, or 1D with length ``ny = z.shape[0]``.\n            If not specified are assumed to be ``np.arange(ny)``. Must be\n            ordered monotonically.\n\n        z (array-like[float] of shape (ny, nx)) :\n            A 2D NumPy array of gridded values to calculate the contours\n            of.  May be a masked array, and any invalid values (``np.inf``\n            or ``np.nan``) will also be masked out.\n\n        levels (array-like[float]) :\n            The z-levels to calculate the contours at, must be increasing.\n            Contour lines are calculated at each level and filled contours\n            are calculated between each adjacent pair of levels so the\n            number of sets of contour lines is ``len(levels)`` and the\n            number of sets of filled contour polygons is ``len(levels)-1``.\n\n        **visuals: |fill properties|, |hatch properties| and |line properties|\n            Fill and hatch properties are used for filled contours, line\n            properties for line contours. If using vectorized properties\n            then the correct number must be used, ``len(levels)`` for line\n            properties and ``len(levels)-1`` for fill and hatch properties.\n\n            ``fill_color`` and ``line_color`` are more flexible in that\n            they will accept longer sequences and interpolate them to the\n            required number using :func:`~bokeh.palettes.linear_palette`,\n            and also accept palette collections (dictionaries mapping from\n            integer length to color sequence) such as\n            `bokeh.palettes.Cividis`.\n\n    '
    levels = _validate_levels(levels)
    if len(levels) < 2:
        want_fill = False
    nlevels = len(levels)
    want_line = visuals.get('line_color', None) is not None
    if want_line:
        visuals['line_color'] = _color(visuals['line_color'], nlevels)
        line_cds = ColumnDataSource()
        _process_sequence_literals(MultiLine, visuals, line_cds, False)
        line_visuals = {}
        for name in LineProps.properties():
            prop = visuals.pop(name, None)
            if prop is not None:
                line_visuals[name] = prop
    else:
        visuals.pop('line_color', None)
    want_fill = visuals.get('fill_color', None) is not None
    if want_fill:
        visuals['fill_color'] = _color(visuals['fill_color'], nlevels - 1)
        if 'hatch_color' in visuals:
            visuals['hatch_color'] = _color(visuals['hatch_color'], nlevels - 1)
        fill_cds = ColumnDataSource()
        _process_sequence_literals(MultiPolygons, visuals, fill_cds, False)
    else:
        visuals.pop('fill_color', None)
    unknown = visuals.keys() - FillProps.properties() - HatchProps.properties()
    if unknown:
        raise ValueError(f"Unknown keyword arguments in 'from_contour': {', '.join(unknown)}")
    new_contour_data = contour_data(x=x, y=y, z=z, levels=levels, want_fill=want_fill, want_line=want_line)
    contour_renderer = ContourRenderer(fill_renderer=GlyphRenderer(glyph=MultiPolygons(), data_source=ColumnDataSource()), line_renderer=GlyphRenderer(glyph=MultiLine(), data_source=ColumnDataSource()), levels=list(levels))
    contour_renderer.set_data(new_contour_data)
    if new_contour_data.fill_data:
        glyph = contour_renderer.fill_renderer.glyph
        for (name, value) in visuals.items():
            setattr(glyph, name, value)
        cds = contour_renderer.fill_renderer.data_source
        for (name, value) in fill_cds.data.items():
            cds.add(value, name)
        glyph.line_alpha = 0
        glyph.line_width = 0
    if new_contour_data.line_data:
        glyph = contour_renderer.line_renderer.glyph
        for (name, value) in line_visuals.items():
            setattr(glyph, name, value)
        cds = contour_renderer.line_renderer.data_source
        for (name, value) in line_cds.data.items():
            cds.add(value, name)
    return contour_renderer

@dataclass(frozen=True)
class SingleFillCoords:
    """ Coordinates for filled contour polygons between a lower and upper level.

    The first list contains a list for each polygon. The second list contains
    a separate NumPy array for each boundary of that polygon; the first array
    is always the outer boundary, subsequent arrays are holes.
    """
    xs: list[list[np.ndarray]]
    ys: list[list[np.ndarray]]

@dataclass(frozen=True)
class SingleLineCoords:
    """ Coordinates for contour lines at a single contour level.

    The x and y coordinates are stored in a single NumPy array each, with a
    np.nan separating each line.
    """
    xs: np.ndarray
    ys: np.ndarray

def _color(color: ContourColorOrPalette, n: int) -> ContourColor:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(color, dict):
        return _palette_from_collection(color, n)
    if isinstance(color, Sequence) and (not isinstance(color, (bytes, str))) and (len(color) != n):
        return interp_palette(color, n)
    return color

def _contour_coords(x: ArrayLike | None, y: ArrayLike | None, z: ArrayLike | np.ma.MaskedArray | None, levels: ArrayLike, want_fill: bool, want_line: bool) -> ContourCoords:
    if False:
        while True:
            i = 10
    '\n    Return the (xs, ys) coords of filled and/or line contours.\n    '
    if not want_fill and (not want_line):
        raise RuntimeError('Neither fill nor line requested in _contour_coords')
    from contourpy import FillType, LineType, contour_generator
    cont_gen = contour_generator(x, y, z, line_type=LineType.ChunkCombinedOffset, fill_type=FillType.OuterOffset)
    fill_coords = None
    if want_fill:
        all_xs = []
        all_ys = []
        for i in range(len(levels) - 1):
            filled = cont_gen.filled(levels[i], levels[i + 1])
            coords = _filled_to_coords(filled)
            all_xs.append(coords.xs)
            all_ys.append(coords.ys)
        fill_coords = FillCoords(all_xs, all_ys)
    line_coords = None
    if want_line:
        all_xs = []
        all_ys = []
        for level in levels:
            lines = cont_gen.lines(level)
            coords = _lines_to_coords(lines)
            all_xs.append(coords.xs)
            all_ys.append(coords.ys)
        line_coords = LineCoords(all_xs, all_ys)
    return ContourCoords(fill_coords, line_coords)

def _filled_to_coords(filled) -> SingleFillCoords:
    if False:
        i = 10
        return i + 15
    xs = []
    ys = []
    for (points, offsets) in zip(*filled):
        n = len(offsets) - 1
        xs.append([points[offsets[i]:offsets[i + 1], 0] for i in range(n)])
        ys.append([points[offsets[i]:offsets[i + 1], 1] for i in range(n)])
    return SingleFillCoords(xs, ys)

def _lines_to_coords(lines) -> SingleLineCoords:
    if False:
        i = 10
        return i + 15
    points = lines[0][0]
    if points is None:
        empty = np.empty(0)
        return SingleLineCoords(empty, empty)
    offsets = lines[1][0]
    npoints = len(points)
    nlines = len(offsets) - 1
    xs = np.empty(npoints + nlines - 1)
    ys = np.empty(npoints + nlines - 1)
    for i in range(nlines):
        start = offsets[i]
        end = offsets[i + 1]
        if i > 0:
            xs[start + i - 1] = np.nan
            ys[start + i - 1] = np.nan
        xs[start + i:end + i] = points[start:end, 0]
        ys[start + i:end + i] = points[start:end, 1]
    return SingleLineCoords(xs, ys)

def _palette_from_collection(collection: PaletteCollection, n: int) -> Palette:
    if False:
        while True:
            i = 10
    if len(collection) < 1:
        raise ValueError('PaletteCollection is empty')
    palette = collection.get(n, None)
    if palette is not None:
        return palette
    max_key = max(collection.keys())
    if isinstance(max_key, int) and n > max_key:
        return interp_palette(collection[max_key], n)
    min_key = min(collection.keys())
    if isinstance(min_key, int) and n < min_key:
        return interp_palette(collection[min_key], n)
    raise ValueError(f'Unable to extract or interpolate palette of length {n} from PaletteCollection')

def _validate_levels(levels: ArrayLike | None) -> NDArray[float]:
    if False:
        for i in range(10):
            print('nop')
    levels = np.asarray(levels)
    if levels.ndim == 0 or len(levels) == 0:
        raise ValueError('No contour levels specified')
    if len(levels) > 1 and np.diff(levels).min() <= 0.0:
        raise ValueError('Contour levels must be increasing')
    return levels