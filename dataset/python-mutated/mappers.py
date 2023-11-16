""" Models for mapping values from one range or space to another in the client.

Mappers (as opposed to scales) are not presumed to be invertible.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from .. import palettes
from ..core.enums import Palette
from ..core.has_props import abstract
from ..core.properties import Bool, Color, Either, Enum, FactorSeq, Float, HatchPatternType, Instance, Int, List, MarkerType, Nullable, Seq, String, Tuple
from ..core.validation import error, warning
from ..core.validation.errors import WEIGHTED_STACK_COLOR_MAPPER_LABEL_LENGTH_MISMATCH
from ..core.validation.warnings import PALETTE_LENGTH_FACTORS_MISMATCH
from .transforms import Transform
__all__ = ('Mapper', 'ColorMapper', 'CategoricalMapper', 'CategoricalColorMapper', 'CategoricalMarkerMapper', 'CategoricalPatternMapper', 'ContinuousColorMapper', 'LinearColorMapper', 'LogColorMapper', 'EqHistColorMapper', 'StackColorMapper', 'WeightedStackColorMapper')

@abstract
class Mapper(Transform):
    """ Base class for mappers.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)

@abstract
class ColorMapper(Mapper):
    """ Base class for color mapper types.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        if len(args) == 1:
            kwargs['palette'] = args[0]
        super().__init__(**kwargs)
    palette = Seq(Color, help='\n    A sequence of colors to use as the target palette for mapping.\n\n    This property can also be set as a ``String``, to the name of any of the\n    palettes shown in :ref:`bokeh.palettes`.\n    ').accepts(Enum(Palette), lambda pal: getattr(palettes, pal))
    nan_color = Color(default='gray', help='\n    Color to be used if data is NaN or otherwise not mappable.\n    ')

@abstract
class CategoricalMapper(Mapper):
    """ Base class for mappers that map categorical factors to other values.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    factors = FactorSeq(help='\n    A sequence of factors / categories that map to the some target range. For\n    example the following color mapper:\n\n    .. code-block:: python\n\n        mapper = CategoricalColorMapper(palette=["red", "blue"], factors=["foo", "bar"])\n\n    will map the factor ``"foo"`` to red and the factor ``"bar"`` to blue.\n    ')
    start = Int(default=0, help='\n    A start index to "slice" data factors with before mapping.\n\n    For example, if the data to color map consists of 2-level factors such\n    as ``["2016", "sales"]`` and ``["2016", "marketing"]``, then setting\n    ``start=1`` will perform color mapping only based on the second sub-factor\n    (i.e. in this case based on the department ``"sales"`` or ``"marketing"``)\n    ')
    end = Nullable(Int, help='\n    A start index to "slice" data factors with before mapping.\n\n    For example, if the data to color map consists of 2-level factors such\n    as ``["2016", "sales"]`` and ``["2017", "marketing"]``, then setting\n    ``end=1`` will perform color mapping only based on the first sub-factor\n    (i.e. in this case based on the year ``"2016"`` or ``"2017"``)\n\n    If ``None`` then all sub-factors from ``start`` to the end of the\n    factor will be used for color mapping.\n    ')

class CategoricalColorMapper(CategoricalMapper, ColorMapper):
    """ Map categorical factors to colors.

    Values that are passed to this mapper that are not in the factors list
    will be mapped to ``nan_color``.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)

    @warning(PALETTE_LENGTH_FACTORS_MISMATCH)
    def _check_palette_length(self):
        if False:
            for i in range(10):
                print('nop')
        palette = self.palette
        factors = self.factors
        if len(palette) < len(factors):
            extra_factors = factors[len(palette):]
            return f'{extra_factors} will be assigned to `nan_color` {self.nan_color}'

class CategoricalMarkerMapper(CategoricalMapper):
    """ Map categorical factors to marker types.

    Values that are passed to this mapper that are not in the factors list
    will be mapped to ``default_value``.

    .. note::
        This mappers is primarily only useful with the ``Scatter`` marker
        glyph that be parameterized by marker type.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    markers = Seq(MarkerType, help='\n    A sequence of marker types to use as the target for mapping.\n    ')
    default_value = MarkerType(default='circle', help='\n    A marker type to use in case an unrecognized factor is passed in to be\n    mapped.\n    ')

class CategoricalPatternMapper(CategoricalMapper):
    """ Map categorical factors to hatch fill patterns.

    Values that are passed to this mapper that are not in the factors list
    will be mapped to ``default_value``.

    Added in version 1.1.1

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    patterns = Seq(HatchPatternType, help='\n    A sequence of marker types to use as the target for mapping.\n    ')
    default_value = HatchPatternType(default=' ', help='\n    A hatch pattern to use in case an unrecognized factor is passed in to be\n    mapped.\n    ')

@abstract
class ContinuousColorMapper(ColorMapper):
    """ Base class for continuous color mapper types.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    domain = List(Tuple(Instance('bokeh.models.renderers.GlyphRenderer'), Either(String, List(String))), default=[], help='\n    A collection of glyph renderers to pool data from for establishing data metrics.\n    If empty, mapped data will be used instead.\n    ')
    low = Nullable(Float, help='\n    The minimum value of the range to map into the palette. Values below\n    this are clamped to ``low``. If ``None``, the value is inferred from data.\n    ')
    high = Nullable(Float, help='\n    The maximum value of the range to map into the palette. Values above\n    this are clamped to ``high``. If ``None``, the value is inferred from data.\n    ')
    low_color = Nullable(Color, help='\n    Color to be used if data is lower than ``low`` value. If None,\n    values lower than ``low`` are mapped to the first color in the palette.\n    ')
    high_color = Nullable(Color, help='\n    Color to be used if data is higher than ``high`` value. If None,\n    values higher than ``high`` are mapped to the last color in the palette.\n    ')

class LinearColorMapper(ContinuousColorMapper):
    """ Map numbers in a range [*low*, *high*] linearly into a sequence of
    colors (a palette).

    For example, if the range is [0, 99] and the palette is
    ``['red', 'green', 'blue']``, the values would be mapped as follows::

             x < 0  : 'red'     # values < low are clamped
        0 <= x < 33 : 'red'
       33 <= x < 66 : 'green'
       66 <= x < 99 : 'blue'
       99 <= x      : 'blue'    # values > high are clamped

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)

class LogColorMapper(ContinuousColorMapper):
    """ Map numbers in a range [*low*, *high*] into a sequence of colors
    (a palette) on a natural logarithm scale.

    For example, if the range is [0, 25] and the palette is
    ``['red', 'green', 'blue']``, the values would be mapped as follows::

                x < 0     : 'red'     # values < low are clamped
       0     <= x < 2.72  : 'red'     # math.e ** 1
       2.72  <= x < 7.39  : 'green'   # math.e ** 2
       7.39  <= x < 20.09 : 'blue'    # math.e ** 3
       20.09 <= x         : 'blue'    # values > high are clamped

    .. warning::
        The ``LogColorMapper`` only works for images with scalar values that are
        non-negative.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)

@abstract
class ScanningColorMapper(ContinuousColorMapper):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)

class EqHistColorMapper(ScanningColorMapper):
    """

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    bins = Int(default=256 * 256, help='Number of histogram bins')
    rescale_discrete_levels = Bool(default=False, help='\n    If there are only a few discrete levels in the values that are color\n    mapped then ``rescale_discrete_levels=True`` decreases the lower limit of\n    the span so that the values are rendered towards the top end of the\n    palette.\n    ')

@abstract
class StackColorMapper(ColorMapper):
    """ Abstract base class for color mappers that operate on ``ImageStack``
    glyphs.

    These map 3D data arrays of shape ``(ny, nx, nstack)`` to 2D RGBA images
    of shape ``(ny, nx)``.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)

class WeightedStackColorMapper(StackColorMapper):
    """ Maps 3D data arrays of shape ``(ny, nx, nstack)`` to 2D RGBA images
    of shape ``(ny, nx)`` using a palette of length ``nstack``.

    The mapping occurs in two stages. Firstly the RGB values are calculated
    using a weighted sum of the palette colors in the ``nstack`` direction.
    Then the alpha values are calculated using the ``alpha_mapper`` applied to
    the sum of the array in the ``nstack`` direction.

    The RGB values calculated by the ``alpha_mapper`` are ignored by the color
    mapping but are used in any ``ColorBar`` that is displayed.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    alpha_mapper = Instance(ContinuousColorMapper, help='\n    Color mapper used to calculate the alpha values of the mapped data.\n    ')
    color_baseline = Nullable(Float, help='\n    Baseline value used for the weights when calculating the weighted sum of\n    palette colors. If ``None`` then the minimum of the supplied data is used\n    meaning that values at this minimum have a weight of zero and do not\n    contribute to the weighted sum. As a special case, if all data for a\n    particular output pixel are at the color baseline then the color is an\n    evenly weighted average of the colors corresponding to all such values,\n    to avoid the color being undefined.\n    ')
    stack_labels = Nullable(Seq(String), help='\n    An optional sequence of strings to use as labels for the ``nstack`` stacks.\n    If set, the number of labels should match the number of stacks and hence\n    also the number of palette colors.\n\n    The labels are used in hover tooltips for ``ImageStack`` glyphs that use a\n    ``WeightedStackColorMapper`` as their color mapper.\n    ')

    @error(WEIGHTED_STACK_COLOR_MAPPER_LABEL_LENGTH_MISMATCH)
    def _check_label_length(self):
        if False:
            i = 10
            return i + 15
        if self.stack_labels is not None:
            nlabel = len(self.stack_labels)
            npalette = len(self.palette)
            if nlabel > npalette:
                self.stack_labels = self.stack_labels[:npalette]
                return f'{nlabel} != {npalette}, removing unwanted stack_labels'
            elif nlabel < npalette:
                self.stack_labels = list(self.stack_labels) + [''] * (npalette - nlabel)
                return f'{nlabel} != {npalette}, padding with empty strings'