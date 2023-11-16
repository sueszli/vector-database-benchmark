from warnings import warn
from ..doctools import document
from ..exceptions import PlotnineError, PlotnineWarning
from ..utils import alias
from .scale_continuous import scale_continuous
from .scale_discrete import scale_discrete
shapes = ('o', '^', 's', 'D', 'v', '*', 'p', '8', '<', 'h', '>', 'H', 'd')
unfilled_shapes = ('+', 'x', '.', '1', '2', '3', '4', ',', '_', '|', 0, 1, 2, 3, 4, 5, 6, 7)
FILLED_SHAPES = set(shapes)
UNFILLED_SHAPES = set(unfilled_shapes)

@document
class scale_shape(scale_discrete):
    """
    Scale for shapes

    Parameters
    ----------
    unfilled : bool
        If ``True``, then all shapes will have no interiors
        that can be a filled.
    {superclass_parameters}
    """
    _aesthetics = ['shape']

    def __init__(self, unfilled=False, **kwargs):
        if False:
            while True:
                i = 10
        from mizani.palettes import manual_pal
        if unfilled:
            self.palette = manual_pal(unfilled_shapes)
        else:
            self.palette = manual_pal(shapes)
        scale_discrete.__init__(self, **kwargs)

@document
class scale_shape_ordinal(scale_shape):
    """
    Scale for shapes

    Parameters
    ----------
    {superclass_parameters}
    """
    _aesthetics = ['shape']

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        warn('Using shapes for an ordinal variable is not advised.', PlotnineWarning)
        super().__init__(**kwargs)

class scale_shape_continuous(scale_continuous):
    """
    Continuous scale for shapes

    This is not a valid type of scale.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        raise PlotnineError('A continuous variable can not be mapped to shape')
alias('scale_shape_discrete', scale_shape)