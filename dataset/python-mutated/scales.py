"""

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ..core.has_props import abstract
from ..core.properties import Instance, Required
from .transforms import Transform
__all__ = ('CategoricalScale', 'CompositeScale', 'LinearScale', 'LogScale', 'Scale')

@abstract
class Scale(Transform):
    """ Base class for ``Scale`` models that represent an invertible
    computation to be carried out on the client-side.

    JavaScript implementations should implement the following methods:

    .. code-block

        compute(x: number): number {
            # compute and return the transform of a single value
        }

        v_compute(xs: Arrayable<number>): Arrayable<number> {
            # compute and return the transform of an array of values
        }

        invert(sx: number): number {
            # compute and return the inverse transform of a single value
        }

        v_invert(sxs: Arrayable<number>): Arrayable<number> {
            # compute and return the inverse transform of an array of values
        }

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)

class ContinuousScale(Scale):
    """ Represent a scale transformation between continuous ranges.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)

class LinearScale(ContinuousScale):
    """ Represent a linear scale transformation between continuous ranges.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)

class LogScale(ContinuousScale):
    """ Represent a log scale transformation between continuous ranges.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)

class CategoricalScale(Scale):
    """ Represent a scale transformation between a categorical source range and
    continuous target range.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)

class CompositeScale(Scale):
    """ Represent a composition of two scales, which useful for defining
    sub-coordinate systems.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    source_scale = Required(Instance(Scale), help='\n    The source scale.\n    ')
    target_scale = Required(Instance(Scale), help='\n    The target scale.\n    ')