""" Provide wildcard properties.

The Any and AnyRef properties can be used to hold values without performing
any validation.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import typing
from .bases import Init, Property
__all__ = ('Any', 'AnyRef')

class Any(Property[typing.Any]):
    """ Accept all values.

    The ``Any`` property does not do any validation or transformation.

    Args:
        default (obj or None, optional) :
            A default value for attributes created from this property to
            have (default: None)

        help (str or None, optional) :
            A documentation string for this property. It will be automatically
            used by the :ref:`bokeh.sphinxext.bokeh_prop` extension when
            generating Spinx documentation. (default: None)

    Example:

        .. code-block:: python

            >>> class AnyModel(HasProps):
            ...     prop = Any()
            ...

            >>> m = AnyModel()

            >>> m.prop = True

            >>> m.prop = 10

            >>> m.prop = 3.14

            >>> m.prop = "foo"

            >>> m.prop = [1, 2, 3]

    """

    def __init__(self, default: Init[typing.Any]=None, help: str | None=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__(default=default, help=help)

class AnyRef(Any):
    """ Accept all values and force reference discovery. """

    @property
    def has_ref(self) -> bool:
        if False:
            while True:
                i = 10
        return True