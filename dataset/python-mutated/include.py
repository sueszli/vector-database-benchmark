"""

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from copy import copy
from typing import TypeVar
from ..has_props import HasProps
from .descriptor_factory import PropertyDescriptorFactory
from .descriptors import PropertyDescriptor
__all__ = ('Include',)
T = TypeVar('T')

class Include(PropertyDescriptorFactory[T]):
    """ Include "mix-in" property collection in a Bokeh model.

    See :ref:`bokeh.core.property_mixins` for more details.

    """

    def __init__(self, delegate: type[HasProps], *, help: str='', prefix: str | None=None) -> None:
        if False:
            return 10
        if not (isinstance(delegate, type) and issubclass(delegate, HasProps)):
            raise ValueError(f'expected a subclass of HasProps, got {delegate!r}')
        self.delegate = delegate
        self.help = help
        self.prefix = prefix + '_' if prefix else ''

    def make_descriptors(self, _base_name: str) -> list[PropertyDescriptor[T]]:
        if False:
            return 10
        descriptors = []
        for descriptor in self.delegate.descriptors():
            prop = copy(descriptor.property)
            prop.__doc__ = self.help.format(prop=descriptor.name.replace('_', ' '))
            descriptors += prop.make_descriptors(self.prefix + descriptor.name)
        return descriptors