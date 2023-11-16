""" Auxiliary graphical models for aiding glyphs, guide renderers, etc.

"""
import logging
log = logging.getLogger(__name__)
from ..core.has_props import abstract
from ..core.properties import Enum, Instance, Required
from ..model import Model
__all__ = ('Decoration', 'Marking')

@abstract
class Marking(Model):
    """ Base class for graphical markings, e.g. arrow heads.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)

class Decoration(Model):
    """ Indicates a positioned marker, e.g. at a node of a glyph.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    marking = Instance(Marking, help='\n    The graphical marking associated with this decoration, e.g. an arrow head.\n    ')
    node = Required(Enum('start', 'middle', 'end'), help='\n    The placement of the marking on the parent graphical object.\n    ')