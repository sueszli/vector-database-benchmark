"""

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ..core.enums import TextureRepetition
from ..core.has_props import abstract
from ..core.properties import Enum, Required, String
from ..model import Model
__all__ = ('CanvasTexture', 'ImageURLTexture', 'Texture')

@abstract
class Texture(Model):
    """ Base class for ``Texture`` models that represent fill patterns.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    repetition = Enum(TextureRepetition, default='repeat', help='\n\n    ')

class CanvasTexture(Texture):
    """

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    code = Required(String, help='\n    A snippet of JavaScript code to execute in the browser.\n\n    ')

class ImageURLTexture(Texture):
    """

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    url = Required(String, help='\n    A URL to a drawable resource like image, video, etc.\n\n    ')