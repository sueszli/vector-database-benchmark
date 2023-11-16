"""
    This implementation of Event is triggered when an Image has been processed using a Do instruction
"""
from decimal import Decimal
from PIL import Image as PILImage
from borb.pdf.canvas.canvas_graphics_state import CanvasGraphicsState
from borb.pdf.canvas.event.event_listener import Event

class ImageRenderEvent(Event):
    """
    This implementation of Event is triggered when an Image has been processed using a Do instruction
    """

    def __init__(self, graphics_state: CanvasGraphicsState, image: PILImage):
        if False:
            for i in range(10):
                print('nop')
        self._image: PILImage = image
        v = graphics_state.ctm.cross(Decimal(0), Decimal(0), Decimal(1))
        self._x: Decimal = v[0]
        self._y: Decimal = v[1]
        v = graphics_state.ctm.cross(Decimal(1), Decimal(1), Decimal(0))
        self._width: Decimal = max(abs(v[0]), Decimal(1))
        self._height: Decimal = max(abs(v[1]), Decimal(1))

    def get_height(self) -> Decimal:
        if False:
            i = 10
            return i + 15
        '\n        Get the height of the (scaled) Image\n        '
        return self._height

    def get_image(self) -> PILImage:
        if False:
            return 10
        '\n        Get the (source) Image\n        This Image may have different dimensions than\n        how it is displayed in the PDF\n        '
        return self._image

    def get_width(self) -> Decimal:
        if False:
            i = 10
            return i + 15
        '\n        Get the width of the (scaled) Image\n        '
        return self._width

    def get_x(self) -> Decimal:
        if False:
            print('Hello World!')
        '\n        Get the x-coordinate at which the Image is drawn\n        '
        return self._x

    def get_y(self) -> Decimal:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the y-coordinate at which the Image is drawn\n        '
        return self._y