from typing import Any, Optional
from flet_core.canvas.shape import Shape
from flet_core.control import OptionalNumber
from flet_core.painting import Paint

class Oval(Shape):

    def __init__(self, x: OptionalNumber=None, y: OptionalNumber=None, width: OptionalNumber=None, height: OptionalNumber=None, paint: Optional[Paint]=None, ref=None, visible: Optional[bool]=None, disabled: Optional[bool]=None, data: Any=None):
        if False:
            i = 10
            return i + 15
        Shape.__init__(self, ref=ref, visible=visible, disabled=disabled, data=data)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.paint = paint

    def _get_control_name(self):
        if False:
            print('Hello World!')
        return 'oval'

    def _before_build_command(self):
        if False:
            print('Hello World!')
        super()._before_build_command()
        self._set_attr_json('paint', self.__paint)

    @property
    def x(self) -> OptionalNumber:
        if False:
            i = 10
            return i + 15
        return self._get_attr('x')

    @x.setter
    def x(self, value: OptionalNumber):
        if False:
            return 10
        self._set_attr('x', value)

    @property
    def y(self) -> OptionalNumber:
        if False:
            print('Hello World!')
        return self._get_attr('y')

    @y.setter
    def y(self, value: OptionalNumber):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('y', value)

    @property
    def width(self) -> OptionalNumber:
        if False:
            print('Hello World!')
        return self._get_attr('width')

    @width.setter
    def width(self, value: OptionalNumber):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('width', value)

    @property
    def height(self) -> OptionalNumber:
        if False:
            print('Hello World!')
        return self._get_attr('height')

    @height.setter
    def height(self, value: OptionalNumber):
        if False:
            i = 10
            return i + 15
        self._set_attr('height', value)

    @property
    def paint(self) -> Optional[Paint]:
        if False:
            print('Hello World!')
        return self.__paint

    @paint.setter
    def paint(self, value: Optional[Paint]):
        if False:
            while True:
                i = 10
        self.__paint = value