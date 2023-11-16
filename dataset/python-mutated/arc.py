from typing import Any, Optional
from flet_core.canvas.shape import Shape
from flet_core.control import OptionalNumber
from flet_core.painting import Paint

class Arc(Shape):

    def __init__(self, x: OptionalNumber=None, y: OptionalNumber=None, width: OptionalNumber=None, height: OptionalNumber=None, start_angle: OptionalNumber=None, sweep_angle: OptionalNumber=None, use_center: Optional[bool]=None, paint: Optional[Paint]=None, ref=None, visible: Optional[bool]=None, disabled: Optional[bool]=None, data: Any=None):
        if False:
            print('Hello World!')
        Shape.__init__(self, ref=ref, visible=visible, disabled=disabled, data=data)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.start_angle = start_angle
        self.sweep_angle = sweep_angle
        self.use_center = use_center
        self.paint = paint

    def _get_control_name(self):
        if False:
            print('Hello World!')
        return 'arc'

    def _before_build_command(self):
        if False:
            print('Hello World!')
        super()._before_build_command()
        self._set_attr_json('paint', self.__paint)

    @property
    def x(self) -> OptionalNumber:
        if False:
            while True:
                i = 10
        return self._get_attr('x')

    @x.setter
    def x(self, value: OptionalNumber):
        if False:
            return 10
        self._set_attr('x', value)

    @property
    def y(self) -> OptionalNumber:
        if False:
            for i in range(10):
                print('nop')
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
            print('Hello World!')
        self._set_attr('width', value)

    @property
    def height(self) -> OptionalNumber:
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('height')

    @height.setter
    def height(self, value: OptionalNumber):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('height', value)

    @property
    def start_angle(self) -> OptionalNumber:
        if False:
            i = 10
            return i + 15
        return self._get_attr('startAngle')

    @start_angle.setter
    def start_angle(self, value: OptionalNumber):
        if False:
            return 10
        self._set_attr('startAngle', value)

    @property
    def sweep_angle(self) -> OptionalNumber:
        if False:
            while True:
                i = 10
        return self._get_attr('sweepAngle')

    @sweep_angle.setter
    def sweep_angle(self, value: OptionalNumber):
        if False:
            return 10
        self._set_attr('sweepAngle', value)

    @property
    def use_center(self) -> Optional[bool]:
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('useCenter', data_type='bool', def_value=False)

    @use_center.setter
    def use_center(self, value: Optional[bool]):
        if False:
            i = 10
            return i + 15
        self._set_attr('useCenter', value)

    @property
    def paint(self) -> Optional[Paint]:
        if False:
            print('Hello World!')
        return self.__paint

    @paint.setter
    def paint(self, value: Optional[Paint]):
        if False:
            print('Hello World!')
        self.__paint = value