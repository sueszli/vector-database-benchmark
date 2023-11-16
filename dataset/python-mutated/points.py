from enum import Enum
from typing import Any, List, Optional
from flet_core.canvas.shape import Shape
from flet_core.painting import Paint
from flet_core.types import OffsetValue

class PointMode(Enum):
    POINTS = 'points'
    LINES = 'lines'
    POLYGON = 'polygon'

class Points(Shape):

    def __init__(self, points: Optional[List[OffsetValue]]=None, point_mode: Optional[PointMode]=None, paint: Optional[Paint]=None, ref=None, visible: Optional[bool]=None, disabled: Optional[bool]=None, data: Any=None):
        if False:
            print('Hello World!')
        Shape.__init__(self, ref=ref, visible=visible, disabled=disabled, data=data)
        self.points = points
        self.point_mode = point_mode
        self.paint = paint

    def _get_control_name(self):
        if False:
            return 10
        return 'points'

    def _before_build_command(self):
        if False:
            while True:
                i = 10
        super()._before_build_command()
        self._set_attr_json('points', self.__points)
        self._set_attr_json('paint', self.__paint)

    @property
    def point_mode(self) -> Optional[PointMode]:
        if False:
            for i in range(10):
                print('nop')
        return self.__point_mode

    @point_mode.setter
    def point_mode(self, value: Optional[PointMode]):
        if False:
            i = 10
            return i + 15
        self.__point_mode = value
        self._set_attr('pointMode', value.value if value is not None else None)

    @property
    def points(self) -> Optional[List[OffsetValue]]:
        if False:
            return 10
        return self.__points

    @points.setter
    def points(self, value: Optional[List[OffsetValue]]):
        if False:
            for i in range(10):
                print('nop')
        self.__points = value if value is not None else []

    @property
    def paint(self) -> Optional[Paint]:
        if False:
            return 10
        return self.__paint

    @paint.setter
    def paint(self, value: Optional[Paint]):
        if False:
            print('Hello World!')
        self.__paint = value