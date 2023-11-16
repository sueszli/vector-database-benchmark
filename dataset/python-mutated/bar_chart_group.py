from typing import Any, List, Optional
from flet_core.charts.bar_chart_rod import BarChartRod
from flet_core.control import Control, OptionalNumber
from flet_core.ref import Ref

class BarChartGroup(Control):

    def __init__(self, x: Optional[int]=None, bar_rods: Optional[List[BarChartRod]]=None, ref: Optional[Ref]=None, disabled: Optional[bool]=None, visible: Optional[bool]=None, data: Any=None, group_vertically: Optional[bool]=None, bars_space: OptionalNumber=None):
        if False:
            return 10
        Control.__init__(self, ref=ref, disabled=disabled, visible=visible, data=data)
        self.x = x
        self.bar_rods = bar_rods
        self.group_vertically = group_vertically
        self.bars_space = bars_space

    def _get_control_name(self):
        if False:
            print('Hello World!')
        return 'group'

    def _before_build_command(self):
        if False:
            print('Hello World!')
        super()._before_build_command()

    def _get_children(self):
        if False:
            print('Hello World!')
        return self.__bar_rods

    @property
    def bar_rods(self):
        if False:
            while True:
                i = 10
        return self.__bar_rods

    @bar_rods.setter
    def bar_rods(self, value):
        if False:
            return 10
        self.__bar_rods = value if value is not None else []

    @property
    def x(self) -> Optional[int]:
        if False:
            print('Hello World!')
        return self._get_attr('x', data_type='int')

    @x.setter
    def x(self, value: Optional[int]):
        if False:
            print('Hello World!')
        self._set_attr('x', value)

    @property
    def group_vertically(self) -> Optional[bool]:
        if False:
            return 10
        return self._get_attr('groupVertically', data_type='bool', def_value=False)

    @group_vertically.setter
    def group_vertically(self, value: Optional[bool]):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('groupVertically', value)

    @property
    def bars_space(self) -> OptionalNumber:
        if False:
            return 10
        return self._get_attr('barsSpace', data_type='float')

    @bars_space.setter
    def bars_space(self, value: OptionalNumber):
        if False:
            print('Hello World!')
        self._set_attr('barsSpace', value)