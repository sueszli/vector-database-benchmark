from typing import Any, Optional
from flet_core.border import BorderSide
from flet_core.control import Control, OptionalNumber
from flet_core.ref import Ref

class BarChartRodStackItem(Control):

    def __init__(self, from_y: OptionalNumber=None, to_y: OptionalNumber=None, color: Optional[str]=None, border_side: Optional[BorderSide]=None, ref: Optional[Ref]=None, disabled: Optional[bool]=None, visible: Optional[bool]=None, data: Any=None):
        if False:
            while True:
                i = 10
        Control.__init__(self, ref=ref, disabled=disabled, visible=visible, data=data)
        self.from_y = from_y
        self.to_y = to_y
        self.color = color
        self.border_side = border_side

    def _get_control_name(self):
        if False:
            i = 10
            return i + 15
        return 'stack_item'

    def _before_build_command(self):
        if False:
            print('Hello World!')
        super()._before_build_command()
        self._set_attr_json('borderSide', self.__border_side)

    @property
    def from_y(self) -> OptionalNumber:
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('fromY', data_type='float')

    @from_y.setter
    def from_y(self, value: OptionalNumber):
        if False:
            i = 10
            return i + 15
        self._set_attr('fromY', value)

    @property
    def to_y(self) -> OptionalNumber:
        if False:
            i = 10
            return i + 15
        return self._get_attr('toY', data_type='float')

    @to_y.setter
    def to_y(self, value: OptionalNumber):
        if False:
            i = 10
            return i + 15
        self._set_attr('toY', value)

    @property
    def color(self) -> Optional[str]:
        if False:
            print('Hello World!')
        return self._get_attr('color')

    @color.setter
    def color(self, value: Optional[str]):
        if False:
            return 10
        self._set_attr('color', value)

    @property
    def border_side(self) -> Optional[BorderSide]:
        if False:
            print('Hello World!')
        return self.__border_side

    @border_side.setter
    def border_side(self, value: Optional[BorderSide]):
        if False:
            for i in range(10):
                print('nop')
        self.__border_side = value