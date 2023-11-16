from typing import Any, Optional
from flet_core.control import Control, OptionalNumber
from flet_core.ref import Ref

class ChartAxisLabel(Control):

    def __init__(self, ref: Optional[Ref]=None, disabled: Optional[bool]=None, visible: Optional[bool]=None, data: Any=None, value: OptionalNumber=None, label: Optional[Control]=None):
        if False:
            for i in range(10):
                print('nop')
        Control.__init__(self, ref=ref, disabled=disabled, visible=visible, data=data)
        self.value = value
        self.label = label

    def _get_control_name(self):
        if False:
            i = 10
            return i + 15
        return 'l'

    def _get_children(self):
        if False:
            return 10
        children = []
        if self.__label:
            children.append(self.__label)
        return children

    @property
    def value(self) -> OptionalNumber:
        if False:
            return 10
        return self._get_attr('value', data_type='float', def_value=1.0)

    @value.setter
    def value(self, value: OptionalNumber):
        if False:
            print('Hello World!')
        self._set_attr('value', value)

    @property
    def label(self) -> Optional[Control]:
        if False:
            while True:
                i = 10
        return self.__label

    @label.setter
    def label(self, value: Optional[Control]):
        if False:
            for i in range(10):
                print('nop')
        self.__label = value