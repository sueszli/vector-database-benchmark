from typing import Any, Optional
from flet_core.border import BorderSide
from flet_core.control import Control, OptionalNumber
from flet_core.ref import Ref
from flet_core.text_style import TextStyle

class PieChartSection(Control):

    def __init__(self, value: OptionalNumber=None, ref: Optional[Ref]=None, disabled: Optional[bool]=None, visible: Optional[bool]=None, data: Any=None, radius: OptionalNumber=None, color: Optional[str]=None, border_side: Optional[BorderSide]=None, title: Optional[str]=None, title_style: Optional[TextStyle]=None, title_position: OptionalNumber=None, badge: Optional[Control]=None, badge_position: OptionalNumber=None):
        if False:
            for i in range(10):
                print('nop')
        Control.__init__(self, ref=ref, disabled=disabled, visible=visible, data=data)
        self.value = value
        self.radius = radius
        self.color = color
        self.border_side = border_side
        self.title = title
        self.title_style = title_style
        self.title_position = title_position
        self.badge = badge
        self.badge_position = badge_position

    def _get_control_name(self):
        if False:
            return 10
        return 'section'

    def _before_build_command(self):
        if False:
            while True:
                i = 10
        super()._before_build_command()
        self._set_attr_json('borderSide', self.__border_side)
        self._set_attr_json('titleStyle', self.__title_style)

    def _get_children(self):
        if False:
            while True:
                i = 10
        children = []
        if self.__badge:
            self.__badge._set_attr_internal('n', 'badge')
            children.append(self.__badge)
        return children

    @property
    def value(self) -> OptionalNumber:
        if False:
            print('Hello World!')
        return self._get_attr('value', data_type='float')

    @value.setter
    def value(self, value: OptionalNumber):
        if False:
            print('Hello World!')
        self._set_attr('value', value)

    @property
    def radius(self) -> OptionalNumber:
        if False:
            return 10
        return self._get_attr('radius', data_type='float')

    @radius.setter
    def radius(self, value: OptionalNumber):
        if False:
            while True:
                i = 10
        self._set_attr('radius', value)

    @property
    def border_side(self) -> Optional[BorderSide]:
        if False:
            print('Hello World!')
        return self.__border_side

    @border_side.setter
    def border_side(self, value: Optional[BorderSide]):
        if False:
            while True:
                i = 10
        self.__border_side = value

    @property
    def color(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        return self._get_attr('color')

    @color.setter
    def color(self, value: Optional[str]):
        if False:
            while True:
                i = 10
        self._set_attr('color', value)

    @property
    def badge(self) -> Optional[Control]:
        if False:
            for i in range(10):
                print('nop')
        return self.__badge

    @badge.setter
    def badge(self, value: Optional[Control]):
        if False:
            i = 10
            return i + 15
        self.__badge = value

    @property
    def badge_position(self) -> OptionalNumber:
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('badgePosition', data_type='float')

    @badge_position.setter
    def badge_position(self, value: OptionalNumber):
        if False:
            print('Hello World!')
        self._set_attr('badgePosition', value)

    @property
    def title(self):
        if False:
            i = 10
            return i + 15
        return self._get_attr('title')

    @title.setter
    def title(self, value: Optional[str]):
        if False:
            while True:
                i = 10
        self._set_attr('title', value)

    @property
    def title_style(self):
        if False:
            print('Hello World!')
        return self.__title_style

    @title_style.setter
    def title_style(self, value: Optional[TextStyle]):
        if False:
            while True:
                i = 10
        self.__title_style = value

    @property
    def title_position(self) -> OptionalNumber:
        if False:
            while True:
                i = 10
        return self._get_attr('titlePosition', data_type='float', def_value=1.0)

    @title_position.setter
    def title_position(self, value: OptionalNumber):
        if False:
            while True:
                i = 10
        self._set_attr('titlePosition', value)