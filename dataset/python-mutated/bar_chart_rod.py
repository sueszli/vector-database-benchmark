from typing import Any, List, Optional
from flet_core.border import BorderSide
from flet_core.charts.bar_chart_rod_stack_item import BarChartRodStackItem
from flet_core.control import Control, OptionalNumber
from flet_core.gradients import Gradient
from flet_core.ref import Ref
from flet_core.text_style import TextStyle
from flet_core.types import BorderRadiusValue, TextAlign, TextAlignString

class BarChartRod(Control):

    def __init__(self, rod_stack_items: Optional[List[BarChartRodStackItem]]=None, ref: Optional[Ref]=None, disabled: Optional[bool]=None, visible: Optional[bool]=None, data: Any=None, from_y: OptionalNumber=None, to_y: OptionalNumber=None, width: OptionalNumber=None, color: Optional[str]=None, gradient: Optional[Gradient]=None, border_radius: BorderRadiusValue=None, border_side: Optional[BorderSide]=None, bg_from_y: OptionalNumber=None, bg_to_y: OptionalNumber=None, bg_color: Optional[str]=None, bg_gradient: Optional[Gradient]=None, selected: Optional[bool]=None, show_tooltip: Optional[bool]=None, tooltip: Optional[str]=None, tooltip_style: Optional[TextStyle]=None, tooltip_align: TextAlign=TextAlign.NONE):
        if False:
            i = 10
            return i + 15
        Control.__init__(self, ref=ref, disabled=disabled, visible=visible, data=data)
        self.rod_stack_items = rod_stack_items
        self.from_y = from_y
        self.to_y = to_y
        self.width = width
        self.color = color
        self.gradient = gradient
        self.border_side = border_side
        self.border_radius = border_radius
        self.bg_from_y = bg_from_y
        self.bg_to_y = bg_to_y
        self.bg_color = bg_color
        self.bg_gradient = bg_gradient
        self.selected = selected
        self.show_tooltip = show_tooltip
        self.tooltip = tooltip
        self.tooltip_align = tooltip_align
        self.tooltip_style = tooltip_style

    def _get_control_name(self):
        if False:
            return 10
        return 'rod'

    def _before_build_command(self):
        if False:
            i = 10
            return i + 15
        super()._before_build_command()
        self._set_attr_json('gradient', self.__gradient)
        self._set_attr_json('borderSide', self.__border_side)
        self._set_attr_json('borderRadius', self.__border_radius)
        self._set_attr_json('bgGradient', self.__bg_gradient)

    def _get_children(self):
        if False:
            while True:
                i = 10
        return self.__rod_stack_items

    @property
    def rod_stack_items(self):
        if False:
            i = 10
            return i + 15
        return self.__rod_stack_items

    @rod_stack_items.setter
    def rod_stack_items(self, value):
        if False:
            print('Hello World!')
        self.__rod_stack_items = value if value is not None else []

    @property
    def from_y(self) -> OptionalNumber:
        if False:
            while True:
                i = 10
        return self._get_attr('fromY', data_type='float')

    @from_y.setter
    def from_y(self, value: OptionalNumber):
        if False:
            return 10
        self._set_attr('fromY', value)

    @property
    def to_y(self) -> OptionalNumber:
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('toY', data_type='float')

    @to_y.setter
    def to_y(self, value: OptionalNumber):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('toY', value)

    @property
    def width(self) -> OptionalNumber:
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('width', data_type='float')

    @width.setter
    def width(self, value: OptionalNumber):
        if False:
            print('Hello World!')
        self._set_attr('width', value)

    @property
    def color(self) -> Optional[str]:
        if False:
            return 10
        return self._get_attr('color')

    @color.setter
    def color(self, value: Optional[str]):
        if False:
            while True:
                i = 10
        self._set_attr('color', value)

    @property
    def border_side(self) -> Optional[BorderSide]:
        if False:
            print('Hello World!')
        return self.__border_side

    @border_side.setter
    def border_side(self, value: Optional[BorderSide]):
        if False:
            print('Hello World!')
        self.__border_side = value

    @property
    def border_radius(self) -> Optional[BorderRadiusValue]:
        if False:
            print('Hello World!')
        return self.__border_radius

    @border_radius.setter
    def border_radius(self, value: Optional[BorderRadiusValue]):
        if False:
            print('Hello World!')
        self.__border_radius = value

    @property
    def gradient(self) -> Optional[Gradient]:
        if False:
            while True:
                i = 10
        return self.__gradient

    @gradient.setter
    def gradient(self, value: Optional[Gradient]):
        if False:
            return 10
        self.__gradient = value

    @property
    def bg_from_y(self) -> OptionalNumber:
        if False:
            i = 10
            return i + 15
        return self._get_attr('bgFromY', data_type='float')

    @bg_from_y.setter
    def bg_from_y(self, value: OptionalNumber):
        if False:
            while True:
                i = 10
        self._set_attr('bgFromY', value)

    @property
    def bg_to_y(self) -> OptionalNumber:
        if False:
            while True:
                i = 10
        return self._get_attr('bgToY', data_type='float')

    @bg_to_y.setter
    def bg_to_y(self, value: OptionalNumber):
        if False:
            return 10
        self._set_attr('bgToY', value)

    @property
    def bg_color(self) -> Optional[str]:
        if False:
            return 10
        return self._get_attr('bgColor')

    @bg_color.setter
    def bg_color(self, value: Optional[str]):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('bgColor', value)

    @property
    def bg_gradient(self) -> Optional[Gradient]:
        if False:
            print('Hello World!')
        return self.__bg_gradient

    @bg_gradient.setter
    def bg_gradient(self, value: Optional[Gradient]):
        if False:
            return 10
        self.__bg_gradient = value

    @property
    def selected(self) -> Optional[bool]:
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('selected', data_type='bool', def_value=False)

    @selected.setter
    def selected(self, value: Optional[bool]):
        if False:
            return 10
        self._set_attr('selected', value)

    @property
    def show_tooltip(self) -> Optional[bool]:
        if False:
            print('Hello World!')
        return self._get_attr('showTooltip', data_type='bool', def_value=True)

    @show_tooltip.setter
    def show_tooltip(self, value: Optional[bool]):
        if False:
            i = 10
            return i + 15
        self._set_attr('showTooltip', value)

    @property
    def tooltip(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        return self._get_attr('tooltip')

    @tooltip.setter
    def tooltip(self, value: Optional[str]):
        if False:
            print('Hello World!')
        self._set_attr('tooltip', value)

    @property
    def tooltip_align(self) -> TextAlign:
        if False:
            while True:
                i = 10
        return self.__tooltip_align

    @tooltip_align.setter
    def tooltip_align(self, value: TextAlign):
        if False:
            for i in range(10):
                print('nop')
        self.__tooltip_align = value
        if isinstance(value, TextAlign):
            self._set_attr('tooltipAlign', value.value)
        else:
            self.__set_tooltip_align(value)

    def __set_tooltip_align(self, value: TextAlignString):
        if False:
            print('Hello World!')
        self._set_attr('tooltipAlign', value)

    @property
    def tooltip_style(self):
        if False:
            i = 10
            return i + 15
        return self.__tooltip_style

    @tooltip_style.setter
    def tooltip_style(self, value: Optional[TextStyle]):
        if False:
            print('Hello World!')
        self.__tooltip_style = value