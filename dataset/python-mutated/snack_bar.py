from enum import Enum
from typing import Any, Optional
from flet_core.control import Control, OptionalNumber
from flet_core.ref import Ref
from flet_core.types import MarginValue, PaddingValue

class SnackBarBehavior(Enum):
    FIXED = 'fixed'
    FLOATING = 'floating'

class DismissDirection(Enum):
    NONE = 'none'
    VERTICAL = 'vertical'
    HORIZONTAL = 'horizontal'
    END_TO_START = 'endToStart'
    START_TO_END = 'startToEnd'
    UP = 'up'
    DOWN = 'down'

class SnackBar(Control):
    """
    A lightweight message with an optional action which briefly displays at the bottom of the screen.

    Example:
    ```
    import flet as ft

    class Data:
        def __init__(self) -> None:
            self.counter = 0

    d = Data()

    def main(page):

        page.snack_bar = ft.SnackBar(
            content=ft.Text("Hello, world!"),
            action="Alright!",
        )
        page.snack_bar.open = True

        def on_click(e):
            page.snack_bar = ft.SnackBar(ft.Text(f"Hello {d.counter}"))
            page.snack_bar.open = True
            d.counter += 1
            page.update()

        page.add(ft.ElevatedButton("Open SnackBar", on_click=on_click))

    ft.app(target=main)
    ```

    -----

    Online docs: https://flet.dev/docs/controls/snackbar
    """

    def __init__(self, content: Control, ref: Optional[Ref]=None, disabled: Optional[bool]=None, visible: Optional[bool]=None, data: Any=None, open: bool=False, behavior: Optional[SnackBarBehavior]=None, dismiss_direction: Optional[DismissDirection]=None, show_close_icon: Optional[bool]=False, action: Optional[str]=None, action_color: Optional[str]=None, close_icon_color: Optional[str]=None, bgcolor: Optional[str]=None, duration: Optional[int]=None, margin: MarginValue=None, padding: PaddingValue=None, width: OptionalNumber=None, elevation: OptionalNumber=None, on_action=None):
        if False:
            return 10
        Control.__init__(self, ref=ref, disabled=disabled, visible=visible, data=data)
        self.open = open
        self.behavior = behavior
        self.dismiss_direction = dismiss_direction
        self.show_close_icon = show_close_icon
        self.close_icon_color = close_icon_color
        self.margin = margin
        self.padding = padding
        self.width = width
        self.content = content
        self.action = action
        self.action_color = action_color
        self.bgcolor = bgcolor
        self.duration = duration
        self.elevation = elevation
        self.on_action = on_action

    def _get_control_name(self):
        if False:
            return 10
        return 'snackbar'

    def _get_children(self):
        if False:
            for i in range(10):
                print('nop')
        children = []
        if self.__content:
            self.__content._set_attr_internal('n', 'content')
            children.append(self.__content)
        return children

    def _before_build_command(self):
        if False:
            for i in range(10):
                print('nop')
        super()._before_build_command()
        self._set_attr_json('margin', self.__margin)
        self._set_attr_json('padding', self.__padding)

    @property
    def open(self) -> Optional[bool]:
        if False:
            print('Hello World!')
        return self._get_attr('open', data_type='bool', def_value=False)

    @open.setter
    def open(self, value: Optional[bool]):
        if False:
            i = 10
            return i + 15
        self._set_attr('open', value)

    @property
    def show_close_icon(self) -> Optional[bool]:
        if False:
            return 10
        return self._get_attr('showCloseIcon', data_type='bool', def_value=False)

    @show_close_icon.setter
    def show_close_icon(self, value: Optional[bool]):
        if False:
            while True:
                i = 10
        self._set_attr('showCloseIcon', value)

    @property
    def content(self) -> Control:
        if False:
            while True:
                i = 10
        return self.__content

    @content.setter
    def content(self, value: Control):
        if False:
            print('Hello World!')
        self.__content = value

    @property
    def action(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('action')

    @action.setter
    def action(self, value):
        if False:
            return 10
        self._set_attr('action', value)

    @property
    def action_color(self):
        if False:
            while True:
                i = 10
        return self._get_attr('actionColor')

    @action_color.setter
    def action_color(self, value):
        if False:
            i = 10
            return i + 15
        self._set_attr('actionColor', value)

    @property
    def bgcolor(self):
        if False:
            while True:
                i = 10
        return self._get_attr('bgColor')

    @bgcolor.setter
    def bgcolor(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('bgColor', value)

    @property
    def close_icon_color(self):
        if False:
            while True:
                i = 10
        return self._get_attr('closeIconColor')

    @close_icon_color.setter
    def close_icon_color(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('closeIconColor', value)

    @property
    def duration(self) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        return self._get_attr('duration')

    @duration.setter
    def duration(self, value: Optional[int]):
        if False:
            i = 10
            return i + 15
        self._set_attr('duration', value)

    @property
    def behavior(self) -> Optional[SnackBarBehavior]:
        if False:
            i = 10
            return i + 15
        return self.__behavior

    @behavior.setter
    def behavior(self, value: Optional[SnackBarBehavior]):
        if False:
            i = 10
            return i + 15
        self.__behavior = value
        self._set_attr('behavior', value.value if isinstance(value, SnackBarBehavior) else value)

    @property
    def dismiss_direction(self) -> Optional[DismissDirection]:
        if False:
            i = 10
            return i + 15
        return self.__dismiss_direction

    @dismiss_direction.setter
    def dismiss_direction(self, value: Optional[DismissDirection]):
        if False:
            for i in range(10):
                print('nop')
        self.__dismiss_direction = value
        self._set_attr('dismissDirection', value.value if isinstance(value, DismissDirection) else value)

    @property
    def padding(self) -> PaddingValue:
        if False:
            i = 10
            return i + 15
        return self.__padding

    @padding.setter
    def padding(self, value: PaddingValue):
        if False:
            i = 10
            return i + 15
        self.__padding = value

    @property
    def margin(self) -> MarginValue:
        if False:
            print('Hello World!')
        return self.__margin

    @margin.setter
    def margin(self, value: MarginValue):
        if False:
            i = 10
            return i + 15
        self.__margin = value

    @property
    def width(self) -> OptionalNumber:
        if False:
            i = 10
            return i + 15
        return self._get_attr('width')

    @width.setter
    def width(self, value: OptionalNumber):
        if False:
            while True:
                i = 10
        self._set_attr('width', value)

    @property
    def elevation(self) -> OptionalNumber:
        if False:
            print('Hello World!')
        return self._get_attr('elevation')

    @elevation.setter
    def elevation(self, value: OptionalNumber):
        if False:
            while True:
                i = 10
        self._set_attr('elevation', value)

    @property
    def on_action(self):
        if False:
            while True:
                i = 10
        return self._get_event_handler('action')

    @on_action.setter
    def on_action(self, handler):
        if False:
            while True:
                i = 10
        self._add_event_handler('action', handler)