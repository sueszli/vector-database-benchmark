from typing import Any, Optional, Union
from flet_core.constrained_control import ConstrainedControl
from flet_core.control import Control, OptionalNumber
from flet_core.ref import Ref
from flet_core.types import AnimationValue, OffsetValue, ResponsiveNumber, RotateValue, ScaleValue

class WindowDragArea(ConstrainedControl):
    """
    A control for drag to move, maximize and restore application window.

    When you have hidden the title bar with `page.window_title_bar_hidden`, you can add this control to move the window position.

    Example:
    ```
    import flet as ft

    def main(page: ft.Page):
        page.window_title_bar_hidden = True
        page.window_title_bar_buttons_hidden = True

        page.add(
            ft.Row(
                [
                    ft.WindowDragArea(ft.Container(ft.Text("Drag this area to move, maximize and restore application window."), bgcolor=ft.colors.AMBER_300, padding=10), expand=True),
                    ft.IconButton(ft.icons.CLOSE, on_click=lambda _: page.window_close())
                ]
            )
        )

    ft.app(target=main)
    ```

    -----

    Online docs: https://flet.dev/docs/controls/windowdragarea
    """

    def __init__(self, content: Optional[Control]=None, ref: Optional[Ref]=None, width: OptionalNumber=None, height: OptionalNumber=None, left: OptionalNumber=None, top: OptionalNumber=None, right: OptionalNumber=None, bottom: OptionalNumber=None, expand: Union[None, bool, int]=None, col: Optional[ResponsiveNumber]=None, opacity: OptionalNumber=None, rotate: RotateValue=None, scale: ScaleValue=None, offset: OffsetValue=None, aspect_ratio: OptionalNumber=None, animate_opacity: AnimationValue=None, animate_size: AnimationValue=None, animate_position: AnimationValue=None, animate_rotation: AnimationValue=None, animate_scale: AnimationValue=None, animate_offset: AnimationValue=None, on_animation_end=None, tooltip: Optional[str]=None, visible: Optional[bool]=None, disabled: Optional[bool]=None, data: Any=None, maximizable: Optional[bool]=None):
        if False:
            print('Hello World!')
        ConstrainedControl.__init__(self, ref=ref, width=width, height=height, left=left, top=top, right=right, bottom=bottom, expand=expand, col=col, opacity=opacity, rotate=rotate, scale=scale, offset=offset, aspect_ratio=aspect_ratio, animate_opacity=animate_opacity, animate_size=animate_size, animate_position=animate_position, animate_rotation=animate_rotation, animate_scale=animate_scale, animate_offset=animate_offset, on_animation_end=on_animation_end, tooltip=tooltip, visible=visible, disabled=disabled, data=data)
        self.__content: Optional[Control] = None
        self.content = content
        self.maximizable = maximizable

    def _get_control_name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'windowDragArea'

    def _get_children(self):
        if False:
            return 10
        children = []
        if self.__content:
            self.__content._set_attr_internal('n', 'content')
            children.append(self.__content)
        return children

    @property
    def content(self):
        if False:
            while True:
                i = 10
        return self.__content

    @content.setter
    def content(self, value):
        if False:
            print('Hello World!')
        self.__content = value

    @property
    def maximizable(self) -> Optional[bool]:
        if False:
            i = 10
            return i + 15
        return self._get_attr('maximizable', data_type='bool', def_value=True)

    @maximizable.setter
    def maximizable(self, value: Optional[bool]):
        if False:
            i = 10
            return i + 15
        self._set_attr('maximizable', value)