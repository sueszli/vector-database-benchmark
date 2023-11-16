import time
from typing import Any, Optional, Union
from flet_core.buttons import ButtonStyle
from flet_core.constrained_control import ConstrainedControl
from flet_core.control import Control, OptionalNumber
from flet_core.ref import Ref
from flet_core.types import AnimationValue, OffsetValue, ResponsiveNumber, RotateValue, ScaleValue

class IconButton(ConstrainedControl):
    """
    An icon button is a round button with an icon in the middle that reacts to touches by filling with color (ink).

    Icon buttons are commonly used in the toolbars, but they can be used in many other places as well.

    Example:
    ```
    import flet as ft

    def main(page: ft.Page):
        page.title = "Icon buttons"
        page.add(
            ft.Row(
                [
                    ft.IconButton(
                        icon=ft.icons.PAUSE_CIRCLE_FILLED_ROUNDED,
                        icon_color="blue400",
                        icon_size=20,
                        tooltip="Pause record",
                    ),
                    ft.IconButton(
                        icon=ft.icons.DELETE_FOREVER_ROUNDED,
                        icon_color="pink600",
                        icon_size=40,
                        tooltip="Delete record",
                    ),
                ]
            ),
        )

    ft.app(target=main)
    ```

    -----

    Online docs: https://flet.dev/docs/controls/iconbutton
    """

    def __init__(self, icon: Optional[str]=None, ref: Optional[Ref]=None, key: Optional[str]=None, width: OptionalNumber=None, height: OptionalNumber=None, left: OptionalNumber=None, top: OptionalNumber=None, right: OptionalNumber=None, bottom: OptionalNumber=None, expand: Union[None, bool, int]=None, col: Optional[ResponsiveNumber]=None, opacity: OptionalNumber=None, rotate: RotateValue=None, scale: ScaleValue=None, offset: OffsetValue=None, aspect_ratio: OptionalNumber=None, animate_opacity: AnimationValue=None, animate_size: AnimationValue=None, animate_position: AnimationValue=None, animate_rotation: AnimationValue=None, animate_scale: AnimationValue=None, animate_offset: AnimationValue=None, on_animation_end=None, tooltip: Optional[str]=None, visible: Optional[bool]=None, disabled: Optional[bool]=None, data: Any=None, icon_size: OptionalNumber=None, icon_color: Optional[str]=None, selected_icon: Optional[str]=None, selected_icon_color: Optional[str]=None, selected: Optional[bool]=None, bgcolor: Optional[str]=None, style: Optional[ButtonStyle]=None, content: Optional[Control]=None, autofocus: Optional[bool]=None, url: Optional[str]=None, url_target: Optional[str]=None, on_click=None, on_focus=None, on_blur=None):
        if False:
            for i in range(10):
                print('nop')
        ConstrainedControl.__init__(self, ref=ref, key=key, width=width, height=height, left=left, top=top, right=right, bottom=bottom, expand=expand, col=col, opacity=opacity, rotate=rotate, scale=scale, offset=offset, aspect_ratio=aspect_ratio, animate_opacity=animate_opacity, animate_size=animate_size, animate_position=animate_position, animate_rotation=animate_rotation, animate_scale=animate_scale, animate_offset=animate_offset, on_animation_end=on_animation_end, tooltip=tooltip, visible=visible, disabled=disabled, data=data)
        self.icon = icon
        self.icon_size = icon_size
        self.icon_color = icon_color
        self.selected_icon = selected_icon
        self.selected_icon_color = selected_icon_color
        self.selected = selected
        self.bgcolor = bgcolor
        self.style = style
        self.content = content
        self.autofocus = autofocus
        self.url = url
        self.url_target = url_target
        self.on_click = on_click
        self.on_focus = on_focus
        self.on_blur = on_blur

    def _get_control_name(self):
        if False:
            return 10
        return 'iconbutton'

    def _before_build_command(self):
        if False:
            for i in range(10):
                print('nop')
        super()._before_build_command()
        if self.__style is not None:
            self.__style.side = self._wrap_attr_dict(self.__style.side)
            self.__style.shape = self._wrap_attr_dict(self.__style.shape)
        self._set_attr_json('style', self.__style)

    def _get_children(self):
        if False:
            i = 10
            return i + 15
        if self.__content is None:
            return []
        self.__content._set_attr_internal('n', 'content')
        return [self.__content]

    def focus(self):
        if False:
            i = 10
            return i + 15
        self._set_attr_json('focus', str(time.time()))
        self.update()

    async def focus_async(self):
        self._set_attr_json('focus', str(time.time()))
        await self.update_async()

    @property
    def icon(self):
        if False:
            print('Hello World!')
        return self._get_attr('icon')

    @icon.setter
    def icon(self, value):
        if False:
            i = 10
            return i + 15
        self._set_attr('icon', value)

    @property
    def selected_icon(self):
        if False:
            while True:
                i = 10
        return self._get_attr('selectedIcon')

    @selected_icon.setter
    def selected_icon(self, value):
        if False:
            i = 10
            return i + 15
        self._set_attr('selectedIcon', value)

    @property
    def icon_size(self):
        if False:
            while True:
                i = 10
        return self._get_attr('iconSize')

    @icon_size.setter
    def icon_size(self, value):
        if False:
            print('Hello World!')
        self._set_attr('iconSize', value)

    @property
    def icon_color(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('iconColor')

    @icon_color.setter
    def icon_color(self, value):
        if False:
            i = 10
            return i + 15
        self._set_attr('iconColor', value)

    @property
    def selected_icon_color(self):
        if False:
            while True:
                i = 10
        return self._get_attr('selectedIconColor')

    @selected_icon_color.setter
    def selected_icon_color(self, value):
        if False:
            return 10
        self._set_attr('selectedIconColor', value)

    @property
    def bgcolor(self):
        if False:
            i = 10
            return i + 15
        return self._get_attr('bgcolor')

    @bgcolor.setter
    def bgcolor(self, value):
        if False:
            i = 10
            return i + 15
        self._set_attr('bgcolor', value)

    @property
    def selected(self) -> Optional[bool]:
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('selected', data_type='bool', def_value=False)

    @selected.setter
    def selected(self, value: Optional[bool]):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('selected', value)

    @property
    def style(self) -> Optional[ButtonStyle]:
        if False:
            i = 10
            return i + 15
        return self.__style

    @style.setter
    def style(self, value: Optional[ButtonStyle]):
        if False:
            for i in range(10):
                print('nop')
        self.__style = value

    @property
    def url(self):
        if False:
            return 10
        return self._get_attr('url')

    @url.setter
    def url(self, value):
        if False:
            return 10
        self._set_attr('url', value)

    @property
    def url_target(self):
        if False:
            print('Hello World!')
        return self._get_attr('urlTarget')

    @url_target.setter
    def url_target(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('urlTarget', value)

    @property
    def on_click(self):
        if False:
            print('Hello World!')
        return self._get_event_handler('click')

    @on_click.setter
    def on_click(self, handler):
        if False:
            print('Hello World!')
        self._add_event_handler('click', handler)

    @property
    def content(self) -> Optional[Control]:
        if False:
            print('Hello World!')
        return self.__content

    @content.setter
    def content(self, value: Optional[Control]):
        if False:
            i = 10
            return i + 15
        self.__content = value

    @property
    def autofocus(self) -> Optional[bool]:
        if False:
            print('Hello World!')
        return self._get_attr('autofocus', data_type='bool', def_value=False)

    @autofocus.setter
    def autofocus(self, value: Optional[bool]):
        if False:
            return 10
        self._set_attr('autofocus', value)

    @property
    def on_focus(self):
        if False:
            i = 10
            return i + 15
        return self._get_event_handler('focus')

    @on_focus.setter
    def on_focus(self, handler):
        if False:
            while True:
                i = 10
        self._add_event_handler('focus', handler)

    @property
    def on_blur(self):
        if False:
            i = 10
            return i + 15
        return self._get_event_handler('blur')

    @on_blur.setter
    def on_blur(self, handler):
        if False:
            for i in range(10):
                print('nop')
        self._add_event_handler('blur', handler)