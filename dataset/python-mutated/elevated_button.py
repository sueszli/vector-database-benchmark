import time
from typing import Any, Optional, Union
from flet_core.buttons import ButtonStyle
from flet_core.constrained_control import ConstrainedControl
from flet_core.control import Control, OptionalNumber
from flet_core.ref import Ref
from flet_core.types import AnimationValue, OffsetValue, ResponsiveNumber, RotateValue, ScaleValue

class ElevatedButton(ConstrainedControl):
    """
    Elevated buttons are essentially filled tonal buttons with a shadow. To prevent shadow creep, only use them when absolutely necessary, such as when the button requires visual separation from a patterned background.

    Example:
    ```
    import flet as ft

    def main(page: ft.Page):
        page.title = "Basic elevated buttons"
        page.add(
            ft.ElevatedButton(text="Elevated button"),
            ft.ElevatedButton("Disabled button", disabled=True),
        )

    ft.app(target=main)
    ```

    -----

    Online docs: https://flet.dev/docs/controls/elevatedbutton
    """

    def __init__(self, text: Optional[str]=None, ref: Optional[Ref]=None, key: Optional[str]=None, width: OptionalNumber=None, height: OptionalNumber=None, left: OptionalNumber=None, top: OptionalNumber=None, right: OptionalNumber=None, bottom: OptionalNumber=None, expand: Union[None, bool, int]=None, col: Optional[ResponsiveNumber]=None, opacity: OptionalNumber=None, rotate: RotateValue=None, scale: ScaleValue=None, offset: OffsetValue=None, aspect_ratio: OptionalNumber=None, animate_opacity: AnimationValue=None, animate_size: AnimationValue=None, animate_position: AnimationValue=None, animate_rotation: AnimationValue=None, animate_scale: AnimationValue=None, animate_offset: AnimationValue=None, on_animation_end=None, tooltip: Optional[str]=None, visible: Optional[bool]=None, disabled: Optional[bool]=None, data: Any=None, color: Optional[str]=None, bgcolor: Optional[str]=None, elevation: OptionalNumber=None, style: Optional[ButtonStyle]=None, icon: Optional[str]=None, icon_color: Optional[str]=None, content: Optional[Control]=None, autofocus: Optional[bool]=None, url: Optional[str]=None, url_target: Optional[str]=None, on_click=None, on_long_press=None, on_hover=None, on_focus=None, on_blur=None):
        if False:
            return 10
        ConstrainedControl.__init__(self, ref=ref, key=key, width=width, height=height, left=left, top=top, right=right, bottom=bottom, expand=expand, col=col, opacity=opacity, rotate=rotate, scale=scale, offset=offset, aspect_ratio=aspect_ratio, animate_opacity=animate_opacity, animate_size=animate_size, animate_position=animate_position, animate_rotation=animate_rotation, animate_scale=animate_scale, animate_offset=animate_offset, on_animation_end=on_animation_end, tooltip=tooltip, visible=visible, disabled=disabled, data=data)
        self.__color = None
        self.__bgcolor = None
        self.__elevation = None
        self.text = text
        self.color = color
        self.bgcolor = bgcolor
        self.elevation = elevation
        self.style = style
        self.icon = icon
        self.icon_color = icon_color
        self.content = content
        self.autofocus = autofocus
        self.url = url
        self.url_target = url_target
        self.on_click = on_click
        self.on_long_press = on_long_press
        self.on_hover = on_hover
        self.on_focus = on_focus
        self.on_blur = on_blur

    def _get_control_name(self):
        if False:
            print('Hello World!')
        return 'elevatedbutton'

    def _before_build_command(self):
        if False:
            for i in range(10):
                print('nop')
        super()._before_build_command()
        if self.__color is not None or self.__bgcolor is not None or self.__elevation is not None:
            if self.__style is None:
                self.__style = ButtonStyle()
            if self.__style.color != self.__color or self.disabled:
                self.__style.color = self.__color if not self.disabled else None
            if self.__style.bgcolor != self.__bgcolor or self.disabled:
                self.__style.bgcolor = self.__bgcolor if not self.disabled else None
            if self.__style.elevation != self.__elevation:
                self.__style.elevation = self.__elevation
        if self.__style is not None:
            self.__style.side = self._wrap_attr_dict(self.__style.side)
            self.__style.shape = self._wrap_attr_dict(self.__style.shape)
        self._set_attr_json('style', self.__style)

    def _get_children(self):
        if False:
            return 10
        if self.__content is None:
            return []
        self.__content._set_attr_internal('n', 'content')
        return [self.__content]

    def focus(self):
        if False:
            print('Hello World!')
        self._set_attr_json('focus', str(time.time()))
        self.update()

    async def focus_async(self):
        self._set_attr_json('focus', str(time.time()))
        await self.update_async()

    @property
    def text(self):
        if False:
            return 10
        return self._get_attr('text')

    @text.setter
    def text(self, value):
        if False:
            return 10
        self._set_attr('text', value)

    @property
    def color(self):
        if False:
            print('Hello World!')
        return self.__color

    @color.setter
    def color(self, value):
        if False:
            while True:
                i = 10
        self.__color = value

    @property
    def bgcolor(self):
        if False:
            print('Hello World!')
        return self.__bgcolor

    @bgcolor.setter
    def bgcolor(self, value):
        if False:
            print('Hello World!')
        self.__bgcolor = value

    @property
    def elevation(self) -> OptionalNumber:
        if False:
            for i in range(10):
                print('nop')
        return self.__elevation

    @elevation.setter
    def elevation(self, value: OptionalNumber):
        if False:
            i = 10
            return i + 15
        self.__elevation = value

    @property
    def style(self) -> Optional[ButtonStyle]:
        if False:
            print('Hello World!')
        return self.__style

    @style.setter
    def style(self, value: Optional[ButtonStyle]):
        if False:
            i = 10
            return i + 15
        self.__style = value

    @property
    def icon(self):
        if False:
            i = 10
            return i + 15
        return self._get_attr('icon')

    @icon.setter
    def icon(self, value):
        if False:
            return 10
        self._set_attr('icon', value)

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
    def url(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('url')

    @url.setter
    def url(self, value):
        if False:
            print('Hello World!')
        self._set_attr('url', value)

    @property
    def url_target(self):
        if False:
            return 10
        return self._get_attr('urlTarget')

    @url_target.setter
    def url_target(self, value):
        if False:
            i = 10
            return i + 15
        self._set_attr('urlTarget', value)

    @property
    def on_click(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_event_handler('click')

    @on_click.setter
    def on_click(self, handler):
        if False:
            while True:
                i = 10
        self._add_event_handler('click', handler)

    @property
    def on_long_press(self):
        if False:
            i = 10
            return i + 15
        return self._get_event_handler('long_press')

    @on_long_press.setter
    def on_long_press(self, handler):
        if False:
            for i in range(10):
                print('nop')
        self._add_event_handler('long_press', handler)
        self._set_attr('onLongPress', True if handler is not None else None)

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
            for i in range(10):
                print('nop')
        self._set_attr('autofocus', value)

    @property
    def on_hover(self):
        if False:
            i = 10
            return i + 15
        return self._get_event_handler('hover')

    @on_hover.setter
    def on_hover(self, handler):
        if False:
            return 10
        self._add_event_handler('hover', handler)
        self._set_attr('onHover', True if handler is not None else None)

    @property
    def on_focus(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_event_handler('focus')

    @on_focus.setter
    def on_focus(self, handler):
        if False:
            return 10
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
            return 10
        self._add_event_handler('blur', handler)