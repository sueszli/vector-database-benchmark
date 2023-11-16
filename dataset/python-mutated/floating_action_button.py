from typing import Any, Optional, Union
from flet_core.buttons import OutlinedBorder
from flet_core.constrained_control import ConstrainedControl
from flet_core.control import Control, OptionalNumber
from flet_core.ref import Ref
from flet_core.types import AnimationValue, OffsetValue, ResponsiveNumber, RotateValue, ScaleValue

class FloatingActionButton(ConstrainedControl):
    """
    A floating action button is a circular icon button that hovers over content to promote a primary action in the application. Floating action button is usually set to `page.floating_action_button`, but can also be added as a regular control at any place on a page.

    Example:
    ```
    import flet as ft

    def main(page: ft.Page):
        page.title = "Floating Action Button"
        page.theme_mode = ft.ThemeMode.LIGHT
        page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        page.auto_scroll = True
        page.scroll = ft.ScrollMode.HIDDEN
        page.appbar = ft.AppBar(
            title=ft.Text(
                "Floating Action Button", weight=ft.FontWeight.BOLD, color=ft.colors.BLACK87
            ),
            bgcolor=ft.colors.BLUE,
            center_title=True,
            actions=[
                ft.IconButton(ft.icons.MENU, tooltip="Menu", icon_color=ft.colors.BLACK87)
            ],
            color=ft.colors.WHITE,
        )

        # keeps track of the number of tiles already added
        page.count = 0

        def fab_pressed(e):
            page.add(ft.ListTile(title=ft.Text(f"Tile {page.count}")))
            page.show_snack_bar(
                ft.SnackBar(ft.Text("Tile was added successfully!"), open=True)
            )
            page.count += 1

        page.floating_action_button = ft.FloatingActionButton(
            icon=ft.icons.ADD, on_click=fab_pressed, bgcolor=ft.colors.LIME_300
        )
        page.add(ft.Text("Press the FAB to add a tile!"))

    ft.app(target=main)
    ```

    -----

    Online docs: https://flet.dev/docs/controls/floatingactionbutton
    """

    def __init__(self, text: Optional[str]=None, ref: Optional[Ref]=None, key: Optional[str]=None, width: OptionalNumber=None, height: OptionalNumber=None, left: OptionalNumber=None, top: OptionalNumber=None, right: OptionalNumber=None, bottom: OptionalNumber=None, expand: Union[None, bool, int]=None, col: Optional[ResponsiveNumber]=None, opacity: OptionalNumber=None, rotate: RotateValue=None, scale: ScaleValue=None, offset: OffsetValue=None, aspect_ratio: OptionalNumber=None, animate_opacity: AnimationValue=None, animate_size: AnimationValue=None, animate_position: AnimationValue=None, animate_rotation: AnimationValue=None, animate_scale: AnimationValue=None, animate_offset: AnimationValue=None, on_animation_end=None, tooltip: Optional[str]=None, visible: Optional[bool]=None, disabled: Optional[bool]=None, data: Any=None, icon: Optional[str]=None, bgcolor: Optional[str]=None, content: Optional[Control]=None, autofocus: Optional[bool]=None, shape: Optional[OutlinedBorder]=None, mini: Optional[bool]=None, url: Optional[str]=None, url_target: Optional[str]=None, on_click=None):
        if False:
            i = 10
            return i + 15
        ConstrainedControl.__init__(self, ref=ref, key=key, width=width, height=height, left=left, top=top, right=right, bottom=bottom, expand=expand, col=col, opacity=opacity, rotate=rotate, scale=scale, offset=offset, aspect_ratio=aspect_ratio, animate_opacity=animate_opacity, animate_size=animate_size, animate_position=animate_position, animate_rotation=animate_rotation, animate_scale=animate_scale, animate_offset=animate_offset, on_animation_end=on_animation_end, tooltip=tooltip, visible=visible, disabled=disabled, data=data)
        self.text = text
        self.icon = icon
        self.bgcolor = bgcolor
        self.content = content
        self.autofocus = autofocus
        self.shape = shape
        self.mini = mini
        self.url = url
        self.url_target = url_target
        self.on_click = on_click

    def _get_control_name(self):
        if False:
            return 10
        return 'floatingactionbutton'

    def _before_build_command(self):
        if False:
            print('Hello World!')
        super()._before_build_command()
        self._set_attr_json('shape', self.__shape)

    def _get_children(self):
        if False:
            print('Hello World!')
        if self.__content is None:
            return []
        self.__content._set_attr_internal('n', 'content')
        return [self.__content]

    @property
    def text(self):
        if False:
            i = 10
            return i + 15
        return self._get_attr('text')

    @text.setter
    def text(self, value):
        if False:
            while True:
                i = 10
        self._set_attr('text', value)

    @property
    def icon(self):
        if False:
            while True:
                i = 10
        return self._get_attr('icon')

    @icon.setter
    def icon(self, value):
        if False:
            i = 10
            return i + 15
        self._set_attr('icon', value)

    @property
    def bgcolor(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('bgcolor')

    @bgcolor.setter
    def bgcolor(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('bgcolor', value)

    @property
    def url(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('url')

    @url.setter
    def url(self, value):
        if False:
            i = 10
            return i + 15
        self._set_attr('url', value)

    @property
    def url_target(self):
        if False:
            while True:
                i = 10
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
            for i in range(10):
                print('nop')
        return self._get_event_handler('click')

    @on_click.setter
    def on_click(self, handler):
        if False:
            i = 10
            return i + 15
        self._add_event_handler('click', handler)

    @property
    def content(self) -> Optional[Control]:
        if False:
            return 10
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
            i = 10
            return i + 15
        self._set_attr('autofocus', value)

    @property
    def shape(self) -> Optional[OutlinedBorder]:
        if False:
            return 10
        return self.__shape

    @shape.setter
    def shape(self, value: Optional[OutlinedBorder]):
        if False:
            return 10
        self.__shape = value

    @property
    def mini(self) -> Optional[bool]:
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('mini', data_type='bool', def_value=False)

    @mini.setter
    def mini(self, value: Optional[bool]):
        if False:
            return 10
        self._set_attr('mini', value)