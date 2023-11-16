from enum import Enum
from typing import Any, List, Optional, Union
from flet_core.constrained_control import ConstrainedControl
from flet_core.control import Control, OptionalNumber
from flet_core.ref import Ref
from flet_core.types import PaddingValue
from flet_core.buttons import OutlinedBorder

class NavigationDrawerDestination(Control):
    """
    Displays an icon with a label, for use in NavigationDrawer destinations.

    """

    def __init__(self, ref: Optional[Ref]=None, icon: Optional[str]=None, icon_content: Optional[Control]=None, label: Optional[str]=None, selected_icon: Optional[str]=None, selected_icon_content: Optional[Control]=None):
        if False:
            while True:
                i = 10
        Control.__init__(self, ref=ref)
        self.label = label
        self.icon = icon
        self.__icon_content: Optional[Control] = None
        self.icon_content = icon_content
        self.selected_icon = selected_icon
        self.__selected_icon_content: Optional[Control] = None
        self.selected_icon_content = selected_icon_content

    def _get_control_name(self):
        if False:
            print('Hello World!')
        return 'navigationdrawerdestination'

    def _get_children(self):
        if False:
            return 10
        children = []
        if self.__icon_content:
            self.__icon_content._set_attr_internal('n', 'icon_content')
            children.append(self.__icon_content)
        if self.__selected_icon_content:
            self.__selected_icon_content._set_attr_internal('n', 'selected_icon_content')
            children.append(self.__selected_icon_content)
        return children

    @property
    def icon(self):
        if False:
            while True:
                i = 10
        return self._get_attr('icon')

    @icon.setter
    def icon(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('icon', value)

    @property
    def icon_content(self) -> Optional[Control]:
        if False:
            return 10
        return self.__icon_content

    @icon_content.setter
    def icon_content(self, value: Optional[Control]):
        if False:
            for i in range(10):
                print('nop')
        self.__icon_content = value

    @property
    def selected_icon(self):
        if False:
            return 10
        return self._get_attr('selectedIcon')

    @selected_icon.setter
    def selected_icon(self, value):
        if False:
            return 10
        self._set_attr('selectedIcon', value)

    @property
    def selected_icon_content(self) -> Optional[Control]:
        if False:
            return 10
        return self.__selected_icon_content

    @selected_icon_content.setter
    def selected_icon_content(self, value: Optional[Control]):
        if False:
            while True:
                i = 10
        self.__selected_icon_content = value

    @property
    def label(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('label')

    @label.setter
    def label(self, value):
        if False:
            return 10
        self._set_attr('label', value)

class NavigationDrawer(Control):
    """
    Material Design Navigation Drawer component.

    Navigation Drawer is a panel slides in horizontally from the left or right edge of a page to show primary destinations in an app.

    Example:

    ```
    import flet as ft


    def main(page: ft.Page):
        def item_selected_left(e):
            print(e.control.selected_index)

        page.drawer = ft.NavigationDrawer(
            elevation=40,
            indicator_color=ft.colors.GREEN_200,
            indicator_shape=ft.StadiumBorder(),
            shadow_color=ft.colors.GREEN_900,
            surface_tint_color=ft.colors.GREEN,
            selected_index=-1,
            on_change=item_selected_left,
            controls=[
                ft.Container(height=12),
                ft.NavigationDrawerDestination(
                    label="Item 1",
                    icon=ft.icons.ABC,
                    selected_icon_content=ft.Icon(ft.icons.ACCESS_ALARM),
                ),
                ft.Divider(thickness=2),
                ft.NavigationDrawerDestination(
                    icon_content=ft.Icon(ft.icons.MAIL),
                    label="Item 2",
                    selected_icon=ft.icons.PHISHING,
                ),
                ft.NavigationDrawerDestination(
                    icon_content=ft.Icon(ft.icons.PHONE),
                    label="Item 3",
                    selected_icon=ft.icons.PHISHING,
                ),
            ],
        )

        end_drawer = ft.NavigationDrawer(
            controls=[
                ft.NavigationDrawerDestination(
                    icon=ft.icons.ADD_TO_HOME_SCREEN_SHARP, label="Item 1"
                ),
                ft.NavigationDrawerDestination(icon=ft.icons.ADD_COMMENT, label="Item 2"),
            ],
        )

        def show_drawer(e):
            page.drawer.open = True
            page.drawer.update()

        def show_end_drawer(e):
            page.show_end_drawer(end_drawer)

        page.add(
            ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                controls=[
                    ft.ElevatedButton("Show drawer", on_click=show_drawer),
                    ft.ElevatedButton("Show end drawer", on_click=show_end_drawer),
                ],
            )
        )


    ft.app(main)

    ```

    -----

    Online docs: https://flet.dev/docs/controls/navigationdrawer
    """

    def __init__(self, ref: Optional[Ref]=None, disabled: Optional[bool]=None, visible: Optional[bool]=None, data: Any=None, open: bool=False, controls: Optional[List[Control]]=None, selected_index: Optional[int]=None, bgcolor: Optional[str]=None, elevation: OptionalNumber=None, indicator_color: Optional[str]=None, indicator_shape: Optional[OutlinedBorder]=None, shadow_color: Optional[str]=None, surface_tint_color: Optional[str]=None, tile_padding: PaddingValue=None, on_change=None, on_dismiss=None):
        if False:
            i = 10
            return i + 15
        Control.__init__(self, ref=ref, visible=visible, disabled=disabled, data=data)
        self.open = open
        self.controls = controls
        self.selected_index = selected_index
        self.bgcolor = bgcolor
        self.elevation = elevation
        self.indicator_color = indicator_color
        self.indicator_shape = indicator_shape
        self.shadow_color = shadow_color
        self.surface_tint_color = surface_tint_color
        self.tile_padding = tile_padding
        self.on_change = on_change
        self.on_dismiss = on_dismiss

    def _get_control_name(self):
        if False:
            for i in range(10):
                print('nop')
        return 'navigationdrawer'

    def _before_build_command(self):
        if False:
            return 10
        super()._before_build_command()
        self._set_attr_json('indicatorShape', self.__indicator_shape)
        self._set_attr_json('tilePadding', self.__tile_padding)

    def _get_children(self):
        if False:
            i = 10
            return i + 15
        children = []
        children.extend(self.__controls)
        return children

    @property
    def open(self) -> Optional[bool]:
        if False:
            return 10
        return self._get_attr('open', data_type='bool', def_value=False)

    @open.setter
    def open(self, value: Optional[bool]):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('open', value)

    @property
    def controls(self) -> Optional[List[Control]]:
        if False:
            print('Hello World!')
        return self.__controls

    @controls.setter
    def controls(self, value: Optional[List[Control]]):
        if False:
            for i in range(10):
                print('nop')
        self.__controls = value if value is not None else []

    @property
    def selected_index(self) -> Optional[int]:
        if False:
            return 10
        return self._get_attr('selectedIndex', data_type='int', def_value=0)

    @selected_index.setter
    def selected_index(self, value: Optional[int]):
        if False:
            return 10
        self._set_attr('selectedIndex', value)

    @property
    def bgcolor(self):
        if False:
            return 10
        return self._get_attr('bgcolor')

    @bgcolor.setter
    def bgcolor(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('bgcolor', value)

    @property
    def elevation(self) -> OptionalNumber:
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('elevation')

    @elevation.setter
    def elevation(self, value: OptionalNumber):
        if False:
            return 10
        self._set_attr('elevation', value)

    @property
    def indicator_color(self):
        if False:
            return 10
        return self._get_attr('indicatorColor')

    @indicator_color.setter
    def indicator_color(self, value):
        if False:
            print('Hello World!')
        self._set_attr('indicatorColor', value)

    @property
    def indicator_shape(self) -> Optional[OutlinedBorder]:
        if False:
            while True:
                i = 10
        return self.__indicator_shape

    @indicator_shape.setter
    def indicator_shape(self, value: Optional[OutlinedBorder]):
        if False:
            while True:
                i = 10
        self.__indicator_shape = value

    @property
    def shadow_color(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('shadowColor')

    @shadow_color.setter
    def shadow_color(self, value):
        if False:
            i = 10
            return i + 15
        self._set_attr('shadowColor', value)

    @property
    def surface_tint_color(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('surfaceTintColor')

    @surface_tint_color.setter
    def surface_tint_color(self, value):
        if False:
            i = 10
            return i + 15
        self._set_attr('surfaceTintColor', value)

    @property
    def tile_padding(self) -> PaddingValue:
        if False:
            while True:
                i = 10
        return self.__tile_padding

    @tile_padding.setter
    def tile_padding(self, value: PaddingValue):
        if False:
            i = 10
            return i + 15
        self.__tile_padding = value

    @property
    def on_change(self):
        if False:
            i = 10
            return i + 15
        return self._get_event_handler('change')

    @on_change.setter
    def on_change(self, handler):
        if False:
            for i in range(10):
                print('nop')
        self._add_event_handler('change', handler)

    @property
    def on_dismiss(self):
        if False:
            return 10
        return self._get_event_handler('dismiss')

    @on_dismiss.setter
    def on_dismiss(self, handler):
        if False:
            print('Hello World!')
        self._add_event_handler('dismiss', handler)