from typing import List, Optional
from flet_core.control import Control, OptionalNumber
from flet_core.ref import Ref

class AppBar(Control):
    """
    A material design app bar.

    Example:
    ```
    import flet as ft

    def main(page: ft.Page):
        def check_item_clicked(e):
            e.control.checked = not e.control.checked
            page.update()

        page.appbar = ft.AppBar(
            leading=ft.Icon(ft.icons.PALETTE),
            leading_width=40,
            title=ft.Text("AppBar Example"),
            center_title=False,
            bgcolor=ft.colors.SURFACE_VARIANT,
            actions=[
                ft.IconButton(ft.icons.WB_SUNNY_OUTLINED),
                ft.IconButton(ft.icons.FILTER_3),
                ft.PopupMenuButton(
                    items=[
                        ft.PopupMenuItem(text="Item 1"),
                        ft.PopupMenuItem(),  # divider
                        ft.PopupMenuItem(
                            text="Checked item", checked=False, on_click=check_item_clicked
                        ),
                    ]
                ),
            ],
        )
        page.add(ft.Text("Body!"))

    ft.app(target=main)

    ```

    -----

    Online docs: https://flet.dev/docs/controls/appbar
    """

    def __init__(self, ref: Optional[Ref]=None, leading: Optional[Control]=None, leading_width: OptionalNumber=None, automatically_imply_leading: Optional[bool]=None, title: Optional[Control]=None, center_title: Optional[bool]=None, toolbar_height: OptionalNumber=None, color: Optional[str]=None, bgcolor: Optional[str]=None, elevation: OptionalNumber=None, actions: Optional[List[Control]]=None):
        if False:
            i = 10
            return i + 15
        Control.__init__(self, ref=ref)
        self.__leading: Optional[Control] = None
        self.__title: Optional[Control] = None
        self.__actions: List[Control] = []
        self.leading = leading
        self.leading_width = leading_width
        self.automatically_imply_leading = automatically_imply_leading
        self.title = title
        self.center_title = center_title
        self.toolbar_height = toolbar_height
        self.color = color
        self.bgcolor = bgcolor
        self.elevation = elevation
        self.actions = actions

    def _get_control_name(self):
        if False:
            return 10
        return 'appbar'

    def _get_children(self):
        if False:
            print('Hello World!')
        children = []
        if self.__leading:
            self.__leading._set_attr_internal('n', 'leading')
            children.append(self.__leading)
        if self.__title:
            self.__title._set_attr_internal('n', 'title')
            children.append(self.__title)
        for action in self.__actions:
            action._set_attr_internal('n', 'action')
            children.append(action)
        return children

    @property
    def leading(self) -> Optional[Control]:
        if False:
            return 10
        return self.__leading

    @leading.setter
    def leading(self, value: Optional[Control]):
        if False:
            print('Hello World!')
        "\n        A Control to display before the toolbar's title.\n\n        Typically the leading control is an Icon or an IconButton.\n        "
        self.__leading = value

    @property
    def leading_width(self) -> OptionalNumber:
        if False:
            for i in range(10):
                print('nop')
        return self._get_attr('leadingWidth')

    @leading_width.setter
    def leading_width(self, value: OptionalNumber):
        if False:
            while True:
                i = 10
        self._set_attr('leadingWidth', value)

    @property
    def automatically_imply_leading(self) -> Optional[bool]:
        if False:
            print('Hello World!')
        return self._get_attr('automaticallyImplyLeading', data_type='bool', def_value=True)

    @automatically_imply_leading.setter
    def automatically_imply_leading(self, value: Optional[bool]):
        if False:
            print('Hello World!')
        self._set_attr('automaticallyImplyLeading', value)

    @property
    def title(self) -> Optional[Control]:
        if False:
            print('Hello World!')
        return self.__title

    @title.setter
    def title(self, value: Optional[Control]):
        if False:
            print('Hello World!')
        self.__title = value

    @property
    def center_title(self) -> Optional[bool]:
        if False:
            i = 10
            return i + 15
        return self._get_attr('centerTitle', data_type='bool', def_value=False)

    @center_title.setter
    def center_title(self, value: Optional[bool]):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('centerTitle', value)

    @property
    def toolbar_height(self) -> OptionalNumber:
        if False:
            print('Hello World!')
        return self._get_attr('toolbarHeight')

    @toolbar_height.setter
    def toolbar_height(self, value: OptionalNumber):
        if False:
            i = 10
            return i + 15
        self._set_attr('toolbarHeight', value)

    @property
    def color(self):
        if False:
            while True:
                i = 10
        return self._get_attr('color')

    @color.setter
    def color(self, value):
        if False:
            while True:
                i = 10
        self._set_attr('color', value)

    @property
    def bgcolor(self):
        if False:
            print('Hello World!')
        return self._get_attr('bgcolor')

    @bgcolor.setter
    def bgcolor(self, value):
        if False:
            i = 10
            return i + 15
        self._set_attr('bgcolor', value)

    @property
    def elevation(self) -> OptionalNumber:
        if False:
            return 10
        return self._get_attr('elevation')

    @elevation.setter
    def elevation(self, value: OptionalNumber):
        if False:
            print('Hello World!')
        self._set_attr('elevation', value)

    @property
    def actions(self):
        if False:
            while True:
                i = 10
        return self.__actions

    @actions.setter
    def actions(self, value):
        if False:
            i = 10
            return i + 15
        self.__actions = value if value is not None else []