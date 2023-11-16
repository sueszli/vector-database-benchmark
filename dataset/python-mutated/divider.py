from typing import Any, Optional
from flet_core.control import Control, OptionalNumber
from flet_core.ref import Ref

class Divider(Control):
    """
    A thin horizontal line, with padding on either side.

    In the material design language, this represents a divider.

    Example:
    ```
    import flet as ft


    def main(page: ft.Page):

        page.add(
            ft.Column(
                [
                    ft.Container(
                        bgcolor=ft.colors.AMBER,
                        alignment=ft.alignment.center,
                        expand=True,
                    ),
                    ft.Divider(),
                    ft.Container(
                        bgcolor=ft.colors.PINK, alignment=ft.alignment.center, expand=True
                    ),
                ],
                spacing=0,
                expand=True,
            ),
        )


    ft.app(target=main)

    ```

    -----

    Online docs: https://flet.dev/docs/controls/divider
    """

    def __init__(self, ref: Optional[Ref]=None, opacity: OptionalNumber=None, visible: Optional[bool]=None, data: Any=None, height: OptionalNumber=None, thickness: OptionalNumber=None, color: Optional[str]=None):
        if False:
            while True:
                i = 10
        Control.__init__(self, ref=ref, opacity=opacity, visible=visible, data=data)
        self.height = height
        self.thickness = thickness
        self.color = color

    def _get_control_name(self):
        if False:
            print('Hello World!')
        return 'divider'

    @property
    def height(self) -> OptionalNumber:
        if False:
            while True:
                i = 10
        return self._get_attr('height')

    @height.setter
    def height(self, value: OptionalNumber):
        if False:
            i = 10
            return i + 15
        self._set_attr('height', value)

    @property
    def thickness(self) -> OptionalNumber:
        if False:
            return 10
        return self._get_attr('thickness')

    @thickness.setter
    def thickness(self, value: OptionalNumber):
        if False:
            return 10
        self._set_attr('thickness', value)

    @property
    def color(self):
        if False:
            print('Hello World!')
        return self._get_attr('color')

    @color.setter
    def color(self, value):
        if False:
            print('Hello World!')
        self._set_attr('color', value)