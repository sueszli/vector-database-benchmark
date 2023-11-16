from typing import Any, List, Optional
from flet_core.control import Control
from flet_core.ref import Ref

class BottomSheet(Control):
    """
    A modal bottom sheet is an alternative to a menu or a dialog and prevents the user from interacting with the rest of the app.

    Example:
    ```
    import flet as ft

    def main(page: ft.Page):
        def bs_dismissed(e):
            print("Dismissed!")

        def show_bs(e):
            bs.open = True
            bs.update()

        def close_bs(e):
            bs.open = False
            bs.update()

        bs = ft.BottomSheet(
            ft.Container(
                ft.Column(
                    [
                        ft.Text("This is sheet's content!"),
                        ft.ElevatedButton("Close bottom sheet", on_click=close_bs),
                    ],
                    tight=True,
                ),
                padding=10,
            ),
            open=True,
            on_dismiss=bs_dismissed,
        )
        page.overlay.append(bs)
        page.add(ft.ElevatedButton("Display bottom sheet", on_click=show_bs))

    ft.app(target=main)
    ```

    -----

    Online docs: https://flet.dev/docs/controls/bottomsheet
    """

    def __init__(self, content: Optional[Control]=None, ref: Optional[Ref]=None, disabled: Optional[bool]=None, visible: Optional[bool]=None, data: Any=None, open: bool=False, dismissible: Optional[bool]=None, enable_drag: Optional[bool]=None, show_drag_handle: Optional[bool]=None, use_safe_area: Optional[bool]=None, is_scroll_controlled: Optional[bool]=None, maintain_bottom_view_insets_padding: Optional[bool]=None, on_dismiss=None):
        if False:
            print('Hello World!')
        Control.__init__(self, ref=ref, disabled=disabled, visible=visible, data=data)
        self.__content: Optional[Control] = None
        self.open = open
        self.dismissible = dismissible
        self.enable_drag = enable_drag
        self.show_drag_handle = show_drag_handle
        self.use_safe_area = use_safe_area
        self.is_scroll_controlled = is_scroll_controlled
        self.content = content
        self.maintain_bottom_view_insets_padding = maintain_bottom_view_insets_padding
        self.on_dismiss = on_dismiss

    def _get_control_name(self):
        if False:
            return 10
        return 'bottomsheet'

    def _get_children(self):
        if False:
            i = 10
            return i + 15
        children = []
        if self.__content:
            self.__content._set_attr_internal('n', 'content')
            children.append(self.__content)
        return children

    @property
    def open(self) -> Optional[bool]:
        if False:
            i = 10
            return i + 15
        return self._get_attr('open', data_type='bool', def_value=False)

    @open.setter
    def open(self, value: Optional[bool]):
        if False:
            for i in range(10):
                print('nop')
        self._set_attr('open', value)

    @property
    def dismissible(self) -> Optional[bool]:
        if False:
            return 10
        return self._get_attr('dismissible', data_type='bool', def_value=True)

    @dismissible.setter
    def dismissible(self, value: Optional[bool]):
        if False:
            while True:
                i = 10
        self._set_attr('dismissible', value)

    @property
    def enable_drag(self) -> Optional[bool]:
        if False:
            return 10
        return self._get_attr('enableDrag', data_type='bool', def_value=False)

    @enable_drag.setter
    def enable_drag(self, value: Optional[bool]):
        if False:
            i = 10
            return i + 15
        self._set_attr('enableDrag', value)

    @property
    def show_drag_handle(self) -> Optional[bool]:
        if False:
            return 10
        return self._get_attr('showDragHandle', data_type='bool', def_value=False)

    @show_drag_handle.setter
    def show_drag_handle(self, value: Optional[bool]):
        if False:
            return 10
        self._set_attr('showDragHandle', value)

    @property
    def use_safe_area(self) -> Optional[bool]:
        if False:
            i = 10
            return i + 15
        return self._get_attr('useSafeArea', data_type='bool', def_value=True)

    @use_safe_area.setter
    def use_safe_area(self, value: Optional[bool]):
        if False:
            print('Hello World!')
        self._set_attr('useSafeArea', value)

    @property
    def is_scroll_controlled(self) -> Optional[bool]:
        if False:
            while True:
                i = 10
        return self._get_attr('isScrollControlled', data_type='bool', def_value=False)

    @is_scroll_controlled.setter
    def is_scroll_controlled(self, value: Optional[bool]):
        if False:
            while True:
                i = 10
        self._set_attr('isScrollControlled', value)

    @property
    def maintain_bottom_view_insets_padding(self) -> Optional[bool]:
        if False:
            while True:
                i = 10
        return self._get_attr('maintainBottomViewInsetsPadding', data_type='bool', def_value=True)

    @maintain_bottom_view_insets_padding.setter
    def maintain_bottom_view_insets_padding(self, value: Optional[bool]):
        if False:
            return 10
        self._set_attr('maintainBottomViewInsetsPadding', value)

    @property
    def content(self):
        if False:
            return 10
        return self.__content

    @content.setter
    def content(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.__content = value

    @property
    def on_dismiss(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_event_handler('dismiss')

    @on_dismiss.setter
    def on_dismiss(self, handler):
        if False:
            return 10
        self._add_event_handler('dismiss', handler)