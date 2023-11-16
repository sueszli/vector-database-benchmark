from typing import Any, List, Optional
from flet_core.buttons import OutlinedBorder
from flet_core.control import Control
from flet_core.ref import Ref
from flet_core.types import MainAxisAlignment, MainAxisAlignmentString, PaddingValue

class AlertDialog(Control):
    """
    An alert dialog informs the user about situations that require acknowledgement. An alert dialog has an optional title and an optional list of actions. The title is displayed above the content and the actions are displayed below the content.

    Example:
    ```
    import flet as ft

    def main(page: ft.Page):
        page.title = "AlertDialog examples"

        dlg = ft.AlertDialog(
            title=ft.Text("Hello, you!"), on_dismiss=lambda e: print("Dialog dismissed!")
        )

        def close_dlg(e):
            dlg_modal.open = False
            page.update()

        dlg_modal = ft.AlertDialog(
            modal=True,
            title=ft.Text("Please confirm"),
            content=ft.Text("Do you really want to delete all those files?"),
            actions=[
                ft.TextButton("Yes", on_click=close_dlg),
                ft.TextButton("No", on_click=close_dlg),
            ],
            actions_alignment=ft.MainAxisAlignment.END,
            on_dismiss=lambda e: print("Modal dialog dismissed!"),
        )

        def open_dlg(e):
            page.dialog = dlg
            dlg.open = True
            page.update()

        def open_dlg_modal(e):
            page.dialog = dlg_modal
            dlg_modal.open = True
            page.update()

        page.add(
            ft.ElevatedButton("Open dialog", on_click=open_dlg),
            ft.ElevatedButton("Open modal dialog", on_click=open_dlg_modal),
        )

    ft.app(target=main)
    ```
    -----

    Online docs: https://flet.dev/docs/controls/alertdialog
    """

    def __init__(self, ref: Optional[Ref]=None, disabled: Optional[bool]=None, visible: Optional[bool]=None, data: Any=None, open: bool=False, modal: bool=False, title: Optional[Control]=None, title_padding: PaddingValue=None, content: Optional[Control]=None, content_padding: PaddingValue=None, actions: Optional[List[Control]]=None, actions_padding: PaddingValue=None, actions_alignment: MainAxisAlignment=MainAxisAlignment.NONE, shape: Optional[OutlinedBorder]=None, inset_padding: PaddingValue=None, on_dismiss=None):
        if False:
            return 10
        Control.__init__(self, ref=ref, disabled=disabled, visible=visible, data=data)
        self.__title: Optional[Control] = None
        self.__content: Optional[Control] = None
        self.__actions: List[Control] = []
        self.open = open
        self.modal = modal
        self.title = title
        self.title_padding = title_padding
        self.content = content
        self.content_padding = content_padding
        self.actions = actions
        self.actions_padding = actions_padding
        self.actions_alignment = actions_alignment
        self.shape = shape
        self.inset_padding = inset_padding
        self.on_dismiss = on_dismiss

    def _get_control_name(self):
        if False:
            return 10
        return 'alertdialog'

    def _before_build_command(self):
        if False:
            return 10
        super()._before_build_command()
        self._set_attr_json('actionsPadding', self.__actions_padding)
        self._set_attr_json('contentPadding', self.__content_padding)
        self._set_attr_json('titlePadding', self.__title_padding)
        self._set_attr_json('shape', self.__shape)
        self._set_attr_json('insetPadding', self.__inset_padding)

    def _get_children(self):
        if False:
            while True:
                i = 10
        children = []
        if self.__title:
            self.__title._set_attr_internal('n', 'title')
            children.append(self.__title)
        if self.__content:
            self.__content._set_attr_internal('n', 'content')
            children.append(self.__content)
        for action in self.__actions:
            action._set_attr_internal('n', 'action')
            children.append(action)
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
            while True:
                i = 10
        self._set_attr('open', value)

    @property
    def modal(self) -> Optional[bool]:
        if False:
            return 10
        return self._get_attr('modal', data_type='bool', def_value=False)

    @modal.setter
    def modal(self, value: Optional[bool]):
        if False:
            print('Hello World!')
        self._set_attr('modal', value)

    @property
    def title(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__title

    @title.setter
    def title(self, value):
        if False:
            print('Hello World!')
        self.__title = value

    @property
    def title_padding(self) -> PaddingValue:
        if False:
            print('Hello World!')
        return self.__title_padding

    @title_padding.setter
    def title_padding(self, value: PaddingValue):
        if False:
            while True:
                i = 10
        self.__title_padding = value

    @property
    def content(self):
        if False:
            return 10
        return self.__content

    @content.setter
    def content(self, value):
        if False:
            i = 10
            return i + 15
        self.__content = value

    @property
    def content_padding(self) -> PaddingValue:
        if False:
            i = 10
            return i + 15
        return self.__content_padding

    @content_padding.setter
    def content_padding(self, value: PaddingValue):
        if False:
            i = 10
            return i + 15
        self.__content_padding = value

    @property
    def actions(self):
        if False:
            i = 10
            return i + 15
        return self.__actions

    @actions.setter
    def actions(self, value):
        if False:
            return 10
        self.__actions = value if value is not None else []

    @property
    def actions_padding(self) -> PaddingValue:
        if False:
            print('Hello World!')
        return self.__actions_padding

    @actions_padding.setter
    def actions_padding(self, value: PaddingValue):
        if False:
            while True:
                i = 10
        self.__actions_padding = value

    @property
    def actions_alignment(self) -> MainAxisAlignment:
        if False:
            while True:
                i = 10
        return self.__actions_alignment

    @actions_alignment.setter
    def actions_alignment(self, value: MainAxisAlignment):
        if False:
            print('Hello World!')
        self.__actions_alignment = value
        if isinstance(value, MainAxisAlignment):
            self._set_attr('actionsAlignment', value.value)
        else:
            self.__set_actions_alignment(value)

    def __set_actions_alignment(self, value: MainAxisAlignmentString):
        if False:
            while True:
                i = 10
        self._set_attr('actionsAlignment', value)

    @property
    def shape(self) -> Optional[OutlinedBorder]:
        if False:
            print('Hello World!')
        return self.__shape

    @shape.setter
    def shape(self, value: Optional[OutlinedBorder]):
        if False:
            while True:
                i = 10
        self.__shape = value

    @property
    def inset_padding(self) -> PaddingValue:
        if False:
            print('Hello World!')
        return self.__inset_padding

    @inset_padding.setter
    def inset_padding(self, value: PaddingValue):
        if False:
            while True:
                i = 10
        self.__inset_padding = value

    @property
    def on_dismiss(self):
        if False:
            for i in range(10):
                print('nop')
        return self._get_event_handler('dismiss')

    @on_dismiss.setter
    def on_dismiss(self, handler):
        if False:
            i = 10
            return i + 15
        self._add_event_handler('dismiss', handler)