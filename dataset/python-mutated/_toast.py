"""Widgets for showing notification messages in toasts."""
from __future__ import annotations
from typing import ClassVar
from rich.console import RenderableType
from rich.text import Text
from .. import on
from ..containers import Container
from ..css.query import NoMatches
from ..events import Click, Mount
from ..notifications import Notification, Notifications
from ._static import Static

class ToastHolder(Container, inherit_css=False):
    """Container that holds a single toast.

    Used to control the alignment of each of the toasts in the main toast
    container.
    """
    DEFAULT_CSS = '\n    ToastHolder {\n        align-horizontal: right;\n        width: 1fr;\n        height: auto;\n        visibility: hidden;\n    }\n    '

class Toast(Static, inherit_css=False):
    """A widget for displaying short-lived notifications."""
    DEFAULT_CSS = '\n    Toast {\n        width: 60;\n        max-width: 50%;\n        height: auto;\n        visibility: visible;\n        margin-top: 1;\n        padding: 1 1;\n        background: $panel;\n        tint: white 5%;\n        link-background: initial;\n        link-color: $text;\n        link-style: underline;\n        link-hover-background: $accent;\n        link-hover-color: $text;\n        link-hover-style: bold not underline;\n    }\n\n    .toast--title {\n        text-style: bold;\n    }\n\n    Toast {\n        border-right: wide $background;\n    }\n\n    Toast.-information {\n        border-left: wide $success;\n    }\n\n    Toast.-information .toast--title {\n        color: $success-darken-1;\n    }\n\n    Toast.-warning {\n        border-left: wide $warning;\n    }\n\n    Toast.-warning .toast--title {\n        color: $warning-darken-1;\n    }\n\n    Toast.-error {\n        border-left: wide $error;\n    }\n\n    Toast.-error .toast--title {\n       color: $error-darken-1;\n    }\n    '
    COMPONENT_CLASSES: ClassVar[set[str]] = {'toast--title'}
    '\n    | Class | Description |\n    | :- | :- |\n    | `toast--title` | Targets the title of the toast. |\n    '

    def __init__(self, notification: Notification) -> None:
        if False:
            i = 10
            return i + 15
        'Initialise the toast.\n\n        Args:\n            notification: The notification to show in the toast.\n        '
        super().__init__(classes=f'-{notification.severity}')
        self._notification = notification
        self._timeout = notification.time_left

    def render(self) -> RenderableType:
        if False:
            while True:
                i = 10
        "Render the toast's content.\n\n        Returns:\n            A Rich renderable for the title and content of the Toast.\n        "
        notification = self._notification
        if notification.title:
            header_style = self.get_component_rich_style('toast--title')
            notification_text = Text.assemble((notification.title, header_style), '\n', Text.from_markup(notification.message))
        else:
            notification_text = Text.assemble(Text.from_markup(notification.message))
        return notification_text

    def _on_mount(self, _: Mount) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the time running once the toast is mounted.'
        self.set_timer(self._timeout, self._expire)

    @on(Click)
    def _expire(self) -> None:
        if False:
            return 10
        'Remove the toast once the timer has expired.'
        self.app._unnotify(self._notification, refresh=False)
        (self.parent if isinstance(self.parent, ToastHolder) else self).remove()

class ToastRack(Container, inherit_css=False):
    """A container for holding toasts."""
    DEFAULT_CSS = '\n    ToastRack {\n        layer: _toastrack;\n        width: 1fr;\n        height: auto;\n        dock: top;\n        align: right bottom;\n        visibility: hidden;\n        layout: vertical;\n        overflow-y: scroll;\n        margin-bottom: 1;\n        margin-right: 1;\n    }\n    '

    @staticmethod
    def _toast_id(notification: Notification) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Create a Textual-DOM-internal ID for the given notification.\n\n        Args:\n            notification: The notification to create the ID for.\n\n        Returns:\n            An ID for the notification that can be used within the DOM.\n        '
        return f'--textual-toast-{notification.identity}'

    def show(self, notifications: Notifications) -> None:
        if False:
            return 10
        'Show the notifications as toasts.\n\n        Args:\n            notifications: The notifications to show.\n        '
        for toast in self.query(Toast):
            if toast._notification not in notifications:
                toast.remove()
        new_toasts: list[Notification] = []
        for notification in notifications:
            try:
                _ = self.get_child_by_id(self._toast_id(notification))
            except NoMatches:
                if not notification.has_expired:
                    new_toasts.append(notification)
        if new_toasts:
            self.mount_all((ToastHolder(Toast(toast), id=self._toast_id(toast)) for toast in new_toasts))
            self.call_later(self.scroll_end, animate=False, force=True)