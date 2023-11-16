"""Alert dialog components."""
from __future__ import annotations
from typing import Any, Union
from reflex.components.component import Component
from reflex.components.libs.chakra import ChakraComponent, LiteralAlertDialogSize
from reflex.components.media.icon import Icon
from reflex.vars import Var

class AlertDialog(ChakraComponent):
    """Provides context and state for the dialog."""
    tag = 'AlertDialog'
    is_open: Var[bool]
    least_destructive_ref: Var[str]
    allow_pinch_zoom: Var[bool]
    auto_focus: Var[bool]
    block_scroll_on_mount: Var[bool]
    close_on_esc: Var[bool]
    close_on_overlay_click: Var[bool]
    is_centered: Var[bool]
    lock_focus_across_frames: Var[bool]
    preserve_scroll_bar_gap: Var[bool]
    return_focus_on_close: Var[bool]
    size: Var[LiteralAlertDialogSize]
    use_inert: Var[bool]

    def get_event_triggers(self) -> dict[str, Union[Var, Any]]:
        if False:
            while True:
                i = 10
        'Get the event triggers for the component.\n\n        Returns:\n            The event triggers.\n        '
        return {**super().get_event_triggers(), 'on_close': lambda : [], 'on_close_complete': lambda : [], 'on_esc': lambda : [], 'on_overlay_click': lambda : []}

    @classmethod
    def create(cls, *children, header=None, body=None, footer=None, close_button=None, **props) -> Component:
        if False:
            for i in range(10):
                print('nop')
        'Create an alert dialog component.\n\n        Args:\n            *children: The children of the alert dialog component.\n            header: The header of the alert dialog.\n            body: The body of the alert dialog.\n            footer: The footer of the alert dialog.\n            close_button: The close button of the alert dialog.\n            **props: The properties of the alert dialog component.\n\n        Raises:\n            AttributeError: if there is a conflict between the props used.\n\n        Returns:\n            The alert dialog component.\n        '
        if len(children) == 0:
            contents = []
            if header:
                contents.append(AlertDialogHeader.create(header))
            if body:
                contents.append(AlertDialogBody.create(body))
            if footer:
                contents.append(AlertDialogFooter.create(footer))
            if props.get('on_close'):
                if not close_button:
                    close_button = Icon.create(tag='close')
                contents.append(AlertDialogCloseButton.create(close_button))
            elif close_button:
                raise AttributeError('Close button can not be used if on_close event handler is not defined')
            children = [AlertDialogOverlay.create(AlertDialogContent.create(*contents))]
        return super().create(*children, **props)

class AlertDialogBody(ChakraComponent):
    """Should contain the description announced by screen readers."""
    tag = 'AlertDialogBody'

class AlertDialogHeader(ChakraComponent):
    """Should contain the title announced by screen readers."""
    tag = 'AlertDialogHeader'

class AlertDialogFooter(ChakraComponent):
    """Should contain the events of the dialog."""
    tag = 'AlertDialogFooter'

class AlertDialogContent(ChakraComponent):
    """The wrapper for the alert dialog's content."""
    tag = 'AlertDialogContent'

class AlertDialogOverlay(ChakraComponent):
    """The dimmed overlay behind the dialog."""
    tag = 'AlertDialogOverlay'

class AlertDialogCloseButton(ChakraComponent):
    """The button that closes the dialog."""
    tag = 'AlertDialogCloseButton'