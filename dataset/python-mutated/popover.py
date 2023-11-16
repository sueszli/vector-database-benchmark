"""Popover components."""
from __future__ import annotations
from typing import Any, Union
from reflex.components.component import Component
from reflex.components.libs.chakra import ChakraComponent, LiteralChakraDirection, LiteralMenuStrategy, LiteralPopOverTrigger
from reflex.vars import Var

class Popover(ChakraComponent):
    """The wrapper that provides props, state, and context to its children."""
    tag = 'Popover'
    arrow_padding: Var[int]
    arrow_shadow_color: Var[str]
    arrow_size: Var[int]
    auto_focus: Var[bool]
    boundary: Var[str]
    close_on_blur: Var[bool]
    close_on_esc: Var[bool]
    default_is_open: Var[bool]
    direction: Var[LiteralChakraDirection]
    flip: Var[bool]
    gutter: Var[int]
    id_: Var[str]
    is_lazy: Var[bool]
    lazy_behavior: Var[str]
    is_open: Var[bool]
    match_width: Var[bool]
    placement: Var[str]
    prevent_overflow: Var[bool]
    return_focus_on_close: Var[bool]
    strategy: Var[LiteralMenuStrategy]
    trigger: Var[LiteralPopOverTrigger]

    def get_event_triggers(self) -> dict[str, Union[Var, Any]]:
        if False:
            i = 10
            return i + 15
        'Get the event triggers for the component.\n\n        Returns:\n            The event triggers.\n        '
        return {**super().get_event_triggers(), 'on_close': lambda : [], 'on_open': lambda : []}

    @classmethod
    def create(cls, *children, trigger=None, header=None, body=None, footer=None, use_close_button=False, **props) -> Component:
        if False:
            while True:
                i = 10
        'Create a popover component.\n\n        Args:\n            *children: The children of the component.\n            trigger: The trigger that opens the popover.\n            header: The header of the popover.\n            body: The body of the popover.\n            footer: The footer of the popover.\n            use_close_button: Whether to add a close button on the popover.\n            **props: The properties of the component.\n\n        Returns:\n            The popover component.\n        '
        if len(children) == 0:
            contents = []
            trigger = PopoverTrigger.create(trigger)
            if header:
                contents.append(PopoverHeader.create(header))
            if body:
                contents.append(PopoverBody.create(body))
            if footer:
                contents.append(PopoverFooter.create(footer))
            if use_close_button:
                contents.append(PopoverCloseButton.create())
            children = [trigger, PopoverContent.create(*contents)]
        return super().create(*children, **props)

class PopoverContent(ChakraComponent):
    """The popover itself."""
    tag = 'PopoverContent'

class PopoverHeader(ChakraComponent):
    """The header of the popover."""
    tag = 'PopoverHeader'

class PopoverFooter(ChakraComponent):
    """Display a popover footer."""
    tag = 'PopoverFooter'

class PopoverBody(ChakraComponent):
    """The body of the popover."""
    tag = 'PopoverBody'

class PopoverArrow(ChakraComponent):
    """A visual arrow that points to the reference (or trigger)."""
    tag = 'PopoverArrow'

class PopoverCloseButton(ChakraComponent):
    """A button to close the popover."""
    tag = 'PopoverCloseButton'

class PopoverAnchor(ChakraComponent):
    """Used to wrap the position-reference element."""
    tag = 'PopoverAnchor'

class PopoverTrigger(ChakraComponent):
    """Used to wrap the reference (or trigger) element."""
    tag = 'PopoverTrigger'