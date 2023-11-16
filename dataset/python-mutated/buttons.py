""" Various kinds of button widgets.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from typing import TYPE_CHECKING, Callable
from ...core.enums import ButtonType
from ...core.has_props import HasProps, abstract
from ...core.properties import Bool, Either, Enum, Instance, List, Nullable, Override, Required, String, Tuple
from ...events import ButtonClick, MenuItemClick
from ..callbacks import Callback
from ..ui.icons import BuiltinIcon, Icon
from ..ui.tooltips import Tooltip
from .widget import Widget
if TYPE_CHECKING:
    from ...util.callback_manager import EventCallback
__all__ = ('AbstractButton', 'Button', 'ButtonLike', 'Dropdown', 'HelpButton', 'Toggle')

@abstract
class ButtonLike(HasProps):
    """ Shared properties for button-like widgets.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    button_type = Enum(ButtonType, help="\n    A style for the button, signifying it's role. Possible values are one of the\n    following:\n\n    .. bokeh-plot::\n        :source-position: none\n\n        from bokeh.core.enums import ButtonType\n        from bokeh.io import show\n        from bokeh.layouts import column\n        from bokeh.models import Button\n\n        show(column(\n            [Button(label=button_type, button_type=button_type) for button_type in ButtonType]\n            ))\n    ")

@abstract
class AbstractButton(Widget, ButtonLike):
    """ A base class that defines common properties for all button types.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    label = String('Button', help='\n    The text label for the button to display.\n    ')
    icon = Nullable(Instance(Icon), help="\n    An optional image appearing to the left of button's text. An instance of\n    :class:`~bokeh.models.Icon` (such as :class:`~bokeh.models.BuiltinIcon`,\n    :class:`~bokeh.models.SVGIcon`, or :class:`~bokeh.models.TablerIcon`).`\n    ")

class Button(AbstractButton):
    """ A click button.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    label = Override(default='Button')

    def on_click(self, handler: EventCallback) -> None:
        if False:
            while True:
                i = 10
        ' Set up a handler for button clicks.\n\n        Args:\n            handler (func) : handler function to call when button is clicked.\n\n        Returns:\n            None\n\n        '
        self.on_event(ButtonClick, handler)

    def js_on_click(self, handler: Callback) -> None:
        if False:
            while True:
                i = 10
        ' Set up a JavaScript handler for button clicks. '
        self.js_on_event(ButtonClick, handler)

class Toggle(AbstractButton):
    """ A two-state toggle button.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    label = Override(default='Toggle')
    active = Bool(False, help='\n    The state of the toggle button.\n    ')

    def on_click(self, handler: Callable[[bool], None]) -> None:
        if False:
            for i in range(10):
                print('nop')
        ' Set up a handler for button state changes (clicks).\n\n        Args:\n            handler (func) : handler function to call when button is toggled.\n\n        Returns:\n            None\n        '
        self.on_change('active', lambda attr, old, new: handler(new))

    def js_on_click(self, handler: Callback) -> None:
        if False:
            i = 10
            return i + 15
        ' Set up a JavaScript handler for button state changes (clicks). '
        self.js_on_change('active', handler)

class Dropdown(AbstractButton):
    """ A dropdown button.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    label = Override(default='Dropdown')
    split = Bool(default=False, help='\n    ')
    menu = List(Nullable(Either(String, Tuple(String, Either(String, Instance(Callback))))), help="\n    Button's dropdown menu consisting of entries containing item's text and\n    value name. Use ``None`` as a menu separator.\n    ")

    def on_click(self, handler: EventCallback) -> None:
        if False:
            print('Hello World!')
        ' Set up a handler for button or menu item clicks.\n\n        Args:\n            handler (func) : handler function to call when button is activated.\n\n        Returns:\n            None\n\n        '
        self.on_event(ButtonClick, handler)
        self.on_event(MenuItemClick, handler)

    def js_on_click(self, handler: Callback) -> None:
        if False:
            for i in range(10):
                print('nop')
        ' Set up a JavaScript handler for button or menu item clicks. '
        self.js_on_event(ButtonClick, handler)
        self.js_on_event(MenuItemClick, handler)

class HelpButton(AbstractButton):
    """ A button with a help symbol that displays additional text when hovered
    over or clicked.
    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    tooltip = Required(Instance(Tooltip), help="\n    A tooltip with plain text or rich HTML contents, providing general help or\n    description of a widget's or component's function.\n    ")
    label = Override(default='')
    icon = Override(default=lambda : BuiltinIcon('help', size=18))
    button_type = Override(default='default')