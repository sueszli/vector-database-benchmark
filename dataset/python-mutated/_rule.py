from __future__ import annotations
from rich.text import Text
from typing_extensions import Literal
from ..app import RenderResult
from ..css._error_tools import friendly_list
from ..reactive import Reactive, reactive
from ..widget import Widget
RuleOrientation = Literal['horizontal', 'vertical']
'The valid orientations of the rule widget.'
LineStyle = Literal['ascii', 'blank', 'dashed', 'double', 'heavy', 'hidden', 'none', 'solid', 'thick']
'The valid line styles of the rule widget.'
_VALID_RULE_ORIENTATIONS = {'horizontal', 'vertical'}
_VALID_LINE_STYLES = {'ascii', 'blank', 'dashed', 'double', 'heavy', 'hidden', 'none', 'solid', 'thick'}
_HORIZONTAL_LINE_CHARS: dict[LineStyle, str] = {'ascii': '-', 'blank': ' ', 'dashed': '╍', 'double': '═', 'heavy': '━', 'hidden': ' ', 'none': ' ', 'solid': '─', 'thick': '█'}
_VERTICAL_LINE_CHARS: dict[LineStyle, str] = {'ascii': '|', 'blank': ' ', 'dashed': '╏', 'double': '║', 'heavy': '┃', 'hidden': ' ', 'none': ' ', 'solid': '│', 'thick': '█'}

class InvalidRuleOrientation(Exception):
    """Exception raised for an invalid rule orientation."""

class InvalidLineStyle(Exception):
    """Exception raised for an invalid rule line style."""

class Rule(Widget, can_focus=False):
    """A rule widget to separate content, similar to a `<hr>` HTML tag."""
    DEFAULT_CSS = '\n    Rule {\n        color: $primary;\n    }\n\n    Rule.-horizontal {\n        min-height: 1;\n        max-height: 1;\n        margin: 1 0;\n    }\n\n    Rule.-vertical {\n        min-width: 1;\n        max-width: 1;\n        margin: 0 2;\n    }\n    '
    orientation: Reactive[RuleOrientation] = reactive[RuleOrientation]('horizontal')
    'The orientation of the rule.'
    line_style: Reactive[LineStyle] = reactive[LineStyle]('solid')
    'The line style of the rule.'

    def __init__(self, orientation: RuleOrientation='horizontal', line_style: LineStyle='solid', *, name: str | None=None, id: str | None=None, classes: str | None=None, disabled: bool=False) -> None:
        if False:
            print('Hello World!')
        'Initialize a rule widget.\n\n        Args:\n            orientation: The orientation of the rule.\n            line_style: The line style of the rule.\n            name: The name of the widget.\n            id: The ID of the widget in the DOM.\n            classes: The CSS classes of the widget.\n            disabled: Whether the widget is disabled or not.\n        '
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self.orientation = orientation
        self.line_style = line_style

    def render(self) -> RenderResult:
        if False:
            return 10
        rule_char: str
        if self.orientation == 'vertical':
            rule_char = _VERTICAL_LINE_CHARS[self.line_style]
            return Text(rule_char * self.size.height)
        elif self.orientation == 'horizontal':
            rule_char = _HORIZONTAL_LINE_CHARS[self.line_style]
            return Text(rule_char * self.size.width)
        else:
            raise InvalidRuleOrientation(f'Valid rule orientations are {friendly_list(_VALID_RULE_ORIENTATIONS)}')

    def watch_orientation(self, old_orientation: RuleOrientation, orientation: RuleOrientation) -> None:
        if False:
            return 10
        self.remove_class(f'-{old_orientation}')
        self.add_class(f'-{orientation}')

    def validate_orientation(self, orientation: RuleOrientation) -> RuleOrientation:
        if False:
            for i in range(10):
                print('nop')
        if orientation not in _VALID_RULE_ORIENTATIONS:
            raise InvalidRuleOrientation(f'Valid rule orientations are {friendly_list(_VALID_RULE_ORIENTATIONS)}')
        return orientation

    def validate_line_style(self, style: LineStyle) -> LineStyle:
        if False:
            i = 10
            return i + 15
        if style not in _VALID_LINE_STYLES:
            raise InvalidLineStyle(f'Valid rule line styles are {friendly_list(_VALID_LINE_STYLES)}')
        return style

    @classmethod
    def horizontal(cls, line_style: LineStyle='solid', name: str | None=None, id: str | None=None, classes: str | None=None, disabled: bool=False) -> Rule:
        if False:
            print('Hello World!')
        'Utility constructor for creating a horizontal rule.\n\n        Args:\n            line_style: The line style of the rule.\n            name: The name of the widget.\n            id: The ID of the widget in the DOM.\n            classes: The CSS classes of the widget.\n            disabled: Whether the widget is disabled or not.\n\n        Returns:\n            A rule widget with horizontal orientation.\n        '
        return Rule(orientation='horizontal', line_style=line_style, name=name, id=id, classes=classes, disabled=disabled)

    @classmethod
    def vertical(cls, line_style: LineStyle='solid', name: str | None=None, id: str | None=None, classes: str | None=None, disabled: bool=False) -> Rule:
        if False:
            while True:
                i = 10
        'Utility constructor for creating a vertical rule.\n\n        Args:\n            line_style: The line style of the rule.\n            name: The name of the widget.\n            id: The ID of the widget in the DOM.\n            classes: The CSS classes of the widget.\n            disabled: Whether the widget is disabled or not.\n\n        Returns:\n            A rule widget with vertical orientation.\n        '
        return Rule(orientation='vertical', line_style=line_style, name=name, id=id, classes=classes, disabled=disabled)