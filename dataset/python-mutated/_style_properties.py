"""
Style properties are descriptors which allow the ``Styles`` object to accept different types when
setting attributes. This gives the developer more freedom in how to express style information.

Descriptors also play nicely with Mypy, which is aware that attributes can have different types
when setting and getting.
"""
from __future__ import annotations
from operator import attrgetter
from typing import TYPE_CHECKING, Generic, Iterable, NamedTuple, Sequence, TypeVar, cast
import rich.errors
import rich.repr
from rich.style import Style
from typing_extensions import TypeAlias
from .._border import normalize_border_value
from ..color import Color, ColorParseError
from ..geometry import Spacing, SpacingDimensions, clamp
from ._error_tools import friendly_list
from ._help_text import border_property_help_text, color_property_help_text, fractional_property_help_text, layout_property_help_text, offset_property_help_text, scalar_help_text, spacing_wrong_number_of_values_help_text, string_enum_help_text, style_flags_property_help_text
from .constants import NULL_SPACING, VALID_STYLE_FLAGS
from .errors import StyleTypeError, StyleValueError
from .scalar import NULL_SCALAR, UNIT_SYMBOL, Scalar, ScalarOffset, ScalarParseError, Unit, get_symbols, percentage_string_to_float
from .transition import Transition
if TYPE_CHECKING:
    from .._layout import Layout
    from .styles import StylesBase
from .types import AlignHorizontal, AlignVertical, DockEdge, EdgeType
BorderDefinition: TypeAlias = 'Sequence[tuple[EdgeType, str | Color] | None] | tuple[EdgeType, str | Color]'
PropertyGetType = TypeVar('PropertyGetType')
PropertySetType = TypeVar('PropertySetType')

class GenericProperty(Generic[PropertyGetType, PropertySetType]):
    """Descriptor that abstracts away common machinery for other style descriptors.

    Args:
        default: The default value (or a factory thereof) of the property.
        layout: Whether to refresh the node layout on value change.
        refresh_children: Whether to refresh the node children on value change.
    """

    def __init__(self, default: PropertyGetType, layout: bool=False, refresh_children: bool=False) -> None:
        if False:
            return 10
        self.default = default
        self.layout = layout
        self.refresh_children = refresh_children

    def validate_value(self, value: object) -> PropertyGetType:
        if False:
            print('Hello World!')
        'Validate the setter value.\n\n        Args:\n            value: The value being set.\n\n        Returns:\n            The value to be set.\n        '
        return cast(PropertyGetType, value)

    def __set_name__(self, owner: StylesBase, name: str) -> None:
        if False:
            while True:
                i = 10
        self.name = name

    def __get__(self, obj: StylesBase, objtype: type[StylesBase] | None=None) -> PropertyGetType:
        if False:
            return 10
        return cast(PropertyGetType, obj.get_rule(self.name, self.default))

    def __set__(self, obj: StylesBase, value: PropertySetType | None) -> None:
        if False:
            return 10
        _rich_traceback_omit = True
        if value is None:
            obj.clear_rule(self.name)
            obj.refresh(layout=self.layout, children=self.refresh_children)
            return
        new_value = self.validate_value(value)
        if obj.set_rule(self.name, new_value):
            obj.refresh(layout=self.layout, children=self.refresh_children)

class IntegerProperty(GenericProperty[int, int]):

    def validate_value(self, value: object) -> int:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(value, (int, float)):
            return int(value)
        else:
            raise StyleValueError(f'Expected a number here, got f{value}')

class BooleanProperty(GenericProperty[bool, bool]):
    """A property that requires a True or False value."""

    def validate_value(self, value: object) -> bool:
        if False:
            while True:
                i = 10
        return bool(value)

class ScalarProperty:
    """Descriptor for getting and setting scalar properties. Scalars are numeric values with a unit, e.g. "50vh"."""

    def __init__(self, units: set[Unit] | None=None, percent_unit: Unit=Unit.WIDTH, allow_auto: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        self.units: set[Unit] = units or {*UNIT_SYMBOL}
        self.percent_unit = percent_unit
        self.allow_auto = allow_auto
        super().__init__()

    def __set_name__(self, owner: StylesBase, name: str) -> None:
        if False:
            i = 10
            return i + 15
        self.name = name

    def __get__(self, obj: StylesBase, objtype: type[StylesBase] | None=None) -> Scalar | None:
        if False:
            while True:
                i = 10
        "Get the scalar property.\n\n        Args:\n            obj: The ``Styles`` object.\n            objtype: The ``Styles`` class.\n\n        Returns:\n            The Scalar object or ``None`` if it's not set.\n        "
        return cast('Scalar | None', obj.get_rule(self.name))

    def __set__(self, obj: StylesBase, value: float | int | Scalar | str | None) -> None:
        if False:
            print('Hello World!')
        'Set the scalar property.\n\n        Args:\n            obj: The ``Styles`` object.\n            value: The value to set the scalar property to.\n                You can directly pass a float or int value, which will be interpreted with\n                a default unit of Cells. You may also provide a string such as ``"50%"``,\n                as you might do when writing CSS. If a string with no units is supplied,\n                Cells will be used as the unit. Alternatively, you can directly supply\n                a ``Scalar`` object.\n\n        Raises:\n            StyleValueError: If the value is of an invalid type, uses an invalid unit, or\n                cannot be parsed for any other reason.\n        '
        _rich_traceback_omit = True
        if value is None:
            obj.clear_rule(self.name)
            obj.refresh(layout=True)
            return
        if isinstance(value, (int, float)):
            new_value = Scalar(float(value), Unit.CELLS, Unit.WIDTH)
        elif isinstance(value, Scalar):
            new_value = value
        elif isinstance(value, str):
            try:
                new_value = Scalar.parse(value)
            except ScalarParseError:
                raise StyleValueError(f'unable to parse scalar from {value!r}', help_text=scalar_help_text(property_name=self.name, context='inline'))
        else:
            raise StyleValueError('expected float, int, Scalar, or None')
        if new_value is not None and new_value.unit == Unit.AUTO and (not self.allow_auto):
            raise StyleValueError("'auto' not allowed here")
        if new_value is not None and new_value.unit != Unit.AUTO:
            if new_value.unit not in self.units:
                raise StyleValueError(f'{self.name} units must be one of {friendly_list(get_symbols(self.units))}')
            if new_value.is_percent:
                new_value = Scalar(float(new_value.value), self.percent_unit, Unit.WIDTH)
        if obj.set_rule(self.name, new_value):
            obj.refresh(layout=True)

class ScalarListProperty:
    """Descriptor for lists of scalars.

    Args:
        percent_unit: The dimension to which percentage scalars will be relative to.
        refresh_children: Whether to refresh the node children on value change.
    """

    def __init__(self, percent_unit: Unit, refresh_children: bool=False) -> None:
        if False:
            while True:
                i = 10
        self.percent_unit = percent_unit
        self.refresh_children = refresh_children

    def __set_name__(self, owner: StylesBase, name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.name = name

    def __get__(self, obj: StylesBase, objtype: type[StylesBase] | None=None) -> tuple[Scalar, ...] | None:
        if False:
            for i in range(10):
                print('nop')
        return cast('tuple[Scalar, ...]', obj.get_rule(self.name))

    def __set__(self, obj: StylesBase, value: str | Iterable[str | float] | None) -> None:
        if False:
            print('Hello World!')
        if value is None:
            obj.clear_rule(self.name)
            obj.refresh(layout=True, children=self.refresh_children)
            return
        parse_values: Iterable[str | float]
        if isinstance(value, str):
            parse_values = value.split()
        else:
            parse_values = value
        scalars = []
        for parse_value in parse_values:
            if isinstance(parse_value, (int, float)):
                scalars.append(Scalar.from_number(parse_value))
            else:
                scalars.append(Scalar.parse(parse_value, self.percent_unit) if isinstance(parse_value, str) else parse_value)
        if obj.set_rule(self.name, tuple(scalars)):
            obj.refresh(layout=True, children=self.refresh_children)

class BoxProperty:
    """Descriptor for getting and setting outlines and borders along a single edge.
    For example "border-right", "outline-bottom", etc.
    """

    def __init__(self, default_color: Color) -> None:
        if False:
            while True:
                i = 10
        self._default_color = default_color

    def __set_name__(self, owner: StylesBase, name: str) -> None:
        if False:
            print('Hello World!')
        self.name = name
        (_type, edge) = name.split('_')
        self._type = _type
        self.edge = edge

    def __get__(self, obj: StylesBase, objtype: type[StylesBase] | None=None) -> tuple[EdgeType, Color]:
        if False:
            i = 10
            return i + 15
        'Get the box property.\n\n        Args:\n            obj: The ``Styles`` object.\n            objtype: The ``Styles`` class.\n\n        Returns:\n            A ``tuple[EdgeType, Style]`` containing the string type of the box and\n                it\'s style. Example types are "rounded", "solid", and "dashed".\n        '
        return cast('tuple[EdgeType, Color]', obj.get_rule(self.name) or ('', self._default_color))

    def __set__(self, obj: StylesBase, border: tuple[EdgeType, str | Color] | None):
        if False:
            for i in range(10):
                print('nop')
        'Set the box property.\n\n        Args:\n            obj: The ``Styles`` object.\n            value: A 2-tuple containing the type of box to use,\n                e.g. "dashed", and the ``Style`` to be used. You can supply the ``Style`` directly, or pass a\n                ``str`` (e.g. ``"blue on #f0f0f0"`` ) or ``Color`` instead.\n\n        Raises:\n            StyleSyntaxError: If the string supplied for the color has invalid syntax.\n        '
        _rich_traceback_omit = True
        if border is None:
            if obj.clear_rule(self.name):
                obj.refresh(layout=True)
        else:
            (_type, color) = border
            if _type in ('none', 'hidden'):
                _type = ''
            new_value = border
            if isinstance(color, str):
                try:
                    new_value = (_type, Color.parse(color))
                except ColorParseError as error:
                    raise StyleValueError(str(error), help_text=border_property_help_text(self.name, context='inline'))
            elif isinstance(color, Color):
                new_value = (_type, color)
            current_value: tuple[str, Color] = cast('tuple[str, Color]', obj.get_rule(self.name))
            has_edge = bool(current_value and current_value[0])
            new_edge = bool(_type)
            if obj.set_rule(self.name, new_value):
                obj.refresh(layout=has_edge != new_edge)

@rich.repr.auto
class Edges(NamedTuple):
    """Stores edges for border / outline."""
    top: tuple[EdgeType, Color]
    right: tuple[EdgeType, Color]
    bottom: tuple[EdgeType, Color]
    left: tuple[EdgeType, Color]

    def __bool__(self) -> bool:
        if False:
            i = 10
            return i + 15
        ((top, _), (right, _), (bottom, _), (left, _)) = self
        return bool(top or right or bottom or left)

    def __rich_repr__(self) -> rich.repr.Result:
        if False:
            i = 10
            return i + 15
        (top, right, bottom, left) = self
        if top[0]:
            yield ('top', top)
        if right[0]:
            yield ('right', right)
        if bottom[0]:
            yield ('bottom', bottom)
        if left[0]:
            yield ('left', left)

    @property
    def spacing(self) -> Spacing:
        if False:
            for i in range(10):
                print('nop')
        'Get spacing created by borders.\n\n        Returns:\n            Spacing for top, right, bottom, and left.\n        '
        ((top, _), (right, _), (bottom, _), (left, _)) = self
        return Spacing(1 if top else 0, 1 if right else 0, 1 if bottom else 0, 1 if left else 0)

class BorderProperty:
    """Descriptor for getting and setting full borders and outlines.

    Args:
        layout: True if the layout should be refreshed after setting, False otherwise.
    """

    def __init__(self, layout: bool) -> None:
        if False:
            return 10
        self._layout = layout

    def __set_name__(self, owner: StylesBase, name: str) -> None:
        if False:
            i = 10
            return i + 15
        self.name = name
        self._properties = (f'{name}_top', f'{name}_right', f'{name}_bottom', f'{name}_left')
        self._get_properties = attrgetter(*self._properties)

    def __get__(self, obj: StylesBase, objtype: type[StylesBase] | None=None) -> Edges:
        if False:
            return 10
        'Get the border.\n\n        Args:\n            obj: The ``Styles`` object.\n            objtype: The ``Styles`` class.\n\n        Returns:\n            An ``Edges`` object describing the type and style of each edge.\n        '
        return Edges(*self._get_properties(obj))

    def __set__(self, obj: StylesBase, border: BorderDefinition | None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the border.\n\n        Args:\n            obj: The ``Styles`` object.\n            border:\n                A ``tuple[EdgeType, str | Color | Style]`` representing the type of box to use and the ``Style`` to apply\n                to the box.\n                Alternatively, you can supply a sequence of these tuples and they will be applied per-edge.\n                If the sequence is of length 1, all edges will be decorated according to the single element.\n                If the sequence is length 2, the first ``tuple`` will be applied to the top and bottom edges.\n                If the sequence is length 4, the tuples will be applied to the edges in the order: top, right, bottom, left.\n\n        Raises:\n            StyleValueError: When the supplied ``tuple`` is not of valid length (1, 2, or 4).\n        '
        _rich_traceback_omit = True
        (top, right, bottom, left) = self._properties
        border_spacing = Edges(*self._get_properties(obj)).spacing

        def check_refresh() -> None:
            if False:
                i = 10
                return i + 15
            'Check if an update requires a layout'
            if not self._layout:
                obj.refresh()
            else:
                layout = Edges(*self._get_properties(obj)).spacing != border_spacing
                obj.refresh(layout=layout)
        if border is None:
            clear_rule = obj.clear_rule
            clear_rule(top)
            clear_rule(right)
            clear_rule(bottom)
            clear_rule(left)
            check_refresh()
            return
        if isinstance(border, tuple) and len(border) == 2:
            _border = normalize_border_value(border)
            setattr(obj, top, _border)
            setattr(obj, right, _border)
            setattr(obj, bottom, _border)
            setattr(obj, left, _border)
            check_refresh()
            return
        count = len(border)
        if count == 1:
            _border = normalize_border_value(border[0])
            setattr(obj, top, _border)
            setattr(obj, right, _border)
            setattr(obj, bottom, _border)
            setattr(obj, left, _border)
        elif count == 2:
            (_border1, _border2) = (normalize_border_value(border[0]), normalize_border_value(border[1]))
            setattr(obj, top, _border1)
            setattr(obj, bottom, _border1)
            setattr(obj, right, _border2)
            setattr(obj, left, _border2)
        elif count == 4:
            (_border1, _border2, _border3, _border4) = (normalize_border_value(border[0]), normalize_border_value(border[1]), normalize_border_value(border[2]), normalize_border_value(border[3]))
            setattr(obj, top, _border1)
            setattr(obj, right, _border2)
            setattr(obj, bottom, _border3)
            setattr(obj, left, _border4)
        else:
            raise StyleValueError('expected 1, 2, or 4 values', help_text=border_property_help_text(self.name, context='inline'))
        check_refresh()

class SpacingProperty:
    """Descriptor for getting and setting spacing properties (e.g. padding and margin)."""

    def __set_name__(self, owner: StylesBase, name: str) -> None:
        if False:
            while True:
                i = 10
        self.name = name

    def __get__(self, obj: StylesBase, objtype: type[StylesBase] | None=None) -> Spacing:
        if False:
            return 10
        'Get the Spacing.\n\n        Args:\n            obj: The ``Styles`` object.\n            objtype: The ``Styles`` class.\n\n        Returns:\n            The Spacing. If unset, returns the null spacing ``(0, 0, 0, 0)``.\n        '
        return cast(Spacing, obj.get_rule(self.name, NULL_SPACING))

    def __set__(self, obj: StylesBase, spacing: SpacingDimensions | None):
        if False:
            for i in range(10):
                print('nop')
        'Set the Spacing.\n\n        Args:\n            obj: The ``Styles`` object.\n            style: You can supply the ``Style`` directly, or a\n                string (e.g. ``"blue on #f0f0f0"``).\n\n        Raises:\n            ValueError: When the value is malformed,\n                e.g. a ``tuple`` with a length that is not 1, 2, or 4.\n        '
        _rich_traceback_omit = True
        if spacing is None:
            if obj.clear_rule(self.name):
                obj.refresh(layout=True)
        else:
            try:
                unpacked_spacing = Spacing.unpack(spacing)
            except ValueError as error:
                raise StyleValueError(str(error), help_text=spacing_wrong_number_of_values_help_text(property_name=self.name, num_values_supplied=1 if isinstance(spacing, int) else len(spacing), context='inline'))
            if obj.set_rule(self.name, unpacked_spacing):
                obj.refresh(layout=True)

class DockProperty:
    """Descriptor for getting and setting the dock property. The dock property
    allows you to specify which edge you want to fix a Widget to.
    """

    def __get__(self, obj: StylesBase, objtype: type[StylesBase] | None=None) -> DockEdge:
        if False:
            return 10
        'Get the Dock property.\n\n        Args:\n            obj: The ``Styles`` object.\n            objtype: The ``Styles`` class.\n\n        Returns:\n            The dock name as a string, or "" if the rule is not set.\n        '
        return cast(DockEdge, obj.get_rule('dock', ''))

    def __set__(self, obj: StylesBase, dock_name: str | None):
        if False:
            print('Hello World!')
        'Set the Dock property.\n\n        Args:\n            obj: The ``Styles`` object.\n            dock_name: The name of the dock to attach this widget to.\n        '
        _rich_traceback_omit = True
        if obj.set_rule('dock', dock_name):
            obj.refresh(layout=True)

class LayoutProperty:
    """Descriptor for getting and setting layout."""

    def __set_name__(self, owner: StylesBase, name: str) -> None:
        if False:
            return 10
        self.name = name

    def __get__(self, obj: StylesBase, objtype: type[StylesBase] | None=None) -> Layout | None:
        if False:
            print('Hello World!')
        '\n        Args:\n            obj: The Styles object.\n            objtype: The Styles class.\n        Returns:\n            The ``Layout`` object.\n        '
        return cast('Layout | None', obj.get_rule(self.name))

    def __set__(self, obj: StylesBase, layout: str | Layout | None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Args:\n            obj: The Styles object.\n            layout: The layout to use. You can supply the name of the layout\n                or a ``Layout`` object.\n        '
        from ..layouts.factory import Layout
        from ..layouts.factory import MissingLayout, get_layout
        _rich_traceback_omit = True
        if layout is None:
            if obj.clear_rule('layout'):
                obj.refresh(layout=True, children=True)
        elif isinstance(layout, Layout):
            if obj.set_rule('layout', layout):
                obj.refresh(layout=True, children=True)
        else:
            try:
                layout_object = get_layout(layout)
            except MissingLayout as error:
                raise StyleValueError(str(error), help_text=layout_property_help_text(self.name, context='inline'))
            if obj.set_rule('layout', layout_object):
                obj.refresh(layout=True, children=True)

class OffsetProperty:
    """Descriptor for getting and setting the offset property.
    Offset consists of two values, x and y, that a widget's position
    will be adjusted by before it is rendered.
    """

    def __set_name__(self, owner: StylesBase, name: str) -> None:
        if False:
            print('Hello World!')
        self.name = name

    def __get__(self, obj: StylesBase, objtype: type[StylesBase] | None=None) -> ScalarOffset:
        if False:
            return 10
        'Get the offset.\n\n        Args:\n            obj: The ``Styles`` object.\n            objtype: The ``Styles`` class.\n\n        Returns:\n            The ``ScalarOffset`` indicating the adjustment that\n                will be made to widget position prior to it being rendered.\n        '
        return cast('ScalarOffset', obj.get_rule(self.name, NULL_SCALAR))

    def __set__(self, obj: StylesBase, offset: tuple[int | str, int | str] | ScalarOffset | None):
        if False:
            print('Hello World!')
        'Set the offset.\n\n        Args:\n            obj: The ``Styles`` class.\n            offset: A ScalarOffset object, or a 2-tuple of the form ``(x, y)`` indicating\n                the x and y offsets. When the ``tuple`` form is used, x and y can be specified\n                as either ``int`` or ``str``. The string format allows you to also specify\n                any valid scalar unit e.g. ``("0.5vw", "0.5vh")``.\n\n        Raises:\n            ScalarParseError: If any of the string values supplied in the 2-tuple cannot\n                be parsed into a Scalar. For example, if you specify a non-existent unit.\n        '
        _rich_traceback_omit = True
        if offset is None:
            if obj.clear_rule(self.name):
                obj.refresh(layout=True)
        elif isinstance(offset, ScalarOffset):
            if obj.set_rule(self.name, offset):
                obj.refresh(layout=True)
        else:
            (x, y) = offset
            try:
                scalar_x = Scalar.parse(x, Unit.WIDTH) if isinstance(x, str) else Scalar(float(x), Unit.CELLS, Unit.WIDTH)
                scalar_y = Scalar.parse(y, Unit.HEIGHT) if isinstance(y, str) else Scalar(float(y), Unit.CELLS, Unit.HEIGHT)
            except ScalarParseError as error:
                raise StyleValueError(str(error), help_text=offset_property_help_text(context='inline'))
            _offset = ScalarOffset(scalar_x, scalar_y)
            if obj.set_rule(self.name, _offset):
                obj.refresh(layout=True)

class StringEnumProperty:
    """Descriptor for getting and setting string properties and ensuring that the set
    value belongs in the set of valid values.

    Args:
        valid_values: The set of valid values that the descriptor can take.
        default: The default value (or a factory thereof) of the property.
        layout: Whether to refresh the node layout on value change.
        refresh_children: Whether to refresh the node children on value change.
    """

    def __init__(self, valid_values: set[str], default: str, layout: bool=False, refresh_children: bool=False, refresh_parent: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        self._valid_values = valid_values
        self._default = default
        self._layout = layout
        self._refresh_children = refresh_children
        self._refresh_parent = refresh_parent

    def __set_name__(self, owner: StylesBase, name: str) -> None:
        if False:
            return 10
        self.name = name

    def __get__(self, obj: StylesBase, objtype: type[StylesBase] | None=None) -> str:
        if False:
            return 10
        "Get the string property, or the default value if it's not set.\n\n        Args:\n            obj: The ``Styles`` object.\n            objtype: The ``Styles`` class.\n\n        Returns:\n            The string property value.\n        "
        return cast(str, obj.get_rule(self.name, self._default))

    def _before_refresh(self, obj: StylesBase, value: str | None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Do any housekeeping before asking for a layout refresh after a value change.'

    def __set__(self, obj: StylesBase, value: str | None=None):
        if False:
            for i in range(10):
                print('nop')
        'Set the string property and ensure it is in the set of allowed values.\n\n        Args:\n            obj: The ``Styles`` object.\n            value: The string value to set the property to.\n\n        Raises:\n            StyleValueError: If the value is not in the set of valid values.\n        '
        _rich_traceback_omit = True
        if value is None:
            if obj.clear_rule(self.name):
                self._before_refresh(obj, value)
                obj.refresh(layout=self._layout, children=self._refresh_children, parent=self._refresh_parent)
        else:
            if value not in self._valid_values:
                raise StyleValueError(f'{self.name} must be one of {friendly_list(self._valid_values)} (received {value!r})', help_text=string_enum_help_text(self.name, valid_values=list(self._valid_values), context='inline'))
            if obj.set_rule(self.name, value):
                self._before_refresh(obj, value)
                obj.refresh(layout=self._layout, children=self._refresh_children, parent=self._refresh_parent)

class OverflowProperty(StringEnumProperty):
    """Descriptor for overflow styles that forces widgets to refresh scrollbars."""

    def _before_refresh(self, obj: StylesBase, value: str | None) -> None:
        if False:
            i = 10
            return i + 15
        from ..widget import Widget
        if isinstance(obj.node, Widget):
            obj.node._refresh_scrollbars()

class NameProperty:
    """Descriptor for getting and setting name properties."""

    def __set_name__(self, owner: StylesBase, name: str) -> None:
        if False:
            print('Hello World!')
        self.name = name

    def __get__(self, obj: StylesBase, objtype: type[StylesBase] | None) -> str:
        if False:
            print('Hello World!')
        'Get the name property.\n\n        Args:\n            obj: The ``Styles`` object.\n            objtype: The ``Styles`` class.\n\n        Returns:\n            The name.\n        '
        return cast(str, obj.get_rule(self.name, ''))

    def __set__(self, obj: StylesBase, name: str | None):
        if False:
            print('Hello World!')
        'Set the name property.\n\n        Args:\n            obj: The ``Styles`` object.\n            name: The name to set the property to.\n\n        Raises:\n            StyleTypeError: If the value is not a ``str``.\n        '
        _rich_traceback_omit = True
        if name is None:
            if obj.clear_rule(self.name):
                obj.refresh(layout=True)
        else:
            if not isinstance(name, str):
                raise StyleTypeError(f'{self.name} must be a str')
            if obj.set_rule(self.name, name):
                obj.refresh(layout=True)

class NameListProperty:

    def __set_name__(self, owner: StylesBase, name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.name = name

    def __get__(self, obj: StylesBase, objtype: type[StylesBase] | None=None) -> tuple[str, ...]:
        if False:
            while True:
                i = 10
        return cast('tuple[str, ...]', obj.get_rule(self.name, ()))

    def __set__(self, obj: StylesBase, names: str | tuple[str] | None=None):
        if False:
            while True:
                i = 10
        _rich_traceback_omit = True
        if names is None:
            if obj.clear_rule(self.name):
                obj.refresh(layout=True)
        elif isinstance(names, str):
            if obj.set_rule(self.name, tuple((name.strip().lower() for name in names.split(' ')))):
                obj.refresh(layout=True)
        elif isinstance(names, tuple):
            if obj.set_rule(self.name, names):
                obj.refresh(layout=True)

class ColorProperty:
    """Descriptor for getting and setting color properties."""

    def __init__(self, default_color: Color | str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._default_color = Color.parse(default_color)

    def __set_name__(self, owner: StylesBase, name: str) -> None:
        if False:
            i = 10
            return i + 15
        self.name = name

    def __get__(self, obj: StylesBase, objtype: type[StylesBase] | None=None) -> Color:
        if False:
            return 10
        'Get a ``Color``.\n\n        Args:\n            obj: The ``Styles`` object.\n            objtype: The ``Styles`` class.\n\n        Returns:\n            The Color.\n        '
        return cast(Color, obj.get_rule(self.name, self._default_color))

    def __set__(self, obj: StylesBase, color: Color | str | None):
        if False:
            for i in range(10):
                print('nop')
        'Set the Color.\n\n        Args:\n            obj: The ``Styles`` object.\n            color: The color to set. Pass a ``Color`` instance directly,\n                or pass a ``str`` which will be parsed into a color (e.g. ``"red""``, ``"rgb(20, 50, 80)"``,\n                ``"#f4e32d"``).\n\n        Raises:\n            ColorParseError: When the color string is invalid.\n        '
        _rich_traceback_omit = True
        if color is None:
            if obj.clear_rule(self.name):
                obj.refresh(children=True)
        elif isinstance(color, Color):
            if obj.set_rule(self.name, color):
                obj.refresh(children=True)
        elif isinstance(color, str):
            alpha = 1.0
            parsed_color = Color(255, 255, 255)
            for token in color.split():
                if token.endswith('%'):
                    try:
                        alpha = percentage_string_to_float(token)
                    except ValueError:
                        raise StyleValueError(f"invalid percentage value '{token}'")
                    continue
                try:
                    parsed_color = Color.parse(token)
                except ColorParseError as error:
                    raise StyleValueError(f"Invalid color value '{token}'", help_text=color_property_help_text(self.name, context='inline', error=error))
            parsed_color = parsed_color.with_alpha(alpha)
            if obj.set_rule(self.name, parsed_color):
                obj.refresh(children=True)
        else:
            raise StyleValueError(f'Invalid color value {color}')

class StyleFlagsProperty:
    """Descriptor for getting and set style flag properties (e.g. ``bold italic underline``)."""

    def __set_name__(self, owner: StylesBase, name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.name = name

    def __get__(self, obj: StylesBase, objtype: type[StylesBase] | None=None) -> Style:
        if False:
            i = 10
            return i + 15
        'Get the ``Style``.\n\n        Args:\n            obj: The ``Styles`` object.\n            objtype: The ``Styles`` class.\n\n        Returns:\n            The ``Style`` object.\n        '
        return cast(Style, obj.get_rule(self.name, Style.null()))

    def __set__(self, obj: StylesBase, style_flags: Style | str | None):
        if False:
            while True:
                i = 10
        'Set the style using a style flag string.\n\n        Args:\n            obj: The ``Styles`` object.\n            style_flags: The style flags to set as a string. For example,\n                ``"bold italic"``.\n\n        Raises:\n            StyleValueError: If the value is an invalid style flag.\n        '
        _rich_traceback_omit = True
        if style_flags is None:
            if obj.clear_rule(self.name):
                obj.refresh(children=True)
        elif isinstance(style_flags, Style):
            if obj.set_rule(self.name, style_flags):
                obj.refresh(children=True)
        else:
            words = [word.strip() for word in style_flags.split(' ')]
            valid_word = VALID_STYLE_FLAGS.__contains__
            for word in words:
                if not valid_word(word):
                    raise StyleValueError(f'unknown word {word!r} in style flags', help_text=style_flags_property_help_text(self.name, word, context='inline'))
            try:
                style = Style.parse(style_flags)
            except rich.errors.StyleSyntaxError as error:
                if 'none' in words and len(words) > 1:
                    raise StyleValueError("cannot mix 'none' with other style flags", help_text=style_flags_property_help_text(self.name, ' '.join(words), context='inline')) from None
                raise error from None
            if obj.set_rule(self.name, style):
                obj.refresh(children=True)

class TransitionsProperty:
    """Descriptor for getting transitions properties"""

    def __get__(self, obj: StylesBase, objtype: type[StylesBase] | None=None) -> dict[str, Transition]:
        if False:
            return 10
        'Get a mapping of properties to the transitions applied to them.\n\n        Args:\n            obj: The ``Styles`` object.\n            objtype: The ``Styles`` class.\n\n        Returns:\n            A ``dict`` mapping property names to the ``Transition`` applied to them.\n                e.g. ``{"offset": Transition(...), ...}``. If no transitions have been set, an empty ``dict``\n                is returned.\n        '
        return cast('dict[str, Transition]', obj.get_rule('transitions', {}))

    def __set__(self, obj: StylesBase, transitions: dict[str, Transition] | None) -> None:
        if False:
            for i in range(10):
                print('nop')
        _rich_traceback_omit = True
        if transitions is None:
            obj.clear_rule('transitions')
        else:
            obj.set_rule('transitions', transitions.copy())

class FractionalProperty:
    """Property that can be set either as a float (e.g. 0.1) or a
    string percentage (e.g. '10%'). Values will be clamped to the range (0, 1).
    """

    def __init__(self, default: float=1.0, children: bool=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Args:\n            default: Default value if the rule wasn't explicitly set.\n            children: If `True`, then updating this value will also refresh children.\n                Otherwise only this widget will be refreshed.\n        "
        self.default = default
        self.children = children

    def __set_name__(self, owner: StylesBase, name: str) -> None:
        if False:
            i = 10
            return i + 15
        self.name = name

    def __get__(self, obj: StylesBase, type: type[StylesBase]) -> float:
        if False:
            while True:
                i = 10
        'Get the property value as a float between 0 and 1.\n\n        Args:\n            obj: The ``Styles`` object.\n            objtype: The ``Styles`` class.\n\n        Returns:\n            The value of the property (in the range (0, 1)).\n        '
        return cast(float, obj.get_rule(self.name, self.default))

    def __set__(self, obj: StylesBase, value: float | str | None) -> None:
        if False:
            while True:
                i = 10
        "Set the property value, clamping it between 0 and 1.\n\n        Args:\n            obj: The Styles object.\n            value: The value to set as a float between 0 and 1, or\n                as a percentage string such as '10%'.\n        "
        _rich_traceback_omit = True
        name = self.name
        if value is None:
            if obj.clear_rule(name):
                obj.refresh(children=self.children)
            return
        if isinstance(value, (int, float)):
            float_value = float(value)
        elif isinstance(value, str) and value.endswith('%'):
            float_value = float(Scalar.parse(value).value) / 100
        else:
            raise StyleValueError(f"{self.name} must be a str (e.g. '10%') or a float (e.g. 0.1)", help_text=fractional_property_help_text(name, context='inline'))
        if obj.set_rule(name, clamp(float_value, 0, 1)):
            obj.refresh(children=self.children)

class AlignProperty:
    """Combines the horizontal and vertical alignment properties in to a single property."""

    def __set_name__(self, owner: StylesBase, name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.horizontal = f'{name}_horizontal'
        self.vertical = f'{name}_vertical'

    def __get__(self, obj: StylesBase, type: type[StylesBase]) -> tuple[AlignHorizontal, AlignVertical]:
        if False:
            print('Hello World!')
        horizontal = getattr(obj, self.horizontal)
        vertical = getattr(obj, self.vertical)
        return (horizontal, vertical)

    def __set__(self, obj: StylesBase, value: tuple[AlignHorizontal, AlignVertical]) -> None:
        if False:
            return 10
        (horizontal, vertical) = value
        setattr(obj, self.horizontal, horizontal)
        setattr(obj, self.vertical, vertical)