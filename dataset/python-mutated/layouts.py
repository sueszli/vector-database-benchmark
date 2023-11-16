""" Functions for arranging bokeh layout objects.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import math
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Literal, Sequence, TypeVar, Union, overload
from .core.enums import Location, LocationType, SizingModeType
from .core.property.singletons import Undefined, UndefinedType
from .models import Column, CopyTool, ExamineTool, FlexBox, FullscreenTool, GridBox, GridPlot, LayoutDOM, Plot, Row, SaveTool, Spacer, Tool, Toolbar, ToolProxy, UIElement
from .util.dataclasses import dataclass
from .util.warnings import warn
if TYPE_CHECKING:
    from typing_extensions import TypeAlias
__all__ = ('column', 'grid', 'gridplot', 'layout', 'row', 'Spacer')
if TYPE_CHECKING:
    ToolbarOptions = Literal['logo', 'autohide', 'active_drag', 'active_inspect', 'active_scroll', 'active_tap', 'active_multi']

@overload
def row(children: list[UIElement], *, sizing_mode: SizingModeType | None=None, **kwargs: Any) -> Row:
    if False:
        print('Hello World!')
    ...

@overload
def row(*children: UIElement, sizing_mode: SizingModeType | None=None, **kwargs: Any) -> Row:
    if False:
        i = 10
        return i + 15
    ...

def row(*children: UIElement | list[UIElement], sizing_mode: SizingModeType | None=None, **kwargs: Any) -> Row:
    if False:
        while True:
            i = 10
    ' Create a row of Bokeh Layout objects. Forces all objects to\n    have the same sizing_mode, which is required for complex layouts to work.\n\n    Args:\n        children (list of :class:`~bokeh.models.LayoutDOM` ): A list of instances for\n            the row. Can be any of the following - |Plot|,\n            :class:`~bokeh.models.Widget`,\n            :class:`~bokeh.models.Row`,\n            :class:`~bokeh.models.Column`,\n            :class:`~bokeh.models.Spacer`.\n\n        sizing_mode (``"fixed"``, ``"stretch_both"``, ``"scale_width"``, ``"scale_height"``, ``"scale_both"`` ): How\n            will the items in the layout resize to fill the available space.\n            Default is ``"fixed"``. For more information on the different\n            modes see :attr:`~bokeh.models.LayoutDOM.sizing_mode`\n            description on :class:`~bokeh.models.LayoutDOM`.\n\n    Returns:\n        Row: A row of LayoutDOM objects all with the same sizing_mode.\n\n    Examples:\n\n        >>> row(plot1, plot2)\n        >>> row(children=[widgets, plot], sizing_mode=\'stretch_both\')\n    '
    _children = _parse_children_arg(*children, children=kwargs.pop('children', None))
    _handle_child_sizing(_children, sizing_mode, widget='row')
    return Row(children=_children, sizing_mode=sizing_mode, **kwargs)

@overload
def column(children: list[UIElement], *, sizing_mode: SizingModeType | None=None, **kwargs: Any) -> Column:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def column(*children: UIElement, sizing_mode: SizingModeType | None=None, **kwargs: Any) -> Column:
    if False:
        i = 10
        return i + 15
    ...

def column(*children: UIElement | list[UIElement], sizing_mode: SizingModeType | None=None, **kwargs: Any) -> Column:
    if False:
        i = 10
        return i + 15
    ' Create a column of Bokeh Layout objects. Forces all objects to\n    have the same sizing_mode, which is required for complex layouts to work.\n\n    Args:\n        children (list of :class:`~bokeh.models.LayoutDOM` ): A list of instances for\n            the column. Can be any of the following - |Plot|,\n            :class:`~bokeh.models.Widget`,\n            :class:`~bokeh.models.Row`,\n            :class:`~bokeh.models.Column`,\n            :class:`~bokeh.models.Spacer`.\n\n        sizing_mode (``"fixed"``, ``"stretch_both"``, ``"scale_width"``, ``"scale_height"``, ``"scale_both"`` ): How\n            will the items in the layout resize to fill the available space.\n            Default is ``"fixed"``. For more information on the different\n            modes see :attr:`~bokeh.models.LayoutDOM.sizing_mode`\n            description on :class:`~bokeh.models.LayoutDOM`.\n\n    Returns:\n        Column: A column of LayoutDOM objects all with the same sizing_mode.\n\n    Examples:\n\n        >>> column(plot1, plot2)\n        >>> column(children=[widgets, plot], sizing_mode=\'stretch_both\')\n    '
    _children = _parse_children_arg(*children, children=kwargs.pop('children', None))
    _handle_child_sizing(_children, sizing_mode, widget='column')
    return Column(children=_children, sizing_mode=sizing_mode, **kwargs)

def layout(*args: UIElement, children: list[UIElement] | None=None, sizing_mode: SizingModeType | None=None, **kwargs: Any) -> Column:
    if False:
        for i in range(10):
            print('nop')
    ' Create a grid-based arrangement of Bokeh Layout objects.\n\n    Args:\n        children (list of lists of :class:`~bokeh.models.LayoutDOM` ): A list of lists of instances\n            for a grid layout. Can be any of the following - |Plot|,\n            :class:`~bokeh.models.Widget`,\n            :class:`~bokeh.models.Row`,\n            :class:`~bokeh.models.Column`,\n            :class:`~bokeh.models.Spacer`.\n\n        sizing_mode (``"fixed"``, ``"stretch_both"``, ``"scale_width"``, ``"scale_height"``, ``"scale_both"`` ): How\n            will the items in the layout resize to fill the available space.\n            Default is ``"fixed"``. For more information on the different\n            modes see :attr:`~bokeh.models.LayoutDOM.sizing_mode`\n            description on :class:`~bokeh.models.LayoutDOM`.\n\n    Returns:\n        Column: A column of ``Row`` layouts of the children, all with the same sizing_mode.\n\n    Examples:\n\n        >>> layout([[plot_1, plot_2], [plot_3, plot_4]])\n        >>> layout(\n                children=[\n                    [widget_1, plot_1],\n                    [slider],\n                    [widget_2, plot_2, plot_3]\n                ],\n                sizing_mode=\'fixed\',\n            )\n\n    '
    _children = _parse_children_arg(*args, children=children)
    return _create_grid(_children, sizing_mode, **kwargs)

def gridplot(children: list[list[UIElement | None]], *, sizing_mode: SizingModeType | None=None, toolbar_location: LocationType | None='above', ncols: int | None=None, width: int | None=None, height: int | None=None, toolbar_options: dict[ToolbarOptions, Any] | None=None, merge_tools: bool=True) -> GridPlot:
    if False:
        i = 10
        return i + 15
    ' Create a grid of plots rendered on separate canvases.\n\n    The ``gridplot`` function builds a single toolbar for all the plots in the\n    grid. ``gridplot`` is designed to layout a set of plots. For general\n    grid layout, use the :func:`~bokeh.layouts.layout` function.\n\n    Args:\n        children (list of lists of |Plot|): An array of plots to display in a\n            grid, given as a list of lists of Plot objects. To leave a position\n            in the grid empty, pass None for that position in the children list.\n            OR list of |Plot| if called with ncols.\n\n        sizing_mode (``"fixed"``, ``"stretch_both"``, ``"scale_width"``, ``"scale_height"``, ``"scale_both"`` ): How\n            will the items in the layout resize to fill the available space.\n            Default is ``"fixed"``. For more information on the different\n            modes see :attr:`~bokeh.models.LayoutDOM.sizing_mode`\n            description on :class:`~bokeh.models.LayoutDOM`.\n\n        toolbar_location (``above``, ``below``, ``left``, ``right`` ): Where the\n            toolbar will be located, with respect to the grid. Default is\n            ``above``. If set to None, no toolbar will be attached to the grid.\n\n        ncols (int, optional): Specify the number of columns you would like in your grid.\n            You must only pass an un-nested list of plots (as opposed to a list of lists of plots)\n            when using ncols.\n\n        width (int, optional): The width you would like all your plots to be\n\n        height (int, optional): The height you would like all your plots to be.\n\n        toolbar_options (dict, optional) : A dictionary of options that will be\n            used to construct the grid\'s toolbar (an instance of\n            :class:`~bokeh.models.Toolbar`). If none is supplied,\n            Toolbar\'s defaults will be used.\n\n        merge_tools (``True``, ``False``): Combine tools from all child plots into\n            a single toolbar.\n\n    Returns:\n        GridPlot:\n\n    Examples:\n\n        >>> gridplot([[plot_1, plot_2], [plot_3, plot_4]])\n        >>> gridplot([plot_1, plot_2, plot_3, plot_4], ncols=2, width=200, height=100)\n        >>> gridplot(\n                children=[[plot_1, plot_2], [None, plot_3]],\n                toolbar_location=\'right\'\n                sizing_mode=\'fixed\',\n                toolbar_options=dict(logo=\'gray\')\n            )\n\n    '
    if toolbar_options is None:
        toolbar_options = {}
    if toolbar_location:
        if not hasattr(Location, toolbar_location):
            raise ValueError(f'Invalid value of toolbar_location: {toolbar_location}')
    children = _parse_children_arg(children=children)
    if ncols:
        if any((isinstance(child, list) for child in children)):
            raise ValueError('Cannot provide a nested list when using ncols')
        children = list(_chunks(children, ncols))
    if not children:
        children = []
    toolbars: list[Toolbar] = []
    items: list[tuple[UIElement, int, int]] = []
    for (y, row) in enumerate(children):
        for (x, item) in enumerate(row):
            if item is None:
                continue
            elif isinstance(item, LayoutDOM):
                if merge_tools:
                    for plot in item.select(dict(type=Plot)):
                        toolbars.append(plot.toolbar)
                        plot.toolbar_location = None
                if width is not None:
                    item.width = width
                if height is not None:
                    item.height = height
                if sizing_mode is not None and _has_auto_sizing(item):
                    item.sizing_mode = sizing_mode
                items.append((item, y, x))
            elif isinstance(item, UIElement):
                continue
            else:
                raise ValueError('Only UIElement and LayoutDOM items can be inserted into a grid')

    def merge(cls: type[Tool], group: list[Tool]) -> Tool | ToolProxy | None:
        if False:
            for i in range(10):
                print('nop')
        if issubclass(cls, (SaveTool, CopyTool, ExamineTool, FullscreenTool)):
            return cls()
        else:
            return None
    tools: list[Tool | ToolProxy] = []
    for toolbar in toolbars:
        tools.extend(toolbar.tools)
    if merge_tools:
        tools = group_tools(tools, merge=merge)
    logos = [toolbar.logo for toolbar in toolbars]
    autohides = [toolbar.autohide for toolbar in toolbars]
    active_drags = [toolbar.active_drag for toolbar in toolbars]
    active_inspects = [toolbar.active_inspect for toolbar in toolbars]
    active_scrolls = [toolbar.active_scroll for toolbar in toolbars]
    active_taps = [toolbar.active_tap for toolbar in toolbars]
    active_multis = [toolbar.active_multi for toolbar in toolbars]
    V = TypeVar('V')

    def assert_unique(values: list[V], name: ToolbarOptions) -> V | UndefinedType:
        if False:
            i = 10
            return i + 15
        if name in toolbar_options:
            return toolbar_options[name]
        n = len(set(values))
        if n == 0:
            return Undefined
        elif n > 1:
            warn(f"found multiple competing values for 'toolbar.{name}' property; using the latest value")
        return values[-1]
    logo = assert_unique(logos, 'logo')
    autohide = assert_unique(autohides, 'autohide')
    active_drag = assert_unique(active_drags, 'active_drag')
    active_inspect = assert_unique(active_inspects, 'active_inspect')
    active_scroll = assert_unique(active_scrolls, 'active_scroll')
    active_tap = assert_unique(active_taps, 'active_tap')
    active_multi = assert_unique(active_multis, 'active_multi')
    toolbar = Toolbar(tools=tools, logo=logo, autohide=autohide, active_drag=active_drag, active_inspect=active_inspect, active_scroll=active_scroll, active_tap=active_tap, active_multi=active_multi)
    gp = GridPlot(children=items, toolbar=toolbar, toolbar_location=toolbar_location, sizing_mode=sizing_mode)
    return gp

@overload
def grid(children: list[UIElement | list[UIElement | list[Any]]], *, sizing_mode: SizingModeType | None=...) -> GridBox:
    if False:
        i = 10
        return i + 15
    ...

@overload
def grid(children: Row | Column, *, sizing_mode: SizingModeType | None=...) -> GridBox:
    if False:
        return 10
    ...

@overload
def grid(children: list[UIElement | None], *, sizing_mode: SizingModeType | None=..., nrows: int) -> GridBox:
    if False:
        print('Hello World!')
    ...

@overload
def grid(children: list[UIElement | None], *, sizing_mode: SizingModeType | None=..., ncols: int) -> GridBox:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def grid(children: list[UIElement | None], *, sizing_mode: SizingModeType | None=..., nrows: int, ncols: int) -> GridBox:
    if False:
        return 10
    ...

@overload
def grid(children: str, *, sizing_mode: SizingModeType | None=...) -> GridBox:
    if False:
        for i in range(10):
            print('nop')
    ...

def grid(children: Any=[], sizing_mode: SizingModeType | None=None, nrows: int | None=None, ncols: int | None=None) -> GridBox:
    if False:
        i = 10
        return i + 15
    "\n    Conveniently create a grid of layoutable objects.\n\n    Grids are created by using ``GridBox`` model. This gives the most control over\n    the layout of a grid, but is also tedious and may result in unreadable code in\n    practical applications. ``grid()`` function remedies this by reducing the level\n    of control, but in turn providing a more convenient API.\n\n    Supported patterns:\n\n    1. Nested lists of layoutable objects. Assumes the top-level list represents\n       a column and alternates between rows and columns in subsequent nesting\n       levels. One can use ``None`` for padding purpose.\n\n       >>> grid([p1, [[p2, p3], p4]])\n       GridBox(children=[\n           (p1, 0, 0, 1, 2),\n           (p2, 1, 0, 1, 1),\n           (p3, 2, 0, 1, 1),\n           (p4, 1, 1, 2, 1),\n       ])\n\n    2. Nested ``Row`` and ``Column`` instances. Similar to the first pattern, just\n       instead of using nested lists, it uses nested ``Row`` and ``Column`` models.\n       This can be much more readable that the former. Note, however, that only\n       models that don't have ``sizing_mode`` set are used.\n\n       >>> grid(column(p1, row(column(p2, p3), p4)))\n       GridBox(children=[\n           (p1, 0, 0, 1, 2),\n           (p2, 1, 0, 1, 1),\n           (p3, 2, 0, 1, 1),\n           (p4, 1, 1, 2, 1),\n       ])\n\n    3. Flat list of layoutable objects. This requires ``nrows`` and/or ``ncols`` to\n       be set. The input list will be rearranged into a 2D array accordingly. One\n       can use ``None`` for padding purpose.\n\n       >>> grid([p1, p2, p3, p4], ncols=2)\n       GridBox(children=[\n           (p1, 0, 0, 1, 1),\n           (p2, 0, 1, 1, 1),\n           (p3, 1, 0, 1, 1),\n           (p4, 1, 1, 1, 1),\n       ])\n\n    "

    @dataclass
    class row:
        children: list[row | col]

    @dataclass
    class col:
        children: list[row | col]

    @dataclass
    class Item:
        layout: LayoutDOM
        r0: int
        c0: int
        r1: int
        c1: int

    @dataclass
    class Grid:
        nrows: int
        ncols: int
        items: list[Item]

    def flatten(layout) -> GridBox:
        if False:
            while True:
                i = 10

        def gcd(a: int, b: int) -> int:
            if False:
                for i in range(10):
                    print('nop')
            (a, b) = (abs(a), abs(b))
            while b != 0:
                (a, b) = (b, a % b)
            return a

        def lcm(a: int, *rest: int) -> int:
            if False:
                while True:
                    i = 10
            for b in rest:
                a = a * b // gcd(a, b)
            return a

        def nonempty(child: Grid) -> bool:
            if False:
                print('Hello World!')
            return child.nrows != 0 and child.ncols != 0

        def _flatten(layout: row | col | LayoutDOM) -> Grid:
            if False:
                print('Hello World!')
            if isinstance(layout, row):
                children = list(filter(nonempty, map(_flatten, layout.children)))
                if not children:
                    return Grid(0, 0, [])
                nrows = lcm(*[child.nrows for child in children])
                ncols = sum((child.ncols for child in children))
                items: list[Item] = []
                offset = 0
                for child in children:
                    factor = nrows // child.nrows
                    for i in child.items:
                        items.append(Item(i.layout, factor * i.r0, i.c0 + offset, factor * i.r1, i.c1 + offset))
                    offset += child.ncols
                return Grid(nrows, ncols, items)
            elif isinstance(layout, col):
                children = list(filter(nonempty, map(_flatten, layout.children)))
                if not children:
                    return Grid(0, 0, [])
                nrows = sum((child.nrows for child in children))
                ncols = lcm(*[child.ncols for child in children])
                items = []
                offset = 0
                for child in children:
                    factor = ncols // child.ncols
                    for i in child.items:
                        items.append(Item(i.layout, i.r0 + offset, factor * i.c0, i.r1 + offset, factor * i.c1))
                    offset += child.nrows
                return Grid(nrows, ncols, items)
            else:
                return Grid(1, 1, [Item(layout, 0, 0, 1, 1)])
        grid = _flatten(layout)
        children = []
        for i in grid.items:
            if i.layout is not None:
                children.append((i.layout, i.r0, i.c0, i.r1 - i.r0, i.c1 - i.c0))
        return GridBox(children=children)
    layout: row | col
    if isinstance(children, list):
        if nrows is not None or ncols is not None:
            N = len(children)
            if ncols is None:
                ncols = math.ceil(N / nrows)
            layout = col([row(children[i:i + ncols]) for i in range(0, N, ncols)])
        else:

            def traverse(children: list[LayoutDOM], level: int=0):
                if False:
                    print('Hello World!')
                if isinstance(children, list):
                    container = col if level % 2 == 0 else row
                    return container([traverse(child, level + 1) for child in children])
                else:
                    return children
            layout = traverse(children)
    elif isinstance(children, LayoutDOM):

        def is_usable(child: LayoutDOM) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            return _has_auto_sizing(child) and child.spacing == 0

        def traverse(item: LayoutDOM, top_level: bool=False):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(item, FlexBox) and (top_level or is_usable(item)):
                container = col if isinstance(item, Column) else row
                return container(list(map(traverse, item.children)))
            else:
                return item
        layout = traverse(children, top_level=True)
    elif isinstance(children, str):
        raise NotImplementedError
    else:
        raise ValueError('expected a list, string or model')
    grid = flatten(layout)
    if sizing_mode is not None:
        grid.sizing_mode = sizing_mode
        for child in grid.children:
            layout = child[0]
            if _has_auto_sizing(layout):
                layout.sizing_mode = sizing_mode
    return grid
T = TypeVar('T', bound=Tool)
MergeFn: TypeAlias = Callable[[type[T], list[T]], Union[Tool, ToolProxy, None]]

def group_tools(tools: list[Tool | ToolProxy], *, merge: MergeFn[Tool] | None=None, ignore: set[str] | None=None) -> list[Tool | ToolProxy]:
    if False:
        print('Hello World!')
    ' Group common tools into tool proxies. '

    @dataclass
    class ToolEntry:
        tool: Tool
        props: Any
    by_type: defaultdict[type[Tool], list[ToolEntry]] = defaultdict(list)
    computed: list[Tool | ToolProxy] = []
    if ignore is None:
        ignore = {'overlay', 'renderers'}
    for tool in tools:
        if isinstance(tool, ToolProxy):
            computed.append(tool)
        else:
            props = tool.properties_with_values()
            for attr in ignore:
                if attr in props:
                    del props[attr]
            by_type[tool.__class__].append(ToolEntry(tool, props))
    for (cls, entries) in by_type.items():
        if merge is not None:
            merged = merge(cls, [entry.tool for entry in entries])
            if merged is not None:
                computed.append(merged)
                continue
        while entries:
            (head, *tail) = entries
            group: list[Tool] = [head.tool]
            for item in list(tail):
                if item.props == head.props:
                    group.append(item.tool)
                    entries.remove(item)
            entries.remove(head)
            if len(group) == 1:
                computed.append(group[0])
            elif merge is not None and (tool := merge(cls, group)) is not None:
                computed.append(tool)
            else:
                computed.append(ToolProxy(tools=group))
    return computed

def _has_auto_sizing(item: LayoutDOM) -> bool:
    if False:
        i = 10
        return i + 15
    return item.sizing_mode is None and item.width_policy == 'auto' and (item.height_policy == 'auto')
L = TypeVar('L', bound=LayoutDOM)

def _parse_children_arg(*args: L | list[L], children: list[L] | None=None) -> list[L]:
    if False:
        for i in range(10):
            print('nop')
    if len(args) > 0 and children is not None:
        raise ValueError("'children' keyword cannot be used with positional arguments")
    if not children:
        if len(args) == 1:
            [arg] = args
            if isinstance(arg, list):
                return arg
        return list(args)
    return children

def _handle_child_sizing(children: list[UIElement], sizing_mode: SizingModeType | None, *, widget: str) -> None:
    if False:
        return 10
    for item in children:
        if isinstance(item, UIElement):
            continue
        if not isinstance(item, LayoutDOM):
            raise ValueError(f'Only LayoutDOM items can be inserted into a {widget}. Tried to insert: {item} of type {type(item)}')
        if sizing_mode is not None and _has_auto_sizing(item):
            item.sizing_mode = sizing_mode

def _create_grid(iterable: Iterable[UIElement | list[UIElement]], sizing_mode: SizingModeType | None, layer: int=0, **kwargs) -> Row | Column:
    if False:
        for i in range(10):
            print('nop')
    'Recursively create grid from input lists.'
    return_list: list[UIElement] = []
    for item in iterable:
        if isinstance(item, list):
            return_list.append(_create_grid(item, sizing_mode, layer + 1))
        elif isinstance(item, LayoutDOM):
            if sizing_mode is not None and _has_auto_sizing(item):
                item.sizing_mode = sizing_mode
            return_list.append(item)
        elif isinstance(item, UIElement):
            return_list.append(item)
        else:
            raise ValueError(f'Only LayoutDOM items can be inserted into a layout.\n                Tried to insert: {item} of type {type(item)}')
    if layer % 2 == 0:
        return column(children=return_list, sizing_mode=sizing_mode, **kwargs)
    else:
        return row(children=return_list, sizing_mode=sizing_mode, **kwargs)
I = TypeVar('I')

def _chunks(l: Sequence[I], ncols: int) -> Iterator[Sequence[I]]:
    if False:
        for i in range(10):
            print('nop')
    'Yield successive n-sized chunks from list, l.'
    assert isinstance(ncols, int), 'ncols must be an integer'
    for i in range(0, len(l), ncols):
        yield l[i:i + ncols]