from __future__ import annotations
from functools import lru_cache
from typing import TYPE_CHECKING, Iterable, Tuple, cast
from rich.console import Console
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from .color import Color
from .css.types import AlignHorizontal, EdgeStyle, EdgeType
if TYPE_CHECKING:
    from typing_extensions import TypeAlias
INNER = 1
OUTER = 2
BORDER_CHARS: dict[EdgeType, tuple[tuple[str, str, str], tuple[str, str, str], tuple[str, str, str]]] = {'': ((' ', ' ', ' '), (' ', ' ', ' '), (' ', ' ', ' ')), 'ascii': (('+', '-', '+'), ('|', ' ', '|'), ('+', '-', '+')), 'none': ((' ', ' ', ' '), (' ', ' ', ' '), (' ', ' ', ' ')), 'hidden': ((' ', ' ', ' '), (' ', ' ', ' '), (' ', ' ', ' ')), 'blank': ((' ', ' ', ' '), (' ', ' ', ' '), (' ', ' ', ' ')), 'round': (('╭', '─', '╮'), ('│', ' ', '│'), ('╰', '─', '╯')), 'solid': (('┌', '─', '┐'), ('│', ' ', '│'), ('└', '─', '┘')), 'double': (('╔', '═', '╗'), ('║', ' ', '║'), ('╚', '═', '╝')), 'dashed': (('┏', '╍', '┓'), ('╏', ' ', '╏'), ('┗', '╍', '┛')), 'heavy': (('┏', '━', '┓'), ('┃', ' ', '┃'), ('┗', '━', '┛')), 'inner': (('▗', '▄', '▖'), ('▐', ' ', '▌'), ('▝', '▀', '▘')), 'outer': (('▛', '▀', '▜'), ('▌', ' ', '▐'), ('▙', '▄', '▟')), 'thick': (('█', '▀', '█'), ('█', ' ', '█'), ('█', '▄', '█')), 'hkey': (('▔', '▔', '▔'), (' ', ' ', ' '), ('▁', '▁', '▁')), 'vkey': (('▏', ' ', '▕'), ('▏', ' ', '▕'), ('▏', ' ', '▕')), 'tall': (('▊', '▔', '▎'), ('▊', ' ', '▎'), ('▊', '▁', '▎')), 'panel': (('▊', '█', '▎'), ('▊', ' ', '▎'), ('▊', '▁', '▎')), 'wide': (('▁', '▁', '▁'), ('▎', ' ', '▊'), ('▔', '▔', '▔'))}
BORDER_LOCATIONS: dict[EdgeType, tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]] = {'': ((0, 0, 0), (0, 0, 0), (0, 0, 0)), 'ascii': ((0, 0, 0), (0, 0, 0), (0, 0, 0)), 'none': ((0, 0, 0), (0, 0, 0), (0, 0, 0)), 'hidden': ((0, 0, 0), (0, 0, 0), (0, 0, 0)), 'blank': ((0, 0, 0), (0, 0, 0), (0, 0, 0)), 'round': ((0, 0, 0), (0, 0, 0), (0, 0, 0)), 'solid': ((0, 0, 0), (0, 0, 0), (0, 0, 0)), 'double': ((0, 0, 0), (0, 0, 0), (0, 0, 0)), 'dashed': ((0, 0, 0), (0, 0, 0), (0, 0, 0)), 'heavy': ((0, 0, 0), (0, 0, 0), (0, 0, 0)), 'inner': ((1, 1, 1), (1, 1, 1), (1, 1, 1)), 'outer': ((0, 0, 0), (0, 0, 0), (0, 0, 0)), 'thick': ((0, 0, 0), (0, 0, 0), (0, 0, 0)), 'hkey': ((0, 0, 0), (0, 0, 0), (0, 0, 0)), 'vkey': ((0, 0, 0), (0, 0, 0), (0, 0, 0)), 'tall': ((2, 0, 1), (2, 0, 1), (2, 0, 1)), 'panel': ((2, 0, 1), (2, 0, 1), (2, 0, 1)), 'wide': ((1, 1, 1), (0, 1, 3), (1, 1, 1))}
BORDER_TITLE_FLIP: dict[str, tuple[bool, bool]] = {'panel': (True, False)}
BORDER_LABEL_LOCATIONS: dict[EdgeType, tuple[int, int]] = {edge_type: (locations[0][1], locations[2][1]) for (edge_type, locations) in BORDER_LOCATIONS.items()}
INVISIBLE_EDGE_TYPES = cast('frozenset[EdgeType]', frozenset(('', 'none', 'hidden')))
BorderValue: TypeAlias = Tuple[EdgeType, Color]
BoxSegments: TypeAlias = Tuple[Tuple[Segment, Segment, Segment], Tuple[Segment, Segment, Segment], Tuple[Segment, Segment, Segment]]
Borders: TypeAlias = Tuple[EdgeStyle, EdgeStyle, EdgeStyle, EdgeStyle]

@lru_cache(maxsize=1024)
def get_box(name: EdgeType, inner_style: Style, outer_style: Style, style: Style) -> BoxSegments:
    if False:
        print('Hello World!')
    'Get segments used to render a box.\n\n    Args:\n        name: Name of the box type.\n        inner_style: The inner style (widget background)\n        outer_style: The outer style (parent background)\n        style: Widget style\n\n    Returns:\n        A tuple of 3 Segment triplets.\n    '
    _Segment = Segment
    ((top1, top2, top3), (mid1, mid2, mid3), (bottom1, bottom2, bottom3)) = BORDER_CHARS[name]
    ((ltop1, ltop2, ltop3), (lmid1, lmid2, lmid3), (lbottom1, lbottom2, lbottom3)) = BORDER_LOCATIONS[name]
    inner = inner_style + style
    outer = outer_style + style
    styles = (inner, outer, Style.from_color(outer.bgcolor, inner.color), Style.from_color(inner.bgcolor, outer.color))
    return ((_Segment(top1, styles[ltop1]), _Segment(top2, styles[ltop2]), _Segment(top3, styles[ltop3])), (_Segment(mid1, styles[lmid1]), _Segment(mid2, styles[lmid2]), _Segment(mid3, styles[lmid3])), (_Segment(bottom1, styles[lbottom1]), _Segment(bottom2, styles[lbottom2]), _Segment(bottom3, styles[lbottom3])))

def render_border_label(label: tuple[Text, Style], is_title: bool, name: EdgeType, width: int, inner_style: Style, outer_style: Style, style: Style, console: Console, has_left_corner: bool, has_right_corner: bool) -> Iterable[Segment]:
    if False:
        print('Hello World!')
    'Render a border label (the title or subtitle) with optional markup.\n\n    The styling that may be embedded in the label will be reapplied after taking into\n    account the inner, outer, and border-specific, styles.\n\n    Args:\n        label: Tuple of label and style to render in the border.\n        is_title: Whether we are rendering the title (`True`) or the subtitle (`False`).\n        name: Name of the box type.\n        width: The width, in cells, of the space available for the whole edge.\n            This is the total space that may also be needed for the border corners and\n            the whitespace padding around the (sub)title. Thus, the effective space\n            available for the border label is:\n            - `width` if no corner is needed;\n            - `width - 2` if one corner is needed; and\n            - `width - 4` if both corners are needed.\n        inner_style: The inner style (widget background).\n        outer_style: The outer style (parent background).\n        style: Widget style.\n        console: The console that will render the markup in the label.\n        has_left_corner: Whether the border edge will have to render a left corner.\n        has_right_corner: Whether the border edge will have to render a right corner.\n\n    Returns:\n        A list of segments that represent the full label and surrounding padding.\n    '
    corners_needed = has_left_corner + has_right_corner
    cells_reserved = 2 * corners_needed
    (text_label, label_style) = label
    if not text_label.cell_len or width <= cells_reserved:
        return
    text_label = text_label.copy()
    text_label.truncate(width - cells_reserved, overflow='ellipsis')
    if has_left_corner:
        text_label.pad_left(1)
    if has_right_corner:
        text_label.pad_right(1)
    text_label.stylize_before(label_style)
    label_style_location = BORDER_LABEL_LOCATIONS[name][0 if is_title else 1]
    (flip_top, flip_bottom) = BORDER_TITLE_FLIP.get(name, (False, False))
    inner = inner_style + style
    outer = outer_style + style
    base_style: Style
    if label_style_location == 0:
        base_style = inner
    elif label_style_location == 1:
        base_style = outer
    elif label_style_location == 2:
        base_style = Style.from_color(outer.bgcolor, inner.color)
    elif label_style_location == 3:
        base_style = Style.from_color(inner.bgcolor, outer.color)
    else:
        assert False
    if flip_top and is_title or (flip_bottom and (not is_title)):
        base_style = base_style.without_color + Style.from_color(base_style.bgcolor, base_style.color)
    text_label.stylize_before(base_style + label_style)
    segments = text_label.render(console)
    yield from segments

def render_row(box_row: tuple[Segment, Segment, Segment], width: int, left: bool, right: bool, label_segments: Iterable[Segment], label_alignment: AlignHorizontal='left') -> Iterable[Segment]:
    if False:
        return 10
    'Compose a box row with its padded label.\n\n    This is the function that actually does the work that `render_row` is intended\n    to do, but we have many lists of segments flowing around, so it becomes easier\n    to yield the segments bit by bit, and the aggregate everything into a list later.\n\n    Args:\n        box_row: Corners and side segments.\n        width: Total width of resulting line.\n        left: Render left corner.\n        right: Render right corner.\n        label_segments: The segments that make up the label.\n        label_alignment: Where to horizontally align the label.\n\n    Returns:\n        An iterable of segments.\n    '
    (box1, box2, box3) = box_row
    corners_needed = left + right
    label_segments_list = list(label_segments)
    label_length = sum((segment.cell_length for segment in label_segments_list), 0)
    space_available = max(0, width - corners_needed - label_length)
    if left:
        yield box1
    if not space_available:
        yield from label_segments_list
    elif not label_length:
        yield Segment(box2.text * space_available, box2.style)
    elif label_alignment == 'left' or label_alignment == 'right':
        edge = Segment(box2.text * (space_available - 1), box2.style)
        if label_alignment == 'left':
            yield Segment(box2.text, box2.style)
            yield from label_segments_list
            yield edge
        else:
            yield edge
            yield from label_segments_list
            yield Segment(box2.text, box2.style)
    elif label_alignment == 'center':
        length_on_left = space_available // 2
        length_on_right = space_available - length_on_left
        yield Segment(box2.text * length_on_left, box2.style)
        yield from label_segments_list
        yield Segment(box2.text * length_on_right, box2.style)
    else:
        assert False
    if right:
        yield box3
_edge_type_normalization_table: dict[EdgeType, EdgeType] = {'none': '', 'hidden': ''}

def normalize_border_value(value: BorderValue) -> BorderValue:
    if False:
        print('Hello World!')
    return (_edge_type_normalization_table.get(value[0], value[0]), value[1])