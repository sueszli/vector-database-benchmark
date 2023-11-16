from __future__ import annotations
import functools
from dataclasses import dataclass
from itertools import chain, zip_longest
from operator import itemgetter
from typing import Any, Callable, ClassVar, Generic, Iterable, NamedTuple, TypeVar, cast
import rich.repr
from rich.console import RenderableType
from rich.padding import Padding
from rich.protocol import is_renderable
from rich.segment import Segment
from rich.style import Style
from rich.text import Text, TextType
from typing_extensions import Literal, Self, TypeAlias
from .. import events
from .._cache import LRUCache
from .._segment_tools import line_crop
from .._two_way_dict import TwoWayDict
from .._types import SegmentLines
from ..binding import Binding, BindingType
from ..color import Color
from ..coordinate import Coordinate
from ..geometry import Region, Size, Spacing, clamp
from ..message import Message
from ..reactive import Reactive
from ..render import measure
from ..renderables.styled import Styled
from ..scroll_view import ScrollView
from ..strip import Strip
from ..widget import PseudoClasses
CellCacheKey: TypeAlias = 'tuple[RowKey, ColumnKey, Style, bool, bool, bool, int, PseudoClasses]'
LineCacheKey: TypeAlias = 'tuple[int, int, int, int, Coordinate, Coordinate, Style, CursorType, bool, int, PseudoClasses]'
RowCacheKey: TypeAlias = 'tuple[RowKey, int, Style, Coordinate, Coordinate, CursorType, bool, bool, int, PseudoClasses]'
CursorType = Literal['cell', 'row', 'column', 'none']
'The valid types of cursors for [`DataTable.cursor_type`][textual.widgets.DataTable.cursor_type].'
CellType = TypeVar('CellType')
_DEFAULT_CELL_X_PADDING = 1
'Default padding to use on each side of a column in the data table.'

class CellDoesNotExist(Exception):
    """The cell key/index was invalid.

    Raised when the coordinates or cell key provided does not exist
    in the DataTable (e.g. out of bounds index, invalid key)"""

class RowDoesNotExist(Exception):
    """Raised when the row index or row key provided does not exist
    in the DataTable (e.g. out of bounds index, invalid key)"""

class ColumnDoesNotExist(Exception):
    """Raised when the column index or column key provided does not exist
    in the DataTable (e.g. out of bounds index, invalid key)"""

class DuplicateKey(Exception):
    """The key supplied already exists.

    Raised when the RowKey or ColumnKey provided already refers to
    an existing row or column in the DataTable. Keys must be unique."""

@functools.total_ordering
class StringKey:
    """An object used as a key in a mapping.

    It can optionally wrap a string,
    and lookups into a map using the object behave the same as lookups using
    the string itself."""
    value: str | None

    def __init__(self, value: str | None=None):
        if False:
            i = 10
            return i + 15
        self.value = value

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(self.value) if self.value is not None else id(self)

    def __eq__(self, other: object) -> bool:
        if False:
            while True:
                i = 10
        if isinstance(other, str):
            return self.value == other
        elif isinstance(other, StringKey):
            if self.value is not None and other.value is not None:
                return self.value == other.value
            else:
                return hash(self) == hash(other)
        else:
            return NotImplemented

    def __lt__(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, str):
            return self.value < other
        elif isinstance(other, StringKey):
            return self.value < other.value
        else:
            return NotImplemented

    def __rich_repr__(self):
        if False:
            print('Hello World!')
        yield ('value', self.value)

class RowKey(StringKey):
    """Uniquely identifies a row in the DataTable.

    Even if the visual location
    of the row changes due to sorting or other modifications, a key will always
    refer to the same row."""

class ColumnKey(StringKey):
    """Uniquely identifies a column in the DataTable.

    Even if the visual location
    of the column changes due to sorting or other modifications, a key will always
    refer to the same column."""

class CellKey(NamedTuple):
    """A unique identifier for a cell in the DataTable.

    A cell key is a `(row_key, column_key)` tuple.

    Even if the cell changes
    visual location (i.e. moves to a different coordinate in the table), this key
    can still be used to retrieve it, regardless of where it currently is."""
    row_key: RowKey
    "The key of this cell's row."
    column_key: ColumnKey
    "The key of this cell's column."

    def __rich_repr__(self):
        if False:
            while True:
                i = 10
        yield ('row_key', self.row_key)
        yield ('column_key', self.column_key)

def default_cell_formatter(obj: object) -> RenderableType:
    if False:
        i = 10
        return i + 15
    'Convert a cell into a Rich renderable for display.\n\n    Args:\n        obj: Data for a cell.\n\n    Returns:\n        A renderable to be displayed which represents the data.\n    '
    if isinstance(obj, str):
        return Text.from_markup(obj)
    if isinstance(obj, float):
        return f'{obj:.2f}'
    if not is_renderable(obj):
        return str(obj)
    return cast(RenderableType, obj)

@dataclass
class Column:
    """Metadata for a column in the DataTable."""
    key: ColumnKey
    label: Text
    width: int = 0
    content_width: int = 0
    auto_width: bool = False

    def get_render_width(self, data_table: DataTable[Any]) -> int:
        if False:
            print('Hello World!')
        'Width, in cells, required to render the column with padding included.\n\n        Args:\n            data_table: The data table where the column will be rendered.\n\n        Returns:\n            The width, in cells, required to render the column with padding included.\n        '
        return 2 * data_table.cell_padding + (self.content_width if self.auto_width else self.width)

@dataclass
class Row:
    """Metadata for a row in the DataTable."""
    key: RowKey
    height: int
    label: Text | None = None
    auto_height: bool = False

class RowRenderables(NamedTuple):
    """Container for a row, which contains an optional label and some data cells."""
    label: RenderableType | None
    cells: list[RenderableType]

class DataTable(ScrollView, Generic[CellType], can_focus=True):
    """A tabular widget that contains data."""
    BINDINGS: ClassVar[list[BindingType]] = [Binding('enter', 'select_cursor', 'Select', show=False), Binding('up', 'cursor_up', 'Cursor Up', show=False), Binding('down', 'cursor_down', 'Cursor Down', show=False), Binding('right', 'cursor_right', 'Cursor Right', show=False), Binding('left', 'cursor_left', 'Cursor Left', show=False), Binding('pageup', 'page_up', 'Page Up', show=False), Binding('pagedown', 'page_down', 'Page Down', show=False)]
    '\n    | Key(s) | Description |\n    | :- | :- |\n    | enter | Select cells under the cursor. |\n    | up | Move the cursor up. |\n    | down | Move the cursor down. |\n    | right | Move the cursor right. |\n    | left | Move the cursor left. |\n    '
    COMPONENT_CLASSES: ClassVar[set[str]] = {'datatable--cursor', 'datatable--hover', 'datatable--fixed', 'datatable--fixed-cursor', 'datatable--header', 'datatable--header-cursor', 'datatable--header-hover', 'datatable--odd-row', 'datatable--even-row'}
    '\n    | Class | Description |\n    | :- | :- |\n    | `datatable--cursor` | Target the cursor. |\n    | `datatable--hover` | Target the cells under the hover cursor. |\n    | `datatable--fixed` | Target fixed columns and fixed rows. |\n    | `datatable--fixed-cursor` | Target highlighted and fixed columns or header. |\n    | `datatable--header` | Target the header of the data table. |\n    | `datatable--header-cursor` | Target cells highlighted by the cursor. |\n    | `datatable--header-hover` | Target hovered header or row label cells. |\n    | `datatable--even-row` | Target even rows (row indices start at 0). |\n    | `datatable--odd-row` | Target odd rows (row indices start at 0). |\n    '
    DEFAULT_CSS = '\n    DataTable:dark {\n        background: initial;\n    }\n    DataTable {\n        background: $surface ;\n        color: $text;\n        height: auto;\n        max-height: 100vh;\n    }\n    DataTable > .datatable--header {\n        text-style: bold;\n        background: $primary;\n        color: $text;\n    }\n    DataTable > .datatable--fixed {\n        background: $primary 50%;\n        color: $text;\n    }\n\n    DataTable > .datatable--odd-row {\n\n    }\n\n    DataTable > .datatable--even-row {\n        background: $primary 10%;\n    }\n\n    DataTable > .datatable--cursor {\n        background: $secondary;\n        color: $text;\n    }\n\n    DataTable > .datatable--fixed-cursor {\n        background: $secondary 92%;\n        color: $text;\n    }\n\n    DataTable > .datatable--header-cursor {\n        background: $secondary-darken-1;\n        color: $text;\n    }\n\n    DataTable > .datatable--header-hover {\n        background: $secondary 30%;\n    }\n\n    DataTable:dark > .datatable--even-row {\n        background: $primary 15%;\n    }\n\n    DataTable > .datatable--hover {\n        background: $secondary 20%;\n    }\n    '
    show_header = Reactive(True)
    show_row_labels = Reactive(True)
    fixed_rows = Reactive(0)
    fixed_columns = Reactive(0)
    zebra_stripes = Reactive(False)
    header_height = Reactive(1)
    show_cursor = Reactive(True)
    cursor_type: Reactive[CursorType] = Reactive[CursorType]('cell')
    'The type of the cursor of the `DataTable`.'
    cell_padding = Reactive(_DEFAULT_CELL_X_PADDING)
    'Horizontal padding between cells, applied on each side of each cell.'
    cursor_coordinate: Reactive[Coordinate] = Reactive(Coordinate(0, 0), repaint=False, always_update=True)
    'Current cursor [`Coordinate`][textual.coordinate.Coordinate].\n\n    This can be set programmatically or changed via the method\n    [`move_cursor`][textual.widgets.DataTable.move_cursor].\n    '
    hover_coordinate: Reactive[Coordinate] = Reactive(Coordinate(0, 0), repaint=False, always_update=True)
    'The coordinate of the `DataTable` that is being hovered.'

    class CellHighlighted(Message):
        """Posted when the cursor moves to highlight a new cell.

        This is only relevant when the `cursor_type` is `"cell"`.
        It's also posted when the cell cursor is
        re-enabled (by setting `show_cursor=True`), and when the cursor type is
        changed to `"cell"`. Can be handled using `on_data_table_cell_highlighted` in
        a subclass of `DataTable` or in a parent widget in the DOM.
        """

        def __init__(self, data_table: DataTable, value: CellType, coordinate: Coordinate, cell_key: CellKey) -> None:
            if False:
                i = 10
                return i + 15
            self.data_table = data_table
            'The data table.'
            self.value: CellType = value
            'The value in the highlighted cell.'
            self.coordinate: Coordinate = coordinate
            'The coordinate of the highlighted cell.'
            self.cell_key: CellKey = cell_key
            'The key for the highlighted cell.'
            super().__init__()

        def __rich_repr__(self) -> rich.repr.Result:
            if False:
                return 10
            yield ('value', self.value)
            yield ('coordinate', self.coordinate)
            yield ('cell_key', self.cell_key)

        @property
        def control(self) -> DataTable:
            if False:
                return 10
            'Alias for the data table.'
            return self.data_table

    class CellSelected(Message):
        """Posted by the `DataTable` widget when a cell is selected.

        This is only relevant when the `cursor_type` is `"cell"`. Can be handled using
        `on_data_table_cell_selected` in a subclass of `DataTable` or in a parent
        widget in the DOM.
        """

        def __init__(self, data_table: DataTable, value: CellType, coordinate: Coordinate, cell_key: CellKey) -> None:
            if False:
                i = 10
                return i + 15
            self.data_table = data_table
            'The data table.'
            self.value: CellType = value
            'The value in the cell that was selected.'
            self.coordinate: Coordinate = coordinate
            'The coordinate of the cell that was selected.'
            self.cell_key: CellKey = cell_key
            'The key for the selected cell.'
            super().__init__()

        def __rich_repr__(self) -> rich.repr.Result:
            if False:
                print('Hello World!')
            yield ('value', self.value)
            yield ('coordinate', self.coordinate)
            yield ('cell_key', self.cell_key)

        @property
        def control(self) -> DataTable:
            if False:
                while True:
                    i = 10
            'Alias for the data table.'
            return self.data_table

    class RowHighlighted(Message):
        """Posted when a row is highlighted.

        This message is only posted when the
        `cursor_type` is set to `"row"`. Can be handled using
        `on_data_table_row_highlighted` in a subclass of `DataTable` or in a parent
        widget in the DOM.
        """

        def __init__(self, data_table: DataTable, cursor_row: int, row_key: RowKey) -> None:
            if False:
                i = 10
                return i + 15
            self.data_table = data_table
            'The data table.'
            self.cursor_row: int = cursor_row
            'The y-coordinate of the cursor that highlighted the row.'
            self.row_key: RowKey = row_key
            'The key of the row that was highlighted.'
            super().__init__()

        def __rich_repr__(self) -> rich.repr.Result:
            if False:
                while True:
                    i = 10
            yield ('cursor_row', self.cursor_row)
            yield ('row_key', self.row_key)

        @property
        def control(self) -> DataTable:
            if False:
                print('Hello World!')
            'Alias for the data table.'
            return self.data_table

    class RowSelected(Message):
        """Posted when a row is selected.

        This message is only posted when the
        `cursor_type` is set to `"row"`. Can be handled using
        `on_data_table_row_selected` in a subclass of `DataTable` or in a parent
        widget in the DOM.
        """

        def __init__(self, data_table: DataTable, cursor_row: int, row_key: RowKey) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.data_table = data_table
            'The data table.'
            self.cursor_row: int = cursor_row
            'The y-coordinate of the cursor that made the selection.'
            self.row_key: RowKey = row_key
            'The key of the row that was selected.'
            super().__init__()

        def __rich_repr__(self) -> rich.repr.Result:
            if False:
                i = 10
                return i + 15
            yield ('cursor_row', self.cursor_row)
            yield ('row_key', self.row_key)

        @property
        def control(self) -> DataTable:
            if False:
                while True:
                    i = 10
            'Alias for the data table.'
            return self.data_table

    class ColumnHighlighted(Message):
        """Posted when a column is highlighted.

        This message is only posted when the
        `cursor_type` is set to `"column"`. Can be handled using
        `on_data_table_column_highlighted` in a subclass of `DataTable` or in a parent
        widget in the DOM.
        """

        def __init__(self, data_table: DataTable, cursor_column: int, column_key: ColumnKey) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.data_table = data_table
            'The data table.'
            self.cursor_column: int = cursor_column
            'The x-coordinate of the column that was highlighted.'
            self.column_key = column_key
            'The key of the column that was highlighted.'
            super().__init__()

        def __rich_repr__(self) -> rich.repr.Result:
            if False:
                return 10
            yield ('cursor_column', self.cursor_column)
            yield ('column_key', self.column_key)

        @property
        def control(self) -> DataTable:
            if False:
                i = 10
                return i + 15
            'Alias for the data table.'
            return self.data_table

    class ColumnSelected(Message):
        """Posted when a column is selected.

        This message is only posted when the
        `cursor_type` is set to `"column"`. Can be handled using
        `on_data_table_column_selected` in a subclass of `DataTable` or in a parent
        widget in the DOM.
        """

        def __init__(self, data_table: DataTable, cursor_column: int, column_key: ColumnKey) -> None:
            if False:
                print('Hello World!')
            self.data_table = data_table
            'The data table.'
            self.cursor_column: int = cursor_column
            'The x-coordinate of the column that was selected.'
            self.column_key = column_key
            'The key of the column that was selected.'
            super().__init__()

        def __rich_repr__(self) -> rich.repr.Result:
            if False:
                for i in range(10):
                    print('nop')
            yield ('cursor_column', self.cursor_column)
            yield ('column_key', self.column_key)

        @property
        def control(self) -> DataTable:
            if False:
                return 10
            'Alias for the data table.'
            return self.data_table

    class HeaderSelected(Message):
        """Posted when a column header/label is clicked."""

        def __init__(self, data_table: DataTable, column_key: ColumnKey, column_index: int, label: Text):
            if False:
                while True:
                    i = 10
            self.data_table = data_table
            'The data table.'
            self.column_key = column_key
            'The key for the column.'
            self.column_index = column_index
            'The index for the column.'
            self.label = label
            'The text of the label.'
            super().__init__()

        def __rich_repr__(self) -> rich.repr.Result:
            if False:
                print('Hello World!')
            yield ('column_key', self.column_key)
            yield ('column_index', self.column_index)
            yield ('label', self.label.plain)

        @property
        def control(self) -> DataTable:
            if False:
                return 10
            'Alias for the data table.'
            return self.data_table

    class RowLabelSelected(Message):
        """Posted when a row label is clicked."""

        def __init__(self, data_table: DataTable, row_key: RowKey, row_index: int, label: Text):
            if False:
                return 10
            self.data_table = data_table
            'The data table.'
            self.row_key = row_key
            'The key for the column.'
            self.row_index = row_index
            'The index for the column.'
            self.label = label
            'The text of the label.'
            super().__init__()

        def __rich_repr__(self) -> rich.repr.Result:
            if False:
                i = 10
                return i + 15
            yield ('row_key', self.row_key)
            yield ('row_index', self.row_index)
            yield ('label', self.label.plain)

        @property
        def control(self) -> DataTable:
            if False:
                i = 10
                return i + 15
            'Alias for the data table.'
            return self.data_table

    def __init__(self, *, show_header: bool=True, show_row_labels: bool=True, fixed_rows: int=0, fixed_columns: int=0, zebra_stripes: bool=False, header_height: int=1, show_cursor: bool=True, cursor_foreground_priority: Literal['renderable', 'css']='css', cursor_background_priority: Literal['renderable', 'css']='renderable', cursor_type: CursorType='cell', cell_padding: int=_DEFAULT_CELL_X_PADDING, name: str | None=None, id: str | None=None, classes: str | None=None, disabled: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        'Initialises a widget to display tabular data.\n\n        Args:\n            show_header: Whether the table header should be visible or not.\n            show_row_labels: Whether the row labels should be shown or not.\n            fixed_rows: The number of rows, counting from the top, that should be fixed\n                and still visible when the user scrolls down.\n            fixed_columns: The number of columns, counting from the left, that should be\n                fixed and still visible when the user scrolls right.\n            zebra_stripes: Enables or disables a zebra effect applied to the background\n                color of the rows of the table, where alternate colors are styled\n                differently to improve the readability of the table.\n            header_height: The height, in number of cells, of the data table header.\n            show_cursor: Whether the cursor should be visible when navigating the data\n                table or not.\n            cursor_foreground_priority: If the data associated with a cell is an\n                arbitrary renderable with a set foreground color, this determines whether\n                that color is prioritized over the cursor component class or not.\n            cursor_background_priority: If the data associated with a cell is an\n                arbitrary renderable with a set background color, this determines whether\n                that color is prioritized over the cursor component class or not.\n            cursor_type: The type of cursor to be used when navigating the data table\n                with the keyboard.\n            cell_padding: The number of cells added on each side of each column. Setting\n                this value to zero will likely make your table very heard to read.\n            name: The name of the widget.\n            id: The ID of the widget in the DOM.\n            classes: The CSS classes for the widget.\n            disabled: Whether the widget is disabled or not.\n        '
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        self._data: dict[RowKey, dict[ColumnKey, CellType]] = {}
        'Contains the cells of the table, indexed by row key and column key.\n        The final positioning of a cell on screen cannot be determined solely by this\n        structure. Instead, we must check _row_locations and _column_locations to find\n        where each cell currently resides in space.'
        self.columns: dict[ColumnKey, Column] = {}
        'Metadata about the columns of the table, indexed by their key.'
        self.rows: dict[RowKey, Row] = {}
        'Metadata about the rows of the table, indexed by their key.'
        self._row_locations: TwoWayDict[RowKey, int] = TwoWayDict({})
        'Maps row keys to row indices which represent row order.'
        self._column_locations: TwoWayDict[ColumnKey, int] = TwoWayDict({})
        'Maps column keys to column indices which represent column order.'
        self._row_render_cache: LRUCache[RowCacheKey, tuple[SegmentLines, SegmentLines]] = LRUCache(1000)
        'For each row (a row can have a height of multiple lines), we maintain a\n        cache of the fixed and scrollable lines within that row to minimise how often\n        we need to re-render it. '
        self._cell_render_cache: LRUCache[CellCacheKey, SegmentLines] = LRUCache(10000)
        'Cache for individual cells.'
        self._line_cache: LRUCache[LineCacheKey, Strip] = LRUCache(1000)
        'Cache for lines within rows.'
        self._offset_cache: LRUCache[int, list[tuple[RowKey, int]]] = LRUCache(1)
        'Cached y_offset - key is update_count - see y_offsets property for more\n        information '
        self._ordered_row_cache: LRUCache[tuple[int, int], list[Row]] = LRUCache(1)
        'Caches row ordering - key is (num_rows, update_count).'
        self._pseudo_class_state = PseudoClasses(False, False, False)
        'The pseudo-class state is used as part of cache keys to ensure that, for example,\n        when we lose focus on the DataTable, rules which apply to :focus are invalidated\n        and we prevent lingering styles.'
        self._require_update_dimensions: bool = False
        'Set to re-calculate dimensions on idle.'
        self._new_rows: set[RowKey] = set()
        'Tracking newly added rows to be used in calculation of dimensions on idle.'
        self._updated_cells: set[CellKey] = set()
        'Track which cells were updated, so that we can refresh them once on idle.'
        self._show_hover_cursor = False
        'Used to hide the mouse hover cursor when the user uses the keyboard.'
        self._update_count = 0
        'Number of update (INCLUDING SORT) operations so far. Used for cache invalidation.'
        self._header_row_key = RowKey()
        'The header is a special row - not part of the data. Retrieve via this key.'
        self._label_column_key = ColumnKey()
        'The column containing row labels is not part of the data. This key identifies it.'
        self._labelled_row_exists = False
        'Whether or not the user has supplied any rows with labels.'
        self._label_column = Column(self._label_column_key, Text(), auto_width=True)
        'The largest content width out of all row labels in the table.'
        self.show_header = show_header
        'Show/hide the header row (the row of column labels).'
        self.show_row_labels = show_row_labels
        'Show/hide the column containing the labels of rows.'
        self.header_height = header_height
        'The height of the header row (the row of column labels).'
        self.fixed_rows = fixed_rows
        'The number of rows to fix (prevented from scrolling).'
        self.fixed_columns = fixed_columns
        'The number of columns to fix (prevented from scrolling).'
        self.zebra_stripes = zebra_stripes
        'Apply zebra effect on row backgrounds (light, dark, light, dark, ...).'
        self.show_cursor = show_cursor
        'Show/hide both the keyboard and hover cursor.'
        self.cursor_foreground_priority = cursor_foreground_priority
        'Should we prioritize the cursor component class CSS foreground or the renderable foreground\n         in the event where a cell contains a renderable with a foreground color.'
        self.cursor_background_priority = cursor_background_priority
        'Should we prioritize the cursor component class CSS background or the renderable background\n         in the event where a cell contains a renderable with a background color.'
        self.cursor_type = cursor_type
        'The type of cursor of the `DataTable`.'
        self.cell_padding = cell_padding
        'Horizontal padding between cells, applied on each side of each cell.'

    @property
    def hover_row(self) -> int:
        if False:
            i = 10
            return i + 15
        'The index of the row that the mouse cursor is currently hovering above.'
        return self.hover_coordinate.row

    @property
    def hover_column(self) -> int:
        if False:
            while True:
                i = 10
        'The index of the column that the mouse cursor is currently hovering above.'
        return self.hover_coordinate.column

    @property
    def cursor_row(self) -> int:
        if False:
            while True:
                i = 10
        'The index of the row that the DataTable cursor is currently on.'
        return self.cursor_coordinate.row

    @property
    def cursor_column(self) -> int:
        if False:
            i = 10
            return i + 15
        'The index of the column that the DataTable cursor is currently on.'
        return self.cursor_coordinate.column

    @property
    def row_count(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'The number of rows currently present in the DataTable.'
        return len(self.rows)

    @property
    def _y_offsets(self) -> list[tuple[RowKey, int]]:
        if False:
            print('Hello World!')
        'Contains a 2-tuple for each line (not row!) of the DataTable. Given a\n        y-coordinate, we can index into this list to find which row that y-coordinate\n        lands on, and the y-offset *within* that row. The length of the returned list\n        is therefore the total height of all rows within the DataTable.'
        y_offsets = []
        if self._update_count in self._offset_cache:
            y_offsets = self._offset_cache[self._update_count]
        else:
            for row in self.ordered_rows:
                y_offsets += [(row.key, y) for y in range(row.height)]
            self._offset_cache[self._update_count] = y_offsets
        return y_offsets

    @property
    def _total_row_height(self) -> int:
        if False:
            i = 10
            return i + 15
        'The total height of all rows within the DataTable'
        return len(self._y_offsets)

    def update_cell(self, row_key: RowKey | str, column_key: ColumnKey | str, value: CellType, *, update_width: bool=False) -> None:
        if False:
            print('Hello World!')
        'Update the cell identified by the specified row key and column key.\n\n        Args:\n            row_key: The key identifying the row.\n            column_key: The key identifying the column.\n            value: The new value to put inside the cell.\n            update_width: Whether to resize the column width to accommodate\n                for the new cell content.\n\n        Raises:\n            CellDoesNotExist: When the supplied `row_key` and `column_key`\n                cannot be found in the table.\n        '
        if isinstance(row_key, str):
            row_key = RowKey(row_key)
        if isinstance(column_key, str):
            column_key = ColumnKey(column_key)
        try:
            self._data[row_key][column_key] = value
        except KeyError:
            raise CellDoesNotExist(f'No cell exists for row_key={row_key!r}, column_key={column_key!r}.') from None
        self._update_count += 1
        if update_width:
            self._updated_cells.add(CellKey(row_key, column_key))
            self._require_update_dimensions = True
        self.refresh()

    def update_cell_at(self, coordinate: Coordinate, value: CellType, *, update_width: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        'Update the content inside the cell currently occupying the given coordinate.\n\n        Args:\n            coordinate: The coordinate to update the cell at.\n            value: The new value to place inside the cell.\n            update_width: Whether to resize the column width to accommodate\n                for the new cell content.\n        '
        if not self.is_valid_coordinate(coordinate):
            raise CellDoesNotExist(f'Coordinate {coordinate!r} is invalid.')
        (row_key, column_key) = self.coordinate_to_cell_key(coordinate)
        self.update_cell(row_key, column_key, value, update_width=update_width)

    def get_cell(self, row_key: RowKey | str, column_key: ColumnKey | str) -> CellType:
        if False:
            return 10
        'Given a row key and column key, return the value of the corresponding cell.\n\n        Args:\n            row_key: The row key of the cell.\n            column_key: The column key of the cell.\n\n        Returns:\n            The value of the cell identified by the row and column keys.\n        '
        try:
            cell_value = self._data[row_key][column_key]
        except KeyError:
            raise CellDoesNotExist(f'No cell exists for row_key={row_key!r}, column_key={column_key!r}.')
        return cell_value

    def get_cell_at(self, coordinate: Coordinate) -> CellType:
        if False:
            i = 10
            return i + 15
        'Get the value from the cell occupying the given coordinate.\n\n        Args:\n            coordinate: The coordinate to retrieve the value from.\n\n        Returns:\n            The value of the cell at the coordinate.\n\n        Raises:\n            CellDoesNotExist: If there is no cell with the given coordinate.\n        '
        (row_key, column_key) = self.coordinate_to_cell_key(coordinate)
        return self.get_cell(row_key, column_key)

    def get_cell_coordinate(self, row_key: RowKey | str, column_key: Column | str) -> Coordinate:
        if False:
            while True:
                i = 10
        'Given a row key and column key, return the corresponding cell coordinate.\n\n        Args:\n            row_key: The row key of the cell.\n            column_key: The column key of the cell.\n\n        Returns:\n            The current coordinate of the cell identified by the row and column keys.\n\n        Raises:\n            CellDoesNotExist: If the specified cell does not exist.\n        '
        if row_key not in self._row_locations or column_key not in self._column_locations:
            raise CellDoesNotExist(f'No cell exists for row_key={row_key!r}, column_key={column_key!r}.')
        row_index = self._row_locations.get(row_key)
        column_index = self._column_locations.get(column_key)
        return Coordinate(row_index, column_index)

    def get_row(self, row_key: RowKey | str) -> list[CellType]:
        if False:
            return 10
        'Get the values from the row identified by the given row key.\n\n        Args:\n            row_key: The key of the row.\n\n        Returns:\n            A list of the values contained within the row.\n\n        Raises:\n            RowDoesNotExist: When there is no row corresponding to the key.\n        '
        if row_key not in self._row_locations:
            raise RowDoesNotExist(f'Row key {row_key!r} is not valid.')
        cell_mapping: dict[ColumnKey, CellType] = self._data.get(row_key, {})
        ordered_row: list[CellType] = [cell_mapping[column.key] for column in self.ordered_columns]
        return ordered_row

    def get_row_at(self, row_index: int) -> list[CellType]:
        if False:
            while True:
                i = 10
        'Get the values from the cells in a row at a given index. This will\n        return the values from a row based on the rows _current position_ in\n        the table.\n\n        Args:\n            row_index: The index of the row.\n\n        Returns:\n            A list of the values contained in the row.\n\n        Raises:\n            RowDoesNotExist: If there is no row with the given index.\n        '
        if not self.is_valid_row_index(row_index):
            raise RowDoesNotExist(f'Row index {row_index!r} is not valid.')
        row_key = self._row_locations.get_key(row_index)
        return self.get_row(row_key)

    def get_row_index(self, row_key: RowKey | str) -> int:
        if False:
            return 10
        'Return the current index for the row identified by row_key.\n\n        Args:\n            row_key: The row key to find the current index of.\n\n        Returns:\n            The current index of the specified row key.\n\n        Raises:\n            RowDoesNotExist: If the row key does not exist.\n        '
        if row_key not in self._row_locations:
            raise RowDoesNotExist(f'No row exists for row_key={row_key!r}')
        return self._row_locations.get(row_key)

    def get_column(self, column_key: ColumnKey | str) -> Iterable[CellType]:
        if False:
            print('Hello World!')
        'Get the values from the column identified by the given column key.\n\n        Args:\n            column_key: The key of the column.\n\n        Returns:\n            A generator which yields the cells in the column.\n\n        Raises:\n            ColumnDoesNotExist: If there is no column corresponding to the key.\n        '
        if column_key not in self._column_locations:
            raise ColumnDoesNotExist(f'Column key {column_key!r} is not valid.')
        data = self._data
        for row_metadata in self.ordered_rows:
            row_key = row_metadata.key
            yield data[row_key][column_key]

    def get_column_at(self, column_index: int) -> Iterable[CellType]:
        if False:
            i = 10
            return i + 15
        'Get the values from the column at a given index.\n\n        Args:\n            column_index: The index of the column.\n\n        Returns:\n            A generator which yields the cells in the column.\n\n        Raises:\n            ColumnDoesNotExist: If there is no column with the given index.\n        '
        if not self.is_valid_column_index(column_index):
            raise ColumnDoesNotExist(f'Column index {column_index!r} is not valid.')
        column_key = self._column_locations.get_key(column_index)
        yield from self.get_column(column_key)

    def get_column_index(self, column_key: ColumnKey | str) -> int:
        if False:
            print('Hello World!')
        'Return the current index for the column identified by column_key.\n\n        Args:\n            column_key: The column key to find the current index of.\n\n        Returns:\n            The current index of the specified column key.\n\n        Raises:\n            ColumnDoesNotExist: If the column key does not exist.\n        '
        if column_key not in self._column_locations:
            raise ColumnDoesNotExist(f'No column exists for column_key={column_key!r}')
        return self._column_locations.get(column_key)

    def _clear_caches(self) -> None:
        if False:
            while True:
                i = 10
        self._row_render_cache.clear()
        self._cell_render_cache.clear()
        self._line_cache.clear()
        self._styles_cache.clear()
        self._offset_cache.clear()
        self._ordered_row_cache.clear()
        self._get_styles_to_render_cell.cache_clear()

    def get_row_height(self, row_key: RowKey) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Given a row key, return the height of that row in terminal cells.\n\n        Args:\n            row_key: The key of the row.\n\n        Returns:\n            The height of the row, measured in terminal character cells.\n        '
        if row_key is self._header_row_key:
            return self.header_height
        return self.rows[row_key].height

    def notify_style_update(self) -> None:
        if False:
            while True:
                i = 10
        self._clear_caches()
        self.refresh()

    def _on_resize(self, _: events.Resize) -> None:
        if False:
            return 10
        self._update_count += 1

    def watch_show_cursor(self, show_cursor: bool) -> None:
        if False:
            return 10
        self._clear_caches()
        if show_cursor and self.cursor_type != 'none':
            self._scroll_cursor_into_view(animate=False)
            if self.cursor_type == 'cell':
                self._highlight_coordinate(self.cursor_coordinate)
            elif self.cursor_type == 'row':
                self._highlight_row(self.cursor_row)
            elif self.cursor_type == 'column':
                self._highlight_column(self.cursor_column)

    def watch_show_header(self, show: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        (width, height) = self.virtual_size
        height_change = self.header_height if show else -self.header_height
        self.virtual_size = Size(width, height + height_change)
        self._scroll_cursor_into_view()
        self._clear_caches()

    def watch_show_row_labels(self, show: bool) -> None:
        if False:
            i = 10
            return i + 15
        (width, height) = self.virtual_size
        column_width = self._label_column.get_render_width(self)
        width_change = column_width if show else -column_width
        self.virtual_size = Size(width + width_change, height)
        self._scroll_cursor_into_view()
        self._clear_caches()

    def watch_fixed_rows(self) -> None:
        if False:
            return 10
        self._clear_caches()

    def watch_fixed_columns(self) -> None:
        if False:
            i = 10
            return i + 15
        self._clear_caches()

    def watch_zebra_stripes(self) -> None:
        if False:
            i = 10
            return i + 15
        self._clear_caches()

    def validate_cell_padding(self, cell_padding: int) -> int:
        if False:
            print('Hello World!')
        return max(cell_padding, 0)

    def watch_cell_padding(self, old_padding: int, new_padding: int) -> None:
        if False:
            i = 10
            return i + 15
        width_change = 2 * (new_padding - old_padding) * len(self.columns)
        (width, height) = self.virtual_size
        self.virtual_size = Size(width + width_change, height)
        self._scroll_cursor_into_view()
        self._clear_caches()

    def watch_hover_coordinate(self, old: Coordinate, value: Coordinate) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.refresh_coordinate(old)
        self.refresh_coordinate(value)

    def watch_cursor_coordinate(self, old_coordinate: Coordinate, new_coordinate: Coordinate) -> None:
        if False:
            while True:
                i = 10
        if old_coordinate != new_coordinate:
            if self.cursor_type == 'cell':
                self.refresh_coordinate(old_coordinate)
                self._highlight_coordinate(new_coordinate)
            elif self.cursor_type == 'row':
                self.refresh_row(old_coordinate.row)
                self._highlight_row(new_coordinate.row)
            elif self.cursor_type == 'column':
                self.refresh_column(old_coordinate.column)
                self._highlight_column(new_coordinate.column)
            self.call_next(self._scroll_cursor_into_view)

    def move_cursor(self, *, row: int | None=None, column: int | None=None, animate: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Move the cursor to the given position.\n\n        Example:\n            ```py\n            datatable = app.query_one(DataTable)\n            datatable.move_cursor(row=4, column=6)\n            # datatable.cursor_coordinate == Coordinate(4, 6)\n            datatable.move_cursor(row=3)\n            # datatable.cursor_coordinate == Coordinate(3, 6)\n            ```\n\n        Args:\n            row: The new row to move the cursor to.\n            column: The new column to move the cursor to.\n            animate: Whether to animate the change of coordinates.\n        '
        (cursor_row, cursor_column) = self.cursor_coordinate
        if row is not None:
            cursor_row = row
        if column is not None:
            cursor_column = column
        destination = Coordinate(cursor_row, cursor_column)
        self.cursor_coordinate = destination
        self.call_after_refresh(self._scroll_cursor_into_view, animate=animate)

    def _highlight_coordinate(self, coordinate: Coordinate) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Apply highlighting to the cell at the coordinate, and post event.'
        self.refresh_coordinate(coordinate)
        try:
            cell_value = self.get_cell_at(coordinate)
        except CellDoesNotExist:
            return
        else:
            cell_key = self.coordinate_to_cell_key(coordinate)
            self.post_message(DataTable.CellHighlighted(self, cell_value, coordinate=coordinate, cell_key=cell_key))

    def coordinate_to_cell_key(self, coordinate: Coordinate) -> CellKey:
        if False:
            for i in range(10):
                print('nop')
        'Return the key for the cell currently occupying this coordinate.\n\n        Args:\n            coordinate: The coordinate to exam the current cell key of.\n\n        Returns:\n            The key of the cell currently occupying this coordinate.\n\n        Raises:\n            CellDoesNotExist: If the coordinate is not valid.\n        '
        if not self.is_valid_coordinate(coordinate):
            raise CellDoesNotExist(f'No cell exists at {coordinate!r}.')
        (row_index, column_index) = coordinate
        row_key = self._row_locations.get_key(row_index)
        column_key = self._column_locations.get_key(column_index)
        return CellKey(row_key, column_key)

    def _highlight_row(self, row_index: int) -> None:
        if False:
            i = 10
            return i + 15
        'Apply highlighting to the row at the given index, and post event.'
        self.refresh_row(row_index)
        is_valid_row = row_index < len(self._data)
        if is_valid_row:
            row_key = self._row_locations.get_key(row_index)
            self.post_message(DataTable.RowHighlighted(self, row_index, row_key))

    def _highlight_column(self, column_index: int) -> None:
        if False:
            return 10
        'Apply highlighting to the column at the given index, and post event.'
        self.refresh_column(column_index)
        if column_index < len(self.columns):
            column_key = self._column_locations.get_key(column_index)
            self.post_message(DataTable.ColumnHighlighted(self, column_index, column_key))

    def validate_cursor_coordinate(self, value: Coordinate) -> Coordinate:
        if False:
            print('Hello World!')
        return self._clamp_cursor_coordinate(value)

    def _clamp_cursor_coordinate(self, coordinate: Coordinate) -> Coordinate:
        if False:
            while True:
                i = 10
        'Clamp a coordinate such that it falls within the boundaries of the table.'
        (row, column) = coordinate
        row = clamp(row, 0, self.row_count - 1)
        column = clamp(column, 0, len(self.columns) - 1)
        return Coordinate(row, column)

    def watch_cursor_type(self, old: str, new: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._set_hover_cursor(False)
        if self.show_cursor:
            self._highlight_cursor()
        if old == 'cell':
            self.refresh_coordinate(self.cursor_coordinate)
        elif old == 'row':
            (row_index, _) = self.cursor_coordinate
            self.refresh_row(row_index)
        elif old == 'column':
            (_, column_index) = self.cursor_coordinate
            self.refresh_column(column_index)
        self._scroll_cursor_into_view()

    def _highlight_cursor(self) -> None:
        if False:
            return 10
        'Applies the appropriate highlighting and raises the appropriate\n        [Row|Column|Cell]Highlighted event for the given cursor coordinate\n        and cursor type.'
        (row_index, column_index) = self.cursor_coordinate
        cursor_type = self.cursor_type
        if cursor_type == 'cell':
            self._highlight_coordinate(self.cursor_coordinate)
        elif cursor_type == 'row':
            self._highlight_row(row_index)
        elif cursor_type == 'column':
            self._highlight_column(column_index)

    @property
    def _row_label_column_width(self) -> int:
        if False:
            i = 10
            return i + 15
        'The render width of the column containing row labels'
        return self._label_column.get_render_width(self) if self._should_render_row_labels else 0

    def _update_column_widths(self, updated_cells: set[CellKey]) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Update the widths of the columns based on the newly updated cell widths.'
        for (row_key, column_key) in updated_cells:
            column = self.columns.get(column_key)
            if column is None:
                continue
            console = self.app.console
            label_width = measure(console, column.label, 1)
            content_width = column.content_width
            cell_value = self._data[row_key][column_key]
            new_content_width = measure(console, default_cell_formatter(cell_value), 1)
            if new_content_width < content_width:
                cells_in_column = self.get_column(column_key)
                cell_widths = [measure(console, default_cell_formatter(cell), 1) for cell in cells_in_column]
                column.content_width = max([*cell_widths, label_width])
            else:
                column.content_width = max(new_content_width, label_width)
        self._require_update_dimensions = True

    def _update_dimensions(self, new_rows: Iterable[RowKey]) -> None:
        if False:
            print('Hello World!')
        'Called to recalculate the virtual (scrollable) size.\n\n        This recomputes column widths and then checks if any of the new rows need\n        to have their height computed.\n\n        Args:\n            new_rows: The new rows that will affect the `DataTable` dimensions.\n        '
        console = self.app.console
        auto_height_rows: list[tuple[int, Row, list[RenderableType]]] = []
        for row_key in new_rows:
            row_index = self._row_locations.get(row_key)
            if row_index is None:
                continue
            row = self.rows.get(row_key)
            assert row is not None
            if row.label is not None:
                self._labelled_row_exists = True
            (row_label, cells_in_row) = self._get_row_renderables(row_index)
            label_content_width = measure(console, row_label, 1) if row_label else 0
            self._label_column.content_width = max(self._label_column.content_width, label_content_width)
            for (column, renderable) in zip(self.ordered_columns, cells_in_row):
                content_width = measure(console, renderable, 1)
                column.content_width = max(column.content_width, content_width)
            if row.auto_height:
                auto_height_rows.append((row_index, row, cells_in_row))
        if auto_height_rows:
            render_cell = self._render_cell
            should_highlight = self._should_highlight
            cursor_type = self.cursor_type
            cursor_location = self.cursor_coordinate
            hover_location = self.hover_coordinate
            base_style = self.rich_style
            fixed_style = self.get_component_styles('datatable--fixed').rich_style + Style.from_meta({'fixed': True})
            ordered_columns = self.ordered_columns
            fixed_columns = self.fixed_columns
            for (row_index, row, cells_in_row) in auto_height_rows:
                height = 0
                row_style = self._get_row_style(row_index, base_style)
                rendered_cells: list[tuple[SegmentLines, int, int]] = []
                for (column_index, column) in enumerate(ordered_columns):
                    style = fixed_style if column_index < fixed_columns else row_style
                    cell_location = Coordinate(row_index, column_index)
                    rendered_cell = render_cell(row_index, column_index, style, column.get_render_width(self), cursor=should_highlight(cursor_location, cell_location, cursor_type), hover=should_highlight(hover_location, cell_location, cursor_type))
                    cell_height = len(rendered_cell)
                    rendered_cells.append((rendered_cell, cell_height, column.get_render_width(self)))
                    height = max(height, cell_height)
                row.height = height
                for (cell_renderable, cell_height, column_width) in rendered_cells:
                    if cell_height < height:
                        first_line_space_style = cell_renderable[0][0].style
                        cell_renderable.extend([[Segment(' ' * column_width, first_line_space_style)] for _ in range(height - cell_height)])
        data_cells_width = sum((column.get_render_width(self) for column in self.columns.values()))
        total_width = data_cells_width + self._row_label_column_width
        header_height = self.header_height if self.show_header else 0
        self.virtual_size = Size(total_width, self._total_row_height + header_height)

    def _get_cell_region(self, coordinate: Coordinate) -> Region:
        if False:
            return 10
        'Get the region of the cell at the given spatial coordinate.'
        if not self.is_valid_coordinate(coordinate):
            return Region(0, 0, 0, 0)
        (row_index, column_index) = coordinate
        row_key = self._row_locations.get_key(row_index)
        row = self.rows[row_key]
        x = sum((column.get_render_width(self) for column in self.ordered_columns[:column_index])) + self._row_label_column_width
        column_key = self._column_locations.get_key(column_index)
        width = self.columns[column_key].get_render_width(self)
        height = row.height
        y = sum((ordered_row.height for ordered_row in self.ordered_rows[:row_index]))
        if self.show_header:
            y += self.header_height
        cell_region = Region(x, y, width, height)
        return cell_region

    def _get_row_region(self, row_index: int) -> Region:
        if False:
            return 10
        'Get the region of the row at the given index.'
        if not self.is_valid_row_index(row_index):
            return Region(0, 0, 0, 0)
        rows = self.rows
        row_key = self._row_locations.get_key(row_index)
        row = rows[row_key]
        row_width = sum((column.get_render_width(self) for column in self.columns.values())) + self._row_label_column_width
        y = sum((ordered_row.height for ordered_row in self.ordered_rows[:row_index]))
        if self.show_header:
            y += self.header_height
        row_region = Region(0, y, row_width, row.height)
        return row_region

    def _get_column_region(self, column_index: int) -> Region:
        if False:
            return 10
        'Get the region of the column at the given index.'
        if not self.is_valid_column_index(column_index):
            return Region(0, 0, 0, 0)
        columns = self.columns
        x = sum((column.get_render_width(self) for column in self.ordered_columns[:column_index])) + self._row_label_column_width
        column_key = self._column_locations.get_key(column_index)
        width = columns[column_key].get_render_width(self)
        header_height = self.header_height if self.show_header else 0
        height = self._total_row_height + header_height
        full_column_region = Region(x, 0, width, height)
        return full_column_region

    def clear(self, columns: bool=False) -> Self:
        if False:
            print('Hello World!')
        'Clear the table.\n\n        Args:\n            columns: Also clear the columns.\n\n        Returns:\n            The `DataTable` instance.\n        '
        self._clear_caches()
        self._y_offsets.clear()
        self._data.clear()
        self.rows.clear()
        self._row_locations = TwoWayDict({})
        if columns:
            self.columns.clear()
            self._column_locations = TwoWayDict({})
        self._require_update_dimensions = True
        self.cursor_coordinate = Coordinate(0, 0)
        self.hover_coordinate = Coordinate(0, 0)
        self._label_column = Column(self._label_column_key, Text(), auto_width=True)
        self._labelled_row_exists = False
        self.refresh()
        self.scroll_x = 0
        self.scroll_y = 0
        self.scroll_target_x = 0
        self.scroll_target_y = 0
        return self

    def add_column(self, label: TextType, *, width: int | None=None, key: str | None=None, default: CellType | None=None) -> ColumnKey:
        if False:
            print('Hello World!')
        'Add a column to the table.\n\n        Args:\n            label: A str or Text object containing the label (shown top of column).\n            width: Width of the column in cells or None to fit content.\n            key: A key which uniquely identifies this column.\n                If None, it will be generated for you.\n            default: The  value to insert into pre-existing rows.\n\n        Returns:\n            Uniquely identifies this column. Can be used to retrieve this column\n                regardless of its current location in the DataTable (it could have moved\n                after being added due to sorting/insertion/deletion of other columns).\n        '
        column_key = ColumnKey(key)
        if column_key in self._column_locations:
            raise DuplicateKey(f'The column key {key!r} already exists.')
        column_index = len(self.columns)
        label = Text.from_markup(label) if isinstance(label, str) else label
        content_width = measure(self.app.console, label, 1)
        if width is None:
            column = Column(column_key, label, content_width, content_width=content_width, auto_width=True)
        else:
            column = Column(column_key, label, width, content_width=content_width)
        self.columns[column_key] = column
        self._column_locations[column_key] = column_index
        for row_key in self.rows.keys():
            self._data[row_key][column_key] = default
            self._updated_cells.add(CellKey(row_key, column_key))
        self._require_update_dimensions = True
        self.check_idle()
        return column_key

    def add_row(self, *cells: CellType, height: int | None=1, key: str | None=None, label: TextType | None=None) -> RowKey:
        if False:
            for i in range(10):
                print('nop')
        'Add a row at the bottom of the DataTable.\n\n        Args:\n            *cells: Positional arguments should contain cell data.\n            height: The height of a row (in lines). Use `None` to auto-detect the optimal\n                height.\n            key: A key which uniquely identifies this row. If None, it will be generated\n                for you and returned.\n            label: The label for the row. Will be displayed to the left if supplied.\n\n        Returns:\n            Unique identifier for this row. Can be used to retrieve this row regardless\n                of its current location in the DataTable (it could have moved after\n                being added due to sorting or insertion/deletion of other rows).\n        '
        row_key = RowKey(key)
        if row_key in self._row_locations:
            raise DuplicateKey(f'The row key {row_key!r} already exists.')
        row_index = self.row_count
        self._row_locations[row_key] = row_index
        self._data[row_key] = {column.key: cell for (column, cell) in zip_longest(self.ordered_columns, cells)}
        label = Text.from_markup(label) if isinstance(label, str) else label
        self.rows[row_key] = Row(row_key, height or 0, label, height is None)
        self._new_rows.add(row_key)
        self._require_update_dimensions = True
        self.cursor_coordinate = self.cursor_coordinate
        cell_now_available = self.row_count == 1 and len(self.columns) > 0
        visible_cursor = self.show_cursor and self.cursor_type != 'none'
        if cell_now_available and visible_cursor:
            self._highlight_cursor()
        self._update_count += 1
        self.check_idle()
        return row_key

    def add_columns(self, *labels: TextType) -> list[ColumnKey]:
        if False:
            print('Hello World!')
        'Add a number of columns.\n\n        Args:\n            *labels: Column headers.\n\n        Returns:\n            A list of the keys for the columns that were added. See\n                the `add_column` method docstring for more information on how\n                these keys are used.\n        '
        column_keys = []
        for label in labels:
            column_key = self.add_column(label, width=None)
            column_keys.append(column_key)
        return column_keys

    def add_rows(self, rows: Iterable[Iterable[CellType]]) -> list[RowKey]:
        if False:
            i = 10
            return i + 15
        'Add a number of rows at the bottom of the DataTable.\n\n        Args:\n            rows: Iterable of rows. A row is an iterable of cells.\n\n        Returns:\n            A list of the keys for the rows that were added. See\n                the `add_row` method docstring for more information on how\n                these keys are used.\n        '
        row_keys = []
        for row in rows:
            row_key = self.add_row(*row)
            row_keys.append(row_key)
        return row_keys

    def remove_row(self, row_key: RowKey | str) -> None:
        if False:
            while True:
                i = 10
        'Remove a row (identified by a key) from the DataTable.\n\n        Args:\n            row_key: The key identifying the row to remove.\n\n        Raises:\n            RowDoesNotExist: If the row key does not exist.\n        '
        if row_key not in self._row_locations:
            raise RowDoesNotExist(f'Row key {row_key!r} is not valid.')
        self._require_update_dimensions = True
        self.check_idle()
        index_to_delete = self._row_locations.get(row_key)
        new_row_locations = TwoWayDict({})
        for row_location_key in self._row_locations:
            row_index = self._row_locations.get(row_location_key)
            if row_index > index_to_delete:
                new_row_locations[row_location_key] = row_index - 1
            elif row_index < index_to_delete:
                new_row_locations[row_location_key] = row_index
        self._row_locations = new_row_locations
        for column_key in self._data.get(row_key):
            self._updated_cells.discard(CellKey(row_key, column_key))
        del self.rows[row_key]
        del self._data[row_key]
        self.cursor_coordinate = self.cursor_coordinate
        self.hover_coordinate = self.hover_coordinate
        self._update_count += 1
        self.refresh(layout=True)

    def remove_column(self, column_key: ColumnKey | str) -> None:
        if False:
            i = 10
            return i + 15
        'Remove a column (identified by a key) from the DataTable.\n\n        Args:\n            column_key: The key identifying the column to remove.\n\n        Raises:\n            ColumnDoesNotExist: If the column key does not exist.\n        '
        if column_key not in self._column_locations:
            raise ColumnDoesNotExist(f'Column key {column_key!r} is not valid.')
        self._require_update_dimensions = True
        self.check_idle()
        index_to_delete = self._column_locations.get(column_key)
        new_column_locations = TwoWayDict({})
        for column_location_key in self._column_locations:
            column_index = self._column_locations.get(column_location_key)
            if column_index > index_to_delete:
                new_column_locations[column_location_key] = column_index - 1
            elif column_index < index_to_delete:
                new_column_locations[column_location_key] = column_index
        self._column_locations = new_column_locations
        del self.columns[column_key]
        for row_key in self._data:
            self._updated_cells.discard(CellKey(row_key, column_key))
            del self._data[row_key][column_key]
        self.cursor_coordinate = self.cursor_coordinate
        self.hover_coordinate = self.hover_coordinate
        self._update_count += 1
        self.refresh(layout=True)

    async def _on_idle(self, _: events.Idle) -> None:
        """Runs when the message pump is empty.

        We use this for some expensive calculations like re-computing dimensions of the
        whole DataTable and re-computing column widths after some cells
        have been updated. This is more efficient in the case of high
        frequency updates, ensuring we only do expensive computations once."""
        if self._updated_cells:
            updated_cells = self._updated_cells.copy()
            self._updated_cells.clear()
            self._update_column_widths(updated_cells)
        if self._require_update_dimensions:
            self._require_update_dimensions = False
            new_rows = self._new_rows.copy()
            self._new_rows.clear()
            self._update_dimensions(new_rows)

    def refresh_coordinate(self, coordinate: Coordinate) -> Self:
        if False:
            print('Hello World!')
        'Refresh the cell at a coordinate.\n\n        Args:\n            coordinate: The coordinate to refresh.\n\n        Returns:\n            The `DataTable` instance.\n        '
        if not self.is_valid_coordinate(coordinate):
            return self
        region = self._get_cell_region(coordinate)
        self._refresh_region(region)
        return self

    def refresh_row(self, row_index: int) -> Self:
        if False:
            i = 10
            return i + 15
        'Refresh the row at the given index.\n\n        Args:\n            row_index: The index of the row to refresh.\n\n        Returns:\n            The `DataTable` instance.\n        '
        if not self.is_valid_row_index(row_index):
            return self
        region = self._get_row_region(row_index)
        self._refresh_region(region)
        return self

    def refresh_column(self, column_index: int) -> Self:
        if False:
            i = 10
            return i + 15
        'Refresh the column at the given index.\n\n        Args:\n            column_index: The index of the column to refresh.\n\n        Returns:\n            The `DataTable` instance.\n        '
        if not self.is_valid_column_index(column_index):
            return self
        region = self._get_column_region(column_index)
        self._refresh_region(region)
        return self

    def _refresh_region(self, region: Region) -> Self:
        if False:
            for i in range(10):
                print('nop')
        "Refresh a region of the DataTable, if it's visible within the window.\n\n        This method will translate the region to account for scrolling.\n\n        Returns:\n            The `DataTable` instance.\n        "
        if not self.window_region.overlaps(region):
            return self
        region = region.translate(-self.scroll_offset)
        self.refresh(region)
        return self

    def is_valid_row_index(self, row_index: int) -> bool:
        if False:
            return 10
        'Return a boolean indicating whether the row_index is within table bounds.\n\n        Args:\n            row_index: The row index to check.\n\n        Returns:\n            True if the row index is within the bounds of the table.\n        '
        return 0 <= row_index < len(self.rows)

    def is_valid_column_index(self, column_index: int) -> bool:
        if False:
            while True:
                i = 10
        'Return a boolean indicating whether the column_index is within table bounds.\n\n        Args:\n            column_index: The column index to check.\n\n        Returns:\n            True if the column index is within the bounds of the table.\n        '
        return 0 <= column_index < len(self.columns)

    def is_valid_coordinate(self, coordinate: Coordinate) -> bool:
        if False:
            print('Hello World!')
        'Return a boolean indicating whether the given coordinate is valid.\n\n        Args:\n            coordinate: The coordinate to validate.\n\n        Returns:\n            True if the coordinate is within the bounds of the table.\n        '
        (row_index, column_index) = coordinate
        return self.is_valid_row_index(row_index) and self.is_valid_column_index(column_index)

    @property
    def ordered_columns(self) -> list[Column]:
        if False:
            i = 10
            return i + 15
        'The list of Columns in the DataTable, ordered as they appear on screen.'
        column_indices = range(len(self.columns))
        column_keys = [self._column_locations.get_key(index) for index in column_indices]
        ordered_columns = [self.columns[key] for key in column_keys]
        return ordered_columns

    @property
    def ordered_rows(self) -> list[Row]:
        if False:
            i = 10
            return i + 15
        'The list of Rows in the DataTable, ordered as they appear on screen.'
        num_rows = self.row_count
        update_count = self._update_count
        cache_key = (num_rows, update_count)
        if cache_key in self._ordered_row_cache:
            ordered_rows = self._ordered_row_cache[cache_key]
        else:
            row_indices = range(num_rows)
            ordered_rows = []
            for row_index in row_indices:
                row_key = self._row_locations.get_key(row_index)
                row = self.rows[row_key]
                ordered_rows.append(row)
            self._ordered_row_cache[cache_key] = ordered_rows
        return ordered_rows

    @property
    def _should_render_row_labels(self) -> bool:
        if False:
            return 10
        'Whether row labels should be rendered or not.'
        return self._labelled_row_exists and self.show_row_labels

    def _get_row_renderables(self, row_index: int) -> RowRenderables:
        if False:
            i = 10
            return i + 15
        'Get renderables for the row currently at the given row index. The renderables\n        returned here have already been passed through the default_cell_formatter.\n\n        Args:\n            row_index: Index of the row.\n\n        Returns:\n            A RowRenderables containing the optional label and the rendered cells.\n        '
        ordered_columns = self.ordered_columns
        if row_index == -1:
            header_row: list[RenderableType] = [column.label for column in ordered_columns]
            return RowRenderables(None, header_row)
        ordered_row = self.get_row_at(row_index)
        empty = Text()
        formatted_row_cells = [Text() if datum is None else default_cell_formatter(datum) or empty for (datum, _) in zip_longest(ordered_row, range(len(self.columns)))]
        label = None
        if self._should_render_row_labels:
            row_metadata = self.rows.get(self._row_locations.get_key(row_index))
            label = default_cell_formatter(row_metadata.label) if row_metadata.label else None
        return RowRenderables(label, formatted_row_cells)

    def _render_cell(self, row_index: int, column_index: int, base_style: Style, width: int, cursor: bool=False, hover: bool=False) -> SegmentLines:
        if False:
            while True:
                i = 10
        'Render the given cell.\n\n        Args:\n            row_index: Index of the row.\n            column_index: Index of the column.\n            base_style: Style to apply.\n            width: Width of the cell.\n            cursor: Is this cell affected by cursor highlighting?\n            hover: Is this cell affected by hover cursor highlighting?\n\n        Returns:\n            A list of segments per line.\n        '
        is_header_cell = row_index == -1
        is_row_label_cell = column_index == -1
        is_fixed_style_cell = not is_header_cell and (not is_row_label_cell) and (row_index < self.fixed_rows or column_index < self.fixed_columns)
        if is_header_cell:
            row_key = self._header_row_key
        else:
            row_key = self._row_locations.get_key(row_index)
        column_key = self._column_locations.get_key(column_index)
        cell_cache_key: CellCacheKey = (row_key, column_key, base_style, cursor, hover, self._show_hover_cursor, self._update_count, self._pseudo_class_state)
        if cell_cache_key not in self._cell_render_cache:
            base_style += Style.from_meta({'row': row_index, 'column': column_index})
            (row_label, row_cells) = self._get_row_renderables(row_index)
            if is_row_label_cell:
                cell = row_label if row_label is not None else ''
            else:
                cell = row_cells[column_index]
            (component_style, post_style) = self._get_styles_to_render_cell(is_header_cell, is_row_label_cell, is_fixed_style_cell, hover, cursor, self.show_cursor, self._show_hover_cursor, self.cursor_foreground_priority == 'css', self.cursor_background_priority == 'css')
            if is_header_cell:
                options = self.app.console.options.update_dimensions(width, self.header_height)
            else:
                row = self.rows[row_key]
                if row.auto_height and row.height == 0:
                    options = self.app.console.options.update_width(width)
                else:
                    options = self.app.console.options.update_dimensions(width, row.height)
            lines = self.app.console.render_lines(Styled(Padding(cell, (0, self.cell_padding)), pre_style=base_style + component_style, post_style=post_style), options)
            self._cell_render_cache[cell_cache_key] = lines
        return self._cell_render_cache[cell_cache_key]

    @functools.lru_cache(maxsize=32)
    def _get_styles_to_render_cell(self, is_header_cell: bool, is_row_label_cell: bool, is_fixed_style_cell: bool, hover: bool, cursor: bool, show_cursor: bool, show_hover_cursor: bool, has_css_foreground_priority: bool, has_css_background_priority: bool) -> tuple[Style, Style]:
        if False:
            i = 10
            return i + 15
        'Auxiliary method to compute styles used to render a given cell.\n\n        Args:\n            is_header_cell: Is this a cell from a header?\n            is_row_label_cell: Is this the label of any given row?\n            is_fixed_style_cell: Should this cell be styled like a fixed cell?\n            hover: Does this cell have the hover pseudo class?\n            cursor: Is this cell covered by the cursor?\n            show_cursor: Do we want to show the cursor in the data table?\n            show_hover_cursor: Do we want to show the mouse hover when using the keyboard\n                to move the cursor?\n            has_css_foreground_priority: `self.cursor_foreground_priority == "css"`?\n            has_css_background_priority: `self.cursor_background_priority == "css"`?\n        '
        get_component = self.get_component_rich_style
        component_style = Style()
        if hover and show_cursor and show_hover_cursor:
            component_style += get_component('datatable--hover')
            if is_header_cell or is_row_label_cell:
                component_style += get_component('datatable--header-hover')
        if cursor and show_cursor:
            cursor_style = get_component('datatable--cursor')
            component_style += cursor_style
            if is_header_cell or is_row_label_cell:
                component_style += get_component('datatable--header-cursor')
            elif is_fixed_style_cell:
                component_style += get_component('datatable--fixed-cursor')
        post_foreground = Style.from_color(color=component_style.color) if has_css_foreground_priority else Style.null()
        post_background = Style.from_color(bgcolor=component_style.bgcolor) if has_css_background_priority else Style.null()
        return (component_style, post_foreground + post_background)

    def _render_line_in_row(self, row_key: RowKey, line_no: int, base_style: Style, cursor_location: Coordinate, hover_location: Coordinate) -> tuple[SegmentLines, SegmentLines]:
        if False:
            for i in range(10):
                print('nop')
        'Render a single line from a row in the DataTable.\n\n        Args:\n            row_key: The identifying key for this row.\n            line_no: Line number (y-coordinate) within row. 0 is the first strip of\n                cells in the row, line_no=1 is the next line in the row, and so on...\n            base_style: Base style of row.\n            cursor_location: The location of the cursor in the DataTable.\n            hover_location: The location of the hover cursor in the DataTable.\n\n        Returns:\n            Lines for fixed cells, and Lines for scrollable cells.\n        '
        cursor_type = self.cursor_type
        show_cursor = self.show_cursor
        cache_key = (row_key, line_no, base_style, cursor_location, hover_location, cursor_type, show_cursor, self._show_hover_cursor, self._update_count, self._pseudo_class_state)
        if cache_key in self._row_render_cache:
            return self._row_render_cache[cache_key]
        should_highlight = self._should_highlight
        render_cell = self._render_cell
        header_style = self.get_component_styles('datatable--header').rich_style
        if row_key in self._row_locations:
            row_index = self._row_locations.get(row_key)
        else:
            row_index = -1
        fixed_row = []
        if self._labelled_row_exists and self.show_row_labels:
            cell_location = Coordinate(row_index, -1)
            label_cell_lines = render_cell(row_index, -1, header_style, width=self._row_label_column_width, cursor=should_highlight(cursor_location, cell_location, cursor_type), hover=should_highlight(hover_location, cell_location, cursor_type))[line_no]
            fixed_row.append(label_cell_lines)
        if self.fixed_columns:
            if row_key is self._header_row_key:
                fixed_style = header_style
            else:
                fixed_style = self.get_component_styles('datatable--fixed').rich_style
                fixed_style += Style.from_meta({'fixed': True})
            for (column_index, column) in enumerate(self.ordered_columns[:self.fixed_columns]):
                cell_location = Coordinate(row_index, column_index)
                fixed_cell_lines = render_cell(row_index, column_index, fixed_style, column.get_render_width(self), cursor=should_highlight(cursor_location, cell_location, cursor_type), hover=should_highlight(hover_location, cell_location, cursor_type))[line_no]
                fixed_row.append(fixed_cell_lines)
        row_style = self._get_row_style(row_index, base_style)
        scrollable_row = []
        for (column_index, column) in enumerate(self.ordered_columns):
            cell_location = Coordinate(row_index, column_index)
            cell_lines = render_cell(row_index, column_index, row_style, column.get_render_width(self), cursor=should_highlight(cursor_location, cell_location, cursor_type), hover=should_highlight(hover_location, cell_location, cursor_type))[line_no]
            scrollable_row.append(cell_lines)
        widget_width = self.size.width
        table_width = sum((column.get_render_width(self) for column in self.ordered_columns[self.fixed_columns:])) + self._row_label_column_width
        remaining_space = max(0, widget_width - table_width)
        background_color = self.background_colors[1]
        faded_color = Color.from_rich_color(row_style.bgcolor).blend(background_color, factor=0.25)
        faded_style = Style.from_color(color=row_style.color, bgcolor=faded_color.rich_color)
        scrollable_row.append([Segment(' ' * remaining_space, faded_style)])
        row_pair = (fixed_row, scrollable_row)
        self._row_render_cache[cache_key] = row_pair
        return row_pair

    def _get_offsets(self, y: int) -> tuple[RowKey, int]:
        if False:
            for i in range(10):
                print('nop')
        'Get row key and line offset for a given line.\n\n        Args:\n            y: Y coordinate relative to DataTable top.\n\n        Returns:\n            Row key and line (y) offset within cell.\n        '
        header_height = self.header_height
        y_offsets = self._y_offsets
        if self.show_header:
            if y < header_height:
                return (self._header_row_key, y)
            y -= header_height
        if y > len(y_offsets):
            raise LookupError('Y coord {y!r} is greater than total height')
        return y_offsets[y]

    def _render_line(self, y: int, x1: int, x2: int, base_style: Style) -> Strip:
        if False:
            i = 10
            return i + 15
        'Render a (possibly cropped) line in to a Strip (a list of segments\n            representing a horizontal line).\n\n        Args:\n            y: Y coordinate of line\n            x1: X start crop.\n            x2: X end crop (exclusive).\n            base_style: Style to apply to line.\n\n        Returns:\n            The Strip which represents this cropped line.\n        '
        width = self.size.width
        try:
            (row_key, y_offset_in_row) = self._get_offsets(y)
        except LookupError:
            return Strip.blank(width, base_style)
        cache_key = (y, x1, x2, width, self.cursor_coordinate, self.hover_coordinate, base_style, self.cursor_type, self._show_hover_cursor, self._update_count, self._pseudo_class_state)
        if cache_key in self._line_cache:
            return self._line_cache[cache_key]
        (fixed, scrollable) = self._render_line_in_row(row_key, y_offset_in_row, base_style, cursor_location=self.cursor_coordinate, hover_location=self.hover_coordinate)
        fixed_width = sum((column.get_render_width(self) for column in self.ordered_columns[:self.fixed_columns]))
        fixed_line: list[Segment] = list(chain.from_iterable(fixed)) if fixed else []
        scrollable_line: list[Segment] = list(chain.from_iterable(scrollable))
        segments = fixed_line + line_crop(scrollable_line, x1 + fixed_width, x2, width)
        strip = Strip(segments).adjust_cell_length(width, base_style).simplify()
        self._line_cache[cache_key] = strip
        return strip

    def render_lines(self, crop: Region) -> list[Strip]:
        if False:
            print('Hello World!')
        self._pseudo_class_state = self.get_pseudo_class_state()
        return super().render_lines(crop)

    def render_line(self, y: int) -> Strip:
        if False:
            return 10
        (width, height) = self.size
        (scroll_x, scroll_y) = self.scroll_offset
        fixed_row_keys: list[RowKey] = [self._row_locations.get_key(row_index) for row_index in range(self.fixed_rows)]
        fixed_rows_height = sum((self.get_row_height(row_key) for row_key in fixed_row_keys))
        if self.show_header:
            fixed_rows_height += self.get_row_height(self._header_row_key)
        if y >= fixed_rows_height:
            y += scroll_y
        return self._render_line(y, scroll_x, scroll_x + width, self.rich_style)

    def _should_highlight(self, cursor: Coordinate, target_cell: Coordinate, type_of_cursor: CursorType) -> bool:
        if False:
            print('Hello World!')
        "Determine if the given cell should be highlighted because of the cursor.\n\n        This auxiliary method takes the cursor position and type into account when\n        determining whether the cell should be highlighted.\n\n        Args:\n            cursor: The current position of the cursor.\n            target_cell: The cell we're checking for the need to highlight.\n            type_of_cursor: The type of cursor that is currently active.\n\n        Returns:\n            Whether or not the given cell should be highlighted.\n        "
        if type_of_cursor == 'cell':
            return cursor == target_cell
        elif type_of_cursor == 'row':
            (cursor_row, _) = cursor
            (cell_row, _) = target_cell
            return cursor_row == cell_row
        elif type_of_cursor == 'column':
            (_, cursor_column) = cursor
            (_, cell_column) = target_cell
            return cursor_column == cell_column
        else:
            return False

    def _get_row_style(self, row_index: int, base_style: Style) -> Style:
        if False:
            for i in range(10):
                print('nop')
        'Gets the Style that should be applied to the row at the given index.\n\n        Args:\n            row_index: The index of the row to style.\n            base_style: The base style to use by default.\n\n        Returns:\n            The appropriate style.\n        '
        if row_index == -1:
            row_style = self.get_component_styles('datatable--header').rich_style
        elif row_index < self.fixed_rows:
            row_style = self.get_component_styles('datatable--fixed').rich_style
        elif self.zebra_stripes:
            component_row_style = 'datatable--odd-row' if row_index % 2 else 'datatable--even-row'
            row_style = self.get_component_styles(component_row_style).rich_style
        else:
            row_style = base_style
        return row_style

    def _on_mouse_move(self, event: events.MouseMove):
        if False:
            print('Hello World!')
        'If the hover cursor is visible, display it by extracting the row\n        and column metadata from the segments present in the cells.'
        self._set_hover_cursor(True)
        meta = event.style.meta
        if not meta:
            self._set_hover_cursor(False)
            return
        if self.show_cursor and self.cursor_type != 'none':
            try:
                self.hover_coordinate = Coordinate(meta['row'], meta['column'])
            except KeyError:
                pass

    def _on_leave(self, _: events.Leave) -> None:
        if False:
            while True:
                i = 10
        self._set_hover_cursor(False)

    def _get_fixed_offset(self) -> Spacing:
        if False:
            return 10
        'Calculate the "fixed offset", that is the space to the top and left\n        that is occupied by fixed rows and columns respectively. Fixed rows and columns\n        are rows and columns that do not participate in scrolling.'
        top = self.header_height if self.show_header else 0
        top += sum((row.height for row in self.ordered_rows[:self.fixed_rows]))
        left = sum((column.get_render_width(self) for column in self.ordered_columns[:self.fixed_columns])) + self._row_label_column_width
        return Spacing(top, 0, 0, left)

    def sort(self, *columns: ColumnKey | str, key: Callable[[Any], Any] | None=None, reverse: bool=False) -> Self:
        if False:
            for i in range(10):
                print('nop')
        'Sort the rows in the `DataTable` by one or more column keys or a\n        key function (or other callable). If both columns and a key function\n        are specified, only data from those columns will sent to the key function.\n\n        Args:\n            columns: One or more columns to sort by the values in.\n            key: A function (or other callable) that returns a key to\n            use for sorting purposes.\n            reverse: If True, the sort order will be reversed.\n\n        Returns:\n            The `DataTable` instance.\n        '

        def key_wrapper(row: tuple[RowKey, dict[ColumnKey | str, CellType]]) -> Any:
            if False:
                while True:
                    i = 10
            (_, row_data) = row
            if columns:
                result = itemgetter(*columns)(row_data)
            else:
                result = tuple(row_data.values())
            if key is not None:
                return key(result)
            return result
        ordered_rows = sorted(self._data.items(), key=key_wrapper, reverse=reverse)
        self._row_locations = TwoWayDict({row_key: new_index for (new_index, (row_key, _)) in enumerate(ordered_rows)})
        self._update_count += 1
        self.refresh()
        return self

    def _scroll_cursor_into_view(self, animate: bool=False) -> None:
        if False:
            print('Hello World!')
        'When the cursor is at a boundary of the DataTable and moves out\n        of view, this method handles scrolling to ensure it remains visible.'
        fixed_offset = self._get_fixed_offset()
        (top, _, _, left) = fixed_offset
        if self.cursor_type == 'row':
            (x, y, width, height) = self._get_row_region(self.cursor_row)
            region = Region(int(self.scroll_x) + left, y, width - left, height)
        elif self.cursor_type == 'column':
            (x, y, width, height) = self._get_column_region(self.cursor_column)
            region = Region(x, int(self.scroll_y) + top, width, height - top)
        else:
            region = self._get_cell_region(self.cursor_coordinate)
        self.scroll_to_region(region, animate=animate, spacing=fixed_offset)

    def _set_hover_cursor(self, active: bool) -> None:
        if False:
            return 10
        'Set whether the hover cursor (the faint cursor you see when you\n        hover the mouse cursor over a cell) is visible or not. Typically,\n        when you interact with the keyboard, you want to switch the hover\n        cursor off.\n\n        Args:\n            active: Display the hover cursor.\n        '
        self._show_hover_cursor = active
        cursor_type = self.cursor_type
        if cursor_type == 'column':
            self.refresh_column(self.hover_column)
        elif cursor_type == 'row':
            self.refresh_row(self.hover_row)
        elif cursor_type == 'cell':
            self.refresh_coordinate(self.hover_coordinate)

    async def _on_click(self, event: events.Click) -> None:
        self._set_hover_cursor(True)
        meta = event.style.meta
        if not meta:
            return
        row_index = meta['row']
        column_index = meta['column']
        is_header_click = self.show_header and row_index == -1
        is_row_label_click = self.show_row_labels and column_index == -1
        if is_header_click:
            column = self.ordered_columns[column_index]
            message = DataTable.HeaderSelected(self, column.key, column_index, label=column.label)
            self.post_message(message)
        elif is_row_label_click:
            row = self.ordered_rows[row_index]
            message = DataTable.RowLabelSelected(self, row.key, row_index, label=row.label)
            self.post_message(message)
        elif self.show_cursor and self.cursor_type != 'none':
            self.cursor_coordinate = Coordinate(row_index, column_index)
            self._post_selected_message()
            self._scroll_cursor_into_view(animate=True)
            event.stop()

    def action_page_down(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Move the cursor one page down.'
        self._set_hover_cursor(False)
        if self.show_cursor and self.cursor_type in ('cell', 'row'):
            height = self.size.height - (self.header_height if self.show_header else 0)
            offset = 0
            rows_to_scroll = 0
            (row_index, column_index) = self.cursor_coordinate
            for ordered_row in self.ordered_rows[row_index:]:
                offset += ordered_row.height
                if offset > height:
                    break
                rows_to_scroll += 1
            self.cursor_coordinate = Coordinate(row_index + rows_to_scroll - 1, column_index)
        else:
            super().action_page_down()

    def action_page_up(self) -> None:
        if False:
            return 10
        'Move the cursor one page up.'
        self._set_hover_cursor(False)
        if self.show_cursor and self.cursor_type in ('cell', 'row'):
            height = self.size.height - (self.header_height if self.show_header else 0)
            offset = 0
            rows_to_scroll = 0
            (row_index, column_index) = self.cursor_coordinate
            for ordered_row in self.ordered_rows[:row_index + 1]:
                offset += ordered_row.height
                if offset > height:
                    break
                rows_to_scroll += 1
            self.cursor_coordinate = Coordinate(row_index - rows_to_scroll + 1, column_index)
        else:
            super().action_page_up()

    def action_scroll_home(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Scroll to the top of the data table.'
        self._set_hover_cursor(False)
        cursor_type = self.cursor_type
        if self.show_cursor and (cursor_type == 'cell' or cursor_type == 'row'):
            (row_index, column_index) = self.cursor_coordinate
            self.cursor_coordinate = Coordinate(0, column_index)
        else:
            super().action_scroll_home()

    def action_scroll_end(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Scroll to the bottom of the data table.'
        self._set_hover_cursor(False)
        cursor_type = self.cursor_type
        if self.show_cursor and (cursor_type == 'cell' or cursor_type == 'row'):
            (row_index, column_index) = self.cursor_coordinate
            self.cursor_coordinate = Coordinate(self.row_count - 1, column_index)
        else:
            super().action_scroll_end()

    def action_cursor_up(self) -> None:
        if False:
            while True:
                i = 10
        self._set_hover_cursor(False)
        cursor_type = self.cursor_type
        if self.show_cursor and (cursor_type == 'cell' or cursor_type == 'row'):
            self.cursor_coordinate = self.cursor_coordinate.up()
        else:
            super().action_scroll_up()

    def action_cursor_down(self) -> None:
        if False:
            return 10
        self._set_hover_cursor(False)
        cursor_type = self.cursor_type
        if self.show_cursor and (cursor_type == 'cell' or cursor_type == 'row'):
            self.cursor_coordinate = self.cursor_coordinate.down()
        else:
            super().action_scroll_down()

    def action_cursor_right(self) -> None:
        if False:
            i = 10
            return i + 15
        self._set_hover_cursor(False)
        cursor_type = self.cursor_type
        if self.show_cursor and (cursor_type == 'cell' or cursor_type == 'column'):
            self.cursor_coordinate = self.cursor_coordinate.right()
            self._scroll_cursor_into_view(animate=True)
        else:
            super().action_scroll_right()

    def action_cursor_left(self) -> None:
        if False:
            print('Hello World!')
        self._set_hover_cursor(False)
        cursor_type = self.cursor_type
        if self.show_cursor and (cursor_type == 'cell' or cursor_type == 'column'):
            self.cursor_coordinate = self.cursor_coordinate.left()
            self._scroll_cursor_into_view(animate=True)
        else:
            super().action_scroll_left()

    def action_select_cursor(self) -> None:
        if False:
            i = 10
            return i + 15
        self._set_hover_cursor(False)
        if self.show_cursor and self.cursor_type != 'none':
            self._post_selected_message()

    def _post_selected_message(self):
        if False:
            i = 10
            return i + 15
        'Post the appropriate message for a selection based on the `cursor_type`.'
        cursor_coordinate = self.cursor_coordinate
        cursor_type = self.cursor_type
        if len(self._data) == 0:
            return
        cell_key = self.coordinate_to_cell_key(cursor_coordinate)
        if cursor_type == 'cell':
            self.post_message(DataTable.CellSelected(self, self.get_cell_at(cursor_coordinate), coordinate=cursor_coordinate, cell_key=cell_key))
        elif cursor_type == 'row':
            (row_index, _) = cursor_coordinate
            (row_key, _) = cell_key
            self.post_message(DataTable.RowSelected(self, row_index, row_key))
        elif cursor_type == 'column':
            (_, column_index) = cursor_coordinate
            (_, column_key) = cell_key
            self.post_message(DataTable.ColumnSelected(self, column_index, column_key))