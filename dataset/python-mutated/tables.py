""" Various kinds of data table (data grid) widgets.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from ...core.enums import AutosizeMode, DateFormat, FontStyle, NumeralLanguage, RoundingFunction, TextAlign
from ...core.has_props import abstract
from ...core.properties import Bool, Color, Either, Enum, Float, Instance, InstanceDefault, Int, List, Nullable, Override, Required, String
from ...model import Model
from ..sources import CDSView, ColumnDataSource, DataSource
from .widget import Widget
__all__ = ('AvgAggregator', 'BooleanFormatter', 'CellFormatter', 'CellEditor', 'CheckboxEditor', 'DataCube', 'DataTable', 'DateEditor', 'DateFormatter', 'GroupingInfo', 'HTMLTemplateFormatter', 'IntEditor', 'MaxAggregator', 'MinAggregator', 'NumberEditor', 'NumberFormatter', 'PercentEditor', 'ScientificFormatter', 'SelectEditor', 'StringEditor', 'StringFormatter', 'SumAggregator', 'TableColumn', 'TableWidget', 'TextEditor', 'TimeEditor')

@abstract
class CellFormatter(Model):
    """ Abstract base class for data table's cell formatters.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)

@abstract
class CellEditor(Model):
    """ Abstract base class for data table's cell editors.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)

@abstract
class RowAggregator(Model):
    """ Abstract base class for data cube's row formatters.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    field_ = String('', help='\n    Refers to the table column being aggregated\n    ')

class StringFormatter(CellFormatter):
    """ Basic string cell formatter.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    font_style = Enum(FontStyle, default='normal', help='\n    An optional text font style, e.g. bold, italic.\n    ')
    text_align = Enum(TextAlign, default='left', help='\n    An optional text align, i.e. left, center or right.\n    ')
    text_color = Nullable(Color, help='\n    An optional text color.\n    ')
    nan_format = String('-', help='\n    Formatting to apply to NaN and None values.\n    ')

class ScientificFormatter(StringFormatter):
    """ Display numeric values from continuous ranges as "basic numbers",
    using scientific notation when appropriate by default.
    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    precision = Int(10, help='\n    How many digits of precision to display.\n    ')
    power_limit_high = Int(5, help='\n    Limit the use of scientific notation to when::\n        log(x) >= power_limit_high\n    ')
    power_limit_low = Int(-3, help='\n    Limit the use of scientific notation to when::\n        log(x) <= power_limit_low\n    ')

class NumberFormatter(StringFormatter):
    """ Number cell formatter.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    format = String('0,0', help="\n    The number format, as defined in the following tables:\n\n    **NUMBERS**:\n\n    ============ ============== ===============\n    Number       Format         String\n    ============ ============== ===============\n    10000        '0,0.0000'     10,000.0000\n    10000.23     '0,0'          10,000\n    10000.23     '+0,0'         +10,000\n    -10000       '0,0.0'        -10,000.0\n    10000.1234   '0.000'        10000.123\n    10000.1234   '0[.]00000'    10000.12340\n    -10000       '(0,0.0000)'   (10,000.0000)\n    -0.23        '.00'          -.23\n    -0.23        '(.00)'        (.23)\n    0.23         '0.00000'      0.23000\n    0.23         '0.0[0000]'    0.23\n    1230974      '0.0a'         1.2m\n    1460         '0 a'          1 k\n    -104000      '0a'           -104k\n    1            '0o'           1st\n    52           '0o'           52nd\n    23           '0o'           23rd\n    100          '0o'           100th\n    ============ ============== ===============\n\n    **CURRENCY**:\n\n    =========== =============== =============\n    Number      Format          String\n    =========== =============== =============\n    1000.234    '$0,0.00'       $1,000.23\n    1000.2      '0,0[.]00 $'    1,000.20 $\n    1001        '$ 0,0[.]00'    $ 1,001\n    -1000.234   '($0,0)'        ($1,000)\n    -1000.234   '$0.00'         -$1000.23\n    1230974     '($ 0.00 a)'    $ 1.23 m\n    =========== =============== =============\n\n    **BYTES**:\n\n    =============== =========== ============\n    Number          Format      String\n    =============== =========== ============\n    100             '0b'        100B\n    2048            '0 b'       2 KB\n    7884486213      '0.0b'      7.3GB\n    3467479682787   '0.000 b'   3.154 TB\n    =============== =========== ============\n\n    **PERCENTAGES**:\n\n    ============= ============= ===========\n    Number        Format        String\n    ============= ============= ===========\n    1             '0%'          100%\n    0.974878234   '0.000%'      97.488%\n    -0.43         '0 %'         -43 %\n    0.43          '(0.000 %)'   43.000 %\n    ============= ============= ===========\n\n    **TIME**:\n\n    ============ ============== ============\n    Number       Format         String\n    ============ ============== ============\n    25           '00:00:00'     0:00:25\n    238          '00:00:00'     0:03:58\n    63846        '00:00:00'     17:44:06\n    ============ ============== ============\n\n    For the complete specification, see http://numbrojs.com/format.html\n    ")
    language = Enum(NumeralLanguage, default='en', help='\n    The language to use for formatting language-specific features (e.g. thousands separator).\n    ')
    rounding = Enum(RoundingFunction, help='\n    Rounding functions (round, floor, ceil) and their synonyms (nearest, rounddown, roundup).\n    ')

class BooleanFormatter(CellFormatter):
    """ Boolean (check mark) cell formatter.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    icon = Enum('check', 'check-circle', 'check-circle-o', 'check-square', 'check-square-o', help='\n    The icon visualizing the check mark.\n    ')

class DateFormatter(StringFormatter):
    """ Date cell formatter.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
    format = Either(Enum(DateFormat), String, default='ISO-8601', help='\n    The date format can be any standard  `strftime`_ format string, as well\n    as any of the following predefined format names:\n\n    ================================================ ================== ===================\n    Format name(s)                                   Format string      Example Output\n    ================================================ ================== ===================\n    ``ATOM`` / ``W3C`` / ``RFC-3339`` / ``ISO-8601`` ``"%Y-%m-%d"``     2014-03-01\n    ``COOKIE``                                       ``"%a, %d %b %Y"`` Sat, 01 Mar 2014\n    ``RFC-850``                                      ``"%A, %d-%b-%y"`` Saturday, 01-Mar-14\n    ``RFC-1123`` / ``RFC-2822``                      ``"%a, %e %b %Y"`` Sat, 1 Mar 2014\n    ``RSS`` / ``RFC-822`` / ``RFC-1036``             ``"%a, %e %b %y"`` Sat, 1 Mar 14\n    ``TIMESTAMP``                                    (ms since epoch)   1393632000000\n    ================================================ ================== ===================\n\n    Note that in the table some of the format names are synonymous, with\n    identical format names separated by slashes.\n\n    This list of supported `strftime`_ format codes is reproduced below.\n\n    %a\n        The abbreviated name of the day of the week according to the\n        current locale.\n\n    %A\n        The full name of the day of the week according to the current\n        locale.\n\n    %b\n        The abbreviated month name according to the current locale.\n\n    %B\n        The full month name according to the current locale.\n\n    %c\n        The preferred date and time representation for the current\n        locale.\n\n    %C\n        The century number (year/100) as a 2-digit integer.\n\n    %d\n        The day of the month as a decimal number (range 01 to 31).\n\n    %D\n        Equivalent to %m/%d/%y.  (Americans should note that in many\n        other countries %d/%m/%y is rather common. This means that in\n        international context this format is ambiguous and should not\n        be used.)\n\n    %e\n        Like %d, the day of the month as a decimal number, but a\n        leading zero is replaced by a space.\n\n    %f\n        Microsecond as a decimal number, zero-padded on the left (range\n        000000-999999). This is an extension to the set of directives\n        available to `timezone`_.\n\n    %F\n        Equivalent to %Y-%m-%d (the ISO 8601 date format).\n\n    %G\n        The ISO 8601 week-based year with century as a decimal number.\n        The 4-digit year corresponding to the ISO week number (see %V).\n        This has the same format and value as %Y, except that if the\n        ISO week number belongs to the previous or next year, that year\n        is used instead.\n\n    %g\n        Like %G, but without century, that is, with a 2-digit year (00-99).\n\n    %h\n        Equivalent to %b.\n\n    %H\n        The hour as a decimal number using a 24-hour clock (range 00\n        to 23).\n\n    %I\n        The hour as a decimal number using a 12-hour clock (range 01\n        to 12).\n\n    %j\n        The day of the year as a decimal number (range 001 to 366).\n\n    %k\n        The hour (24-hour clock) as a decimal number (range 0 to 23).\n        Single digits are preceded by a blank.  (See also %H.)\n\n    %l\n        The hour (12-hour clock) as a decimal number (range 1 to 12).\n        Single digits are preceded by a blank.  (See also %I.)  (TZ)\n\n    %m\n        The month as a decimal number (range 01 to 12).\n\n    %M\n        The minute as a decimal number (range 00 to 59).\n\n    %n\n        A newline character. Bokeh text does not currently support\n        newline characters.\n\n    %N\n        Nanosecond as a decimal number, zero-padded on the left (range\n        000000000-999999999). Supports a padding width specifier, i.e.\n        %3N displays 3 leftmost digits. However, this is only accurate\n        to the millisecond level of precision due to limitations of\n        `timezone`_.\n\n    %p\n        Either "AM" or "PM" according to the given time value, or the\n        corresponding strings for the current locale.  Noon is treated\n        as "PM" and midnight as "AM".\n\n    %P\n        Like %p but in lowercase: "am" or "pm" or a corresponding\n        string for the current locale.\n\n    %r\n        The time in a.m. or p.m. notation.  In the POSIX locale this\n        is equivalent to %I:%M:%S %p.\n\n    %R\n        The time in 24-hour notation (%H:%M). For a version including\n        the seconds, see %T below.\n\n    %s\n        The number of seconds since the Epoch, 1970-01-01 00:00:00\n        +0000 (UTC).\n\n    %S\n        The second as a decimal number (range 00 to 60).  (The range\n        is up to 60 to allow for occasional leap seconds.)\n\n    %t\n        A tab character. Bokeh text does not currently support tab\n        characters.\n\n    %T\n        The time in 24-hour notation (%H:%M:%S).\n\n    %u\n        The day of the week as a decimal, range 1 to 7, Monday being 1.\n        See also %w.\n\n    %U\n        The week number of the current year as a decimal number, range\n        00 to 53, starting with the first Sunday as the first day of\n        week 01.  See also %V and %W.\n\n    %V\n        The ISO 8601 week number (see NOTES) of the current year as a\n        decimal number, range 01 to 53, where week 1 is the first week\n        that has at least 4 days in the new year.  See also %U and %W.\n\n    %w\n        The day of the week as a decimal, range 0 to 6, Sunday being 0.\n        See also %u.\n\n    %W\n        The week number of the current year as a decimal number, range\n        00 to 53, starting with the first Monday as the first day of\n        week 01.\n\n    %x\n        The preferred date representation for the current locale\n        without the time.\n\n    %X\n        The preferred time representation for the current locale\n        without the date.\n\n    %y\n        The year as a decimal number without a century (range 00 to 99).\n\n    %Y\n        The year as a decimal number including the century.\n\n    %z\n        The +hhmm or -hhmm numeric timezone (that is, the hour and\n        minute offset from UTC).\n\n    %Z\n        The timezone name or abbreviation.\n\n    %%\n        A literal \'%\' character.\n\n    .. warning::\n        The client library BokehJS uses the `timezone`_ library to\n        format datetimes. The inclusion of the list below is based on the\n        claim that `timezone`_ makes to support "the full compliment\n        of GNU date format specifiers." However, this claim has not\n        been tested exhaustively against this list. If you find formats\n        that do not function as expected, please submit a `github issue`_,\n        so that the documentation can be updated appropriately.\n\n    .. _strftime: http://man7.org/linux/man-pages/man3/strftime.3.html\n    .. _timezone: http://bigeasy.github.io/timezone/\n    .. _github issue: https://github.com/bokeh/bokeh/issues\n\n    ')

class HTMLTemplateFormatter(CellFormatter):
    """ HTML formatter using a template.
    This uses Underscore's `template` method and syntax.  http://underscorejs.org/#template
    The formatter has access other items in the row via the `dataContext` object passed to the formatter.
    So, for example, if another column in the datasource was named `url`, the template could access it as:

    .. code-block:: jinja

        <a href="<%= url %>"><%= value %></a>

    To use a different set of template delimiters, pass the appropriate values for `evaluate`, `interpolate`,
    or `escape`.  See the Underscore `template` documentation for more information.  http://underscorejs.org/#template

    Example: Simple HTML template to format the column value as code.

    .. code-block:: python

        HTMLTemplateFormatter(template='<code><%= value %></code>')

    Example: Use values from other columns (`manufacturer` and `model`) to build a hyperlink.

    .. code-block:: python

        HTMLTemplateFormatter(template=
            '<a href="https:/www.google.com/search?q=<%= manufacturer %>+<%= model %>" target="_blank"><%= value %></a>'
        )

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            return 10
        super().__init__(*args, **kwargs)
    template = String('<%= value %>', help="\n    Template string to be used by Underscore's template method.\n    ")

class StringEditor(CellEditor):
    """ Basic string cell editor with auto-completion.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    completions = List(String, help='\n    An optional list of completion strings.\n    ')

class TextEditor(CellEditor):
    """ Multi-line string cell editor.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)

class SelectEditor(CellEditor):
    """ Select cell editor.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
    options = List(String, help='\n    The list of options to select from.\n    ')

class PercentEditor(CellEditor):
    """ ``IntEditor`` optimized for editing percentages.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)

class CheckboxEditor(CellEditor):
    """ Boolean value cell editor.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)

class IntEditor(CellEditor):
    """ Spinner-based integer cell editor.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    step = Int(1, help='\n    The major step value.\n    ')

class NumberEditor(CellEditor):
    """ Spinner-based number cell editor.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    step = Float(0.01, help='\n    The major step value.\n    ')

class TimeEditor(CellEditor):
    """ Spinner-based time cell editor.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)

class DateEditor(CellEditor):
    """ Calendar-based date cell editor.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)

class AvgAggregator(RowAggregator):
    """ Simple average across multiple rows.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)

class MinAggregator(RowAggregator):
    """ Smallest value across multiple rows.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)

class MaxAggregator(RowAggregator):
    """ Largest value across multiple rows.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)

class SumAggregator(RowAggregator):
    """ Simple sum across multiple rows.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)

class TableColumn(Model):
    """ Table column widget.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    field = Required(String, help='\n    The name of the field mapping to a column in the data source.\n    ')
    title = Nullable(String, help="\n    The title of this column. If not set, column's data field is\n    used instead.\n    ")
    width = Int(300, help="\n    The width or maximum width (depending on data table's configuration)\n    in pixels of this column.\n    ")
    formatter = Instance(CellFormatter, InstanceDefault(StringFormatter), help='\n    The cell formatter for this column. By default, a simple string\n    formatter is used.\n    ')
    editor = Instance(CellEditor, InstanceDefault(StringEditor), help='\n    The cell editor for this column. By default, a simple string editor\n    is used.\n    ')
    sortable = Bool(True, help='\n    Whether this column is sortable or not. Note that data table has\n    to have sorting enabled to allow sorting in general.\n    ')
    default_sort = Enum('ascending', 'descending', help='\n    The default sorting order. By default ``ascending`` order is used.\n    ')
    visible = Bool(True, help='\n    Whether this column shold be displayed or not.\n    ')

@abstract
class TableWidget(Widget):
    """ Abstract base class for data table (data grid) widgets.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    source = Instance(DataSource, default=InstanceDefault(ColumnDataSource), help='\n    The source of data for the widget.\n    ')
    view = Instance(CDSView, default=InstanceDefault(CDSView), help='\n    A view into the data source to use when rendering table rows. A default view\n    of the entire data source is created if a view is not passed in during\n    initialization.\n    ')

class DataTable(TableWidget):
    """ Two-dimensional grid for visualization and editing large amounts
    of data.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    autosize_mode = Enum(AutosizeMode, default='force_fit', help='\n    Describes the column autosizing mode with one of the following options:\n\n    ``"fit_columns"``\n        Compute column widths based on cell contents but ensure the\n        table fits into the available viewport. This results in no\n        horizontal scrollbar showing up, but data can get unreadable\n        if there is not enough space available.\n\n    ``"fit_viewport"``\n        Adjust the viewport size after computing columns widths based\n        on cell contents.\n\n    ``"force_fit"``\n        Fit columns into available space dividing the table width across\n        the columns equally (equivalent to `fit_columns=True`).\n        This results in no horizontal scrollbar showing up, but data\n        can get unreadable if there is not enough space available.\n\n    ``"none"``\n        Do not automatically compute column widths.\n    ')
    auto_edit = Bool(False, help='\n    When enabled editing mode is enabled after a single click on a\n    table cell.\n    ')
    columns = List(Instance(TableColumn), help='\n    The list of child column widgets.\n    ')
    fit_columns = Nullable(Bool, help="\n    **This is a legacy parameter.** For new development, use the\n    ``autosize_mode`` parameter.\n\n    Whether columns should be fit to the available width. This results in\n    no horizontal scrollbar showing up, but data can get unreadable if there\n    is not enough space available. If set to ``True``, each column's width is\n    understood as maximum width.\n    ")
    frozen_columns = Nullable(Int, help='\n    Integer indicating the number of columns to freeze. If set the first N\n    columns will be frozen which prevents them from scrolling out of frame.\n    ')
    frozen_rows = Nullable(Int, help='\n    Integer indicating the number of rows to freeze. If set the first N\n    rows will be frozen which prevents them from scrolling out of frame,\n    if set to a negative value last N rows will be frozen.\n    ')
    sortable = Bool(True, help="\n    Allows to sort table's contents. By default natural order is preserved.\n    To sort a column, click on it's header. Clicking one more time changes\n    sort direction. Use Ctrl + click to return to natural order. Use\n    Shift + click to sort multiple columns simultaneously.\n    ")
    reorderable = Bool(True, help="\n    Allows the reordering of a table's columns. To reorder a column,\n    click and drag a table's header to the desired location in the table.\n    The columns on either side will remain in their previous order.\n    ")
    editable = Bool(False, help="\n    Allows to edit table's contents. Needs cell editors to be configured on\n    columns that are required to be editable.\n    ")
    selectable = Either(Bool(True), Enum('checkbox'), help="\n    Whether a table's rows can be selected or not. Using ``checkbox`` is\n    equivalent  to ``True``, but makes selection visible through a checkbox\n    for each row,  instead of highlighting rows. Multiple selection is\n    allowed and can be achieved by either clicking multiple checkboxes (if\n    enabled) or using Shift + click on rows.\n    ")
    index_position = Nullable(Int, default=0, help='\n    Where among the list of columns to insert a column displaying the row\n    index. Negative indices are supported, and specify an index position\n    from the end of the list of columns (i.e. standard Python behaviour).\n\n    To prevent the index column from being added, set to None.\n\n    If the absolute value of index_position  is larger than the length of\n    the columns, then the index will appear at the beginning or end, depending\n    on the sign.\n    ')
    index_header = String('#', help='\n    The column header to display for the index column, if it is present.\n    ')
    index_width = Int(40, help='\n    The width of the index column, if present.\n    ')
    scroll_to_selection = Bool(True, help="\n    Whenever a selection is made on the data source, scroll the selected\n    rows into the table's viewport if none of the selected rows are already\n    in the viewport.\n    ")
    header_row = Bool(True, help='\n    Whether to show a header row with column names at the top of the table.\n    ')
    width = Override(default=600)
    height = Override(default=400)
    row_height = Int(25, help='\n    The height of each row in pixels.\n    ')

class GroupingInfo(Model):
    """Describes how to calculate totals and sub-totals

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
    getter = String('', help='\n    References the column which generates the unique keys of this sub-total (groupby).\n    ')
    aggregators = List(Instance(RowAggregator), help='\n    Describes how to aggregate the columns which will populate this sub-total.\n    ')
    collapsed = Bool(False, help='\n    Whether the corresponding sub-total is expanded or collapsed by default.\n    ')

class DataCube(DataTable):
    """Specialized DataTable with collapsing groups, totals, and sub-totals.

    """

    def __init__(self, *args, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
    grouping = List(Instance(GroupingInfo), help='\n    Describe what aggregation operations used to define sub-totals and totals\n    ')
    target = Instance(DataSource, help='\n    Two column datasource (row_indices & labels) describing which rows of the\n    data cubes are expanded or collapsed\n    ')