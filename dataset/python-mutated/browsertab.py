"""Base class for a wrapper over WebView/WebEngineView."""
import enum
import pathlib
import itertools
import functools
import dataclasses
from typing import cast, TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Sequence, Set, Type, Union, Tuple
from qutebrowser.qt import machinery
from qutebrowser.qt.core import pyqtSignal, pyqtSlot, QUrl, QObject, QSizeF, Qt, QEvent, QPoint, QRect
from qutebrowser.qt.gui import QKeyEvent, QIcon, QPixmap
from qutebrowser.qt.widgets import QApplication, QWidget
from qutebrowser.qt.printsupport import QPrintDialog, QPrinter
from qutebrowser.qt.network import QNetworkAccessManager
if TYPE_CHECKING:
    from qutebrowser.qt.webkit import QWebHistory, QWebHistoryItem
    from qutebrowser.qt.webkitwidgets import QWebPage
    from qutebrowser.qt.webenginecore import QWebEngineHistory, QWebEngineHistoryItem, QWebEnginePage
from qutebrowser.keyinput import modeman
from qutebrowser.config import config, websettings
from qutebrowser.utils import utils, objreg, usertypes, log, qtutils, urlutils, message, jinja, version
from qutebrowser.misc import miscwidgets, objects, sessions
from qutebrowser.browser import eventfilter, inspector
from qutebrowser.qt import sip
if TYPE_CHECKING:
    from qutebrowser.browser import webelem
    from qutebrowser.browser.inspector import AbstractWebInspector
    from qutebrowser.browser.webengine.webview import WebEngineView
    from qutebrowser.browser.webkit.webview import WebView
tab_id_gen = itertools.count(0)
_WidgetType = Union['WebView', 'WebEngineView']

def create(win_id: int, private: bool, parent: QWidget=None) -> 'AbstractTab':
    if False:
        print('Hello World!')
    'Get a QtWebKit/QtWebEngine tab object.\n\n    Args:\n        win_id: The window ID where the tab will be shown.\n        private: Whether the tab is a private/off the record tab.\n        parent: The Qt parent to set.\n    '
    mode_manager = modeman.instance(win_id)
    if objects.backend == usertypes.Backend.QtWebEngine:
        from qutebrowser.browser.webengine import webenginetab
        tab_class: Type[AbstractTab] = webenginetab.WebEngineTab
    elif objects.backend == usertypes.Backend.QtWebKit:
        from qutebrowser.browser.webkit import webkittab
        tab_class = webkittab.WebKitTab
    else:
        raise utils.Unreachable(objects.backend)
    return tab_class(win_id=win_id, mode_manager=mode_manager, private=private, parent=parent)

class WebTabError(Exception):
    """Base class for various errors."""

class UnsupportedOperationError(WebTabError):
    """Raised when an operation is not supported with the given backend."""

class TerminationStatus(enum.Enum):
    """How a QtWebEngine renderer process terminated.

    Also see QWebEnginePage::RenderProcessTerminationStatus
    """
    unknown = -1
    normal = 0
    abnormal = 1
    crashed = 2
    killed = 3

@dataclasses.dataclass
class TabData:
    """A simple namespace with a fixed set of attributes.

    Attributes:
        keep_icon: Whether the (e.g. cloned) icon should not be cleared on page
                   load.
        inspector: The QWebInspector used for this webview.
        viewing_source: Set if we're currently showing a source view.
                        Only used when sources are shown via pygments.
        open_target: Where to open the next link.
                     Only used for QtWebKit.
        override_target: Override for open_target for fake clicks (like hints).
                         Only used for QtWebKit.
        pinned: Flag to pin the tab.
        fullscreen: Whether the tab has a video shown fullscreen currently.
        netrc_used: Whether netrc authentication was performed.
        input_mode: current input mode for the tab.
        splitter: InspectorSplitter used to show inspector inside the tab.
    """
    keep_icon: bool = False
    viewing_source: bool = False
    inspector: Optional['AbstractWebInspector'] = None
    open_target: usertypes.ClickTarget = usertypes.ClickTarget.normal
    override_target: Optional[usertypes.ClickTarget] = None
    pinned: bool = False
    fullscreen: bool = False
    netrc_used: bool = False
    input_mode: usertypes.KeyMode = usertypes.KeyMode.normal
    last_navigation: Optional[usertypes.NavigationRequest] = None
    splitter: Optional[miscwidgets.InspectorSplitter] = None

    def should_show_icon(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return config.val.tabs.favicons.show == 'always' or (config.val.tabs.favicons.show == 'pinned' and self.pinned)

class AbstractAction:
    """Attribute ``action`` of AbstractTab for Qt WebActions."""
    action_base: Type[Union['QWebPage.WebAction', 'QWebEnginePage.WebAction']]

    def __init__(self, tab: 'AbstractTab') -> None:
        if False:
            i = 10
            return i + 15
        self._widget = cast(_WidgetType, None)
        self._tab = tab

    def exit_fullscreen(self) -> None:
        if False:
            i = 10
            return i + 15
        'Exit the fullscreen mode.'
        raise NotImplementedError

    def save_page(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Save the current page.'
        raise NotImplementedError

    def run_string(self, name: str) -> None:
        if False:
            return 10
        'Run a webaction based on its name.'
        try:
            member = getattr(self.action_base, name)
        except AttributeError:
            raise WebTabError(f'{name} is not a valid web action!')
        self._widget.triggerPageAction(member)

    def show_source(self, pygments: bool=False) -> None:
        if False:
            print('Hello World!')
        'Show the source of the current page in a new tab.'
        raise NotImplementedError

    def _show_html_source(self, html: str) -> None:
        if False:
            print('Hello World!')
        'Show the given HTML as source page.'
        tb = objreg.get('tabbed-browser', scope='window', window=self._tab.win_id)
        new_tab = tb.tabopen(background=False, related=True)
        new_tab.set_html(html, self._tab.url())
        new_tab.data.viewing_source = True

    def _show_source_fallback(self, source: str) -> None:
        if False:
            return 10
        'Show source with pygments unavailable.'
        html = jinja.render('pre.html', title='Source', content=source, preamble="Note: The optional Pygments dependency wasn't found - showing unhighlighted source.")
        self._show_html_source(html)

    def _show_source_pygments(self) -> None:
        if False:
            while True:
                i = 10

        def show_source_cb(source: str) -> None:
            if False:
                for i in range(10):
                    print('nop')
            "Show source as soon as it's ready."
            try:
                import pygments
                import pygments.lexers
                import pygments.formatters
            except ImportError:
                self._show_source_fallback(source)
                return
            try:
                lexer = pygments.lexers.HtmlLexer()
                formatter = pygments.formatters.HtmlFormatter(full=True, linenos='table')
            except AttributeError:
                self._show_source_fallback(source)
                return
            html = pygments.highlight(source, lexer, formatter)
            self._show_html_source(html)
        self._tab.dump_async(show_source_cb)

class AbstractPrinting(QObject):
    """Attribute ``printing`` of AbstractTab for printing the page."""
    printing_finished = pyqtSignal(bool)
    pdf_printing_finished = pyqtSignal(str, bool)

    def __init__(self, tab: 'AbstractTab', parent: QWidget=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self._widget = cast(_WidgetType, None)
        self._tab = tab
        self._dialog: Optional[QPrintDialog] = None
        self.printing_finished.connect(self._on_printing_finished)
        self.pdf_printing_finished.connect(self._on_pdf_printing_finished)

    @pyqtSlot(bool)
    def _on_printing_finished(self, ok: bool) -> None:
        if False:
            return 10
        if not ok:
            message.error('Printing failed!')
        if self._dialog is not None:
            self._dialog.deleteLater()
            self._dialog = None

    @pyqtSlot(str, bool)
    def _on_pdf_printing_finished(self, path: str, ok: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        if ok:
            message.info(f'Printed to {path}')
        else:
            message.error(f'Printing to {path} failed!')

    def check_pdf_support(self) -> None:
        if False:
            while True:
                i = 10
        "Check whether writing to PDFs is supported.\n\n        If it's not supported (by the current Qt version), a WebTabError is\n        raised.\n        "
        raise NotImplementedError

    def check_preview_support(self) -> None:
        if False:
            i = 10
            return i + 15
        "Check whether showing a print preview is supported.\n\n        If it's not supported (by the current Qt version), a WebTabError is\n        raised.\n        "
        raise NotImplementedError

    def to_pdf(self, path: pathlib.Path) -> None:
        if False:
            i = 10
            return i + 15
        'Print the tab to a PDF with the given filename.'
        raise NotImplementedError

    def to_printer(self, printer: QPrinter) -> None:
        if False:
            return 10
        'Print the tab.\n\n        Args:\n            printer: The QPrinter to print to.\n        '
        raise NotImplementedError

    def _do_print(self) -> None:
        if False:
            print('Hello World!')
        assert self._dialog is not None
        printer = self._dialog.printer()
        assert printer is not None
        self.to_printer(printer)

    def show_dialog(self) -> None:
        if False:
            while True:
                i = 10
        'Print with a QPrintDialog.'
        self._dialog = QPrintDialog(self._tab)
        self._dialog.open(self._do_print)

@dataclasses.dataclass
class SearchMatch:
    """The currently highlighted search match.

    Attributes:
        current: The currently active search match on the page.
                 0 if no search is active or the feature isn't available.
        total: The total number of search matches on the page.
               0 if no search is active or the feature isn't available.
    """
    current: int = 0
    total: int = 0

    def reset(self) -> None:
        if False:
            return 10
        'Reset match counter information.\n\n        Stale information could lead to next_result or prev_result misbehaving.\n        '
        self.current = 0
        self.total = 0

    def is_null(self) -> bool:
        if False:
            return 10
        'Whether the SearchMatch is set to zero.'
        return self.current == 0 and self.total == 0

    def at_limit(self, going_up: bool) -> bool:
        if False:
            print('Hello World!')
        'Whether the SearchMatch is currently at the first/last result.'
        return self.total != 0 and (going_up and self.current == 1 or (not going_up and self.current == self.total))

    def __str__(self) -> str:
        if False:
            return 10
        return f'{self.current}/{self.total}'

class SearchNavigationResult(enum.Enum):
    """The outcome of calling prev_/next_result."""
    found = enum.auto()
    not_found = enum.auto()
    wrapped_bottom = enum.auto()
    wrap_prevented_bottom = enum.auto()
    wrapped_top = enum.auto()
    wrap_prevented_top = enum.auto()

class AbstractSearch(QObject):
    """Attribute ``search`` of AbstractTab for doing searches.

    Attributes:
        text: The last thing this view was searched for.
        search_displayed: Whether we're currently displaying search results in
                          this view.
        match: The currently active search match.
        _flags: The flags of the last search (needs to be set by subclasses).
        _widget: The underlying WebView widget.

    Signals:
        finished: A search has finished. True if the text was found, false otherwise.
        match_changed: The currently active search match has changed.
                       Emits SearchMatch(0, 0) if no search is active.
                       Will not be emitted if search matches are not available.
        cleared: An existing search was cleared.
    """
    finished = pyqtSignal(bool)
    match_changed = pyqtSignal(SearchMatch)
    cleared = pyqtSignal()
    _Callback = Callable[[bool], None]
    _NavCallback = Callable[[SearchNavigationResult], None]

    def __init__(self, tab: 'AbstractTab', parent: QWidget=None):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self._tab = tab
        self._widget = cast(_WidgetType, None)
        self.text: Optional[str] = None
        self.search_displayed = False
        self.match = SearchMatch()

    def _is_case_sensitive(self, ignore_case: usertypes.IgnoreCase) -> bool:
        if False:
            while True:
                i = 10
        'Check if case-sensitivity should be used.\n\n        This assumes self.text is already set properly.\n\n        Arguments:\n            ignore_case: The ignore_case value from the config.\n        '
        assert self.text is not None
        mapping = {usertypes.IgnoreCase.smart: not self.text.islower(), usertypes.IgnoreCase.never: True, usertypes.IgnoreCase.always: False}
        return mapping[ignore_case]

    def search(self, text: str, *, ignore_case: usertypes.IgnoreCase=usertypes.IgnoreCase.never, reverse: bool=False, result_cb: _Callback=None) -> None:
        if False:
            i = 10
            return i + 15
        'Find the given text on the page.\n\n        Args:\n            text: The text to search for.\n            ignore_case: Search case-insensitively.\n            reverse: Reverse search direction.\n            result_cb: Called with a bool indicating whether a match was found.\n        '
        raise NotImplementedError

    def clear(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Clear the current search.'
        raise NotImplementedError

    def prev_result(self, *, wrap: bool=False, callback: _NavCallback=None) -> None:
        if False:
            print('Hello World!')
        'Go to the previous result of the current search.\n\n        Args:\n            wrap: Allow wrapping at the top or bottom of the page.\n            callback: Called with a SearchNavigationResult.\n        '
        raise NotImplementedError

    def next_result(self, *, wrap: bool=False, callback: _NavCallback=None) -> None:
        if False:
            i = 10
            return i + 15
        'Go to the next result of the current search.\n\n        Args:\n            wrap: Allow wrapping at the top or bottom of the page.\n            callback: Called with a SearchNavigationResult.\n        '
        raise NotImplementedError

class AbstractZoom(QObject):
    """Attribute ``zoom`` of AbstractTab for controlling zoom."""

    def __init__(self, tab: 'AbstractTab', parent: QWidget=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self._tab = tab
        self._widget = cast(_WidgetType, None)
        self._default_zoom_changed = False
        self._init_neighborlist()
        config.instance.changed.connect(self._on_config_changed)
        self._zoom_factor = float(config.val.zoom.default) / 100

    @pyqtSlot(str)
    def _on_config_changed(self, option: str) -> None:
        if False:
            print('Hello World!')
        if option in ['zoom.levels', 'zoom.default']:
            if not self._default_zoom_changed:
                factor = float(config.val.zoom.default) / 100
                self.set_factor(factor)
            self._init_neighborlist()

    def _init_neighborlist(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initialize self._neighborlist.\n\n        It is a NeighborList with the zoom levels.'
        levels = config.val.zoom.levels
        self._neighborlist: usertypes.NeighborList[float] = usertypes.NeighborList(levels, mode=usertypes.NeighborList.Modes.edge)
        self._neighborlist.fuzzyval = config.val.zoom.default

    def apply_offset(self, offset: int) -> float:
        if False:
            print('Hello World!')
        'Increase/Decrease the zoom level by the given offset.\n\n        Args:\n            offset: The offset in the zoom level list.\n\n        Return:\n            The new zoom level.\n        '
        level = self._neighborlist.getitem(offset)
        self.set_factor(float(level) / 100, fuzzyval=False)
        return level

    def _set_factor_internal(self, factor: float) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def set_factor(self, factor: float, *, fuzzyval: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Zoom to a given zoom factor.\n\n        Args:\n            factor: The zoom factor as float.\n            fuzzyval: Whether to set the NeighborLists fuzzyval.\n        '
        if fuzzyval:
            self._neighborlist.fuzzyval = int(factor * 100)
        if factor < 0:
            raise ValueError("Can't zoom to factor {}!".format(factor))
        default_zoom_factor = float(config.val.zoom.default) / 100
        self._default_zoom_changed = factor != default_zoom_factor
        self._zoom_factor = factor
        self._set_factor_internal(factor)

    def factor(self) -> float:
        if False:
            return 10
        return self._zoom_factor

    def apply_default(self) -> None:
        if False:
            return 10
        self._set_factor_internal(float(config.val.zoom.default) / 100)

    def reapply(self) -> None:
        if False:
            return 10
        self._set_factor_internal(self._zoom_factor)

class SelectionState(enum.Enum):
    """Possible states of selection in caret mode.

    NOTE: Names need to line up with SelectionState in caret.js!
    """
    none = enum.auto()
    normal = enum.auto()
    line = enum.auto()

class AbstractCaret(QObject):
    """Attribute ``caret`` of AbstractTab for caret browsing."""
    selection_toggled = pyqtSignal(SelectionState)
    follow_selected_done = pyqtSignal()

    def __init__(self, tab: 'AbstractTab', mode_manager: modeman.ModeManager, parent: QWidget=None) -> None:
        if False:
            return 10
        super().__init__(parent)
        self._widget = cast(_WidgetType, None)
        self._mode_manager = mode_manager
        mode_manager.entered.connect(self._on_mode_entered)
        mode_manager.left.connect(self._on_mode_left)
        self._tab = tab

    def _on_mode_entered(self, mode: usertypes.KeyMode) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def _on_mode_left(self, mode: usertypes.KeyMode) -> None:
        if False:
            return 10
        raise NotImplementedError

    def move_to_next_line(self, count: int=1) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def move_to_prev_line(self, count: int=1) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def move_to_next_char(self, count: int=1) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def move_to_prev_char(self, count: int=1) -> None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def move_to_end_of_word(self, count: int=1) -> None:
        if False:
            return 10
        raise NotImplementedError

    def move_to_next_word(self, count: int=1) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def move_to_prev_word(self, count: int=1) -> None:
        if False:
            return 10
        raise NotImplementedError

    def move_to_start_of_line(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def move_to_end_of_line(self) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def move_to_start_of_next_block(self, count: int=1) -> None:
        if False:
            return 10
        raise NotImplementedError

    def move_to_start_of_prev_block(self, count: int=1) -> None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def move_to_end_of_next_block(self, count: int=1) -> None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def move_to_end_of_prev_block(self, count: int=1) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def move_to_start_of_document(self) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def move_to_end_of_document(self) -> None:
        if False:
            return 10
        raise NotImplementedError

    def toggle_selection(self, line: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def drop_selection(self) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def selection(self, callback: Callable[[str], None]) -> None:
        if False:
            return 10
        raise NotImplementedError

    def reverse_selection(self) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def _follow_enter(self, tab: bool) -> None:
        if False:
            while True:
                i = 10
        'Follow a link by faking an enter press.'
        if tab:
            self._tab.fake_key_press(Qt.Key.Key_Enter, modifier=Qt.KeyboardModifier.ControlModifier)
        else:
            self._tab.fake_key_press(Qt.Key.Key_Enter)

    def follow_selected(self, *, tab: bool=False) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError

class AbstractScroller(QObject):
    """Attribute ``scroller`` of AbstractTab to manage scroll position."""
    perc_changed = pyqtSignal(int, int)
    before_jump_requested = pyqtSignal()

    def __init__(self, tab: 'AbstractTab', parent: QWidget=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self._tab = tab
        self._widget = cast(_WidgetType, None)
        if 'log-scroll-pos' in objects.debug_flags:
            self.perc_changed.connect(self._log_scroll_pos_change)

    @pyqtSlot()
    def _log_scroll_pos_change(self) -> None:
        if False:
            print('Hello World!')
        log.webview.vdebug('Scroll position changed to {}'.format(self.pos_px()))

    def _init_widget(self, widget: _WidgetType) -> None:
        if False:
            while True:
                i = 10
        self._widget = widget

    def pos_px(self) -> QPoint:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def pos_perc(self) -> Tuple[int, int]:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def to_perc(self, x: float=None, y: float=None) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def to_point(self, point: QPoint) -> None:
        if False:
            return 10
        raise NotImplementedError

    def to_anchor(self, name: str) -> None:
        if False:
            return 10
        raise NotImplementedError

    def delta(self, x: int=0, y: int=0) -> None:
        if False:
            return 10
        raise NotImplementedError

    def delta_page(self, x: float=0, y: float=0) -> None:
        if False:
            return 10
        raise NotImplementedError

    def up(self, count: int=1) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def down(self, count: int=1) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def left(self, count: int=1) -> None:
        if False:
            return 10
        raise NotImplementedError

    def right(self, count: int=1) -> None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def top(self) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def bottom(self) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def page_up(self, count: int=1) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def page_down(self, count: int=1) -> None:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def at_top(self) -> bool:
        if False:
            return 10
        raise NotImplementedError

    def at_bottom(self) -> bool:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

class AbstractHistoryPrivate:
    """Private API related to the history."""
    _history: Union['QWebHistory', 'QWebEngineHistory']

    def serialize(self) -> bytes:
        if False:
            return 10
        'Serialize into an opaque format understood by self.deserialize.'
        raise NotImplementedError

    def deserialize(self, data: bytes) -> None:
        if False:
            i = 10
            return i + 15
        'Deserialize from a format produced by self.serialize.'
        raise NotImplementedError

    def load_items(self, items: Sequence[sessions.TabHistoryItem]) -> None:
        if False:
            return 10
        'Deserialize from a list of TabHistoryItems.'
        raise NotImplementedError

class AbstractHistory:
    """The history attribute of a AbstractTab."""

    def __init__(self, tab: 'AbstractTab') -> None:
        if False:
            return 10
        self._tab = tab
        self._history = cast(Union['QWebHistory', 'QWebEngineHistory'], None)
        self.private_api = AbstractHistoryPrivate()

    def __len__(self) -> int:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def __iter__(self) -> Iterable[Union['QWebHistoryItem', 'QWebEngineHistoryItem']]:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def _check_count(self, count: int) -> None:
        if False:
            while True:
                i = 10
        'Check whether the count is positive.'
        if count < 0:
            raise WebTabError('count needs to be positive!')

    def current_idx(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def current_item(self) -> Union['QWebHistoryItem', 'QWebEngineHistoryItem']:
        if False:
            return 10
        raise NotImplementedError

    def back(self, count: int=1) -> None:
        if False:
            while True:
                i = 10
        "Go back in the tab's history."
        self._check_count(count)
        idx = self.current_idx() - count
        if idx >= 0:
            self._go_to_item(self._item_at(idx))
        else:
            self._go_to_item(self._item_at(0))
            raise WebTabError('At beginning of history.')

    def forward(self, count: int=1) -> None:
        if False:
            return 10
        "Go forward in the tab's history."
        self._check_count(count)
        idx = self.current_idx() + count
        if idx < len(self):
            self._go_to_item(self._item_at(idx))
        else:
            self._go_to_item(self._item_at(len(self) - 1))
            raise WebTabError('At end of history.')

    def can_go_back(self) -> bool:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def can_go_forward(self) -> bool:
        if False:
            return 10
        raise NotImplementedError

    def _item_at(self, i: int) -> Any:
        if False:
            print('Hello World!')
        raise NotImplementedError

    def _go_to_item(self, item: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def back_items(self) -> List[Any]:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def forward_items(self) -> List[Any]:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

class AbstractElements:
    """Finding and handling of elements on the page."""
    _MultiCallback = Callable[[Sequence['webelem.AbstractWebElement']], None]
    _SingleCallback = Callable[[Optional['webelem.AbstractWebElement']], None]
    _ErrorCallback = Callable[[Exception], None]

    def __init__(self, tab: 'AbstractTab') -> None:
        if False:
            for i in range(10):
                print('nop')
        self._widget = cast(_WidgetType, None)
        self._tab = tab

    def find_css(self, selector: str, callback: _MultiCallback, error_cb: _ErrorCallback, *, only_visible: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        "Find all HTML elements matching a given selector async.\n\n        If there's an error, the callback is called with a webelem.Error\n        instance.\n\n        Args:\n            callback: The callback to be called when the search finished.\n            error_cb: The callback to be called when an error occurred.\n            selector: The CSS selector to search for.\n            only_visible: Only show elements which are visible on screen.\n        "
        raise NotImplementedError

    def find_id(self, elem_id: str, callback: _SingleCallback) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Find the HTML element with the given ID async.\n\n        Args:\n            callback: The callback to be called when the search finished.\n                      Called with a WebEngineElement or None.\n            elem_id: The ID to search for.\n        '
        raise NotImplementedError

    def find_focused(self, callback: _SingleCallback) -> None:
        if False:
            while True:
                i = 10
        'Find the focused element on the page async.\n\n        Args:\n            callback: The callback to be called when the search finished.\n                      Called with a WebEngineElement or None.\n        '
        raise NotImplementedError

    def find_at_pos(self, pos: QPoint, callback: _SingleCallback) -> None:
        if False:
            while True:
                i = 10
        'Find the element at the given position async.\n\n        This is also called "hit test" elsewhere.\n\n        Args:\n            pos: The QPoint to get the element for.\n            callback: The callback to be called when the search finished.\n                      Called with a WebEngineElement or None.\n        '
        raise NotImplementedError

class AbstractAudio(QObject):
    """Handling of audio/muting for this tab."""
    muted_changed = pyqtSignal(bool)
    recently_audible_changed = pyqtSignal(bool)

    def __init__(self, tab: 'AbstractTab', parent: QWidget=None) -> None:
        if False:
            return 10
        super().__init__(parent)
        self._widget = cast(_WidgetType, None)
        self._tab = tab

    def set_muted(self, muted: bool, override: bool=False) -> None:
        if False:
            return 10
        'Set this tab as muted or not.\n\n        Arguments:\n            muted: Whether the tab is currently muted.\n            override: If set to True, muting/unmuting was done manually and\n                      overrides future automatic mute/unmute changes based on\n                      the URL.\n        '
        raise NotImplementedError

    def is_muted(self) -> bool:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def is_recently_audible(self) -> bool:
        if False:
            while True:
                i = 10
        'Whether this tab has had audio playing recently.'
        raise NotImplementedError

class AbstractTabPrivate:
    """Tab-related methods which are only needed in the core.

    Those methods are not part of the API which is exposed to extensions, and
    should ideally be removed at some point in the future.
    """

    def __init__(self, mode_manager: modeman.ModeManager, tab: 'AbstractTab') -> None:
        if False:
            while True:
                i = 10
        self._widget = cast(_WidgetType, None)
        self._tab = tab
        self._mode_manager = mode_manager

    def event_target(self) -> Optional[QWidget]:
        if False:
            print('Hello World!')
        'Return the widget events should be sent to.'
        raise NotImplementedError

    def handle_auto_insert_mode(self, ok: bool) -> None:
        if False:
            while True:
                i = 10
        'Handle `input.insert_mode.auto_load` after loading finished.'
        if not ok or not config.cache['input.insert_mode.auto_load']:
            return
        cur_mode = self._mode_manager.mode
        if cur_mode == usertypes.KeyMode.insert:
            return

        def _auto_insert_mode_cb(elem: Optional['webelem.AbstractWebElement']) -> None:
            if False:
                print('Hello World!')
            'Called from JS after finding the focused element.'
            if elem is None:
                log.webview.debug('No focused element!')
                return
            if elem.is_editable():
                modeman.enter(self._tab.win_id, usertypes.KeyMode.insert, 'load finished', only_if_normal=True)
        self._tab.elements.find_focused(_auto_insert_mode_cb)

    def clear_ssl_errors(self) -> None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def networkaccessmanager(self) -> Optional[QNetworkAccessManager]:
        if False:
            for i in range(10):
                print('nop')
        'Get the QNetworkAccessManager for this tab.\n\n        This is only implemented for QtWebKit.\n        For QtWebEngine, always returns None.\n        '
        raise NotImplementedError

    def shutdown(self) -> None:
        if False:
            return 10
        raise NotImplementedError

    def run_js_sync(self, code: str) -> Any:
        if False:
            i = 10
            return i + 15
        'Run javascript sync.\n\n        Result will be returned when running JS is complete.\n        This is only implemented for QtWebKit.\n        For QtWebEngine, always raises UnsupportedOperationError.\n        '
        raise NotImplementedError

    def _recreate_inspector(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Recreate the inspector when detached to a window.\n\n        This is needed to circumvent a QtWebEngine bug (which wasn't\n        investigated further) which sometimes results in the window not\n        appearing anymore.\n        "
        self._tab.data.inspector = None
        self.toggle_inspector(inspector.Position.window)

    def toggle_inspector(self, position: Optional[inspector.Position]) -> None:
        if False:
            while True:
                i = 10
        'Show/hide (and if needed, create) the web inspector for this tab.'
        tabdata = self._tab.data
        if tabdata.inspector is None:
            assert tabdata.splitter is not None
            tabdata.inspector = self._init_inspector(splitter=tabdata.splitter, win_id=self._tab.win_id)
            self._tab.shutting_down.connect(tabdata.inspector.shutdown)
            tabdata.inspector.recreate.connect(self._recreate_inspector)
            tabdata.inspector.inspect(self._widget.page())
        tabdata.inspector.set_position(position)

    def _init_inspector(self, splitter: 'miscwidgets.InspectorSplitter', win_id: int, parent: QWidget=None) -> 'AbstractWebInspector':
        if False:
            print('Hello World!')
        'Get a WebKitInspector/WebEngineInspector.\n\n        Args:\n            splitter: InspectorSplitter where the inspector can be placed.\n            win_id: The window ID this inspector is associated with.\n            parent: The Qt parent to set.\n        '
        raise NotImplementedError

class AbstractTab(QWidget):
    """An adapter for WebView/WebEngineView representing a single tab."""
    window_close_requested = pyqtSignal()
    link_hovered = pyqtSignal(str)
    load_started = pyqtSignal()
    load_progress = pyqtSignal(int)
    load_finished = pyqtSignal(bool)
    icon_changed = pyqtSignal(QIcon)
    title_changed = pyqtSignal(str)
    pinned_changed = pyqtSignal(bool)
    new_tab_requested = pyqtSignal(QUrl)
    url_changed = pyqtSignal(QUrl)
    contents_size_changed = pyqtSignal(QSizeF)
    fullscreen_requested = pyqtSignal(bool)
    before_load_started = pyqtSignal(QUrl)
    load_status_changed = pyqtSignal(usertypes.LoadStatus)
    shutting_down = pyqtSignal()
    history_item_triggered = pyqtSignal(QUrl, QUrl, str)
    renderer_process_terminated = pyqtSignal(TerminationStatus, int)
    _insecure_hosts: Set[str] = set()
    history: AbstractHistory
    scroller: AbstractScroller
    caret: AbstractCaret
    zoom: AbstractZoom
    search: AbstractSearch
    printing: AbstractPrinting
    action: AbstractAction
    elements: AbstractElements
    audio: AbstractAudio
    private_api: AbstractTabPrivate
    settings: websettings.AbstractSettings

    def __init__(self, *, win_id: int, mode_manager: 'modeman.ModeManager', private: bool, parent: QWidget=None) -> None:
        if False:
            return 10
        utils.unused(mode_manager)
        self.is_private = private
        self.win_id = win_id
        self.tab_id = next(tab_id_gen)
        super().__init__(parent)
        self.registry = objreg.ObjectRegistry()
        tab_registry = objreg.get('tab-registry', scope='window', window=win_id)
        tab_registry[self.tab_id] = self
        objreg.register('tab', self, registry=self.registry)
        self.data = TabData()
        self._layout = miscwidgets.WrapperLayout(self)
        self._widget = cast(_WidgetType, None)
        self._progress = 0
        self._load_status = usertypes.LoadStatus.none
        self._tab_event_filter = eventfilter.TabEventFilter(self, parent=self)
        self.backend: Optional[usertypes.Backend] = None
        self.pending_removal = False
        self.shutting_down.connect(functools.partial(setattr, self, 'pending_removal', True))
        self.before_load_started.connect(self._on_before_load_started)

    def _set_widget(self, widget: _WidgetType) -> None:
        if False:
            i = 10
            return i + 15
        self._widget = widget
        self.data.splitter = miscwidgets.InspectorSplitter(win_id=self.win_id, main_webview=widget)
        self._layout.wrap(self, self.data.splitter)
        self.history._history = widget.history()
        self.history.private_api._history = widget.history()
        self.scroller._init_widget(widget)
        self.caret._widget = widget
        self.zoom._widget = widget
        self.search._widget = widget
        self.printing._widget = widget
        self.action._widget = widget
        self.elements._widget = widget
        self.audio._widget = widget
        self.private_api._widget = widget
        self.settings._settings = widget.settings()
        self._install_event_filter()
        self.zoom.apply_default()

    def _install_event_filter(self) -> None:
        if False:
            return 10
        raise NotImplementedError

    def _set_load_status(self, val: usertypes.LoadStatus) -> None:
        if False:
            print('Hello World!')
        'Setter for load_status.'
        if not isinstance(val, usertypes.LoadStatus):
            raise TypeError('Type {} is no LoadStatus member!'.format(val))
        log.webview.debug('load status for {}: {}'.format(repr(self), val))
        self._load_status = val
        self.load_status_changed.emit(val)

    def send_event(self, evt: QEvent) -> None:
        if False:
            while True:
                i = 10
        'Send the given event to the underlying widget.\n\n        The event will be sent via QApplication.postEvent.\n        Note that a posted event must not be re-used in any way!\n        '
        if getattr(evt, 'posted', False):
            raise utils.Unreachable("Can't re-use an event which was already posted!")
        recipient = self.private_api.event_target()
        if recipient is None:
            log.webview.warning('Unable to find event target!')
            return
        evt.posted = True
        QApplication.postEvent(recipient, evt)

    def navigation_blocked(self) -> bool:
        if False:
            return 10
        'Test if navigation is allowed on the current tab.'
        return self.data.pinned and config.val.tabs.pinned.frozen

    @pyqtSlot(QUrl)
    def _on_before_load_started(self, url: QUrl) -> None:
        if False:
            return 10
        'Adjust the title if we are going to visit a URL soon.'
        qtutils.ensure_valid(url)
        url_string = url.toDisplayString()
        log.webview.debug('Going to start loading: {}'.format(url_string))
        self.title_changed.emit(url_string)

    @pyqtSlot(QUrl)
    def _on_url_changed(self, url: QUrl) -> None:
        if False:
            return 10
        'Update title when URL has changed and no title is available.'
        if url.isValid() and (not self.title()):
            self.title_changed.emit(url.toDisplayString())
        self.url_changed.emit(url)

    @pyqtSlot()
    def _on_load_started(self) -> None:
        if False:
            return 10
        self._progress = 0
        self.data.viewing_source = False
        self._set_load_status(usertypes.LoadStatus.loading)
        self.load_started.emit()

    @pyqtSlot(usertypes.NavigationRequest)
    def _on_navigation_request(self, navigation: usertypes.NavigationRequest) -> None:
        if False:
            return 10
        'Handle common acceptNavigationRequest code.'
        url = utils.elide(navigation.url.toDisplayString(), 100)
        log.webview.debug(f'navigation request: url {url} (current {self.url().toDisplayString()}), type {navigation.navigation_type.name}, is_main_frame {navigation.is_main_frame}')
        if navigation.is_main_frame:
            self.data.last_navigation = navigation
        if not navigation.url.isValid():
            if navigation.navigation_type == navigation.Type.link_clicked:
                msg = urlutils.get_errstring(navigation.url, 'Invalid link clicked')
                message.error(msg)
                self.data.open_target = usertypes.ClickTarget.normal
            log.webview.debug('Ignoring invalid URL {} in acceptNavigationRequest: {}'.format(navigation.url.toDisplayString(), navigation.url.errorString()))
            navigation.accepted = False
        needs_load_workarounds = objects.backend == usertypes.Backend.QtWebEngine and version.qtwebengine_versions().webengine >= utils.VersionNumber(6, 2)
        if needs_load_workarounds and self.url() == QUrl('qute://start/') and (navigation.navigation_type == navigation.Type.form_submitted) and navigation.url.matches(QUrl(config.val.url.searchengines['DEFAULT']), urlutils.FormatOption.REMOVE_QUERY):
            log.webview.debug(f'Working around qute://start loading issue for {navigation.url.toDisplayString()}')
            navigation.accepted = False
            self.load_url(navigation.url)
        if needs_load_workarounds and self.url() == QUrl('qute://bookmarks/') and (navigation.navigation_type == navigation.Type.back_forward):
            log.webview.debug(f'Working around qute://bookmarks loading issue for {navigation.url.toDisplayString()}')
            navigation.accepted = False
            self.load_url(navigation.url)

    @pyqtSlot(bool)
    def _on_load_finished(self, ok: bool) -> None:
        if False:
            i = 10
            return i + 15
        assert self._widget is not None
        if self.is_deleted():
            return
        if sessions.session_manager is not None:
            sessions.session_manager.save_autosave()
        self.load_finished.emit(ok)
        if not self.title():
            self.title_changed.emit(self.url().toDisplayString())
        self.zoom.reapply()

    def _update_load_status(self, ok: bool) -> None:
        if False:
            i = 10
            return i + 15
        'Update the load status after a page finished loading.\n\n        Needs to be called by subclasses to trigger a load status update, e.g.\n        as a response to a loadFinished signal.\n        '
        url = self.url()
        is_https = url.scheme() == 'https'
        if not ok:
            loadstatus = usertypes.LoadStatus.error
        elif is_https and url.host() in self._insecure_hosts:
            loadstatus = usertypes.LoadStatus.warn
        elif is_https:
            loadstatus = usertypes.LoadStatus.success_https
        else:
            loadstatus = usertypes.LoadStatus.success
        self._set_load_status(loadstatus)

    @pyqtSlot()
    def _on_history_trigger(self) -> None:
        if False:
            print('Hello World!')
        'Emit history_item_triggered based on backend-specific signal.'
        raise NotImplementedError

    @pyqtSlot(int)
    def _on_load_progress(self, perc: int) -> None:
        if False:
            while True:
                i = 10
        self._progress = perc
        self.load_progress.emit(perc)

    def url(self, *, requested: bool=False) -> QUrl:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def progress(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self._progress

    def load_status(self) -> usertypes.LoadStatus:
        if False:
            for i in range(10):
                print('nop')
        return self._load_status

    def _load_url_prepare(self, url: QUrl) -> None:
        if False:
            for i in range(10):
                print('nop')
        qtutils.ensure_valid(url)
        self.before_load_started.emit(url)

    def load_url(self, url: QUrl) -> None:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def reload(self, *, force: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def stop(self) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def fake_key_press(self, key: Qt.Key, modifier: Qt.KeyboardModifier=Qt.KeyboardModifier.NoModifier) -> None:
        if False:
            return 10
        'Send a fake key event to this tab.'
        press_evt = QKeyEvent(QEvent.Type.KeyPress, key, modifier, 0, 0, 0)
        release_evt = QKeyEvent(QEvent.Type.KeyRelease, key, modifier, 0, 0, 0)
        self.send_event(press_evt)
        self.send_event(release_evt)

    def dump_async(self, callback: Callable[[str], None], *, plain: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Dump the current page's html asynchronously.\n\n        The given callback will be called with the result when dumping is\n        complete.\n        "
        raise NotImplementedError

    def run_js_async(self, code: str, callback: Callable[[Any], None]=None, *, world: Union[usertypes.JsWorld, int]=None) -> None:
        if False:
            while True:
                i = 10
        'Run javascript async.\n\n        The given callback will be called with the result when running JS is\n        complete.\n\n        Args:\n            code: The javascript code to run.\n            callback: The callback to call with the result, or None.\n            world: A world ID (int or usertypes.JsWorld member) to run the JS\n                   in the main world or in another isolated world.\n        '
        raise NotImplementedError

    def title(self) -> str:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def icon(self) -> QIcon:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def set_html(self, html: str, base_url: QUrl=QUrl()) -> None:
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def set_pinned(self, pinned: bool) -> None:
        if False:
            i = 10
            return i + 15
        self.data.pinned = pinned
        self.pinned_changed.emit(pinned)

    def renderer_process_pid(self) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        "Get the PID of the underlying renderer process.\n\n        Returns None if the PID can't be determined or if getting the PID isn't\n        supported.\n        "
        raise NotImplementedError

    def grab_pixmap(self, rect: QRect=None) -> Optional[QPixmap]:
        if False:
            print('Hello World!')
        'Grab a QPixmap of the displayed page.\n\n        Returns None if we got a null pixmap from Qt.\n        '
        if rect is None:
            pic = self._widget.grab()
        else:
            qtutils.ensure_valid(rect)
            pic = self._widget.grab(rect)
        if pic.isNull():
            return None
        if machinery.IS_QT6:
            pic = cast(QPixmap, pic)
        return pic

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        try:
            qurl = self.url()
            url = qurl.toDisplayString(urlutils.FormatOption.ENCODE_UNICODE)
        except (AttributeError, RuntimeError) as exc:
            url = '<{}>'.format(exc.__class__.__name__)
        else:
            url = utils.elide(url, 100)
        return utils.get_repr(self, tab_id=self.tab_id, url=url)

    def is_deleted(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Check if the tab has been deleted.'
        assert self._widget is not None
        if machinery.IS_QT6:
            widget = cast(QWidget, self._widget)
        else:
            widget = self._widget
        return sip.isdeleted(widget)