"""Wrapper over our (QtWebKit) WebView."""
import re
import functools
import xml.etree.ElementTree
from typing import cast, Iterable, Optional
from qutebrowser.qt.core import pyqtSlot, Qt, QUrl, QPoint, QTimer, QSizeF, QSize
from qutebrowser.qt.gui import QIcon
from qutebrowser.qt.widgets import QWidget
from qutebrowser.qt.webkitwidgets import QWebPage, QWebFrame
from qutebrowser.qt.webkit import QWebSettings, QWebHistory, QWebElement
from qutebrowser.qt.printsupport import QPrinter
from qutebrowser.browser import browsertab, shared
from qutebrowser.browser.webkit import webview, webpage, tabhistory, webkitelem, webkitsettings, webkitinspector
from qutebrowser.browser.webkit.network import networkmanager
from qutebrowser.utils import qtutils, usertypes, utils, log, debug, resources
from qutebrowser.keyinput import modeman
from qutebrowser.qt import sip

class WebKitAction(browsertab.AbstractAction):
    """QtWebKit implementations related to web actions."""
    action_base = QWebPage.WebAction
    _widget: webview.WebView

    def exit_fullscreen(self):
        if False:
            print('Hello World!')
        raise browsertab.UnsupportedOperationError

    def save_page(self):
        if False:
            return 10
        'Save the current page.'
        raise browsertab.UnsupportedOperationError

    def show_source(self, pygments=False):
        if False:
            print('Hello World!')
        self._show_source_pygments()

    def run_string(self, name: str) -> None:
        if False:
            print('Hello World!')
        "Add special cases for new API.\n\n        Those were added to QtWebKit 5.212 (which we enforce), but we don't get\n        the new API from PyQt. Thus, we'll need to use the raw numbers.\n        "
        new_actions = {'RequestClose': QWebPage.WebAction.ToggleVideoFullscreen + 1, 'Unselect': QWebPage.WebAction.ToggleVideoFullscreen + 2}
        if name in new_actions:
            self._widget.triggerPageAction(new_actions[name])
            return
        super().run_string(name)

class WebKitPrinting(browsertab.AbstractPrinting):
    """QtWebKit implementations related to printing."""
    _widget: webview.WebView

    def check_pdf_support(self):
        if False:
            return 10
        pass

    def check_preview_support(self):
        if False:
            i = 10
            return i + 15
        pass

    def to_pdf(self, path):
        if False:
            return 10
        printer = QPrinter()
        printer.setOutputFileName(str(path))
        self._widget.print(printer)
        self.pdf_printing_finished.emit(str(path), True)

    def to_printer(self, printer):
        if False:
            return 10
        self._widget.print(printer)
        self.printing_finished.emit(True)

class WebKitSearch(browsertab.AbstractSearch):
    """QtWebKit implementations related to searching on the page."""
    _widget: webview.WebView

    def __init__(self, tab, parent=None):
        if False:
            return 10
        super().__init__(tab, parent)
        self._flags = self._empty_flags()

    def _empty_flags(self):
        if False:
            return 10
        return QWebPage.FindFlags(0)

    def _args_to_flags(self, reverse, ignore_case):
        if False:
            print('Hello World!')
        flags = self._empty_flags()
        if self._is_case_sensitive(ignore_case):
            flags |= QWebPage.FindFlag.FindCaseSensitively
        if reverse:
            flags |= QWebPage.FindFlag.FindBackward
        return flags

    def _call_cb(self, callback, found, text, flags, caller):
        if False:
            for i in range(10):
                print('nop')
        "Call the given callback if it's non-None.\n\n        Delays the call via a QTimer so the website is re-rendered in between.\n\n        Args:\n            callback: What to call\n            found: If the text was found\n            text: The text searched for\n            flags: The flags searched with\n            caller: Name of the caller.\n        "
        found_text = 'found' if found else "didn't find"
        debug_flags = debug.qflags_key(QWebPage, flags & ~QWebPage.FindFlag.FindWrapsAroundDocument, klass=QWebPage.FindFlag)
        if debug_flags != '0x0000':
            flag_text = 'with flags {}'.format(debug_flags)
        else:
            flag_text = ''
        log.webview.debug(' '.join([caller, found_text, text, flag_text]).strip())
        if callback is not None:
            if caller in ['prev_result', 'next_result']:
                if found:
                    cb_value = browsertab.SearchNavigationResult.found
                elif flags & QWebPage.FindBackward:
                    cb_value = browsertab.SearchNavigationResult.wrap_prevented_top
                else:
                    cb_value = browsertab.SearchNavigationResult.wrap_prevented_bottom
            elif caller == 'search':
                cb_value = found
            else:
                raise utils.Unreachable(caller)
            QTimer.singleShot(0, functools.partial(callback, cb_value))
        self.finished.emit(found)

    def clear(self):
        if False:
            while True:
                i = 10
        if self.search_displayed:
            self.cleared.emit()
        self.search_displayed = False
        self._widget.findText('')
        self._widget.findText('', QWebPage.FindFlag.HighlightAllOccurrences)

    def search(self, text, *, ignore_case=usertypes.IgnoreCase.never, reverse=False, result_cb=None):
        if False:
            i = 10
            return i + 15
        if self.text == text and self.search_displayed:
            log.webview.debug('Ignoring duplicate search request for {}, but resetting flags'.format(text))
            self._flags = self._args_to_flags(reverse, ignore_case)
            return
        self.clear()
        self.text = text
        self.search_displayed = True
        self._flags = self._args_to_flags(reverse, ignore_case)
        found = self._widget.findText(text, self._flags)
        self._widget.findText(text, self._flags | QWebPage.FindFlag.HighlightAllOccurrences)
        self._call_cb(result_cb, found, text, self._flags, 'search')

    def next_result(self, *, wrap=False, callback=None):
        if False:
            i = 10
            return i + 15
        self.search_displayed = True
        flags = QWebPage.FindFlags(int(self._flags))
        if wrap:
            flags |= QWebPage.FindFlag.FindWrapsAroundDocument
        found = self._widget.findText(self.text, flags)
        self._call_cb(callback, found, self.text, flags, 'next_result')

    def prev_result(self, *, wrap=False, callback=None):
        if False:
            for i in range(10):
                print('nop')
        self.search_displayed = True
        flags = QWebPage.FindFlags(int(self._flags))
        if flags & QWebPage.FindFlag.FindBackward:
            flags &= ~QWebPage.FindFlag.FindBackward
        else:
            flags |= QWebPage.FindFlag.FindBackward
        if wrap:
            flags |= QWebPage.FindFlag.FindWrapsAroundDocument
        found = self._widget.findText(self.text, flags)
        self._call_cb(callback, found, self.text, flags, 'prev_result')

class WebKitCaret(browsertab.AbstractCaret):
    """QtWebKit implementations related to moving the cursor/selection."""
    _widget: webview.WebView

    def __init__(self, tab: 'WebKitTab', mode_manager: modeman.ModeManager, parent: QWidget=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(tab, mode_manager, parent)
        self._selection_state = browsertab.SelectionState.none

    @pyqtSlot(usertypes.KeyMode)
    def _on_mode_entered(self, mode):
        if False:
            print('Hello World!')
        if mode != usertypes.KeyMode.caret:
            return
        if self._widget.hasSelection():
            self._selection_state = browsertab.SelectionState.normal
        else:
            self._selection_state = browsertab.SelectionState.none
        self.selection_toggled.emit(self._selection_state)
        settings = self._widget.settings()
        settings.setAttribute(QWebSettings.WebAttribute.CaretBrowsingEnabled, True)
        if self._widget.isVisible():
            self._widget.clearFocus()
            self._widget.setFocus(Qt.FocusReason.OtherFocusReason)
            if self._selection_state is browsertab.SelectionState.none:
                self._widget.page().currentFrame().evaluateJavaScript(resources.read_file('javascript/position_caret.js'))

    @pyqtSlot(usertypes.KeyMode)
    def _on_mode_left(self, _mode):
        if False:
            i = 10
            return i + 15
        settings = self._widget.settings()
        if settings.testAttribute(QWebSettings.WebAttribute.CaretBrowsingEnabled):
            if self._selection_state is not browsertab.SelectionState.none and self._widget.hasSelection():
                self._widget.triggerPageAction(QWebPage.WebAction.MoveToNextChar)
            settings.setAttribute(QWebSettings.WebAttribute.CaretBrowsingEnabled, False)
            self._selection_state = browsertab.SelectionState.none

    def move_to_next_line(self, count=1):
        if False:
            i = 10
            return i + 15
        if self._selection_state is not browsertab.SelectionState.none:
            act = QWebPage.WebAction.SelectNextLine
        else:
            act = QWebPage.WebAction.MoveToNextLine
        for _ in range(count):
            self._widget.triggerPageAction(act)
        if self._selection_state is browsertab.SelectionState.line:
            self._select_line_to_end()

    def move_to_prev_line(self, count=1):
        if False:
            i = 10
            return i + 15
        if self._selection_state is not browsertab.SelectionState.none:
            act = QWebPage.WebAction.SelectPreviousLine
        else:
            act = QWebPage.WebAction.MoveToPreviousLine
        for _ in range(count):
            self._widget.triggerPageAction(act)
        if self._selection_state is browsertab.SelectionState.line:
            self._select_line_to_start()

    def move_to_next_char(self, count=1):
        if False:
            for i in range(10):
                print('nop')
        if self._selection_state is browsertab.SelectionState.normal:
            act = QWebPage.WebAction.SelectNextChar
        elif self._selection_state is browsertab.SelectionState.line:
            return
        else:
            act = QWebPage.WebAction.MoveToNextChar
        for _ in range(count):
            self._widget.triggerPageAction(act)

    def move_to_prev_char(self, count=1):
        if False:
            return 10
        if self._selection_state is browsertab.SelectionState.normal:
            act = QWebPage.WebAction.SelectPreviousChar
        elif self._selection_state is browsertab.SelectionState.line:
            return
        else:
            act = QWebPage.WebAction.MoveToPreviousChar
        for _ in range(count):
            self._widget.triggerPageAction(act)

    def move_to_end_of_word(self, count=1):
        if False:
            print('Hello World!')
        if self._selection_state is browsertab.SelectionState.normal:
            act = [QWebPage.WebAction.SelectNextWord]
            if utils.is_windows:
                act.append(QWebPage.WebAction.SelectPreviousChar)
        elif self._selection_state is browsertab.SelectionState.line:
            return
        else:
            act = [QWebPage.WebAction.MoveToNextWord]
            if utils.is_windows:
                act.append(QWebPage.WebAction.MoveToPreviousChar)
        for _ in range(count):
            for a in act:
                self._widget.triggerPageAction(a)

    def move_to_next_word(self, count=1):
        if False:
            return 10
        if self._selection_state is browsertab.SelectionState.normal:
            act = [QWebPage.WebAction.SelectNextWord]
            if not utils.is_windows:
                act.append(QWebPage.WebAction.SelectNextChar)
        elif self._selection_state is browsertab.SelectionState.line:
            return
        else:
            act = [QWebPage.WebAction.MoveToNextWord]
            if not utils.is_windows:
                act.append(QWebPage.WebAction.MoveToNextChar)
        for _ in range(count):
            for a in act:
                self._widget.triggerPageAction(a)

    def move_to_prev_word(self, count=1):
        if False:
            i = 10
            return i + 15
        if self._selection_state is browsertab.SelectionState.normal:
            act = QWebPage.WebAction.SelectPreviousWord
        elif self._selection_state is browsertab.SelectionState.line:
            return
        else:
            act = QWebPage.WebAction.MoveToPreviousWord
        for _ in range(count):
            self._widget.triggerPageAction(act)

    def move_to_start_of_line(self):
        if False:
            print('Hello World!')
        if self._selection_state is browsertab.SelectionState.normal:
            act = QWebPage.WebAction.SelectStartOfLine
        elif self._selection_state is browsertab.SelectionState.line:
            return
        else:
            act = QWebPage.WebAction.MoveToStartOfLine
        self._widget.triggerPageAction(act)

    def move_to_end_of_line(self):
        if False:
            for i in range(10):
                print('nop')
        if self._selection_state is browsertab.SelectionState.normal:
            act = QWebPage.WebAction.SelectEndOfLine
        elif self._selection_state is browsertab.SelectionState.line:
            return
        else:
            act = QWebPage.WebAction.MoveToEndOfLine
        self._widget.triggerPageAction(act)

    def move_to_start_of_next_block(self, count=1):
        if False:
            for i in range(10):
                print('nop')
        if self._selection_state is not browsertab.SelectionState.none:
            act = [QWebPage.WebAction.SelectNextLine, QWebPage.WebAction.SelectStartOfBlock]
        else:
            act = [QWebPage.WebAction.MoveToNextLine, QWebPage.WebAction.MoveToStartOfBlock]
        for _ in range(count):
            for a in act:
                self._widget.triggerPageAction(a)
        if self._selection_state is browsertab.SelectionState.line:
            self._select_line_to_end()

    def move_to_start_of_prev_block(self, count=1):
        if False:
            i = 10
            return i + 15
        if self._selection_state is not browsertab.SelectionState.none:
            act = [QWebPage.WebAction.SelectPreviousLine, QWebPage.WebAction.SelectStartOfBlock]
        else:
            act = [QWebPage.WebAction.MoveToPreviousLine, QWebPage.WebAction.MoveToStartOfBlock]
        for _ in range(count):
            for a in act:
                self._widget.triggerPageAction(a)
        if self._selection_state is browsertab.SelectionState.line:
            self._select_line_to_start()

    def move_to_end_of_next_block(self, count=1):
        if False:
            while True:
                i = 10
        if self._selection_state is not browsertab.SelectionState.none:
            act = [QWebPage.WebAction.SelectNextLine, QWebPage.WebAction.SelectEndOfBlock]
        else:
            act = [QWebPage.WebAction.MoveToNextLine, QWebPage.WebAction.MoveToEndOfBlock]
        for _ in range(count):
            for a in act:
                self._widget.triggerPageAction(a)
        if self._selection_state is browsertab.SelectionState.line:
            self._select_line_to_end()

    def move_to_end_of_prev_block(self, count=1):
        if False:
            for i in range(10):
                print('nop')
        if self._selection_state is not browsertab.SelectionState.none:
            act = [QWebPage.WebAction.SelectPreviousLine, QWebPage.WebAction.SelectEndOfBlock]
        else:
            act = [QWebPage.WebAction.MoveToPreviousLine, QWebPage.WebAction.MoveToEndOfBlock]
        for _ in range(count):
            for a in act:
                self._widget.triggerPageAction(a)
        if self._selection_state is browsertab.SelectionState.line:
            self._select_line_to_start()

    def move_to_start_of_document(self):
        if False:
            while True:
                i = 10
        if self._selection_state is not browsertab.SelectionState.none:
            act = QWebPage.WebAction.SelectStartOfDocument
        else:
            act = QWebPage.WebAction.MoveToStartOfDocument
        self._widget.triggerPageAction(act)
        if self._selection_state is browsertab.SelectionState.line:
            self._select_line()

    def move_to_end_of_document(self):
        if False:
            for i in range(10):
                print('nop')
        if self._selection_state is not browsertab.SelectionState.none:
            act = QWebPage.WebAction.SelectEndOfDocument
        else:
            act = QWebPage.WebAction.MoveToEndOfDocument
        self._widget.triggerPageAction(act)

    def toggle_selection(self, line=False):
        if False:
            print('Hello World!')
        if line:
            self._selection_state = browsertab.SelectionState.line
            self._select_line()
            self.reverse_selection()
            self._select_line()
            self.reverse_selection()
        elif self._selection_state is not browsertab.SelectionState.normal:
            self._selection_state = browsertab.SelectionState.normal
        else:
            self._selection_state = browsertab.SelectionState.none
        self.selection_toggled.emit(self._selection_state)

    def drop_selection(self):
        if False:
            for i in range(10):
                print('nop')
        self._widget.triggerPageAction(QWebPage.WebAction.MoveToNextChar)

    def selection(self, callback):
        if False:
            return 10
        callback(self._widget.selectedText())

    def reverse_selection(self):
        if False:
            print('Hello World!')
        self._tab.run_js_async('{\n            const sel = window.getSelection();\n            sel.setBaseAndExtent(\n                sel.extentNode, sel.extentOffset, sel.baseNode,\n                sel.baseOffset\n            );\n        }')

    def _select_line(self):
        if False:
            return 10
        self._widget.triggerPageAction(QWebPage.WebAction.SelectStartOfLine)
        self.reverse_selection()
        self._widget.triggerPageAction(QWebPage.WebAction.SelectEndOfLine)
        self.reverse_selection()

    def _select_line_to_end(self):
        if False:
            i = 10
            return i + 15
        if self._js_selection_left_to_right():
            self._widget.triggerPageAction(QWebPage.WebAction.SelectEndOfLine)

    def _select_line_to_start(self):
        if False:
            return 10
        if not self._js_selection_left_to_right():
            self._widget.triggerPageAction(QWebPage.WebAction.SelectStartOfLine)

    def _js_selection_left_to_right(self):
        if False:
            for i in range(10):
                print('nop')
        "Return True iff the selection's direction is left to right."
        return self._tab.private_api.run_js_sync('\n            var sel = window.getSelection();\n            var position = sel.anchorNode.compareDocumentPosition(sel.focusNode);\n            (!position && sel.anchorOffset < sel.focusOffset ||\n                position === Node.DOCUMENT_POSITION_FOLLOWING);\n        ')

    def _follow_selected(self, *, tab=False):
        if False:
            i = 10
            return i + 15
        if QWebSettings.globalSettings().testAttribute(QWebSettings.WebAttribute.JavascriptEnabled):
            if tab:
                self._tab.data.override_target = usertypes.ClickTarget.tab
            self._tab.run_js_async('\n                const aElm = document.activeElement;\n                if (window.getSelection().anchorNode) {\n                    window.getSelection().anchorNode.parentNode.click();\n                } else if (aElm && aElm !== document.body) {\n                    aElm.click();\n                }\n            ')
        else:
            selection = self._widget.selectedHtml()
            if not selection:
                self._follow_enter(tab)
                return
            try:
                selected_element = xml.etree.ElementTree.fromstring('<html>{}</html>'.format(selection)).find('a')
            except xml.etree.ElementTree.ParseError:
                raise browsertab.WebTabError('Could not parse selected element!')
            if selected_element is not None:
                try:
                    href = selected_element.attrib['href']
                except KeyError:
                    raise browsertab.WebTabError('Anchor element without href!')
                url = self._tab.url().resolved(QUrl(href))
                if tab:
                    self._tab.new_tab_requested.emit(url)
                else:
                    self._tab.load_url(url)

    def follow_selected(self, *, tab=False):
        if False:
            i = 10
            return i + 15
        try:
            self._follow_selected(tab=tab)
        finally:
            self.follow_selected_done.emit()

class WebKitZoom(browsertab.AbstractZoom):
    """QtWebKit implementations related to zooming."""
    _widget: webview.WebView

    def _set_factor_internal(self, factor):
        if False:
            for i in range(10):
                print('nop')
        self._widget.setZoomFactor(factor)

class WebKitScroller(browsertab.AbstractScroller):
    """QtWebKit implementations related to scrolling."""
    _widget: webview.WebView

    def pos_px(self):
        if False:
            i = 10
            return i + 15
        return self._widget.page().mainFrame().scrollPosition()

    def pos_perc(self):
        if False:
            i = 10
            return i + 15
        return self._widget.scroll_pos

    def to_point(self, point):
        if False:
            while True:
                i = 10
        self._widget.page().mainFrame().setScrollPosition(point)

    def to_anchor(self, name):
        if False:
            for i in range(10):
                print('nop')
        self._widget.page().mainFrame().scrollToAnchor(name)

    def delta(self, x: int=0, y: int=0) -> None:
        if False:
            for i in range(10):
                print('nop')
        qtutils.check_overflow(x, 'int')
        qtutils.check_overflow(y, 'int')
        self._widget.page().mainFrame().scroll(x, y)

    def delta_page(self, x: float=0.0, y: float=0.0) -> None:
        if False:
            for i in range(10):
                print('nop')
        if y.is_integer():
            y = int(y)
            if y == 0:
                pass
            elif y < 0:
                self.page_up(count=-y)
            elif y > 0:
                self.page_down(count=y)
            y = 0
        if x == 0 and y == 0:
            return
        size = self._widget.page().mainFrame().geometry()
        self.delta(int(x * size.width()), int(y * size.height()))

    def to_perc(self, x=None, y=None):
        if False:
            print('Hello World!')
        if x is None and y == 0:
            self.top()
        elif x is None and y == 100:
            self.bottom()
        else:
            for (val, orientation) in [(x, Qt.Orientation.Horizontal), (y, Qt.Orientation.Vertical)]:
                if val is not None:
                    frame = self._widget.page().mainFrame()
                    maximum = frame.scrollBarMaximum(orientation)
                    if maximum == 0:
                        continue
                    pos = int(maximum * val / 100)
                    pos = qtutils.check_overflow(pos, 'int', fatal=False)
                    frame.setScrollBarValue(orientation, pos)

    def _key_press(self, key, count=1, getter_name=None, direction=None):
        if False:
            while True:
                i = 10
        frame = self._widget.page().mainFrame()
        getter = None if getter_name is None else getattr(frame, getter_name)
        for _ in range(min(count, 5000)):
            if getter is not None and frame.scrollBarValue(direction) == getter(direction):
                return
            self._tab.fake_key_press(key)

    def up(self, count=1):
        if False:
            print('Hello World!')
        self._key_press(Qt.Key.Key_Up, count, 'scrollBarMinimum', Qt.Orientation.Vertical)

    def down(self, count=1):
        if False:
            i = 10
            return i + 15
        self._key_press(Qt.Key.Key_Down, count, 'scrollBarMaximum', Qt.Orientation.Vertical)

    def left(self, count=1):
        if False:
            for i in range(10):
                print('nop')
        self._key_press(Qt.Key.Key_Left, count, 'scrollBarMinimum', Qt.Orientation.Horizontal)

    def right(self, count=1):
        if False:
            print('Hello World!')
        self._key_press(Qt.Key.Key_Right, count, 'scrollBarMaximum', Qt.Orientation.Horizontal)

    def top(self):
        if False:
            for i in range(10):
                print('nop')
        self._key_press(Qt.Key.Key_Home)

    def bottom(self):
        if False:
            while True:
                i = 10
        self._key_press(Qt.Key.Key_End)

    def page_up(self, count=1):
        if False:
            i = 10
            return i + 15
        self._key_press(Qt.Key.Key_PageUp, count, 'scrollBarMinimum', Qt.Orientation.Vertical)

    def page_down(self, count=1):
        if False:
            return 10
        self._key_press(Qt.Key.Key_PageDown, count, 'scrollBarMaximum', Qt.Orientation.Vertical)

    def at_top(self):
        if False:
            print('Hello World!')
        return self.pos_px().y() == 0

    def at_bottom(self):
        if False:
            while True:
                i = 10
        frame = self._widget.page().currentFrame()
        return self.pos_px().y() >= frame.scrollBarMaximum(Qt.Orientation.Vertical)

class WebKitHistoryPrivate(browsertab.AbstractHistoryPrivate):
    """History-related methods which are not part of the extension API."""
    _history: QWebHistory

    def __init__(self, tab: 'WebKitTab') -> None:
        if False:
            for i in range(10):
                print('nop')
        self._tab = tab
        self._history = cast(QWebHistory, None)

    def serialize(self):
        if False:
            return 10
        return qtutils.serialize(self._history)

    def deserialize(self, data):
        if False:
            print('Hello World!')
        qtutils.deserialize(data, self._history)

    def load_items(self, items):
        if False:
            for i in range(10):
                print('nop')
        if items:
            self._tab.before_load_started.emit(items[-1].url)
        (stream, _data, user_data) = tabhistory.serialize(items)
        qtutils.deserialize_stream(stream, self._history)
        for (i, data) in enumerate(user_data):
            self._history.itemAt(i).setUserData(data)
        cur_data = self._history.currentItem().userData()
        if cur_data is not None:
            if 'zoom' in cur_data:
                self._tab.zoom.set_factor(cur_data['zoom'])
            if 'scroll-pos' in cur_data and self._tab.scroller.pos_px() == QPoint(0, 0):
                QTimer.singleShot(0, functools.partial(self._tab.scroller.to_point, cur_data['scroll-pos']))

class WebKitHistory(browsertab.AbstractHistory):
    """QtWebKit implementations related to page history."""

    def __init__(self, tab):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(tab)
        self.private_api = WebKitHistoryPrivate(tab)

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._history)

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self._history.items())

    def current_idx(self):
        if False:
            return 10
        return self._history.currentItemIndex()

    def current_item(self):
        if False:
            print('Hello World!')
        return self._history.currentItem()

    def can_go_back(self):
        if False:
            return 10
        return self._history.canGoBack()

    def can_go_forward(self):
        if False:
            while True:
                i = 10
        return self._history.canGoForward()

    def _item_at(self, i):
        if False:
            for i in range(10):
                print('nop')
        return self._history.itemAt(i)

    def _go_to_item(self, item):
        if False:
            i = 10
            return i + 15
        self._tab.before_load_started.emit(item.url())
        self._history.goToItem(item)

    def back_items(self):
        if False:
            for i in range(10):
                print('nop')
        return self._history.backItems(self._history.count())

    def forward_items(self):
        if False:
            while True:
                i = 10
        return self._history.forwardItems(self._history.count())

class WebKitElements(browsertab.AbstractElements):
    """QtWebKit implementations related to elements on the page."""
    _tab: 'WebKitTab'
    _widget: webview.WebView

    def find_css(self, selector, callback, error_cb, *, only_visible=False):
        if False:
            while True:
                i = 10
        utils.unused(error_cb)
        mainframe = self._widget.page().mainFrame()
        if mainframe is None:
            raise browsertab.WebTabError('No frame focused!')
        elems = []
        frames = webkitelem.get_child_frames(mainframe)
        for f in frames:
            frame_elems = cast(Iterable[QWebElement], f.findAllElements(selector))
            for elem in frame_elems:
                elems.append(webkitelem.WebKitElement(elem, tab=self._tab))
        if only_visible:
            elems = [e for e in elems if e._is_visible(mainframe)]
        callback(elems)

    def find_id(self, elem_id, callback):
        if False:
            i = 10
            return i + 15

        def find_id_cb(elems):
            if False:
                i = 10
                return i + 15
            'Call the real callback with the found elements.'
            if not elems:
                callback(None)
            else:
                callback(elems[0])
        elem_id = re.sub('[^a-zA-Z0-9_-]', '\\\\\\g<0>', elem_id)
        self.find_css('#' + elem_id, find_id_cb, error_cb=lambda exc: None)

    def find_focused(self, callback):
        if False:
            return 10
        frame = cast(Optional[QWebFrame], self._widget.page().currentFrame())
        if frame is None:
            callback(None)
            return
        elem = frame.findFirstElement('*:focus')
        if elem.isNull():
            callback(None)
        else:
            callback(webkitelem.WebKitElement(elem, tab=self._tab))

    def find_at_pos(self, pos, callback):
        if False:
            print('Hello World!')
        assert pos.x() >= 0
        assert pos.y() >= 0
        frame = cast(Optional[QWebFrame], self._widget.page().frameAt(pos))
        if frame is None:
            log.webview.debug('Hit test at {} but frame is None!'.format(pos))
            callback(None)
            return
        hitresult = frame.hitTestContent(pos)
        if hitresult.isNull():
            log.webview.debug('Hit test result is null!')
            callback(None)
            return
        try:
            elem = webkitelem.WebKitElement(hitresult.element(), tab=self._tab)
        except webkitelem.IsNullError:
            log.webview.debug('Hit test result element is null!')
            callback(None)
            return
        callback(elem)

class WebKitAudio(browsertab.AbstractAudio):
    """Dummy handling of audio status for QtWebKit."""

    def set_muted(self, muted: bool, override: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        raise browsertab.WebTabError('Muting is not supported on QtWebKit!')

    def is_muted(self):
        if False:
            print('Hello World!')
        return False

    def is_recently_audible(self):
        if False:
            print('Hello World!')
        return False

class WebKitTabPrivate(browsertab.AbstractTabPrivate):
    """QtWebKit-related methods which aren't part of the public API."""
    _widget: webview.WebView

    def networkaccessmanager(self):
        if False:
            i = 10
            return i + 15
        return self._widget.page().networkAccessManager()

    def clear_ssl_errors(self):
        if False:
            for i in range(10):
                print('nop')
        self.networkaccessmanager().clear_all_ssl_errors()

    def event_target(self):
        if False:
            print('Hello World!')
        return self._widget

    def shutdown(self):
        if False:
            i = 10
            return i + 15
        self._widget.shutdown()

    def run_js_sync(self, code):
        if False:
            print('Hello World!')
        document_element = self._widget.page().mainFrame().documentElement()
        result = document_element.evaluateJavaScript(code)
        return result

    def _init_inspector(self, splitter, win_id, parent=None):
        if False:
            while True:
                i = 10
        return webkitinspector.WebKitInspector(splitter, win_id, parent)

class WebKitTab(browsertab.AbstractTab):
    """A QtWebKit tab in the browser."""
    _widget: webview.WebView

    def __init__(self, *, win_id, mode_manager, private, parent=None):
        if False:
            print('Hello World!')
        super().__init__(win_id=win_id, mode_manager=mode_manager, private=private, parent=parent)
        widget = webview.WebView(win_id=win_id, tab_id=self.tab_id, private=private, tab=self)
        if private:
            self._make_private(widget)
        self.history = WebKitHistory(tab=self)
        self.scroller = WebKitScroller(tab=self, parent=self)
        self.caret = WebKitCaret(mode_manager=mode_manager, tab=self, parent=self)
        self.zoom = WebKitZoom(tab=self, parent=self)
        self.search = WebKitSearch(tab=self, parent=self)
        self.printing = WebKitPrinting(tab=self, parent=self)
        self.elements = WebKitElements(tab=self)
        self.action = WebKitAction(tab=self)
        self.audio = WebKitAudio(tab=self, parent=self)
        self.private_api = WebKitTabPrivate(mode_manager=mode_manager, tab=self)
        self.settings = webkitsettings.WebKitSettings(settings=None)
        self._set_widget(widget)
        self._connect_signals()
        self.backend = usertypes.Backend.QtWebKit

    def _install_event_filter(self):
        if False:
            print('Hello World!')
        self._widget.installEventFilter(self._tab_event_filter)

    def _make_private(self, widget):
        if False:
            print('Hello World!')
        settings = widget.settings()
        settings.setAttribute(QWebSettings.WebAttribute.PrivateBrowsingEnabled, True)

    def load_url(self, url):
        if False:
            print('Hello World!')
        self._load_url_prepare(url)
        self._widget.load(url)

    def url(self, *, requested=False):
        if False:
            print('Hello World!')
        frame = self._widget.page().mainFrame()
        if requested:
            return frame.requestedUrl()
        else:
            return frame.url()

    def dump_async(self, callback, *, plain=False):
        if False:
            i = 10
            return i + 15
        frame = self._widget.page().mainFrame()
        if plain:
            callback(frame.toPlainText())
        else:
            callback(frame.toHtml())

    def run_js_async(self, code, callback=None, *, world=None):
        if False:
            while True:
                i = 10
        if world is not None and world != usertypes.JsWorld.jseval:
            log.webview.warning('Ignoring world ID {}'.format(world))
        result = self.private_api.run_js_sync(code)
        if callback is not None:
            callback(result)

    def icon(self):
        if False:
            for i in range(10):
                print('nop')
        return self._widget.icon()

    def reload(self, *, force=False):
        if False:
            while True:
                i = 10
        if force:
            action = QWebPage.WebAction.ReloadAndBypassCache
        else:
            action = QWebPage.WebAction.Reload
        self._widget.triggerPageAction(action)

    def stop(self):
        if False:
            return 10
        self._widget.stop()

    def title(self):
        if False:
            for i in range(10):
                print('nop')
        return self._widget.title()

    def renderer_process_pid(self) -> Optional[int]:
        if False:
            for i in range(10):
                print('nop')
        return None

    @pyqtSlot()
    def _on_history_trigger(self):
        if False:
            while True:
                i = 10
        url = self.url()
        requested_url = self.url(requested=True)
        self.history_item_triggered.emit(url, requested_url, self.title())

    def set_html(self, html, base_url=QUrl()):
        if False:
            i = 10
            return i + 15
        self._widget.setHtml(html, base_url)

    @pyqtSlot()
    def _on_load_started(self):
        if False:
            i = 10
            return i + 15
        super()._on_load_started()
        nam = self._widget.page().networkAccessManager()
        assert isinstance(nam, networkmanager.NetworkManager), nam
        nam.netrc_used = False
        self.icon_changed.emit(QIcon())

    @pyqtSlot(bool)
    def _on_load_finished(self, ok: bool) -> None:
        if False:
            i = 10
            return i + 15
        super()._on_load_finished(ok)
        self._update_load_status(ok)

    @pyqtSlot()
    def _on_frame_load_finished(self):
        if False:
            while True:
                i = 10
        'Make sure we emit an appropriate status when loading finished.\n\n        While Qt has a bool "ok" attribute for loadFinished, it always is True\n        when using error pages... See\n        https://github.com/qutebrowser/qutebrowser/issues/84\n        '
        page = self._widget.page()
        assert isinstance(page, webpage.BrowserPage), page
        self._on_load_finished(not page.error_occurred)

    @pyqtSlot()
    def _on_webkit_icon_changed(self):
        if False:
            print('Hello World!')
        'Emit iconChanged with a QIcon like QWebEngineView does.'
        if sip.isdeleted(self._widget):
            log.webview.debug('Got _on_webkit_icon_changed for deleted view!')
            return
        self.icon_changed.emit(self._widget.icon())

    @pyqtSlot(QWebFrame)
    def _on_frame_created(self, frame):
        if False:
            while True:
                i = 10
        'Connect the contentsSizeChanged signal of each frame.'
        frame.contentsSizeChanged.connect(self._on_contents_size_changed)

    @pyqtSlot(QSize)
    def _on_contents_size_changed(self, size):
        if False:
            print('Hello World!')
        self.contents_size_changed.emit(QSizeF(size))

    @pyqtSlot(usertypes.NavigationRequest)
    def _on_navigation_request(self, navigation):
        if False:
            print('Hello World!')
        super()._on_navigation_request(navigation)
        if not navigation.accepted:
            return
        log.webview.debug('target {} override {}'.format(self.data.open_target, self.data.override_target))
        if self.data.override_target is not None:
            target = self.data.override_target
            self.data.override_target = None
        else:
            target = self.data.open_target
        if navigation.navigation_type == navigation.Type.link_clicked and target != usertypes.ClickTarget.normal:
            tab = shared.get_tab(self.win_id, target)
            tab.load_url(navigation.url)
            self.data.open_target = usertypes.ClickTarget.normal
            navigation.accepted = False
        if navigation.is_main_frame:
            self.settings.update_for_url(navigation.url)

    @pyqtSlot('QNetworkReply*')
    def _on_ssl_errors(self, reply):
        if False:
            return 10
        self._insecure_hosts.add(reply.url().host())

    def _connect_signals(self):
        if False:
            print('Hello World!')
        view = self._widget
        page = view.page()
        frame = page.mainFrame()
        page.windowCloseRequested.connect(self.window_close_requested)
        page.linkHovered.connect(self.link_hovered)
        page.loadProgress.connect(self._on_load_progress)
        frame.loadStarted.connect(self._on_load_started)
        view.scroll_pos_changed.connect(self.scroller.perc_changed)
        view.titleChanged.connect(self.title_changed)
        view.urlChanged.connect(self._on_url_changed)
        view.shutting_down.connect(self.shutting_down)
        page.networkAccessManager().sslErrors.connect(self._on_ssl_errors)
        frame.loadFinished.connect(self._on_frame_load_finished)
        view.iconChanged.connect(self._on_webkit_icon_changed)
        page.frameCreated.connect(self._on_frame_created)
        frame.contentsSizeChanged.connect(self._on_contents_size_changed)
        frame.initialLayoutCompleted.connect(self._on_history_trigger)
        page.navigation_request.connect(self._on_navigation_request)