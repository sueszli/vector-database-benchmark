"""
Frames browser widget

This is the main widget used in the Debugger plugin
"""
import os.path as osp
import html
from qtpy.QtCore import Signal
from qtpy.QtGui import QAbstractTextDocumentLayout, QTextDocument
from qtpy.QtCore import QSize, Qt, Slot
from qtpy.QtWidgets import QApplication, QStyle, QStyledItemDelegate, QStyleOptionViewItem, QTreeWidgetItem, QVBoxLayout, QWidget, QTreeWidget, QStackedLayout
from spyder.api.config.decorators import on_conf_change
from spyder.api.config.fonts import SpyderFontsMixin, SpyderFontType
from spyder.api.config.mixins import SpyderConfigurationAccessor
from spyder.api.widgets.mixins import SpyderWidgetMixin
from spyder.api.translations import _
from spyder.widgets.helperwidgets import FinderWidget, PaneEmptyWidget

class FramesBrowserState:
    Debug = 'debug'
    DebugWait = 'debugwait'
    Inspect = 'inspect'
    Error = 'error'

class FramesBrowser(QWidget, SpyderWidgetMixin):
    """Frames browser (global debugger widget)"""
    CONF_SECTION = 'debugger'
    sig_edit_goto = Signal(str, int, str)
    '\n    This signal will request to open a file in a given row and column\n    using a code editor.\n\n    Parameters\n    ----------\n    path: str\n        Path to file.\n    row: int\n        Cursor starting row position.\n    word: str\n        Word to select on given row.\n    '
    sig_show_namespace = Signal(dict, object)
    '\n    Show the namespace\n\n    Parameters\n    ----------\n    namespace: dict\n        A namespace view created by spyder_kernels\n    shellwidget: ShellWidget\n        The shellwidget the request originated from\n    '
    sig_update_actions_requested = Signal()
    'Update the widget actions.'
    sig_hide_finder_requested = Signal()
    'Hide the finder widget.'
    sig_load_pdb_file = Signal(str, int)
    '\n    This signal is emitted when Pdb reaches a new line.\n\n    Parameters\n    ----------\n    filename: str\n        The filename the debugger stepped in\n    line_number: int\n        The line number the debugger stepped in\n    '

    def __init__(self, parent, shellwidget, color_scheme):
        if False:
            return 10
        super().__init__(parent)
        self.shellwidget = shellwidget
        self.results_browser = None
        self.color_scheme = color_scheme
        self._persistence = -1
        self.state = None
        self.finder = None
        self.pdb_curindex = None
        self._pdb_state = []

    def pdb_has_stopped(self, fname, lineno):
        if False:
            print('Hello World!')
        'Handle pdb has stopped'
        self.sig_load_pdb_file.emit(fname, lineno)
        if not self.shellwidget._pdb_take_focus:
            self.shellwidget._pdb_take_focus = True
        else:
            self.shellwidget._control.setFocus()

    def set_context_menu(self, context_menu, empty_context_menu):
        if False:
            for i in range(10):
                print('nop')
        'Set the context menus.'
        self.results_browser.menu = context_menu
        self.results_browser.empty_ws_menu = empty_context_menu

    def toggle_finder(self, show):
        if False:
            while True:
                i = 10
        'Show and hide the finder.'
        self.finder.set_visible(show)
        if not show:
            self.results_browser.setFocus()

    def do_find(self, text):
        if False:
            while True:
                i = 10
        'Search for text.'
        if self.results_browser is not None:
            self.results_browser.do_find(text)

    def finder_is_visible(self):
        if False:
            while True:
                i = 10
        'Check if the finder is visible.'
        if self.finder is None:
            return False
        return self.finder.isVisible()

    def set_pane_empty(self, empty):
        if False:
            i = 10
            return i + 15
        if empty:
            self.stack_layout.setCurrentWidget(self.pane_empty)
        else:
            self.stack_layout.setCurrentWidget(self.container)

    def setup(self):
        if False:
            print('Hello World!')
        '\n        Setup the frames browser with provided settings.\n        '
        if self.results_browser is not None:
            return
        self.results_browser = ResultsBrowser(self, self.color_scheme)
        self.results_browser.sig_edit_goto.connect(self.sig_edit_goto)
        self.results_browser.sig_show_namespace.connect(self._show_namespace)
        self.finder = FinderWidget(self)
        self.finder.sig_find_text.connect(self.do_find)
        self.finder.sig_hide_finder_requested.connect(self.sig_hide_finder_requested)
        self.pane_empty = PaneEmptyWidget(self, 'debugger', _('Debugging is not active'), _('Start a debugging session with the ⏯ button, allowing you to step through your code and see the functions here that Python has run.'))
        self.stack_layout = QStackedLayout()
        self.stack_layout.addWidget(self.pane_empty)
        self.setLayout(self.stack_layout)
        self.stack_layout.setContentsMargins(0, 0, 0, 0)
        self.stack_layout.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.results_browser)
        layout.addWidget(self.finder)
        self.container = QWidget(self)
        self.container.setLayout(layout)
        self.stack_layout.addWidget(self.container)

    def _show_namespace(self, namespace):
        if False:
            return 10
        '\n        Request for the given namespace to be shown in the Variable Explorer.\n        '
        self.sig_show_namespace.emit(namespace, self.shellwidget)

    def _show_frames(self, frames, title, state):
        if False:
            for i in range(10):
                print('nop')
        'Set current frames'
        self._persistence = -1
        self.state = state
        self.pdb_curindex = None
        if self.results_browser is not None:
            if frames is not None:
                self.set_pane_empty(False)
            else:
                self.set_pane_empty(True)
            self.results_browser.set_frames(frames)
            self.results_browser.set_title(title)
            try:
                self.results_browser.sig_activated.disconnect(self.set_pdb_index)
            except (TypeError, RuntimeError):
                pass

    def set_pdb_index(self, index):
        if False:
            print('Hello World!')
        'Set pdb index'
        if self.pdb_curindex is None:
            return
        delta_index = self.pdb_curindex - index
        if delta_index > 0:
            command = 'up ' + str(delta_index)
        elif delta_index < 0:
            command = 'down ' + str(-delta_index)
        else:
            return
        self.shellwidget.pdb_execute_command(command)

    def set_from_pdb(self, pdb_stack, curindex):
        if False:
            for i in range(10):
                print('nop')
        'Set frames from pdb stack'
        depth = self.shellwidget.debugging_depth()
        self._pdb_state = self._pdb_state[:depth - 1]
        while len(self._pdb_state) < depth - 1:
            self._pdb_state.append(None)
        self._pdb_state.append((pdb_stack, curindex))

    def show_pdb(self, pdb_stack, curindex):
        if False:
            for i in range(10):
                print('nop')
        'Show pdb frames.'
        self._show_frames({'pdb': pdb_stack}, _('Pdb stack'), FramesBrowserState.Debug)
        self._persistence = 0
        self.pdb_curindex = curindex
        self.set_current_item(0, curindex)
        self.results_browser.sig_activated.connect(self.set_pdb_index)
        self.sig_update_actions_requested.emit()

    def show_exception(self, etype, error, tb):
        if False:
            while True:
                i = 10
        'Set frames from exception'
        self._show_frames({etype.__name__: tb}, _('Exception occured'), FramesBrowserState.Error)
        self.sig_update_actions_requested.emit()

    def show_captured_frames(self, frames):
        if False:
            for i in range(10):
                print('nop')
        'Set from captured frames'
        self._show_frames(frames, _('Snapshot of frames'), FramesBrowserState.Inspect)
        self.sig_update_actions_requested.emit()

    def show_pdb_preview(self, frames):
        if False:
            for i in range(10):
                print('nop')
        'Set from captured frames'
        if 'MainThread' in frames:
            frames = {_('Waiting for debugger'): frames['MainThread']}
        self._show_frames(frames, _('Waiting for debugger'), FramesBrowserState.DebugWait)
        self._persistence = 0
        self.sig_update_actions_requested.emit()

    def clear_if_needed(self):
        if False:
            while True:
                i = 10
        'Execution finished. Clear if it is relevant.'
        if self.shellwidget.is_debugging():
            depth = self.shellwidget.debugging_depth()
            if len(self._pdb_state) > depth - 1:
                pdb_state = self._pdb_state[depth - 1]
                if pdb_state:
                    self.show_pdb(*pdb_state)
                    self._persistence = 0
                    return
        if self._persistence == 0:
            self._show_frames(None, '', None)
            self.sig_update_actions_requested.emit()
        elif self._persistence > 0:
            self._persistence -= 1

    def set_current_item(self, top_idx, sub_index):
        if False:
            print('Hello World!')
        'Set current item'
        if self.results_browser is not None:
            self.results_browser.set_current_item(top_idx, sub_index)

    def on_config_kernel(self):
        if False:
            return 10
        'Ask shellwidget to send Pdb configuration to kernel.'
        self.shellwidget.set_kernel_configuration('pdb', {'breakpoints': self.get_conf('breakpoints', default={}), 'pdb_ignore_lib': self.get_conf('pdb_ignore_lib'), 'pdb_execute_events': self.get_conf('pdb_execute_events'), 'pdb_use_exclamation_mark': self.get_conf('pdb_use_exclamation_mark'), 'pdb_stop_first_line': self.get_conf('pdb_stop_first_line'), 'pdb_publish_stack': True})

    def on_unconfig_kernel(self):
        if False:
            i = 10
            return i + 15
        'Ask shellwidget to stop sending stack.'
        if not self.shellwidget.spyder_kernel_ready:
            return
        self.shellwidget.set_kernel_configuration('pdb', {'pdb_publish_stack': False})

    @on_conf_change(option='pdb_ignore_lib')
    def change_pdb_ignore_lib(self, value):
        if False:
            print('Hello World!')
        self.shellwidget.set_kernel_configuration('pdb', {'pdb_ignore_lib': value})

    @on_conf_change(option='pdb_execute_events')
    def change_pdb_execute_events(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.shellwidget.set_kernel_configuration('pdb', {'pdb_execute_events': value})

    @on_conf_change(option='pdb_use_exclamation_mark')
    def change_pdb_use_exclamation_mark(self, value):
        if False:
            while True:
                i = 10
        self.shellwidget.set_kernel_configuration('pdb', {'pdb_use_exclamation_mark': value})

    @on_conf_change(option='pdb_stop_first_line')
    def change_pdb_stop_first_line(self, value):
        if False:
            return 10
        self.shellwidget.set_kernel_configuration('pdb', {'pdb_stop_first_line': value})

    def set_breakpoints(self):
        if False:
            for i in range(10):
                print('nop')
        'Set current breakpoints.'
        self.shellwidget.set_kernel_configuration('pdb', {'breakpoints': self.get_conf('breakpoints', default={}, section='debugger')})

class LineFrameItem(QTreeWidgetItem):

    def __init__(self, parent, index, filename, line, lineno, name, f_locals, font, color_scheme=None):
        if False:
            i = 10
            return i + 15
        self.index = index
        self.filename = filename
        self.text = line
        self.lineno = lineno
        self.context = name
        self.color_scheme = color_scheme
        self.font = font
        self.locals = f_locals
        QTreeWidgetItem.__init__(self, parent, [self.__repr__()], QTreeWidgetItem.Type)

    def __repr__(self):
        if False:
            while True:
                i = 10
        'Prints item as html.'
        if self.filename is None:
            return '<!-- LineFrameItem --><p><span style="color:{0}">idle</span></p>'.format(self.color_scheme['normal'][0])
        _str = '<!-- LineFrameItem -->' + '<p style="color:\'{0}\';"><b> '.format(self.color_scheme['normal'][0]) + '<span style="color:\'{0}\';">{1}</span>:'.format(self.color_scheme['string'][0], html.escape(osp.basename(self.filename))) + '<span style="color:\'{0}\';">{1}</span></b>'.format(self.color_scheme['number'][0], self.lineno)
        if self.context:
            _str += ' (<span style="color:\'{0}\';">{1}</span>)'.format(self.color_scheme['builtin'][0], html.escape(self.context))
        _str += '    <span style="font-family:{0};'.format(self.font.family()) + 'color:\'{0}\';font-size:50%;"><em>{1}</em></span></p>'.format(self.color_scheme['comment'][0], self.text)
        return _str

    def to_plain_text(self):
        if False:
            i = 10
            return i + 15
        'Represent item as plain text.'
        if self.filename is None:
            return 'idle'
        _str = html.escape(osp.basename(self.filename)) + ':' + str(self.lineno)
        if self.context:
            _str += ' ({})'.format(html.escape(self.context))
        _str += ' {}'.format(self.text)
        return _str

    def __str__(self):
        if False:
            i = 10
            return i + 15
        'String representation.'
        return self.__repr__()

    def __lt__(self, x):
        if False:
            for i in range(10):
                print('nop')
        'Smaller for sorting.'
        return self.index < x.index

    def __ge__(self, x):
        if False:
            while True:
                i = 10
        'Larger or equals for sorting.'
        return self.index >= x.index

class ThreadItem(QTreeWidgetItem):

    def __init__(self, parent, name, text_color):
        if False:
            i = 10
            return i + 15
        self.name = str(name)
        title_format = str('<!-- ThreadItem --><b style="color:{1}">{0}</b>')
        title = title_format.format(name, text_color)
        QTreeWidgetItem.__init__(self, parent, [title], QTreeWidgetItem.Type)
        self.setToolTip(0, self.name)

    def __lt__(self, x):
        if False:
            while True:
                i = 10
        'Smaller for sorting.'
        return self.name < x.name

    def __ge__(self, x):
        if False:
            for i in range(10):
                print('nop')
        'Larger or equals for sorting.'
        return self.name >= x.name

class ItemDelegate(QStyledItemDelegate):

    def __init__(self, parent):
        if False:
            for i in range(10):
                print('nop')
        QStyledItemDelegate.__init__(self, parent)
        self._margin = None

    def paint(self, painter, option, index):
        if False:
            print('Hello World!')
        'Paint the item.'
        options = QStyleOptionViewItem(option)
        self.initStyleOption(options, index)
        style = QApplication.style() if options.widget is None else options.widget.style()
        doc = QTextDocument()
        text = options.text
        doc.setHtml(text)
        doc.setDocumentMargin(0)
        options.text = ''
        style.drawControl(QStyle.CE_ItemViewItem, options, painter)
        ctx = QAbstractTextDocumentLayout.PaintContext()
        textRect = style.subElementRect(QStyle.SE_ItemViewItemText, options, None)
        painter.save()
        painter.translate(textRect.topLeft())
        painter.setClipRect(textRect.translated(-textRect.topLeft()))
        doc.documentLayout().draw(painter, ctx)
        painter.restore()

    def sizeHint(self, option, index):
        if False:
            while True:
                i = 10
        'Get a size hint.'
        options = QStyleOptionViewItem(option)
        self.initStyleOption(options, index)
        doc = QTextDocument()
        doc.setHtml(options.text)
        doc.setTextWidth(options.rect.width())
        size = QSize(int(doc.idealWidth()), int(doc.size().height()))
        return size

class ResultsBrowser(QTreeWidget, SpyderConfigurationAccessor, SpyderFontsMixin):
    CONF_SECTION = 'debugger'
    sig_edit_goto = Signal(str, int, str)
    sig_activated = Signal(int)
    sig_show_namespace = Signal(dict)

    def __init__(self, parent, color_scheme):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.font = self.get_font(SpyderFontType.MonospaceInterface)
        self.data = None
        self.threads = None
        self.color_scheme = color_scheme
        self.text_color = color_scheme['normal'][0]
        self.frames = None
        self.menu = None
        self.empty_ws_menu = None
        self.view_locals_action = None
        self.setItemsExpandable(True)
        self.setColumnCount(1)
        self.set_title('')
        self.setSortingEnabled(False)
        self.setItemDelegate(ItemDelegate(self))
        self.setUniformRowHeights(True)
        self.sortByColumn(0, Qt.AscendingOrder)
        self.header().sectionClicked.connect(self.sort_section)
        self.itemActivated.connect(self.activated)
        self.itemClicked.connect(self.activated)

    def set_title(self, title):
        if False:
            print('Hello World!')
        self.setHeaderLabels([title])

    def activated(self, item):
        if False:
            i = 10
            return i + 15
        'Double-click event.'
        itemdata = self.data.get(id(self.currentItem()))
        if itemdata is not None:
            (filename, lineno) = itemdata
            self.sig_edit_goto.emit(filename, lineno, '')
            self.sig_activated.emit(self.currentItem().index)

    def view_item_locals(self):
        if False:
            while True:
                i = 10
        'View item locals.'
        item = self.currentItem()
        item_has_locals = isinstance(item, LineFrameItem) and item.locals is not None
        if item_has_locals:
            self.sig_show_namespace.emit(item.locals)

    def contextMenuEvent(self, event):
        if False:
            print('Hello World!')
        'Reimplement Qt method'
        if self.menu is None:
            return
        if self.frames:
            self.refresh_menu()
            self.menu.popup(event.globalPos())
            event.accept()
        else:
            self.empty_ws_menu.popup(event.globalPos())
            event.accept()

    def refresh_menu(self):
        if False:
            for i in range(10):
                print('nop')
        'Refresh context menu'
        item = self.currentItem()
        item_has_locals = isinstance(item, LineFrameItem) and item.locals is not None
        self.view_locals_action.setEnabled(item_has_locals)

    @Slot(int)
    def sort_section(self, idx):
        if False:
            for i in range(10):
                print('nop')
        'Sort section'
        self.setSortingEnabled(True)

    def set_current_item(self, top_idx, sub_index):
        if False:
            return 10
        'Set current item.'
        item = self.topLevelItem(top_idx).child(sub_index)
        self.setCurrentItem(item)

    def set_frames(self, frames):
        if False:
            while True:
                i = 10
        'Set frames.'
        self.clear()
        self.threads = {}
        self.data = {}
        self.frames = frames
        if frames is None:
            return
        for (thread_id, stack) in frames.items():
            parent = ThreadItem(self, thread_id, self.text_color)
            parent.setExpanded(True)
            self.threads[thread_id] = parent
            if stack:
                for (idx, frame) in enumerate(stack):
                    item = LineFrameItem(parent, idx, frame.filename, frame.line, frame.lineno, frame.name, frame.locals, self.font, self.color_scheme)
                    self.data[id(item)] = (frame.filename, frame.lineno)
            else:
                item = LineFrameItem(parent, 0, None, '', 0, '', None, self.font, self.color_scheme)

    def do_find(self, text):
        if False:
            print('Hello World!')
        'Update the regex text for the variable finder.'
        for idx in range(self.topLevelItemCount()):
            item = self.topLevelItem(idx)
            all_hidden = True
            for child_idx in range(item.childCount()):
                line_frame = item.child(child_idx)
                if text:
                    match_text = line_frame.to_plain_text().replace(' ', '').lower()
                    if match_text.find(text) == -1:
                        line_frame.setHidden(True)
                    else:
                        line_frame.setHidden(False)
                        all_hidden = False
                else:
                    line_frame.setHidden(False)
                    all_hidden = False
            item.setHidden(all_hidden)