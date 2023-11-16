"""EditorStack Widget"""
import logging
import os
import os.path as osp
import sys
import unicodedata
import qstylizer.style
from qtpy.compat import getsavefilename
from qtpy.QtCore import QFileInfo, Qt, QTimer, Signal, Slot
from qtpy.QtGui import QTextCursor
from qtpy.QtWidgets import QApplication, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QMenu, QVBoxLayout, QWidget, QSizePolicy, QToolBar
from spyder.api.config.mixins import SpyderConfigurationAccessor
from spyder.config.base import _, running_under_pytest
from spyder.config.gui import is_dark_interface
from spyder.config.utils import get_edit_filetypes, get_edit_filters, get_filter, is_kde_desktop, is_anaconda
from spyder.plugins.editor.api.panel import Panel
from spyder.plugins.editor.utils.autosave import AutosaveForStack
from spyder.plugins.editor.utils.editor import get_file_language
from spyder.plugins.editor.widgets import codeeditor
from spyder.plugins.editor.widgets.editorstack.helpers import ThreadManager, FileInfo, StackHistory
from spyder.plugins.editor.widgets.tabswitcher import TabSwitcherWidget
from spyder.plugins.explorer.widgets.explorer import show_in_external_file_explorer
from spyder.plugins.explorer.widgets.utils import fixpath
from spyder.plugins.outlineexplorer.editor import OutlineExplorerProxyEditor
from spyder.plugins.outlineexplorer.api import cell_name
from spyder.py3compat import to_text_string
from spyder.utils import encoding, sourcecode, syntaxhighlighters
from spyder.utils.icon_manager import ima
from spyder.utils.misc import getcwd_or_home
from spyder.utils.palette import QStylePalette
from spyder.utils.qthelpers import add_actions, create_action, create_toolbutton, MENU_SEPARATOR, mimedata2url, set_menu_icons, create_waitspinner
from spyder.utils.stylesheet import PANES_TABBAR_STYLESHEET
from spyder.widgets.tabs import BaseTabs
logger = logging.getLogger(__name__)

class EditorStack(QWidget, SpyderConfigurationAccessor):
    reset_statusbar = Signal()
    readonly_changed = Signal(bool)
    encoding_changed = Signal(str)
    sig_editor_cursor_position_changed = Signal(int, int)
    sig_refresh_eol_chars = Signal(str)
    sig_refresh_formatting = Signal(bool)
    starting_long_process = Signal(str)
    ending_long_process = Signal(str)
    redirect_stdio = Signal(bool)
    update_plugin_title = Signal()
    editor_focus_changed = Signal()
    zoom_in = Signal()
    zoom_out = Signal()
    zoom_reset = Signal()
    sig_open_file = Signal(dict)
    sig_close_file = Signal(str, str)
    file_saved = Signal(str, str, str)
    file_renamed_in_data = Signal(str, str, str)
    opened_files_list_changed = Signal()
    active_languages_stats = Signal(set)
    todo_results_changed = Signal()
    sig_update_code_analysis_actions = Signal()
    refresh_file_dependent_actions = Signal()
    refresh_save_all_action = Signal()
    text_changed_at = Signal(str, int)
    current_file_changed = Signal(str, int, int, int)
    plugin_load = Signal((str,), ())
    edit_goto = Signal(str, int, str)
    sig_split_vertically = Signal()
    sig_split_horizontally = Signal()
    sig_new_file = Signal((str,), ())
    sig_save_as = Signal()
    sig_prev_edit_pos = Signal()
    sig_prev_cursor = Signal()
    sig_next_cursor = Signal()
    sig_prev_warning = Signal()
    sig_next_warning = Signal()
    sig_go_to_definition = Signal(str, int, int)
    sig_perform_completion_request = Signal(str, str, dict)
    sig_option_changed = Signal(str, object)
    sig_save_bookmark = Signal(int)
    sig_load_bookmark = Signal(int)
    sig_save_bookmarks = Signal(str, str)
    sig_codeeditor_created = Signal(object)
    '\n    This signal is emitted when a codeeditor is created.\n\n    Parameters\n    ----------\n    codeeditor: spyder.plugins.editor.widgets.codeeditor.CodeEditor\n        The codeeditor.\n    '
    sig_codeeditor_deleted = Signal(object)
    '\n    This signal is emitted when a codeeditor is closed.\n\n    Parameters\n    ----------\n    codeeditor: spyder.plugins.editor.widgets.codeeditor.CodeEditor\n        The codeeditor.\n    '
    sig_codeeditor_changed = Signal(object)
    '\n    This signal is emitted when the current codeeditor changes.\n\n    Parameters\n    ----------\n    codeeditor: spyder.plugins.editor.widgets.codeeditor.CodeEditor\n        The codeeditor.\n    '
    sig_help_requested = Signal(dict)
    "\n    This signal is emitted to request help on a given object `name`.\n\n    Parameters\n    ----------\n    help_data: dict\n        Dictionary required by the Help pane to render a docstring.\n\n    Examples\n    --------\n    >>> help_data = {\n        'obj_text': str,\n        'name': str,\n        'argspec': str,\n        'note': str,\n        'docstring': str,\n        'force_refresh': bool,\n        'path': str,\n    }\n\n    See Also\n    --------\n    :py:meth:spyder.plugins.editor.widgets.editorstack.EditorStack.send_to_help\n    "

    def __init__(self, parent, actions, use_switcher=True):
        if False:
            print('Hello World!')
        QWidget.__init__(self, parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.threadmanager = ThreadManager(self)
        self.new_window = False
        self.horsplit_action = None
        self.versplit_action = None
        self.close_action = None
        self.__get_split_actions()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.menu = None
        self.switcher_manager = None
        self.tabs = None
        self.tabs_switcher = None
        self.switcher_plugin = None
        switcher_action = None
        symbolfinder_action = None
        if use_switcher and self.get_plugin().main:
            self.switcher_plugin = self.get_plugin().main.switcher
            if self.switcher_plugin:
                switcher_action = self.switcher_plugin.get_action('file switcher')
                symbolfinder_action = self.switcher_plugin.get_action('symbol finder')
        self.stack_history = StackHistory(self)
        self.external_panels = []
        self.setup_editorstack(parent, layout)
        self.find_widget = None
        self.data = []
        copy_absolute_path_action = create_action(self, _('Copy absolute path'), icon=ima.icon('editcopy'), triggered=lambda : self.copy_absolute_path())
        copy_relative_path_action = create_action(self, _('Copy relative path'), icon=ima.icon('editcopy'), triggered=lambda : self.copy_relative_path())
        close_right = create_action(self, _('Close all to the right'), triggered=self.close_all_right)
        close_all_but_this = create_action(self, _('Close all but this'), triggered=self.close_all_but_this)
        sort_tabs = create_action(self, _('Sort tabs alphabetically'), triggered=self.sort_file_tabs_alphabetically)
        if sys.platform == 'darwin':
            text = _('Show in Finder')
        else:
            text = _('Show in external file explorer')
        external_fileexp_action = create_action(self, text, triggered=self.show_in_external_file_explorer, shortcut=self.get_shortcut(context='Editor', name='show in external file explorer'), context=Qt.WidgetShortcut)
        self.menu_actions = actions + [external_fileexp_action, None, switcher_action, symbolfinder_action, copy_absolute_path_action, copy_relative_path_action, None, close_right, close_all_but_this, sort_tabs]
        self.outlineexplorer = None
        self.is_closable = False
        self.new_action = None
        self.open_action = None
        self.save_action = None
        self.revert_action = None
        self.tempfile_path = None
        self.title = _('Editor')
        self.todolist_enabled = True
        self.is_analysis_done = False
        self.linenumbers_enabled = True
        self.blanks_enabled = False
        self.scrollpastend_enabled = False
        self.edgeline_enabled = True
        self.edgeline_columns = (79,)
        self.close_parentheses_enabled = True
        self.close_quotes_enabled = True
        self.add_colons_enabled = True
        self.auto_unindent_enabled = True
        self.indent_chars = ' ' * 4
        self.tab_stop_width_spaces = 4
        self.show_class_func_dropdown = False
        self.help_enabled = False
        self.default_font = None
        self.wrap_enabled = False
        self.tabmode_enabled = False
        self.stripmode_enabled = False
        self.intelligent_backspace_enabled = True
        self.automatic_completions_enabled = True
        self.automatic_completion_chars = 3
        self.automatic_completion_ms = 300
        self.completions_hint_enabled = True
        self.completions_hint_after_ms = 500
        self.hover_hints_enabled = True
        self.format_on_save = False
        self.code_snippets_enabled = True
        self.code_folding_enabled = True
        self.underline_errors_enabled = False
        self.highlight_current_line_enabled = False
        self.highlight_current_cell_enabled = False
        self.occurrence_highlighting_enabled = True
        self.occurrence_highlighting_timeout = 1500
        self.checkeolchars_enabled = True
        self.always_remove_trailing_spaces = False
        self.add_newline = False
        self.remove_trailing_newlines = False
        self.convert_eol_on_save = False
        self.convert_eol_on_save_to = 'LF'
        self.create_new_file_if_empty = True
        self.indent_guides = False
        self.__file_status_flag = False
        color_scheme = 'spyder/dark' if is_dark_interface() else 'spyder'
        if color_scheme not in syntaxhighlighters.COLOR_SCHEME_NAMES:
            color_scheme = syntaxhighlighters.COLOR_SCHEME_NAMES[0]
        self.color_scheme = color_scheme
        self.analysis_timer = QTimer(self)
        self.analysis_timer.setSingleShot(True)
        self.analysis_timer.setInterval(1000)
        self.analysis_timer.timeout.connect(self.analyze_script)
        self.editor_focus_changed.connect(self.update_fname_label)
        self.setAcceptDrops(True)
        self.shortcuts = self.create_shortcuts()
        self.last_closed_files = []
        self.msgbox = None
        self.edit_filetypes = None
        self.edit_filters = None
        self.save_dialog_on_tests = not running_under_pytest()
        self.autosave = AutosaveForStack(self)
        self.last_cell_call = None

    @Slot()
    def show_in_external_file_explorer(self, fnames=None):
        if False:
            i = 10
            return i + 15
        'Show file in external file explorer'
        if fnames is None or isinstance(fnames, bool):
            fnames = self.get_current_filename()
        try:
            show_in_external_file_explorer(fnames)
        except FileNotFoundError as error:
            file = str(error).split("'")[1]
            if 'xdg-open' in file:
                msg_title = _('Warning')
                msg = _("Spyder can't show this file in the external file explorer because the <tt>xdg-utils</tt> package is not available on your system.")
                QMessageBox.information(self, msg_title, msg, QMessageBox.Ok)

    def copy_absolute_path(self):
        if False:
            i = 10
            return i + 15
        'Copy current filename absolute path to the clipboard.'
        QApplication.clipboard().setText(self.get_current_filename())

    def copy_relative_path(self):
        if False:
            for i in range(10):
                print('nop')
        'Copy current filename relative path to the clipboard.'
        file_drive = osp.splitdrive(self.get_current_filename())[0]
        if os.name == 'nt' and osp.splitdrive(getcwd_or_home())[0] != file_drive:
            QMessageBox.warning(self, _('No available relative path'), _('It is not possible to copy a relative path for this file because it is placed in a different drive than your current working directory. Please copy its absolute path.'))
        else:
            base_path = getcwd_or_home()
            if self.get_current_project_path():
                base_path = self.get_current_project_path()
            rel_path = osp.relpath(self.get_current_filename(), base_path).replace(os.sep, '/')
            QApplication.clipboard().setText(rel_path)

    def create_shortcuts(self):
        if False:
            while True:
                i = 10
        'Create local shortcuts'
        inspect = self.config_shortcut(self.inspect_current_object, context='Editor', name='Inspect current object', parent=self)
        gotoline = self.config_shortcut(self.go_to_line, context='Editor', name='Go to line', parent=self)
        tab = self.config_shortcut(lambda : self.tab_navigation_mru(forward=False), context='Editor', name='Go to previous file', parent=self)
        tabshift = self.config_shortcut(self.tab_navigation_mru, context='Editor', name='Go to next file', parent=self)
        prevtab = self.config_shortcut(lambda : self.tabs.tab_navigate(-1), context='Editor', name='Cycle to previous file', parent=self)
        nexttab = self.config_shortcut(lambda : self.tabs.tab_navigate(1), context='Editor', name='Cycle to next file', parent=self)
        new_file = self.config_shortcut(self.sig_new_file[()], context='Editor', name='New file', parent=self)
        open_file = self.config_shortcut(self.plugin_load[()], context='Editor', name='Open file', parent=self)
        save_file = self.config_shortcut(self.save, context='Editor', name='Save file', parent=self)
        save_all = self.config_shortcut(self.save_all, context='Editor', name='Save all', parent=self)
        save_as = self.config_shortcut(self.sig_save_as, context='Editor', name='Save As', parent=self)
        close_all = self.config_shortcut(self.close_all_files, context='Editor', name='Close all', parent=self)
        prev_edit_pos = self.config_shortcut(self.sig_prev_edit_pos, context='Editor', name='Last edit location', parent=self)
        prev_cursor = self.config_shortcut(self.sig_prev_cursor, context='Editor', name='Previous cursor position', parent=self)
        next_cursor = self.config_shortcut(self.sig_next_cursor, context='Editor', name='Next cursor position', parent=self)
        zoom_in_1 = self.config_shortcut(self.zoom_in, context='Editor', name='zoom in 1', parent=self)
        zoom_in_2 = self.config_shortcut(self.zoom_in, context='Editor', name='zoom in 2', parent=self)
        zoom_out = self.config_shortcut(self.zoom_out, context='Editor', name='zoom out', parent=self)
        zoom_reset = self.config_shortcut(self.zoom_reset, context='Editor', name='zoom reset', parent=self)
        close_file_1 = self.config_shortcut(self.close_file, context='Editor', name='close file 1', parent=self)
        close_file_2 = self.config_shortcut(self.close_file, context='Editor', name='close file 2', parent=self)
        go_to_next_cell = self.config_shortcut(self.advance_cell, context='Editor', name='go to next cell', parent=self)
        go_to_previous_cell = self.config_shortcut(lambda : self.advance_cell(reverse=True), context='Editor', name='go to previous cell', parent=self)
        prev_warning = self.config_shortcut(self.sig_prev_warning, context='Editor', name='Previous warning', parent=self)
        next_warning = self.config_shortcut(self.sig_next_warning, context='Editor', name='Next warning', parent=self)
        split_vertically = self.config_shortcut(self.sig_split_vertically, context='Editor', name='split vertically', parent=self)
        split_horizontally = self.config_shortcut(self.sig_split_horizontally, context='Editor', name='split horizontally', parent=self)
        close_split = self.config_shortcut(self.close_split, context='Editor', name='close split panel', parent=self)
        external_fileexp = self.config_shortcut(self.show_in_external_file_explorer, context='Editor', name='show in external file explorer', parent=self)
        return [inspect, gotoline, tab, tabshift, new_file, open_file, save_file, save_all, save_as, close_all, prev_edit_pos, prev_cursor, next_cursor, zoom_in_1, zoom_in_2, zoom_out, zoom_reset, close_file_1, close_file_2, go_to_next_cell, go_to_previous_cell, prev_warning, next_warning, split_vertically, split_horizontally, close_split, prevtab, nexttab, external_fileexp]

    def get_shortcut_data(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns shortcut data, a list of tuples (shortcut, text, default)\n        shortcut (QShortcut or QAction instance)\n        text (string): action/shortcut description\n        default (string): default key sequence\n        '
        return [sc.data for sc in self.shortcuts]

    def setup_editorstack(self, parent, layout):
        if False:
            i = 10
            return i + 15
        "Setup editorstack's layout"
        layout.setSpacing(0)
        self.create_top_widgets()
        layout.addWidget(self.top_toolbar)
        menu_btn = create_toolbutton(self, icon=ima.icon('tooloptions'), tip=_('Options'))
        menu_btn.setStyleSheet(str(PANES_TABBAR_STYLESHEET))
        self.menu = QMenu(self)
        menu_btn.setMenu(self.menu)
        menu_btn.setPopupMode(menu_btn.InstantPopup)
        self.menu.aboutToShow.connect(self.__setup_menu)
        corner_widgets = {Qt.TopRightCorner: [menu_btn]}
        self.tabs = BaseTabs(self, menu=self.menu, menu_use_tooltips=True, corner_widgets=corner_widgets)
        self.tabs.set_close_function(self.close_file)
        self.tabs.tabBar().tabMoved.connect(self.move_editorstack_data)
        self.tabs.setMovable(True)
        self.stack_history.refresh()
        if hasattr(self.tabs, 'setDocumentMode') and (not sys.platform == 'darwin'):
            self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self.current_changed)
        tab_container = QWidget()
        tab_container.setObjectName('tab-container')
        tab_layout = QHBoxLayout(tab_container)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.addWidget(self.tabs)
        layout.addWidget(tab_container)
        if sys.platform == 'darwin':
            self.menu.aboutToHide.connect(lambda menu=self.menu: set_menu_icons(menu, False))

    def create_top_widgets(self):
        if False:
            return 10
        self.fname_label = QLabel()
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.spinner = create_waitspinner(size=16, parent=self.fname_label)
        self.top_toolbar = QToolBar(self)
        self.top_toolbar.addWidget(self.fname_label)
        self.top_toolbar.addWidget(spacer)
        self.top_toolbar.addWidget(self.spinner)
        css = qstylizer.style.StyleSheet()
        css.QToolBar.setValues(margin='0px', padding='4px', borderBottom=f'1px solid {QStylePalette.COLOR_BACKGROUND_4}')
        self.top_toolbar.setStyleSheet(css.toString())

    def hide_tooltip(self):
        if False:
            return 10
        'Hide any open tooltips.'
        for finfo in self.data:
            finfo.editor.hide_tooltip()

    @Slot()
    def update_fname_label(self):
        if False:
            while True:
                i = 10
        'Update file name label.'
        filename = to_text_string(self.get_current_filename())
        if len(filename) > 100:
            shorten_filename = u'...' + filename[-100:]
        else:
            shorten_filename = filename
        self.fname_label.setText(shorten_filename)

    def add_corner_widgets_to_tabbar(self, widgets):
        if False:
            while True:
                i = 10
        self.tabs.add_corner_widgets(widgets)

    @Slot()
    def close_split(self):
        if False:
            while True:
                i = 10
        'Closes the editorstack if it is not the last one opened.'
        if self.is_closable:
            self.close()

    def closeEvent(self, event):
        if False:
            while True:
                i = 10
        'Overrides QWidget closeEvent().'
        self.threadmanager.close_all_threads()
        self.analysis_timer.timeout.disconnect(self.analyze_script)
        if self.outlineexplorer is not None:
            for finfo in self.data:
                self.outlineexplorer.remove_editor(finfo.editor.oe_proxy)
                if finfo.editor.is_cloned:
                    finfo.editor.oe_proxy.deleteLater()
        for finfo in self.data:
            if not finfo.editor.is_cloned:
                finfo.editor.notify_close()
        QWidget.closeEvent(self, event)

    def clone_editor_from(self, other_finfo, set_current):
        if False:
            while True:
                i = 10
        fname = other_finfo.filename
        enc = other_finfo.encoding
        new = other_finfo.newly_created
        finfo = self.create_new_editor(fname, enc, '', set_current=set_current, new=new, cloned_from=other_finfo.editor)
        finfo.set_todo_results(other_finfo.todo_results)
        return finfo.editor

    def clone_from(self, other):
        if False:
            return 10
        'Clone EditorStack from other instance'
        for other_finfo in other.data:
            self.clone_editor_from(other_finfo, set_current=True)
        self.set_stack_index(other.get_stack_index())

    def get_plugin(self):
        if False:
            return 10
        'Get the plugin of the parent widget.'
        return self.parent().plugin

    def get_plugin_title(self):
        if False:
            print('Hello World!')
        'Get the plugin title of the parent widget.'
        return self.get_plugin().get_plugin_title()

    def go_to_line(self, line=None):
        if False:
            for i in range(10):
                print('nop')
        'Go to line dialog'
        if line is not None:
            self.get_current_editor().go_to_line(line)
        elif self.data:
            self.get_current_editor().exec_gotolinedialog()

    def set_bookmark(self, slot_num):
        if False:
            i = 10
            return i + 15
        'Bookmark current position to given slot.'
        if self.data:
            editor = self.get_current_editor()
            editor.add_bookmark(slot_num)

    @Slot()
    @Slot(bool)
    def inspect_current_object(self, clicked=False):
        if False:
            i = 10
            return i + 15
        'Inspect current object in the Help plugin'
        editor = self.get_current_editor()
        editor.sig_display_object_info.connect(self.display_help)
        cursor = None
        offset = editor.get_position('cursor')
        if clicked:
            cursor = editor.get_last_hover_cursor()
            if cursor:
                offset = cursor.position()
            else:
                return
        (line, col) = editor.get_cursor_line_column(cursor)
        editor.request_hover(line, col, offset, show_hint=False, clicked=clicked)

    @Slot(str, bool)
    def display_help(self, help_text, clicked):
        if False:
            while True:
                i = 10
        editor = self.get_current_editor()
        if clicked:
            name = editor.get_last_hover_word()
        else:
            name = editor.get_current_word(help_req=True)
        try:
            editor.sig_display_object_info.disconnect(self.display_help)
        except TypeError:
            pass
        self.send_to_help(name, help_text, force=True)

    def set_closable(self, state):
        if False:
            while True:
                i = 10
        'Parent widget must handle the closable state'
        self.is_closable = state

    def set_io_actions(self, new_action, open_action, save_action, revert_action):
        if False:
            for i in range(10):
                print('nop')
        self.new_action = new_action
        self.open_action = open_action
        self.save_action = save_action
        self.revert_action = revert_action

    def set_find_widget(self, find_widget):
        if False:
            for i in range(10):
                print('nop')
        self.find_widget = find_widget

    def set_outlineexplorer(self, outlineexplorer):
        if False:
            for i in range(10):
                print('nop')
        self.outlineexplorer = outlineexplorer

    def set_tempfile_path(self, path):
        if False:
            for i in range(10):
                print('nop')
        self.tempfile_path = path

    def set_title(self, text):
        if False:
            return 10
        self.title = text

    def set_classfunc_dropdown_visible(self, state):
        if False:
            print('Hello World!')
        self.show_class_func_dropdown = state
        if self.data:
            for finfo in self.data:
                if finfo.editor.is_python_like():
                    finfo.editor.classfuncdropdown.setVisible(state)

    def __update_editor_margins(self, editor):
        if False:
            print('Hello World!')
        editor.linenumberarea.setup_margins(linenumbers=self.linenumbers_enabled, markers=self.has_markers())

    def has_markers(self):
        if False:
            print('Hello World!')
        'Return True if this editorstack has a marker margin for TODOs or\n        code analysis'
        return self.todolist_enabled

    def set_todolist_enabled(self, state, current_finfo=None):
        if False:
            return 10
        self.todolist_enabled = state
        if self.data:
            for finfo in self.data:
                self.__update_editor_margins(finfo.editor)
                finfo.cleanup_todo_results()
                if state and current_finfo is not None:
                    if current_finfo is not finfo:
                        finfo.run_todo_finder()

    def set_linenumbers_enabled(self, state, current_finfo=None):
        if False:
            i = 10
            return i + 15
        self.linenumbers_enabled = state
        if self.data:
            for finfo in self.data:
                self.__update_editor_margins(finfo.editor)

    def set_blanks_enabled(self, state):
        if False:
            i = 10
            return i + 15
        self.blanks_enabled = state
        if self.data:
            for finfo in self.data:
                finfo.editor.set_blanks_enabled(state)

    def set_scrollpastend_enabled(self, state):
        if False:
            return 10
        self.scrollpastend_enabled = state
        if self.data:
            for finfo in self.data:
                finfo.editor.set_scrollpastend_enabled(state)

    def set_edgeline_enabled(self, state):
        if False:
            return 10
        self.edgeline_enabled = state
        if self.data:
            for finfo in self.data:
                finfo.editor.edge_line.set_enabled(state)

    def set_edgeline_columns(self, columns):
        if False:
            for i in range(10):
                print('nop')
        self.edgeline_columns = columns
        if self.data:
            for finfo in self.data:
                finfo.editor.edge_line.set_columns(columns)

    def set_indent_guides(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.indent_guides = state
        if self.data:
            for finfo in self.data:
                finfo.editor.toggle_identation_guides(state)

    def set_close_parentheses_enabled(self, state):
        if False:
            print('Hello World!')
        self.close_parentheses_enabled = state
        if self.data:
            for finfo in self.data:
                finfo.editor.set_close_parentheses_enabled(state)

    def set_close_quotes_enabled(self, state):
        if False:
            print('Hello World!')
        self.close_quotes_enabled = state
        if self.data:
            for finfo in self.data:
                finfo.editor.set_close_quotes_enabled(state)

    def set_add_colons_enabled(self, state):
        if False:
            print('Hello World!')
        self.add_colons_enabled = state
        if self.data:
            for finfo in self.data:
                finfo.editor.set_add_colons_enabled(state)

    def set_auto_unindent_enabled(self, state):
        if False:
            print('Hello World!')
        self.auto_unindent_enabled = state
        if self.data:
            for finfo in self.data:
                finfo.editor.set_auto_unindent_enabled(state)

    def set_indent_chars(self, indent_chars):
        if False:
            while True:
                i = 10
        indent_chars = indent_chars[1:-1]
        self.indent_chars = indent_chars
        if self.data:
            for finfo in self.data:
                finfo.editor.set_indent_chars(indent_chars)

    def set_tab_stop_width_spaces(self, tab_stop_width_spaces):
        if False:
            print('Hello World!')
        self.tab_stop_width_spaces = tab_stop_width_spaces
        if self.data:
            for finfo in self.data:
                finfo.editor.tab_stop_width_spaces = tab_stop_width_spaces
                finfo.editor.update_tab_stop_width_spaces()

    def set_help_enabled(self, state):
        if False:
            i = 10
            return i + 15
        self.help_enabled = state

    def set_default_font(self, font, color_scheme=None):
        if False:
            return 10
        self.default_font = font
        if color_scheme is not None:
            self.color_scheme = color_scheme
        if self.data:
            for finfo in self.data:
                finfo.editor.set_font(font, color_scheme)

    def set_color_scheme(self, color_scheme):
        if False:
            print('Hello World!')
        self.color_scheme = color_scheme
        if self.data:
            for finfo in self.data:
                finfo.editor.set_color_scheme(color_scheme)
                finfo.editor.unhighlight_current_line()
                finfo.editor.unhighlight_current_cell()
                finfo.editor.clear_occurrences()
                if self.highlight_current_line_enabled:
                    finfo.editor.highlight_current_line()
                if self.highlight_current_cell_enabled:
                    finfo.editor.highlight_current_cell()
                if self.occurrence_highlighting_enabled:
                    finfo.editor.mark_occurrences()

    def set_wrap_enabled(self, state):
        if False:
            while True:
                i = 10
        self.wrap_enabled = state
        if self.data:
            for finfo in self.data:
                finfo.editor.toggle_wrap_mode(state)

    def set_tabmode_enabled(self, state):
        if False:
            i = 10
            return i + 15
        self.tabmode_enabled = state
        if self.data:
            for finfo in self.data:
                finfo.editor.set_tab_mode(state)

    def set_stripmode_enabled(self, state):
        if False:
            i = 10
            return i + 15
        self.stripmode_enabled = state
        if self.data:
            for finfo in self.data:
                finfo.editor.set_strip_mode(state)

    def set_intelligent_backspace_enabled(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.intelligent_backspace_enabled = state
        if self.data:
            for finfo in self.data:
                finfo.editor.toggle_intelligent_backspace(state)

    def set_code_snippets_enabled(self, state):
        if False:
            return 10
        self.code_snippets_enabled = state
        if self.data:
            for finfo in self.data:
                finfo.editor.toggle_code_snippets(state)

    def set_code_folding_enabled(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.code_folding_enabled = state
        if self.data:
            for finfo in self.data:
                finfo.editor.toggle_code_folding(state)

    def set_automatic_completions_enabled(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.automatic_completions_enabled = state
        if self.data:
            for finfo in self.data:
                finfo.editor.toggle_automatic_completions(state)

    def set_automatic_completions_after_chars(self, chars):
        if False:
            i = 10
            return i + 15
        self.automatic_completion_chars = chars
        if self.data:
            for finfo in self.data:
                finfo.editor.set_automatic_completions_after_chars(chars)

    def set_completions_hint_enabled(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.completions_hint_enabled = state
        if self.data:
            for finfo in self.data:
                finfo.editor.toggle_completions_hint(state)

    def set_completions_hint_after_ms(self, ms):
        if False:
            return 10
        self.completions_hint_after_ms = ms
        if self.data:
            for finfo in self.data:
                finfo.editor.set_completions_hint_after_ms(ms)

    def set_hover_hints_enabled(self, state):
        if False:
            while True:
                i = 10
        self.hover_hints_enabled = state
        if self.data:
            for finfo in self.data:
                finfo.editor.toggle_hover_hints(state)

    def set_format_on_save(self, state):
        if False:
            print('Hello World!')
        self.format_on_save = state
        if self.data:
            for finfo in self.data:
                finfo.editor.toggle_format_on_save(state)

    def set_occurrence_highlighting_enabled(self, state):
        if False:
            i = 10
            return i + 15
        self.occurrence_highlighting_enabled = state
        if self.data:
            for finfo in self.data:
                finfo.editor.set_occurrence_highlighting(state)

    def set_occurrence_highlighting_timeout(self, timeout):
        if False:
            print('Hello World!')
        self.occurrence_highlighting_timeout = timeout
        if self.data:
            for finfo in self.data:
                finfo.editor.set_occurrence_timeout(timeout)

    def set_underline_errors_enabled(self, state):
        if False:
            print('Hello World!')
        self.underline_errors_enabled = state
        if self.data:
            for finfo in self.data:
                finfo.editor.set_underline_errors_enabled(state)

    def set_highlight_current_line_enabled(self, state):
        if False:
            i = 10
            return i + 15
        self.highlight_current_line_enabled = state
        if self.data:
            for finfo in self.data:
                finfo.editor.set_highlight_current_line(state)

    def set_highlight_current_cell_enabled(self, state):
        if False:
            while True:
                i = 10
        self.highlight_current_cell_enabled = state
        if self.data:
            for finfo in self.data:
                finfo.editor.set_highlight_current_cell(state)

    def set_checkeolchars_enabled(self, state):
        if False:
            i = 10
            return i + 15
        self.checkeolchars_enabled = state

    def set_always_remove_trailing_spaces(self, state):
        if False:
            i = 10
            return i + 15
        self.always_remove_trailing_spaces = state
        if self.data:
            for finfo in self.data:
                finfo.editor.set_remove_trailing_spaces(state)

    def set_add_newline(self, state):
        if False:
            i = 10
            return i + 15
        self.add_newline = state
        if self.data:
            for finfo in self.data:
                finfo.editor.set_add_newline(state)

    def set_remove_trailing_newlines(self, state):
        if False:
            while True:
                i = 10
        self.remove_trailing_newlines = state
        if self.data:
            for finfo in self.data:
                finfo.editor.set_remove_trailing_newlines(state)

    def set_convert_eol_on_save(self, state):
        if False:
            return 10
        'If `state` is `True`, saving files will convert line endings.'
        self.convert_eol_on_save = state

    def set_convert_eol_on_save_to(self, state):
        if False:
            return 10
        "`state` can be one of ('LF', 'CRLF', 'CR')"
        self.convert_eol_on_save_to = state

    def set_current_project_path(self, root_path=None):
        if False:
            while True:
                i = 10
        '\n        Set the current active project root path.\n\n        Parameters\n        ----------\n        root_path: str or None, optional\n            Path to current project root path. Default is None.\n        '
        for finfo in self.data:
            finfo.editor.set_current_project_path(root_path)

    def get_stack_index(self):
        if False:
            print('Hello World!')
        return self.tabs.currentIndex()

    def get_current_finfo(self):
        if False:
            while True:
                i = 10
        if self.data:
            return self.data[self.get_stack_index()]

    def get_current_editor(self):
        if False:
            for i in range(10):
                print('nop')
        return self.tabs.currentWidget()

    def get_stack_count(self):
        if False:
            return 10
        return self.tabs.count()

    def set_stack_index(self, index, instance=None):
        if False:
            i = 10
            return i + 15
        if instance == self or instance is None:
            self.tabs.setCurrentIndex(index)

    def set_tabbar_visible(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.tabs.tabBar().setVisible(state)

    def remove_from_data(self, index):
        if False:
            i = 10
            return i + 15
        self.tabs.blockSignals(True)
        self.tabs.removeTab(index)
        self.data.pop(index)
        self.tabs.blockSignals(False)
        self.update_actions()

    def __modified_readonly_title(self, title, is_modified, is_readonly):
        if False:
            i = 10
            return i + 15
        if is_modified is not None and is_modified:
            title += '*'
        if is_readonly is not None and is_readonly:
            title = '(%s)' % title
        return title

    def get_tab_text(self, index, is_modified=None, is_readonly=None):
        if False:
            while True:
                i = 10
        'Return tab title.'
        files_path_list = [finfo.filename for finfo in self.data]
        fname = self.data[index].filename
        fname = sourcecode.disambiguate_fname(files_path_list, fname)
        return self.__modified_readonly_title(fname, is_modified, is_readonly)

    def get_tab_tip(self, filename, is_modified=None, is_readonly=None):
        if False:
            return 10
        'Return tab menu title'
        text = u'%s — %s'
        text = self.__modified_readonly_title(text, is_modified, is_readonly)
        if self.tempfile_path is not None and filename == encoding.to_unicode_from_fs(self.tempfile_path):
            temp_file_str = to_text_string(_('Temporary file'))
            return text % (temp_file_str, self.tempfile_path)
        else:
            return text % (osp.basename(filename), osp.dirname(filename))

    def add_to_data(self, finfo, set_current, add_where='end'):
        if False:
            i = 10
            return i + 15
        finfo.editor.oe_proxy = None
        index = 0 if add_where == 'start' else len(self.data)
        self.data.insert(index, finfo)
        index = self.data.index(finfo)
        editor = finfo.editor
        self.tabs.insertTab(index, editor, self.get_tab_text(index))
        self.set_stack_title(index, False)
        if set_current:
            self.set_stack_index(index)
            self.current_changed(index)
        self.update_actions()

    def __repopulate_stack(self):
        if False:
            print('Hello World!')
        self.tabs.blockSignals(True)
        self.tabs.clear()
        for finfo in self.data:
            if finfo.newly_created:
                is_modified = True
            else:
                is_modified = None
            index = self.data.index(finfo)
            tab_text = self.get_tab_text(index, is_modified)
            tab_tip = self.get_tab_tip(finfo.filename)
            index = self.tabs.addTab(finfo.editor, tab_text)
            self.tabs.setTabToolTip(index, tab_tip)
        self.tabs.blockSignals(False)

    def rename_in_data(self, original_filename, new_filename):
        if False:
            i = 10
            return i + 15
        index = self.has_filename(original_filename)
        if index is None:
            return
        finfo = self.data[index]
        finfo.editor.notify_close()
        finfo.filename = new_filename
        finfo.editor.filename = new_filename
        original_ext = osp.splitext(original_filename)[1]
        new_ext = osp.splitext(new_filename)[1]
        if original_ext != new_ext:
            txt = to_text_string(finfo.editor.get_text_with_eol())
            language = get_file_language(new_filename, txt)
            finfo.editor.set_language(language, new_filename)
            finfo.editor.run_pygments_highlighter()
            options = {'language': language, 'filename': new_filename, 'codeeditor': finfo.editor}
            self.sig_open_file.emit(options)
            finfo.editor.cleanup_code_analysis()
            finfo.editor.cleanup_folding()
        else:
            finfo.editor.document_did_open()
        set_new_index = index == self.get_stack_index()
        current_fname = self.get_current_filename()
        finfo.editor.filename = new_filename
        new_index = self.data.index(finfo)
        self.__repopulate_stack()
        if set_new_index:
            self.set_stack_index(new_index)
        else:
            self.set_current_filename(current_fname)
        if self.outlineexplorer is not None:
            self.outlineexplorer.file_renamed(finfo.editor.oe_proxy, finfo.filename)
        return new_index

    def set_stack_title(self, index, is_modified):
        if False:
            while True:
                i = 10
        finfo = self.data[index]
        fname = finfo.filename
        is_modified = (is_modified or finfo.newly_created) and (not finfo.default)
        is_readonly = finfo.editor.isReadOnly()
        tab_text = self.get_tab_text(index, is_modified, is_readonly)
        tab_tip = self.get_tab_tip(fname, is_modified, is_readonly)
        if tab_text != self.tabs.tabText(index):
            self.tabs.setTabText(index, tab_text)
        self.tabs.setTabToolTip(index, tab_tip)

    def __setup_menu(self):
        if False:
            print('Hello World!')
        'Setup tab context menu before showing it'
        self.menu.clear()
        if self.data:
            actions = self.menu_actions
        else:
            actions = (self.new_action, self.open_action)
            self.setFocus()
        add_actions(self.menu, list(actions) + self.__get_split_actions())
        self.close_action.setEnabled(self.is_closable)
        if sys.platform == 'darwin':
            set_menu_icons(self.menu, True)

    def __get_split_actions(self):
        if False:
            print('Hello World!')
        if self.parent() is not None:
            plugin = self.parent().plugin
        else:
            plugin = None
        if plugin is not None:
            self.new_window_action = create_action(self, _('New window'), icon=ima.icon('newwindow'), tip=_('Create a new editor window'), triggered=plugin.create_new_window)
        self.versplit_action = create_action(self, _('Split vertically'), icon=ima.icon('versplit'), tip=_('Split vertically this editor window'), triggered=self.sig_split_vertically, shortcut=self.get_shortcut(context='Editor', name='split vertically'), context=Qt.WidgetShortcut)
        self.horsplit_action = create_action(self, _('Split horizontally'), icon=ima.icon('horsplit'), tip=_('Split horizontally this editor window'), triggered=self.sig_split_horizontally, shortcut=self.get_shortcut(context='Editor', name='split horizontally'), context=Qt.WidgetShortcut)
        self.close_action = create_action(self, _('Close this panel'), icon=ima.icon('close_panel'), triggered=self.close_split, shortcut=self.get_shortcut(context='Editor', name='close split panel'), context=Qt.WidgetShortcut)
        actions = [MENU_SEPARATOR, self.versplit_action, self.horsplit_action, self.close_action]
        if self.new_window:
            window = self.window()
            close_window_action = create_action(self, _('Close window'), icon=ima.icon('close_pane'), triggered=window.close)
            actions += [MENU_SEPARATOR, self.new_window_action, close_window_action]
        elif plugin is not None:
            if plugin._undocked_window is not None:
                actions += [MENU_SEPARATOR, plugin._dock_action]
            else:
                actions += [MENU_SEPARATOR, self.new_window_action, plugin._lock_unlock_action, plugin._undock_action, plugin._close_plugin_action]
        return actions

    def reset_orientation(self):
        if False:
            return 10
        self.horsplit_action.setEnabled(True)
        self.versplit_action.setEnabled(True)

    def set_orientation(self, orientation):
        if False:
            while True:
                i = 10
        self.horsplit_action.setEnabled(orientation == Qt.Horizontal)
        self.versplit_action.setEnabled(orientation == Qt.Vertical)

    def update_actions(self):
        if False:
            print('Hello World!')
        state = self.get_stack_count() > 0
        self.horsplit_action.setEnabled(state)
        self.versplit_action.setEnabled(state)

    def get_current_filename(self):
        if False:
            return 10
        if self.data:
            return self.data[self.get_stack_index()].filename

    def get_current_language(self):
        if False:
            while True:
                i = 10
        if self.data:
            return self.data[self.get_stack_index()].editor.language

    def get_current_project_path(self):
        if False:
            print('Hello World!')
        if self.data:
            finfo = self.get_current_finfo()
            if finfo:
                return finfo.editor.current_project_path

    def get_filenames(self):
        if False:
            return 10
        '\n        Return a list with the names of all the files currently opened in\n        the editorstack.\n        '
        return [finfo.filename for finfo in self.data]

    def has_filename(self, filename):
        if False:
            for i in range(10):
                print('nop')
        'Return the self.data index position for the filename.\n\n        Args:\n            filename: Name of the file to search for in self.data.\n\n        Returns:\n            The self.data index for the filename.  Returns None\n            if the filename is not found in self.data.\n        '
        data_filenames = self.get_filenames()
        try:
            return data_filenames.index(filename)
        except ValueError:
            try:
                filename = fixpath(filename)
            except OSError:
                return None
            for (index, editor_filename) in enumerate(data_filenames):
                if filename == fixpath(editor_filename):
                    return index
            return None

    def set_current_filename(self, filename, focus=True):
        if False:
            while True:
                i = 10
        'Set current filename and return the associated editor instance.'
        try:
            index = self.has_filename(filename)
        except (FileNotFoundError, OSError):
            index = None
        if index is not None:
            if focus:
                self.set_stack_index(index)
            editor = self.data[index].editor
            if focus:
                editor.setFocus()
            else:
                self.stack_history.remove_and_append(index)
            return editor

    def is_file_opened(self, filename=None):
        if False:
            for i in range(10):
                print('nop')
        "Return if filename is in the editor stack.\n\n        Args:\n            filename: Name of the file to search for.  If filename is None,\n                then checks if any file is open.\n\n        Returns:\n            True: If filename is None and a file is open.\n            False: If filename is None and no files are open.\n            None: If filename is not None and the file isn't found.\n            integer: Index of file name in editor stack.\n        "
        if filename is None:
            return len(self.data) > 0
        else:
            return self.has_filename(filename)

    def get_index_from_filename(self, filename):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the position index of a file in the tab bar of the editorstack\n        from its name.\n        '
        filenames = [d.filename for d in self.data]
        return filenames.index(filename)

    @Slot(int, int)
    def move_editorstack_data(self, start, end):
        if False:
            for i in range(10):
                print('nop')
        'Reorder editorstack.data so it is synchronized with the tab bar when\n        tabs are moved.'
        if start < 0 or end < 0:
            return
        else:
            steps = abs(end - start)
            direction = (end - start) // steps
        data = self.data
        self.blockSignals(True)
        for i in range(start, end, direction):
            (data[i], data[i + direction]) = (data[i + direction], data[i])
        self.blockSignals(False)
        self.refresh()

    def close_file(self, index=None, force=False):
        if False:
            while True:
                i = 10
        'Close file (index=None -> close current file)\n        Keep current file index unchanged (if current file\n        that is being closed)'
        current_index = self.get_stack_index()
        count = self.get_stack_count()
        if index is None:
            if count > 0:
                index = current_index
            else:
                self.find_widget.set_editor(None)
                return
        new_index = None
        if count > 1:
            if current_index == index:
                new_index = self._get_previous_file_index()
            else:
                new_index = current_index
        can_close_file = self.parent().plugin.can_close_file(self.data[index].filename) if self.parent() else True
        is_ok = force or (self.save_if_changed(cancelable=True, index=index) and can_close_file)
        if is_ok:
            finfo = self.data[index]
            self.threadmanager.close_threads(finfo)
            if self.outlineexplorer is not None:
                self.outlineexplorer.remove_editor(finfo.editor.oe_proxy)
            filename = self.data[index].filename
            self.remove_from_data(index)
            editor = finfo.editor
            editor.notify_close()
            editor.setParent(None)
            editor.completion_widget.setParent(None)
            if self.parent():
                self.get_plugin().unregister_widget_shortcuts(editor)
            self.sig_close_file.emit(str(id(self)), filename)
            self.sig_codeeditor_deleted.emit(editor)
            self.opened_files_list_changed.emit()
            self.sig_update_code_analysis_actions.emit()
            self.refresh_file_dependent_actions.emit()
            self.update_plugin_title.emit()
            if new_index is not None:
                if index < new_index:
                    new_index -= 1
                self.set_stack_index(new_index)
            editor = self.get_current_editor()
            if editor:
                QApplication.processEvents()
                self.__file_status_flag = False
                editor.setFocus()
            self.add_last_closed_file(finfo.filename)
            if finfo.filename in self.autosave.file_hashes:
                del self.autosave.file_hashes[finfo.filename]
        if self.get_stack_count() == 0 and self.create_new_file_if_empty:
            self.sig_new_file[()].emit()
            self.update_fname_label()
            return False
        self.__modify_stack_title()
        return is_ok

    def register_completion_capabilities(self, capabilities, language):
        if False:
            for i in range(10):
                print('nop')
        '\n        Register completion server capabilities across all editors.\n\n        Parameters\n        ----------\n        capabilities: dict\n            Capabilities supported by a language server.\n        language: str\n            Programming language for the language server (it has to be\n            in small caps).\n        '
        for index in range(self.get_stack_count()):
            editor = self.tabs.widget(index)
            if editor.language.lower() == language:
                editor.register_completion_capabilities(capabilities)

    def start_completion_services(self, language):
        if False:
            while True:
                i = 10
        'Notify language server availability to code editors.'
        for index in range(self.get_stack_count()):
            editor = self.tabs.widget(index)
            if editor.language.lower() == language:
                editor.start_completion_services()

    def stop_completion_services(self, language):
        if False:
            return 10
        'Notify language server unavailability to code editors.'
        for index in range(self.get_stack_count()):
            editor = self.tabs.widget(index)
            if editor.language.lower() == language:
                editor.stop_completion_services()

    def close_all_files(self):
        if False:
            i = 10
            return i + 15
        'Close all opened scripts'
        while self.close_file():
            pass

    def close_all_right(self):
        if False:
            print('Hello World!')
        ' Close all files opened to the right '
        num = self.get_stack_index()
        n = self.get_stack_count()
        for __ in range(num, n - 1):
            self.close_file(num + 1)

    def close_all_but_this(self):
        if False:
            i = 10
            return i + 15
        'Close all files but the current one'
        self.close_all_right()
        for __ in range(0, self.get_stack_count() - 1):
            self.close_file(0)

    def sort_file_tabs_alphabetically(self):
        if False:
            while True:
                i = 10
        'Sort open tabs alphabetically.'
        while self.sorted() is False:
            for i in range(0, self.tabs.tabBar().count()):
                if self.tabs.tabBar().tabText(i) > self.tabs.tabBar().tabText(i + 1):
                    self.tabs.tabBar().moveTab(i, i + 1)

    def sorted(self):
        if False:
            return 10
        'Utility function for sort_file_tabs_alphabetically().'
        for i in range(0, self.tabs.tabBar().count() - 1):
            if self.tabs.tabBar().tabText(i) > self.tabs.tabBar().tabText(i + 1):
                return False
        return True

    def add_last_closed_file(self, fname):
        if False:
            i = 10
            return i + 15
        'Add to last closed file list.'
        if fname in self.last_closed_files:
            self.last_closed_files.remove(fname)
        self.last_closed_files.insert(0, fname)
        if len(self.last_closed_files) > 10:
            self.last_closed_files.pop(-1)

    def get_last_closed_files(self):
        if False:
            while True:
                i = 10
        return self.last_closed_files

    def set_last_closed_files(self, fnames):
        if False:
            i = 10
            return i + 15
        self.last_closed_files = fnames

    def save_if_changed(self, cancelable=False, index=None):
        if False:
            i = 10
            return i + 15
        'Ask user to save file if modified.\n\n        Args:\n            cancelable: Show Cancel button.\n            index: File to check for modification.\n\n        Returns:\n            False when save() fails or is cancelled.\n            True when save() is successful, there are no modifications,\n                or user selects No or NoToAll.\n\n        This function controls the message box prompt for saving\n        changed files.  The actual save is performed in save() for\n        each index processed. This function also removes autosave files\n        corresponding to files the user chooses not to save.\n        '
        if index is None:
            indexes = list(range(self.get_stack_count()))
        else:
            indexes = [index]
        buttons = QMessageBox.Yes | QMessageBox.No
        if cancelable:
            buttons |= QMessageBox.Cancel
        unsaved_nb = 0
        for index in indexes:
            if self.data[index].editor.document().isModified():
                unsaved_nb += 1
        if not unsaved_nb:
            return True
        if unsaved_nb > 1:
            buttons |= int(QMessageBox.YesToAll | QMessageBox.NoToAll)
        yes_all = no_all = False
        for index in indexes:
            self.set_stack_index(index)
            try:
                finfo = self.data[index]
            except IndexError:
                return False
            if finfo.filename == self.tempfile_path or yes_all:
                if not self.save(index):
                    return False
            elif no_all:
                self.autosave.remove_autosave_file(finfo)
            elif finfo.editor.document().isModified() and self.save_dialog_on_tests:
                self.msgbox = QMessageBox(QMessageBox.Question, self.title, _('<b>%s</b> has been modified.<br>Do you want to save changes?') % osp.basename(finfo.filename), buttons, parent=self)
                answer = self.msgbox.exec_()
                if answer == QMessageBox.Yes:
                    if not self.save(index):
                        return False
                elif answer == QMessageBox.No:
                    self.autosave.remove_autosave_file(finfo.filename)
                elif answer == QMessageBox.YesToAll:
                    if not self.save(index):
                        return False
                    yes_all = True
                elif answer == QMessageBox.NoToAll:
                    self.autosave.remove_autosave_file(finfo.filename)
                    no_all = True
                elif answer == QMessageBox.Cancel:
                    return False
        return True

    def compute_hash(self, fileinfo):
        if False:
            print('Hello World!')
        'Compute hash of contents of editor.\n\n        Args:\n            fileinfo: FileInfo object associated to editor whose hash needs\n                to be computed.\n\n        Returns:\n            int: computed hash.\n        '
        txt = to_text_string(fileinfo.editor.get_text_with_eol())
        return hash(txt)

    def _write_to_file(self, fileinfo, filename):
        if False:
            for i in range(10):
                print('nop')
        'Low-level function for writing text of editor to file.\n\n        Args:\n            fileinfo: FileInfo object associated to editor to be saved\n            filename: str with filename to save to\n\n        This is a low-level function that only saves the text to file in the\n        correct encoding without doing any error handling.\n        '
        txt = to_text_string(fileinfo.editor.get_text_with_eol())
        fileinfo.encoding = encoding.write(txt, filename, fileinfo.encoding)

    def save(self, index=None, force=False, save_new_files=True):
        if False:
            while True:
                i = 10
        "Write text of editor to a file.\n\n        Args:\n            index: self.data index to save.  If None, defaults to\n                currentIndex().\n            force: Force save regardless of file state.\n\n        Returns:\n            True upon successful save or when file doesn't need to be saved.\n            False if save failed.\n\n        If the text isn't modified and it's not newly created, then the save\n        is aborted.  If the file hasn't been saved before, then save_as()\n        is invoked.  Otherwise, the file is written using the file name\n        currently in self.data.  This function doesn't change the file name.\n        "
        if index is None:
            if not self.get_stack_count():
                return
            index = self.get_stack_index()
        finfo = self.data[index]
        if not (finfo.editor.document().isModified() or finfo.newly_created) and (not force):
            return True
        if not osp.isfile(finfo.filename) and (not force):
            if save_new_files:
                return self.save_as(index=index)
            return True
        if self.always_remove_trailing_spaces and (not self.format_on_save):
            self.remove_trailing_spaces(index)
        if self.remove_trailing_newlines and (not self.format_on_save):
            self.trim_trailing_newlines(index)
        if self.add_newline and (not self.format_on_save):
            self.add_newline_to_file(index)
        if self.convert_eol_on_save:
            osname_lookup = {'LF': 'posix', 'CRLF': 'nt', 'CR': 'mac'}
            osname = osname_lookup[self.convert_eol_on_save_to]
            self.set_os_eol_chars(osname=osname)
        try:
            if self.format_on_save and finfo.editor.formatting_enabled and finfo.editor.is_python():
                format_eventloop = finfo.editor.format_eventloop
                format_timer = finfo.editor.format_timer
                format_timer.setSingleShot(True)
                format_timer.timeout.connect(format_eventloop.quit)
                finfo.editor.sig_stop_operation_in_progress.connect(lambda : self._save_file(finfo))
                finfo.editor.sig_stop_operation_in_progress.connect(format_timer.stop)
                finfo.editor.sig_stop_operation_in_progress.connect(format_eventloop.quit)
                format_timer.start(10000)
                finfo.editor.format_document()
                format_eventloop.exec_()
            else:
                self._save_file(finfo)
            return True
        except EnvironmentError as error:
            self.msgbox = QMessageBox(QMessageBox.Critical, _('Save Error'), _("<b>Unable to save file '%s'</b><br><br>Error message:<br>%s") % (osp.basename(finfo.filename), str(error)), parent=self)
            self.msgbox.exec_()
            return False

    def _save_file(self, finfo):
        if False:
            return 10
        index = self.data.index(finfo)
        self._write_to_file(finfo, finfo.filename)
        file_hash = self.compute_hash(finfo)
        self.autosave.file_hashes[finfo.filename] = file_hash
        self.autosave.remove_autosave_file(finfo.filename)
        finfo.newly_created = False
        self.encoding_changed.emit(finfo.encoding)
        finfo.lastmodified = QFileInfo(finfo.filename).lastModified()
        self.file_saved.emit(str(id(self)), finfo.filename, finfo.filename)
        finfo.editor.document().setModified(False)
        self.modification_changed(index=index)
        self.analyze_script(index=index)
        finfo.editor.notify_save()

    def file_saved_in_other_editorstack(self, original_filename, filename):
        if False:
            for i in range(10):
                print('nop')
        "\n        File was just saved in another editorstack, let's synchronize!\n        This avoids file being automatically reloaded.\n\n        The original filename is passed instead of an index in case the tabs\n        on the editor stacks were moved and are now in a different order - see\n        spyder-ide/spyder#5703.\n        Filename is passed in case file was just saved as another name.\n        "
        index = self.has_filename(original_filename)
        if index is None:
            return
        finfo = self.data[index]
        finfo.newly_created = False
        finfo.filename = to_text_string(filename)
        finfo.lastmodified = QFileInfo(finfo.filename).lastModified()

    def select_savename(self, original_filename):
        if False:
            return 10
        'Select a name to save a file.\n\n        Args:\n            original_filename: Used in the dialog to display the current file\n                    path and name.\n\n        Returns:\n            Normalized path for the selected file name or None if no name was\n            selected.\n        '
        if self.edit_filetypes is None:
            self.edit_filetypes = get_edit_filetypes()
        if self.edit_filters is None:
            self.edit_filters = get_edit_filters()
        if is_kde_desktop() and (not is_anaconda()):
            filters = ''
            selectedfilter = ''
        else:
            filters = self.edit_filters
            selectedfilter = get_filter(self.edit_filetypes, osp.splitext(original_filename)[1])
        self.redirect_stdio.emit(False)
        (filename, _selfilter) = getsavefilename(self, _('Save file'), original_filename, filters=filters, selectedfilter=selectedfilter, options=QFileDialog.HideNameFilterDetails)
        self.redirect_stdio.emit(True)
        if filename:
            return osp.normpath(filename)
        return None

    def save_as(self, index=None):
        if False:
            print('Hello World!')
        'Save file as...\n\n        Args:\n            index: self.data index for the file to save.\n\n        Returns:\n            False if no file name was selected or if save() was unsuccessful.\n            True is save() was successful.\n\n        Gets the new file name from select_savename().  If no name is chosen,\n        then the save_as() aborts.  Otherwise, the current stack is checked\n        to see if the selected name already exists and, if so, then the tab\n        with that name is closed.\n\n        The current stack (self.data) and current tabs are updated with the\n        new name and other file info.  The text is written with the new\n        name using save() and the name change is propagated to the other stacks\n        via the file_renamed_in_data signal.\n        '
        if index is None:
            index = self.get_stack_index()
        finfo = self.data[index]
        original_newly_created = finfo.newly_created
        finfo.newly_created = True
        original_filename = finfo.filename
        filename = self.select_savename(original_filename)
        if filename:
            ao_index = self.has_filename(filename)
            if ao_index is not None and ao_index != index:
                if not self.close_file(ao_index):
                    return
                if ao_index < index:
                    index -= 1
            new_index = self.rename_in_data(original_filename, new_filename=filename)
            self.file_renamed_in_data.emit(original_filename, filename, str(id(self)))
            ok = self.save(index=new_index, force=True)
            self.refresh(new_index)
            self.set_stack_index(new_index)
            return ok
        else:
            finfo.newly_created = original_newly_created
            return False

    def save_copy_as(self, index=None):
        if False:
            print('Hello World!')
        "Save copy of file as...\n\n        Args:\n            index: self.data index for the file to save.\n\n        Returns:\n            False if no file name was selected or if save() was unsuccessful.\n            True is save() was successful.\n\n        Gets the new file name from select_savename().  If no name is chosen,\n        then the save_copy_as() aborts.  Otherwise, the current stack is\n        checked to see if the selected name already exists and, if so, then the\n        tab with that name is closed.\n\n        Unlike save_as(), this calls write() directly instead of using save().\n        The current file and tab aren't changed at all.  The copied file is\n        opened in a new tab.\n        "
        if index is None:
            index = self.get_stack_index()
        finfo = self.data[index]
        original_filename = finfo.filename
        filename = self.select_savename(original_filename)
        if filename:
            ao_index = self.has_filename(filename)
            if ao_index is not None and ao_index != index:
                if not self.close_file(ao_index):
                    return
                if ao_index < index:
                    index -= 1
            try:
                self._write_to_file(finfo, filename)
                self.plugin_load.emit(filename)
                return True
            except EnvironmentError as error:
                self.msgbox = QMessageBox(QMessageBox.Critical, _('Save Error'), _("<b>Unable to save file '%s'</b><br><br>Error message:<br>%s") % (osp.basename(finfo.filename), str(error)), parent=self)
                self.msgbox.exec_()
        else:
            return False

    def save_all(self, save_new_files=True):
        if False:
            while True:
                i = 10
        'Save all opened files.\n\n        Iterate through self.data and call save() on any modified files.\n        '
        all_saved = True
        for index in range(self.get_stack_count()):
            if self.data[index].editor.document().isModified():
                all_saved &= self.save(index, save_new_files=save_new_files)
        return all_saved

    def start_stop_analysis_timer(self):
        if False:
            i = 10
            return i + 15
        self.is_analysis_done = False
        self.analysis_timer.stop()
        self.analysis_timer.start()

    def analyze_script(self, index=None):
        if False:
            for i in range(10):
                print('nop')
        'Analyze current script for TODOs.'
        if self.is_analysis_done:
            return
        if index is None:
            index = self.get_stack_index()
        if self.data and len(self.data) > index:
            finfo = self.data[index]
            if self.todolist_enabled:
                finfo.run_todo_finder()
        self.is_analysis_done = True

    def set_todo_results(self, filename, todo_results):
        if False:
            while True:
                i = 10
        'Synchronize todo results between editorstacks'
        index = self.has_filename(filename)
        if index is None:
            return
        self.data[index].set_todo_results(todo_results)

    def get_todo_results(self):
        if False:
            return 10
        if self.data:
            return self.data[self.get_stack_index()].todo_results

    def current_changed(self, index):
        if False:
            i = 10
            return i + 15
        'Stack index has changed'
        editor = self.get_current_editor()
        if index != -1:
            editor.setFocus()
            logger.debug('Set focus to: %s' % editor.filename)
        else:
            self.reset_statusbar.emit()
        self.opened_files_list_changed.emit()
        self.stack_history.refresh()
        self.stack_history.remove_and_append(index)
        self.sig_codeeditor_changed.emit(editor)
        try:
            logger.debug('Current changed: %d - %s' % (index, self.data[index].editor.filename))
        except IndexError:
            pass
        self.update_plugin_title.emit()
        self.find_widget.set_editor(editor, refresh=False)
        self.find_widget.highlight_matches()
        self.find_widget.update_matches()
        if editor is not None:
            try:
                (line, col) = editor.get_cursor_line_column()
                self.current_file_changed.emit(self.data[index].filename, editor.get_position('cursor'), line, col)
            except IndexError:
                pass

    def _get_previous_file_index(self):
        if False:
            return 10
        'Return the penultimate element of the stack history.'
        try:
            return self.stack_history[-2]
        except IndexError:
            return None

    def tab_navigation_mru(self, forward=True):
        if False:
            return 10
        '\n        Tab navigation with "most recently used" behaviour.\n\n        It\'s fired when pressing \'go to previous file\' or \'go to next file\'\n        shortcuts.\n\n        forward:\n            True: move to next file\n            False: move to previous file\n        '
        self.tabs_switcher = TabSwitcherWidget(self, self.stack_history, self.tabs)
        self.tabs_switcher.show()
        self.tabs_switcher.select_row(1 if forward else -1)
        self.tabs_switcher.setFocus()

    def focus_changed(self):
        if False:
            i = 10
            return i + 15
        'Editor focus has changed'
        fwidget = QApplication.focusWidget()
        for finfo in self.data:
            if fwidget is finfo.editor:
                if finfo.editor.operation_in_progress:
                    self.spinner.start()
                else:
                    self.spinner.stop()
                self.refresh()
        self.editor_focus_changed.emit()

    def _refresh_outlineexplorer(self, index=None, update=True, clear=False):
        if False:
            print('Hello World!')
        'Refresh outline explorer panel'
        oe = self.outlineexplorer
        if oe is None:
            return
        if index is None:
            index = self.get_stack_index()
        if self.data and len(self.data) > index:
            finfo = self.data[index]
            oe.setEnabled(True)
            oe.set_current_editor(finfo.editor.oe_proxy, update=update, clear=clear)
            if index != self.get_stack_index():
                self._refresh_outlineexplorer(update=False)
                return
        self._sync_outlineexplorer_file_order()

    def _sync_outlineexplorer_file_order(self):
        if False:
            print('Hello World!')
        '\n        Order the root file items of the outline explorer as in the tabbar\n        of the current EditorStack.\n        '
        if self.outlineexplorer is not None:
            self.outlineexplorer.treewidget.set_editor_ids_order([finfo.editor.get_document_id() for finfo in self.data])

    def __refresh_statusbar(self, index):
        if False:
            print('Hello World!')
        'Refreshing statusbar widgets'
        if self.data and len(self.data) > index:
            finfo = self.data[index]
            self.encoding_changed.emit(finfo.encoding)
            (line, index) = finfo.editor.get_cursor_line_column()
            self.sig_editor_cursor_position_changed.emit(line, index)

    def __refresh_readonly(self, index):
        if False:
            i = 10
            return i + 15
        if self.data and len(self.data) > index:
            finfo = self.data[index]
            read_only = not QFileInfo(finfo.filename).isWritable()
            if not osp.isfile(finfo.filename):
                read_only = False
            elif os.name == 'nt':
                try:
                    fd = os.open(finfo.filename, os.O_RDWR)
                    os.close(fd)
                except (IOError, OSError):
                    read_only = True
            finfo.editor.setReadOnly(read_only)
            self.readonly_changed.emit(read_only)

    def __check_file_status(self, index):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check if file has been changed in any way outside Spyder.\n\n        Notes\n        -----\n        Possible ways are:\n        * The file was removed, moved or renamed outside Spyder.\n        * The file was modified outside Spyder.\n        '
        if self.__file_status_flag:
            return
        self.__file_status_flag = True
        if len(self.data) <= index:
            index = self.get_stack_index()
        finfo = self.data[index]
        name = osp.basename(finfo.filename)
        if finfo.newly_created:
            pass
        elif not osp.isfile(finfo.filename):
            self.msgbox = QMessageBox(QMessageBox.Warning, self.title, _('The file <b>%s</b> is unavailable.<br><br>It may have been removed, moved or renamed outside Spyder.<br><br>Do you want to close it?') % name, QMessageBox.Yes | QMessageBox.No, self)
            answer = self.msgbox.exec_()
            if answer == QMessageBox.Yes:
                self.close_file(index, force=True)
            else:
                finfo.newly_created = True
                finfo.editor.document().setModified(True)
                self.modification_changed(index=index)
        else:
            lastm = QFileInfo(finfo.filename).lastModified()
            if str(lastm.toString()) != str(finfo.lastmodified.toString()):
                try:
                    if finfo.editor.document().isModified():
                        self.msgbox = QMessageBox(QMessageBox.Question, self.title, _('The file <b>{}</b> has been modified outside Spyder.<br><br>Do you want to reload it and lose all your changes?').format(name), QMessageBox.Yes | QMessageBox.No, self)
                        answer = self.msgbox.exec_()
                        if answer == QMessageBox.Yes:
                            self.reload(index)
                        else:
                            finfo.lastmodified = lastm
                    else:
                        self.reload(index)
                except Exception:
                    self.msgbox = QMessageBox(QMessageBox.Warning, self.title, _('The file <b>{}</b> has been modified outside Spyder but it was not possible to reload it.<br><br>Therefore, it will be closed.').format(name), QMessageBox.Ok, self)
                    self.msgbox.exec_()
                    self.close_file(index, force=True)
        self.__file_status_flag = False

    def __modify_stack_title(self):
        if False:
            while True:
                i = 10
        for (index, finfo) in enumerate(self.data):
            state = finfo.editor.document().isModified()
            self.set_stack_title(index, state)

    def refresh(self, index=None):
        if False:
            i = 10
            return i + 15
        'Refresh tabwidget'
        logger.debug('Refresh EditorStack')
        if index is None:
            index = self.get_stack_index()
        if self.get_stack_count():
            index = self.get_stack_index()
            finfo = self.data[index]
            editor = finfo.editor
            editor.setFocus()
            self._refresh_outlineexplorer(index, update=False)
            self.sig_update_code_analysis_actions.emit()
            self.__refresh_statusbar(index)
            self.__refresh_readonly(index)
            self.__check_file_status(index)
            self.__modify_stack_title()
            self.update_plugin_title.emit()
        else:
            editor = None
        self.modification_changed()
        self.find_widget.set_editor(editor, refresh=False)

    def modification_changed(self, state=None, index=None, editor_id=None):
        if False:
            i = 10
            return i + 15
        "\n        Current editor's modification state has changed\n        --> change tab title depending on new modification state\n        --> enable/disable save/save all actions\n        "
        if editor_id is not None:
            for (index, _finfo) in enumerate(self.data):
                if id(_finfo.editor) == editor_id:
                    break
        self.opened_files_list_changed.emit()
        if index is None:
            index = self.get_stack_index()
        if index == -1:
            return
        finfo = self.data[index]
        if state is None:
            state = finfo.editor.document().isModified() or finfo.newly_created
        self.set_stack_title(index, state)
        self.save_action.setEnabled(state)
        self.refresh_save_all_action.emit()
        eol_chars = finfo.editor.get_line_separator()
        self.refresh_eol_chars(eol_chars)
        self.stack_history.refresh()

    def refresh_eol_chars(self, eol_chars):
        if False:
            for i in range(10):
                print('nop')
        os_name = sourcecode.get_os_name_from_eol_chars(eol_chars)
        self.sig_refresh_eol_chars.emit(os_name)

    def reload(self, index):
        if False:
            for i in range(10):
                print('nop')
        'Reload file from disk.'
        finfo = self.data[index]
        logger.debug('Reloading {}'.format(finfo.filename))
        (txt, finfo.encoding) = encoding.read(finfo.filename)
        finfo.lastmodified = QFileInfo(finfo.filename).lastModified()
        position = finfo.editor.get_position('cursor')
        finfo.editor.set_text(txt)
        finfo.editor.document().setModified(False)
        self.autosave.file_hashes[finfo.filename] = hash(txt)
        finfo.editor.set_cursor_position(position)
        finfo.editor.rehighlight()

    def revert(self):
        if False:
            return 10
        'Revert file from disk.'
        index = self.get_stack_index()
        finfo = self.data[index]
        logger.debug('Reverting {}'.format(finfo.filename))
        filename = finfo.filename
        if finfo.editor.document().isModified():
            self.msgbox = QMessageBox(QMessageBox.Warning, self.title, _('All changes to file <b>%s</b> will be lost.<br>Do you want to revert it from disk?') % osp.basename(filename), QMessageBox.Yes | QMessageBox.No, self)
            answer = self.msgbox.exec_()
            if answer != QMessageBox.Yes:
                return
        try:
            self.reload(index)
        except FileNotFoundError:
            QMessageBox.critical(self, _('Error'), _("File <b>%s</b> is not saved on disk, so it can't be reverted.") % osp.basename(filename), QMessageBox.Ok)

    def create_new_editor(self, fname, enc, txt, set_current, new=False, cloned_from=None, add_where='end'):
        if False:
            print('Hello World!')
        '\n        Create a new editor instance\n        Returns finfo object (instead of editor as in previous releases)\n        '
        editor = codeeditor.CodeEditor(self)
        editor.go_to_definition.connect(lambda fname, line, column: self.sig_go_to_definition.emit(fname, line, column))
        finfo = FileInfo(fname, enc, editor, new, self.threadmanager)
        self.add_to_data(finfo, set_current, add_where)
        finfo.sig_send_to_help.connect(self.send_to_help)
        finfo.sig_show_object_info.connect(self.inspect_current_object)
        finfo.todo_results_changed.connect(self.todo_results_changed)
        finfo.edit_goto.connect(lambda fname, lineno, name: self.edit_goto.emit(fname, lineno, name))
        finfo.sig_save_bookmarks.connect(lambda s1, s2: self.sig_save_bookmarks.emit(s1, s2))
        editor.sig_new_file.connect(self.sig_new_file)
        editor.sig_process_code_analysis.connect(self.sig_update_code_analysis_actions)
        editor.sig_refresh_formatting.connect(self.sig_refresh_formatting)
        editor.sig_save_requested.connect(self.save)
        language = get_file_language(fname, txt)
        editor.setup_editor(linenumbers=self.linenumbers_enabled, show_blanks=self.blanks_enabled, underline_errors=self.underline_errors_enabled, scroll_past_end=self.scrollpastend_enabled, edge_line=self.edgeline_enabled, edge_line_columns=self.edgeline_columns, language=language, markers=self.has_markers(), font=self.default_font, color_scheme=self.color_scheme, wrap=self.wrap_enabled, tab_mode=self.tabmode_enabled, strip_mode=self.stripmode_enabled, intelligent_backspace=self.intelligent_backspace_enabled, automatic_completions=self.automatic_completions_enabled, automatic_completions_after_chars=self.automatic_completion_chars, code_snippets=self.code_snippets_enabled, completions_hint=self.completions_hint_enabled, completions_hint_after_ms=self.completions_hint_after_ms, hover_hints=self.hover_hints_enabled, highlight_current_line=self.highlight_current_line_enabled, highlight_current_cell=self.highlight_current_cell_enabled, occurrence_highlighting=self.occurrence_highlighting_enabled, occurrence_timeout=self.occurrence_highlighting_timeout, close_parentheses=self.close_parentheses_enabled, close_quotes=self.close_quotes_enabled, add_colons=self.add_colons_enabled, auto_unindent=self.auto_unindent_enabled, indent_chars=self.indent_chars, tab_stop_width_spaces=self.tab_stop_width_spaces, cloned_from=cloned_from, filename=fname, show_class_func_dropdown=self.show_class_func_dropdown, indent_guides=self.indent_guides, folding=self.code_folding_enabled, remove_trailing_spaces=self.always_remove_trailing_spaces, remove_trailing_newlines=self.remove_trailing_newlines, add_newline=self.add_newline, format_on_save=self.format_on_save)
        if cloned_from is None:
            editor.set_text(txt)
            editor.document().setModified(False)
        finfo.text_changed_at.connect(lambda fname, position: self.text_changed_at.emit(fname, position))
        editor.sig_cursor_position_changed.connect(self.editor_cursor_position_changed)
        editor.textChanged.connect(self.start_stop_analysis_timer)
        for (panel_class, args, kwargs, position) in self.external_panels:
            self.register_panel(panel_class, *args, position=position, **kwargs)

        def perform_completion_request(lang, method, params):
            if False:
                print('Hello World!')
            self.sig_perform_completion_request.emit(lang, method, params)
        editor.sig_perform_completion_request.connect(perform_completion_request)
        editor.sig_start_operation_in_progress.connect(self.spinner.start)
        editor.sig_stop_operation_in_progress.connect(self.spinner.stop)
        editor.modificationChanged.connect(lambda state: self.modification_changed(state, editor_id=id(editor)))
        editor.focus_in.connect(self.focus_changed)
        editor.zoom_in.connect(self.zoom_in)
        editor.zoom_out.connect(self.zoom_out)
        editor.zoom_reset.connect(self.zoom_reset)
        editor.sig_eol_chars_changed.connect(lambda eol_chars: self.refresh_eol_chars(eol_chars))
        editor.sig_next_cursor.connect(self.sig_next_cursor)
        editor.sig_prev_cursor.connect(self.sig_prev_cursor)
        self.find_widget.set_editor(editor)
        self.refresh_file_dependent_actions.emit()
        self.modification_changed(index=self.data.index(finfo))
        editor.oe_proxy = OutlineExplorerProxyEditor(editor, editor.filename)
        if self.outlineexplorer is not None:
            self.outlineexplorer.register_editor(editor.oe_proxy)
        if cloned_from is not None:
            cloned_from.oe_proxy.sig_outline_explorer_data_changed.connect(editor.oe_proxy.update_outline_info)
            cloned_from.oe_proxy.sig_outline_explorer_data_changed.connect(editor._update_classfuncdropdown)
            cloned_from.oe_proxy.sig_start_outline_spinner.connect(editor.oe_proxy.emit_request_in_progress)
            cloned_from.document_did_change()
        editor.run_pygments_highlighter()
        options = {'language': editor.language, 'filename': editor.filename, 'codeeditor': editor}
        self.sig_open_file.emit(options)
        self.sig_codeeditor_created.emit(editor)
        if self.get_stack_index() == 0:
            self.current_changed(0)
        return finfo

    def editor_cursor_position_changed(self, line, index):
        if False:
            return 10
        'Cursor position of one of the editor in the stack has changed'
        self.sig_editor_cursor_position_changed.emit(line, index)

    @Slot(str, str, bool)
    def send_to_help(self, name, signature, force=False):
        if False:
            i = 10
            return i + 15
        'qstr1: obj_text, qstr2: argpspec, qstr3: note, qstr4: doc_text'
        if not force and (not self.help_enabled):
            return
        editor = self.get_current_editor()
        language = editor.language.lower()
        signature = to_text_string(signature)
        signature = unicodedata.normalize('NFKD', signature)
        parts = signature.split('\n\n')
        definition = parts[0]
        documentation = '\n\n'.join(parts[1:])
        args = ''
        if '(' in definition and language == 'python':
            args = definition[definition.find('('):]
        else:
            documentation = signature
        doc = {'obj_text': '', 'name': name, 'argspec': args, 'note': '', 'docstring': documentation, 'force_refresh': force, 'path': editor.filename}
        self.sig_help_requested.emit(doc)

    def new(self, filename, encoding, text, default_content=False, empty=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create new filename with *encoding* and *text*\n        '
        finfo = self.create_new_editor(filename, encoding, text, set_current=False, new=True)
        finfo.editor.set_cursor_position('eof')
        if not empty:
            finfo.editor.insert_text(os.linesep)
        if default_content:
            finfo.default = True
            finfo.editor.document().setModified(False)
        return finfo

    def load(self, filename, set_current=True, add_where='end', processevents=True):
        if False:
            print('Hello World!')
        '\n        Load filename, create an editor instance and return it.\n\n        This also sets the hash of the loaded file in the autosave component.\n        '
        filename = osp.abspath(to_text_string(filename))
        if processevents:
            self.starting_long_process.emit(_('Loading %s...') % filename)
        try:
            (text, enc) = encoding.read(filename)
        except Exception:
            return
        self.autosave.file_hashes[filename] = hash(text)
        finfo = self.create_new_editor(filename, enc, text, set_current, add_where=add_where)
        index = self.data.index(finfo)
        if processevents:
            self.ending_long_process.emit('')
        if self.isVisible() and self.checkeolchars_enabled and sourcecode.has_mixed_eol_chars(text):
            name = osp.basename(filename)
            self.msgbox = QMessageBox(QMessageBox.Warning, self.title, _('<b>%s</b> contains mixed end-of-line characters.<br>Spyder will fix this automatically.') % name, QMessageBox.Ok, self)
            self.msgbox.exec_()
            self.set_os_eol_chars(index)
        self.is_analysis_done = False
        self.analyze_script(index)
        finfo.editor.set_sync_symbols_and_folding_timeout()
        finfo.editor.unhighlight_current_line()
        if self.highlight_current_line_enabled:
            finfo.editor.highlight_current_line()
        return finfo

    def set_os_eol_chars(self, index=None, osname=None):
        if False:
            print('Hello World!')
        "\n        Sets the EOL character(s) based on the operating system.\n\n        If `osname` is None, then the default line endings for the current\n        operating system will be used.\n\n        `osname` can be one of: 'posix', 'nt', 'mac'.\n        "
        if osname is None:
            if os.name == 'nt':
                osname = 'nt'
            elif sys.platform == 'darwin':
                osname = 'mac'
            else:
                osname = 'posix'
        if index is None:
            index = self.get_stack_index()
        finfo = self.data[index]
        eol_chars = sourcecode.get_eol_chars_from_os_name(osname)
        logger.debug(f'Set OS eol chars {eol_chars} for file {finfo.filename}')
        finfo.editor.set_eol_chars(eol_chars=eol_chars)
        finfo.editor.document().setModified(True)

    def remove_trailing_spaces(self, index=None):
        if False:
            return 10
        'Remove trailing spaces'
        if index is None:
            index = self.get_stack_index()
        finfo = self.data[index]
        logger.debug(f'Remove trailing spaces for file {finfo.filename}')
        finfo.editor.trim_trailing_spaces()

    def trim_trailing_newlines(self, index=None):
        if False:
            return 10
        if index is None:
            index = self.get_stack_index()
        finfo = self.data[index]
        logger.debug(f'Trim trailing new lines for file {finfo.filename}')
        finfo.editor.trim_trailing_newlines()

    def add_newline_to_file(self, index=None):
        if False:
            return 10
        if index is None:
            index = self.get_stack_index()
        finfo = self.data[index]
        logger.debug(f'Add new line to file {finfo.filename}')
        finfo.editor.add_newline_to_file()

    def fix_indentation(self, index=None):
        if False:
            i = 10
            return i + 15
        'Replace tab characters by spaces'
        if index is None:
            index = self.get_stack_index()
        finfo = self.data[index]
        logger.debug(f'Fix indentation for file {finfo.filename}')
        finfo.editor.fix_indentation()

    def format_document_or_selection(self, index=None):
        if False:
            while True:
                i = 10
        if index is None:
            index = self.get_stack_index()
        finfo = self.data[index]
        logger.debug(f'Run formatting in file {finfo.filename}')
        finfo.editor.format_document_or_range()

    def _get_lines_cursor(self, direction):
        if False:
            i = 10
            return i + 15
        ' Select and return all lines from cursor in given direction'
        editor = self.get_current_editor()
        finfo = self.get_current_finfo()
        enc = finfo.encoding
        cursor = editor.textCursor()
        cursor.movePosition(QTextCursor.StartOfLine)
        if direction == 'up':
            cursor.movePosition(QTextCursor.Start, QTextCursor.KeepAnchor)
        elif direction == 'down':
            cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
        selection = editor.get_selection_as_executable_code(cursor)
        if selection:
            (code_text, off_pos, line_col_pos) = selection
            return (code_text.rstrip(), off_pos, line_col_pos, enc)

    def get_to_current_line(self):
        if False:
            return 10
        '\n        Get all lines from the beginning up to, but not including, current\n        line.\n        '
        return self._get_lines_cursor(direction='up')

    def get_from_current_line(self):
        if False:
            return 10
        '\n        Get all lines from and including the current line to the end of\n        the document.\n        '
        return self._get_lines_cursor(direction='down')

    def get_selection(self):
        if False:
            i = 10
            return i + 15
        '\n        Get selected text or current line in console.\n\n        If some text is selected, then execute that text in console.\n\n        If no text is selected, then execute current line, unless current line\n        is empty. Then, advance cursor to next line. If cursor is on last line\n        and that line is not empty, then add a new blank line and move the\n        cursor there. If cursor is on last line and that line is empty, then do\n        not move cursor.\n        '
        editor = self.get_current_editor()
        encoding = self.get_current_finfo().encoding
        selection = editor.get_selection_as_executable_code()
        if selection:
            (text, off_pos, line_col_pos) = selection
            return (text, off_pos, line_col_pos, encoding)
        (line_col_from, line_col_to) = editor.get_current_line_bounds()
        (line_off_from, line_off_to) = editor.get_current_line_offsets()
        line = editor.get_current_line()
        text = line.lstrip()
        return (text, (line_off_from, line_off_to), (line_col_from, line_col_to), encoding)

    def advance_line(self):
        if False:
            for i in range(10):
                print('nop')
        'Advance to the next line.'
        editor = self.get_current_editor()
        if editor.is_cursor_on_last_line() and editor.get_current_line().strip():
            editor.append(editor.get_line_separator())
        editor.move_cursor_to_next('line', 'down')

    def get_current_cell(self):
        if False:
            i = 10
            return i + 15
        'Get current cell attributes.'
        (text, block, off_pos, line_col_pos) = self.get_current_editor().get_cell_as_executable_code()
        encoding = self.get_current_finfo().encoding
        name = cell_name(block)
        return (text, off_pos, line_col_pos, name, encoding)

    def advance_cell(self, reverse=False):
        if False:
            while True:
                i = 10
        'Advance to the next cell.\n\n        reverse = True --> go to previous cell.\n        '
        if not reverse:
            move_func = self.get_current_editor().go_to_next_cell
        else:
            move_func = self.get_current_editor().go_to_previous_cell
        move_func()

    def get_last_cell(self):
        if False:
            for i in range(10):
                print('nop')
        'Run the previous cell again.'
        if self.last_cell_call is None:
            return
        (filename, cell_name) = self.last_cell_call
        index = self.has_filename(filename)
        if index is None:
            return
        editor = self.data[index].editor
        try:
            (text, off_pos, col_pos) = editor.get_cell_code_and_position(cell_name)
            encoding = self.get_current_finfo().encoding
        except RuntimeError:
            return
        return (text, off_pos, col_pos, cell_name, encoding)

    def dragEnterEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        '\n        Reimplemented Qt method.\n\n        Inform Qt about the types of data that the widget accepts.\n        '
        logger.debug('dragEnterEvent was received')
        source = event.mimeData()
        has_urls = source.hasUrls()
        has_text = source.hasText()
        urls = source.urls()
        all_urls = mimedata2url(source)
        logger.debug('Drag event source has_urls: {}'.format(has_urls))
        logger.debug('Drag event source urls: {}'.format(urls))
        logger.debug('Drag event source all_urls: {}'.format(all_urls))
        logger.debug('Drag event source has_text: {}'.format(has_text))
        if has_urls and urls and all_urls:
            text = [encoding.is_text_file(url) for url in all_urls]
            logger.debug('Accept proposed action?: {}'.format(any(text)))
            if any(text):
                event.acceptProposedAction()
            else:
                event.ignore()
        elif source.hasText():
            event.acceptProposedAction()
        elif os.name == 'nt':
            logger.debug('Accept proposed action on Windows')
            event.acceptProposedAction()
        else:
            logger.debug('Ignore drag event')
            event.ignore()

    def dropEvent(self, event):
        if False:
            return 10
        '\n        Reimplement Qt method.\n\n        Unpack dropped data and handle it.\n        '
        logger.debug('dropEvent was received')
        source = event.mimeData()
        if source.hasUrls() and mimedata2url(source):
            files = mimedata2url(source)
            files = [f for f in files if encoding.is_text_file(f)]
            files = set(files or [])
            for fname in files:
                self.plugin_load.emit(fname)
        elif source.hasText():
            editor = self.get_current_editor()
            if editor is not None:
                editor.insert_text(source.text())
        else:
            event.ignore()
        event.acceptProposedAction()

    def register_panel(self, panel_class, *args, position=Panel.Position.LEFT, **kwargs):
        if False:
            return 10
        'Register a panel in all codeeditors.'
        if (panel_class, args, kwargs, position) not in self.external_panels:
            self.external_panels.append((panel_class, args, kwargs, position))
        for finfo in self.data:
            cur_panel = finfo.editor.panels.register(panel_class(*args, **kwargs), position=position)
            if not cur_panel.isVisible():
                cur_panel.setVisible(True)