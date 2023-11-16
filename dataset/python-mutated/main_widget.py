"""
Main Console widget.
"""
import logging
import os
import os.path as osp
import sys
from qtpy.compat import getopenfilename
from qtpy.QtCore import Qt, Signal, Slot
from qtpy.QtWidgets import QAction, QInputDialog, QLineEdit, QVBoxLayout
from qtpy import PYSIDE2
from spyder.api.exceptions import SpyderAPIError
from spyder.api.plugin_registration.registry import PLUGIN_REGISTRY
from spyder.api.translations import _
from spyder.api.widgets.main_widget import PluginMainWidget
from spyder.api.config.decorators import on_conf_change
from spyder.utils.installers import InstallerInternalError
from spyder.config.base import DEV, get_debug_level
from spyder.plugins.console.widgets.internalshell import InternalShell
from spyder.py3compat import to_text_string
from spyder.utils.environ import EnvDialog
from spyder.utils.misc import get_error_match, getcwd_or_home, remove_backslashes
from spyder.utils.qthelpers import DialogManager, mimedata2url
from spyder.widgets.collectionseditor import CollectionsEditor
from spyder.widgets.findreplace import FindReplace
from spyder.widgets.reporterror import SpyderErrorDialog
logger = logging.getLogger(__name__)

class ConsoleWidgetActions:
    Environment = 'environment_action'
    ExternalEditor = 'external_editor_action'
    MaxLineCount = 'max_line_count_action'
    Quit = 'Quit'
    Run = 'run_action'
    SysPath = 'sys_path_action'
    ToggleCodeCompletion = 'toggle_code_completion_action'
    ToggleWrap = 'toggle_wrap_action'

class ConsoleWidgetMenus:
    InternalSettings = 'internal_settings_submenu'

class ConsoleWidgetOptionsMenuSections:
    Run = 'run_section'
    Quit = 'quit_section'

class ConsoleWidgetInternalSettingsSubMenuSections:
    Main = 'main'

class ConsoleWidget(PluginMainWidget):
    sig_edit_goto_requested = Signal(str, int, str)
    sig_focus_changed = Signal()
    sig_refreshed = Signal()
    sig_show_status_requested = Signal(str)
    sig_help_requested = Signal(dict)
    "\n    This signal is emitted to request help on a given object `name`.\n\n    Parameters\n    ----------\n    help_data: dict\n        Example `{'name': str, 'ignore_unknown': bool}`.\n    "

    def __init__(self, name, plugin, parent=None):
        if False:
            print('Hello World!')
        super().__init__(name, plugin, parent)
        logger.info('Initializing...')
        self.error_traceback = ''
        self.dismiss_error = False
        message = _('Spyder Internal Console\n\nThis console is used to report application\ninternal errors and to inspect Spyder\ninternals with the following commands:\n  spy.app, spy.window, dir(spy)\n\nPlease do not use it to run your code\n\n')
        cli_options = plugin.get_command_line_options()
        profile = cli_options.profile
        multithreaded = cli_options.multithreaded
        self.dialog_manager = DialogManager()
        self.error_dlg = None
        self.shell = InternalShell(commands=[], message=message, max_line_count=self.get_conf('max_line_count'), profile=profile, multithreaded=multithreaded)
        self.find_widget = FindReplace(self)
        self.setAcceptDrops(True)
        self.find_widget.set_editor(self.shell)
        self.find_widget.hide()
        self.shell.toggle_wrap_mode(self.get_conf('wrap'))
        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.addWidget(self.shell)
        layout.addWidget(self.find_widget)
        self.setLayout(layout)
        self.shell.sig_help_requested.connect(self.sig_help_requested)
        self.shell.sig_exception_occurred.connect(self.handle_exception)
        self.shell.sig_focus_changed.connect(self.sig_focus_changed)
        self.shell.sig_go_to_error_requested.connect(self.go_to_error)
        self.shell.sig_redirect_stdio_requested.connect(self.sig_redirect_stdio_requested)
        self.shell.sig_refreshed.connect(self.sig_refreshed)
        self.shell.sig_show_status_requested.connect(lambda msg: self.sig_show_status_message.emit(msg, 0))

    def get_title(self):
        if False:
            i = 10
            return i + 15
        return _('Internal console')

    def setup(self):
        if False:
            return 10
        self.quit_action = self.create_action(ConsoleWidgetActions.Quit, text=_('&Quit'), tip=_('Quit'), icon=self.create_icon('exit'), triggered=self.sig_quit_requested, context=Qt.ApplicationShortcut, shortcut_context='_', register_shortcut=True, menurole=QAction.QuitRole)
        run_action = self.create_action(ConsoleWidgetActions.Run, text=_('&Run...'), tip=_('Run a Python file'), icon=self.create_icon('run_small'), triggered=self.run_script)
        environ_action = self.create_action(ConsoleWidgetActions.Environment, text=_('Environment variables...'), tip=_('Show and edit environment variables (for current session)'), icon=self.create_icon('environ'), triggered=self.show_env)
        syspath_action = self.create_action(ConsoleWidgetActions.SysPath, text=_('Show sys.path contents...'), tip=_('Show (read-only) sys.path'), icon=self.create_icon('syspath'), triggered=self.show_syspath)
        buffer_action = self.create_action(ConsoleWidgetActions.MaxLineCount, text=_('Buffer...'), tip=_('Set maximum line count'), triggered=self.change_max_line_count)
        exteditor_action = self.create_action(ConsoleWidgetActions.ExternalEditor, text=_('External editor path...'), tip=_('Set external editor executable path'), triggered=self.change_exteditor)
        wrap_action = self.create_action(ConsoleWidgetActions.ToggleWrap, text=_('Wrap lines'), toggled=lambda val: self.set_conf('wrap', val), initial=self.get_conf('wrap'))
        codecompletion_action = self.create_action(ConsoleWidgetActions.ToggleCodeCompletion, text=_('Automatic code completion'), toggled=lambda val: self.set_conf('codecompletion/auto', val), initial=self.get_conf('codecompletion/auto'))
        internal_settings_menu = self.create_menu(ConsoleWidgetMenus.InternalSettings, _('Internal console settings'), icon=self.create_icon('tooloptions'))
        for item in [buffer_action, wrap_action, codecompletion_action, exteditor_action]:
            self.add_item_to_menu(item, menu=internal_settings_menu, section=ConsoleWidgetInternalSettingsSubMenuSections.Main)
        options_menu = self.get_options_menu()
        for item in [run_action, environ_action, syspath_action, internal_settings_menu]:
            self.add_item_to_menu(item, menu=options_menu, section=ConsoleWidgetOptionsMenuSections.Run)
        self.add_item_to_menu(self.quit_action, menu=options_menu, section=ConsoleWidgetOptionsMenuSections.Quit)
        self.shell.set_external_editor(self.get_conf('external_editor/path'), '')

    @on_conf_change(option='max_line_count')
    def max_line_count_update(self, value):
        if False:
            i = 10
            return i + 15
        self.shell.setMaximumBlockCount(value)

    @on_conf_change(option='wrap')
    def wrap_mode_update(self, value):
        if False:
            return 10
        self.shell.toggle_wrap_mode(value)

    @on_conf_change(option='external_editor/path')
    def external_editor_update(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.shell.set_external_editor(value, '')

    def update_actions(self):
        if False:
            return 10
        pass

    def get_focus_widget(self):
        if False:
            print('Hello World!')
        return self.shell

    def dragEnterEvent(self, event):
        if False:
            return 10
        '\n        Reimplement Qt method.\n\n        Inform Qt about the types of data that the widget accepts.\n        '
        source = event.mimeData()
        if source.hasUrls():
            if mimedata2url(source):
                event.acceptProposedAction()
            else:
                event.ignore()
        elif source.hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if False:
            while True:
                i = 10
        '\n        Reimplement Qt method.\n\n        Unpack dropped data and handle it.\n        '
        source = event.mimeData()
        if source.hasUrls():
            pathlist = mimedata2url(source)
            self.shell.drop_pathlist(pathlist)
        elif source.hasText():
            lines = to_text_string(source.text())
            self.shell.set_cursor_position('eof')
            self.shell.execute_lines(lines)
        event.acceptProposedAction()

    def start_interpreter(self, namespace):
        if False:
            while True:
                i = 10
        '\n        Start internal console interpreter.\n        '
        self.shell.start_interpreter(namespace)

    def set_historylog(self, historylog):
        if False:
            i = 10
            return i + 15
        '\n        Bind historylog instance to this console.\n\n        Not used anymore since v2.0.\n        '
        historylog.add_history(self.shell.history_filename)
        self.shell.sig_append_to_history_requested.connect(historylog.append_to_history)

    def set_help(self, help_plugin):
        if False:
            i = 10
            return i + 15
        '\n        Bind help instance to this console.\n        '
        self.shell.help = help_plugin

    def report_issue(self):
        if False:
            for i in range(10):
                print('nop')
        'Report an issue with the SpyderErrorDialog.'
        self._report_dlg = SpyderErrorDialog(self, is_report=True)
        self._report_dlg.set_color_scheme(self.get_conf('selected', section='appearance'))
        self._report_dlg.show()

    @Slot(dict)
    def handle_exception(self, error_data, sender=None):
        if False:
            while True:
                i = 10
        '\n        Exception occurred in the internal console.\n\n        Show a QDialog or the internal console to warn the user.\n\n        Handle any exception that occurs during Spyder usage.\n\n        Parameters\n        ----------\n        error_data: dict\n            The dictionary containing error data. The expected keys are:\n            >>> error_data= {\n                "text": str,\n                "is_traceback": bool,\n                "repo": str,\n                "title": str,\n                "label": str,\n                "steps": str,\n            }\n        sender: spyder.api.plugins.SpyderPluginV2, optional\n            The sender plugin. Default is None.\n\n        Notes\n        -----\n        The `is_traceback` key indicates if `text` contains plain text or a\n        Python error traceback.\n\n        The `title` and `repo` keys indicate how the error data should\n        customize the report dialog and Github error submission.\n\n        The `label` and `steps` keys allow customizing the content of the\n        error dialog.\n        '
        text = error_data.get('text', None)
        is_traceback = error_data.get('is_traceback', False)
        title = error_data.get('title', '')
        label = error_data.get('label', '')
        steps = error_data.get('steps', '')
        if not text and (not is_traceback) and (self.error_dlg is None) or self.dismiss_error:
            return
        InstallerInternalError(title + text)
        internal_plugins = PLUGIN_REGISTRY.internal_plugins
        is_internal_plugin = True
        if sender is not None:
            sender_name = getattr(sender, 'NAME', getattr(sender, 'CONF_SECTION'))
            is_internal_plugin = sender_name in internal_plugins
        repo = 'spyder-ide/spyder'
        if not is_internal_plugin:
            repo = error_data.get('repo', None)
            if repo is None:
                raise SpyderAPIError(f"External plugin '{sender_name}' does not define 'repo' key in the 'error_data' dictionary in the form my-org/my-repo (only Github is supported).")
            if repo == 'spyder-ide/spyder':
                raise SpyderAPIError(f"External plugin '{sender_name}' 'repo' key needs to be different from the main Spyder repo.")
        if self.get_conf('show_internal_errors', section='main'):
            if self.error_dlg is None:
                self.error_dlg = SpyderErrorDialog(self)
                self.error_dlg.set_color_scheme(self.get_conf('selected', section='appearance'))
                self.error_dlg.rejected.connect(self.remove_error_dlg)
                self.error_dlg.details.sig_go_to_error_requested.connect(self.go_to_error)
            self.error_dlg.set_github_repo_org(repo)
            if title:
                self.error_dlg.set_title(title)
                self.error_dlg.title.setEnabled(False)
            if label:
                self.error_dlg.main_label.setText(label)
                self.error_dlg.submit_btn.setEnabled(True)
            if steps:
                self.error_dlg.steps_text.setText(steps)
                self.error_dlg.set_require_minimum_length(False)
            self.error_dlg.append_traceback(text)
            self.error_dlg.show()
        elif DEV or get_debug_level():
            self.change_visibility(True, True)

    def close_error_dlg(self):
        if False:
            print('Hello World!')
        '\n        Close error dialog.\n        '
        if self.error_dlg:
            self.error_dlg.reject()

    def remove_error_dlg(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove error dialog.\n        '
        if self.error_dlg.dismiss_box.isChecked():
            self.dismiss_error = True
        if PYSIDE2:
            self.error_dlg.disconnect(None, None, None)
        else:
            self.error_dlg.disconnect()
        self.error_dlg = None

    @Slot()
    def show_env(self):
        if False:
            i = 10
            return i + 15
        '\n        Show environment variables.\n        '
        self.dialog_manager.show(EnvDialog(parent=self))

    def get_sys_path(self):
        if False:
            print('Hello World!')
        '\n        Return the `sys.path`.\n        '
        return sys.path

    @Slot()
    def show_syspath(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Show `sys.path`.\n        '
        editor = CollectionsEditor(parent=self)
        editor.setup(sys.path, title='sys.path', readonly=True, icon=self.create_icon('syspath'))
        self.dialog_manager.show(editor)

    @Slot()
    def run_script(self, filename=None, silent=False, args=None):
        if False:
            i = 10
            return i + 15
        '\n        Run a Python script.\n        '
        if filename is None:
            self.shell.interpreter.restore_stds()
            (filename, _selfilter) = getopenfilename(self, _('Run Python file'), getcwd_or_home(), _('Python files') + ' (*.py ; *.pyw ; *.ipy)')
            self.shell.interpreter.redirect_stds()
            if filename:
                os.chdir(osp.dirname(filename))
                filename = osp.basename(filename)
            else:
                return
        logger.debug('Running script with %s', args)
        filename = osp.abspath(filename)
        rbs = remove_backslashes
        command = '%runfile {} --args {}'.format(repr(rbs(filename)), repr(rbs(args)))
        self.change_visibility(True, True)
        self.shell.write(command + '\n')
        self.shell.run_command(command)

    def go_to_error(self, text):
        if False:
            return 10
        '\n        Go to error if relevant.\n        '
        match = get_error_match(to_text_string(text))
        if match:
            (fname, lnb) = match.groups()
            self.edit_script(fname, int(lnb))

    def edit_script(self, filename=None, goto=-1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Edit script.\n        '
        if filename is not None:
            self.shell.external_editor(filename, goto)
            self.sig_edit_goto_requested.emit(osp.abspath(filename), goto, '')

    def execute_lines(self, lines):
        if False:
            print('Hello World!')
        '\n        Execute lines and give focus to shell.\n        '
        self.shell.execute_lines(to_text_string(lines))
        self.shell.setFocus()

    @Slot()
    def change_max_line_count(self, value=None):
        if False:
            for i in range(10):
                print('nop')
        '"\n        Change maximum line count.\n        '
        valid = True
        if value is None:
            (value, valid) = QInputDialog.getInt(self, _('Buffer'), _('Maximum line count'), self.get_conf('max_line_count'), 0, 1000000)
        if valid:
            self.set_conf('max_line_count', value)

    @Slot()
    def change_exteditor(self, path=None):
        if False:
            while True:
                i = 10
        '\n        Change external editor path.\n        '
        valid = True
        if path is None:
            (path, valid) = QInputDialog.getText(self, _('External editor'), _('External editor executable path:'), QLineEdit.Normal, self.get_conf('external_editor/path'))
        if valid:
            self.set_conf('external_editor/path', to_text_string(path))

    def set_exit_function(self, func):
        if False:
            while True:
                i = 10
        '\n        Set the callback function to execute when the `exit_interpreter` is\n        called.\n        '
        self.shell.exitfunc = func

    def set_font(self, font):
        if False:
            i = 10
            return i + 15
        '\n        Set font of the internal shell.\n        '
        self.shell.set_font(font)

    def redirect_stds(self):
        if False:
            while True:
                i = 10
        '\n        Redirect stdout and stderr when using open file dialogs.\n        '
        self.shell.interpreter.redirect_stds()

    def restore_stds(self):
        if False:
            return 10
        '\n        Restore stdout and stderr when using open file dialogs.\n        '
        self.shell.interpreter.restore_stds()

    def set_namespace_item(self, name, item):
        if False:
            print('Hello World!')
        '\n        Add an object to the namespace dictionary of the internal console.\n        '
        self.shell.interpreter.namespace[name] = item

    def exit_interpreter(self):
        if False:
            while True:
                i = 10
        '\n        Exit the internal console interpreter.\n\n        This is equivalent to requesting the main application to quit.\n        '
        self.shell.exit_interpreter()