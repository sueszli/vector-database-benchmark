"""The main window of qutebrowser."""
import binascii
import base64
import itertools
import functools
from typing import List, MutableSequence, Optional, Tuple, cast
from qutebrowser.qt import machinery
from qutebrowser.qt.core import pyqtBoundSignal, pyqtSlot, QRect, QPoint, QTimer, Qt, QCoreApplication, QEventLoop, QByteArray
from qutebrowser.qt.widgets import QWidget, QVBoxLayout, QSizePolicy
from qutebrowser.qt.gui import QPalette
from qutebrowser.commands import runners
from qutebrowser.api import cmdutils
from qutebrowser.config import config, configfiles, stylesheet, websettings
from qutebrowser.utils import message, log, usertypes, qtutils, objreg, utils, jinja, debug
from qutebrowser.mainwindow import messageview, prompt
from qutebrowser.completion import completionwidget, completer
from qutebrowser.keyinput import modeman
from qutebrowser.browser import downloadview, hints, downloads
from qutebrowser.misc import crashsignal, keyhintwidget, sessions, objects
from qutebrowser.qt import sip
win_id_gen = itertools.count(0)

def get_window(*, via_ipc: bool, target: str, no_raise: bool=False) -> 'MainWindow':
    if False:
        print('Hello World!')
    'Helper function for app.py to get a window id.\n\n    Args:\n        via_ipc: Whether the request was made via IPC.\n        target: Where/how to open the window (via setting, command-line or\n                override).\n        no_raise: suppress target window raising\n\n    Return:\n        The MainWindow that was used to open URL\n    '
    if not via_ipc:
        return objreg.get('main-window', scope='window', window=0)
    window = None
    if target not in {'window', 'private-window'}:
        window = get_target_window()
        window.should_raise = target not in {'tab-silent', 'tab-bg-silent'} and (not no_raise)
    is_private = target == 'private-window'
    if window is None:
        window = MainWindow(private=is_private)
        window.should_raise = not no_raise
    return window

def raise_window(window, alert=True):
    if False:
        i = 10
        return i + 15
    'Raise the given MainWindow object.'
    window.setWindowState(window.windowState() & ~Qt.WindowState.WindowMinimized)
    window.setWindowState(window.windowState() | Qt.WindowState.WindowActive)
    window.raise_()
    QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents | QEventLoop.ProcessEventsFlag.ExcludeSocketNotifiers)
    if sip.isdeleted(window):
        return
    window.activateWindow()
    if alert:
        objects.qapp.alert(window)

def get_target_window():
    if False:
        print('Hello World!')
    'Get the target window for new tabs, or None if none exist.'
    getters = {'last-focused': objreg.last_focused_window, 'first-opened': objreg.first_opened_window, 'last-opened': objreg.last_opened_window, 'last-visible': objreg.last_visible_window}
    getter = getters[config.val.new_instance_open_target_window]
    try:
        return getter()
    except objreg.NoWindow:
        return None
_OverlayInfoType = Tuple[QWidget, pyqtBoundSignal, bool, str]

class MainWindow(QWidget):
    """The main window of qutebrowser.

    Adds all needed components to a vbox, initializes sub-widgets and connects
    signals.

    Attributes:
        status: The StatusBar widget.
        tabbed_browser: The TabbedBrowser widget.
        state_before_fullscreen: window state before activation of fullscreen.
        should_raise: Whether the window should be raised/activated when maybe_raise()
                      gets called.
        _downloadview: The DownloadView widget.
        _download_model: The DownloadModel instance.
        _vbox: The main QVBoxLayout.
        _commandrunner: The main CommandRunner instance.
        _overlays: Widgets shown as overlay for the current webpage.
        _private: Whether the window is in private browsing mode.
    """
    STYLESHEET = "\n        HintLabel {\n            background-color: {{ conf.colors.hints.bg }};\n            color: {{ conf.colors.hints.fg }};\n            font: {{ conf.fonts.hints }};\n            border: {{ conf.hints.border }};\n            border-radius: {{ conf.hints.radius }}px;\n            padding-top: {{ conf.hints.padding['top'] }}px;\n            padding-left: {{ conf.hints.padding['left'] }}px;\n            padding-right: {{ conf.hints.padding['right'] }}px;\n            padding-bottom: {{ conf.hints.padding['bottom'] }}px;\n        }\n\n        QToolTip {\n            {% if conf.fonts.tooltip %}\n                font: {{ conf.fonts.tooltip }};\n            {% endif %}\n            {% if conf.colors.tooltip.bg %}\n                background-color: {{ conf.colors.tooltip.bg }};\n            {% endif %}\n            {% if conf.colors.tooltip.fg %}\n                color: {{ conf.colors.tooltip.fg }};\n            {% endif %}\n        }\n\n        QMenu {\n            {% if conf.fonts.contextmenu %}\n                font: {{ conf.fonts.contextmenu }};\n            {% endif %}\n            {% if conf.colors.contextmenu.menu.bg %}\n                background-color: {{ conf.colors.contextmenu.menu.bg }};\n            {% endif %}\n            {% if conf.colors.contextmenu.menu.fg %}\n                color: {{ conf.colors.contextmenu.menu.fg }};\n            {% endif %}\n        }\n\n        QMenu::item:selected {\n            {% if conf.colors.contextmenu.selected.bg %}\n                background-color: {{ conf.colors.contextmenu.selected.bg }};\n            {% endif %}\n            {% if conf.colors.contextmenu.selected.fg %}\n                color: {{ conf.colors.contextmenu.selected.fg }};\n            {% endif %}\n        }\n\n        QMenu::item:disabled {\n            {% if conf.colors.contextmenu.disabled.bg %}\n                background-color: {{ conf.colors.contextmenu.disabled.bg }};\n            {% endif %}\n            {% if conf.colors.contextmenu.disabled.fg %}\n                color: {{ conf.colors.contextmenu.disabled.fg }};\n            {% endif %}\n        }\n    "

    def __init__(self, *, private: bool, geometry: Optional[QByteArray]=None, parent: Optional[QWidget]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Create a new main window.\n\n        Args:\n            geometry: The geometry to load, as a bytes-object (or None).\n            private: Whether the window is in private browsing mode.\n            parent: The parent the window should get.\n        '
        super().__init__(parent)
        from qutebrowser.mainwindow import tabbedbrowser
        from qutebrowser.mainwindow.statusbar import bar
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        if config.val.window.transparent:
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.palette().setColor(QPalette.ColorRole.Window, Qt.GlobalColor.transparent)
        self._overlays: MutableSequence[_OverlayInfoType] = []
        self.win_id = next(win_id_gen)
        self.registry = objreg.ObjectRegistry()
        objreg.window_registry[self.win_id] = self
        objreg.register('main-window', self, scope='window', window=self.win_id)
        tab_registry = objreg.ObjectRegistry()
        objreg.register('tab-registry', tab_registry, scope='window', window=self.win_id)
        self.setWindowTitle('qutebrowser')
        self._vbox = QVBoxLayout(self)
        self._vbox.setContentsMargins(0, 0, 0, 0)
        self._vbox.setSpacing(0)
        self._init_downloadmanager()
        self._downloadview = downloadview.DownloadView(model=self._download_model)
        self.is_private = config.val.content.private_browsing or private
        self.tabbed_browser: tabbedbrowser.TabbedBrowser = tabbedbrowser.TabbedBrowser(win_id=self.win_id, private=self.is_private, parent=self)
        objreg.register('tabbed-browser', self.tabbed_browser, scope='window', window=self.win_id)
        self._init_command_dispatcher()
        self.status = bar.StatusBar(win_id=self.win_id, private=self.is_private, parent=self)
        self._add_widgets()
        self._downloadview.show()
        self._init_completion()
        log.init.debug('Initializing modes...')
        modeman.init(win_id=self.win_id, parent=self)
        self._commandrunner = runners.CommandRunner(self.win_id, partial_match=True, find_similar=True)
        self._keyhint = keyhintwidget.KeyHintView(self.win_id, self)
        self._add_overlay(self._keyhint, self._keyhint.update_geometry)
        self._prompt_container = prompt.PromptContainer(self.win_id, self)
        self._add_overlay(self._prompt_container, self._prompt_container.update_geometry, centered=True, padding=10)
        objreg.register('prompt-container', self._prompt_container, scope='window', window=self.win_id, command_only=True)
        self._prompt_container.hide()
        self._messageview = messageview.MessageView(parent=self)
        self._add_overlay(self._messageview, self._messageview.update_geometry)
        self._init_geometry(geometry)
        self._connect_signals()
        QTimer.singleShot(0, self._connect_overlay_signals)
        config.instance.changed.connect(self._on_config_changed)
        objects.qapp.new_window.emit(self)
        self._set_decoration(config.val.window.hide_decoration)
        self.state_before_fullscreen = self.windowState()
        self.should_raise: bool = False
        stylesheet.set_register(self)

    def _init_geometry(self, geometry):
        if False:
            i = 10
            return i + 15
        'Initialize the window geometry or load it from disk.'
        if geometry is not None:
            self._load_geometry(geometry)
        elif self.win_id == 0:
            self._load_state_geometry()
        else:
            self._set_default_geometry()
        log.init.debug('Initial main window geometry: {}'.format(self.geometry()))

    def _add_overlay(self, widget, signal, *, centered=False, padding=0):
        if False:
            return 10
        self._overlays.append((widget, signal, centered, padding))

    def _update_overlay_geometries(self):
        if False:
            while True:
                i = 10
        'Update the size/position of all overlays.'
        for (w, _signal, centered, padding) in self._overlays:
            self._update_overlay_geometry(w, centered, padding)

    def _update_overlay_geometry(self, widget, centered, padding):
        if False:
            i = 10
            return i + 15
        'Reposition/resize the given overlay.'
        if not widget.isVisible():
            return
        if widget.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding:
            width = self.width() - 2 * padding
            if widget.hasHeightForWidth():
                height = widget.heightForWidth(width)
            else:
                height = widget.sizeHint().height()
            left = padding
        else:
            size_hint = widget.sizeHint()
            width = min(size_hint.width(), self.width() - 2 * padding)
            height = size_hint.height()
            left = (self.width() - width) // 2 if centered else 0
        height_padding = 20
        status_position = config.val.statusbar.position
        if status_position == 'bottom':
            if self.status.isVisible():
                status_height = self.status.height()
                bottom = self.status.geometry().top()
            else:
                status_height = 0
                bottom = self.height()
            top = self.height() - status_height - height
            top = qtutils.check_overflow(top, 'int', fatal=False)
            topleft = QPoint(left, max(height_padding, top))
            bottomright = QPoint(left + width, bottom)
        elif status_position == 'top':
            if self.status.isVisible():
                status_height = self.status.height()
                top = self.status.geometry().bottom()
            else:
                status_height = 0
                top = 0
            topleft = QPoint(left, top)
            bottom = status_height + height
            bottom = qtutils.check_overflow(bottom, 'int', fatal=False)
            bottomright = QPoint(left + width, min(self.height() - height_padding, bottom))
        else:
            raise ValueError('Invalid position {}!'.format(status_position))
        rect = QRect(topleft, bottomright)
        log.misc.debug('new geometry for {!r}: {}'.format(widget, rect))
        if rect.isValid():
            widget.setGeometry(rect)

    def _init_downloadmanager(self):
        if False:
            print('Hello World!')
        log.init.debug('Initializing downloads...')
        qtnetwork_download_manager = objreg.get('qtnetwork-download-manager')
        try:
            webengine_download_manager = objreg.get('webengine-download-manager')
        except KeyError:
            webengine_download_manager = None
        self._download_model = downloads.DownloadModel(qtnetwork_download_manager, webengine_download_manager)
        objreg.register('download-model', self._download_model, scope='window', window=self.win_id, command_only=True)

    def _init_completion(self):
        if False:
            i = 10
            return i + 15
        self._completion = completionwidget.CompletionView(cmd=self.status.cmd, win_id=self.win_id, parent=self)
        completer_obj = completer.Completer(cmd=self.status.cmd, win_id=self.win_id, parent=self._completion)
        self._completion.selection_changed.connect(completer_obj.on_selection_changed)
        objreg.register('completion', self._completion, scope='window', window=self.win_id, command_only=True)
        self._add_overlay(self._completion, self._completion.update_geometry)

    def _init_command_dispatcher(self):
        if False:
            print('Hello World!')
        from qutebrowser.browser import commands
        self._command_dispatcher = commands.CommandDispatcher(self.win_id, self.tabbed_browser)
        objreg.register('command-dispatcher', self._command_dispatcher, command_only=True, scope='window', window=self.win_id)
        widget = self.tabbed_browser.widget
        widget.destroyed.connect(functools.partial(objreg.delete, 'command-dispatcher', scope='window', window=self.win_id))

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return utils.get_repr(self)

    @pyqtSlot(str)
    def _on_config_changed(self, option):
        if False:
            i = 10
            return i + 15
        'Resize the completion if related config options changed.'
        if option == 'statusbar.padding':
            self._update_overlay_geometries()
        elif option == 'downloads.position':
            self._add_widgets()
        elif option == 'statusbar.position':
            self._add_widgets()
            self._update_overlay_geometries()
        elif option == 'window.hide_decoration':
            self._set_decoration(config.val.window.hide_decoration)

    def _add_widgets(self):
        if False:
            while True:
                i = 10
        'Add or re-add all widgets to the VBox.'
        self._vbox.removeWidget(self.tabbed_browser.widget)
        self._vbox.removeWidget(self._downloadview)
        self._vbox.removeWidget(self.status)
        widgets: List[QWidget] = [self.tabbed_browser.widget]
        downloads_position = config.val.downloads.position
        if downloads_position == 'top':
            widgets.insert(0, self._downloadview)
        elif downloads_position == 'bottom':
            widgets.append(self._downloadview)
        else:
            raise ValueError('Invalid position {}!'.format(downloads_position))
        status_position = config.val.statusbar.position
        if status_position == 'top':
            widgets.insert(0, self.status)
        elif status_position == 'bottom':
            widgets.append(self.status)
        else:
            raise ValueError('Invalid position {}!'.format(status_position))
        for widget in widgets:
            self._vbox.addWidget(widget)

    def _load_state_geometry(self):
        if False:
            i = 10
            return i + 15
        'Load the geometry from the state file.'
        try:
            data = configfiles.state['geometry']['mainwindow']
            geom = base64.b64decode(data, validate=True)
        except KeyError:
            self._set_default_geometry()
        except binascii.Error:
            log.init.exception('Error while reading geometry')
            self._set_default_geometry()
        else:
            self._load_geometry(geom)

    def _save_geometry(self):
        if False:
            i = 10
            return i + 15
        'Save the window geometry to the state config.'
        data = self.saveGeometry().data()
        geom = base64.b64encode(data).decode('ASCII')
        configfiles.state['geometry']['mainwindow'] = geom

    def _load_geometry(self, geom):
        if False:
            for i in range(10):
                print('nop')
        'Load geometry from a bytes object.\n\n        If loading fails, loads default geometry.\n        '
        log.init.debug('Loading mainwindow from {!r}'.format(geom))
        ok = self.restoreGeometry(geom)
        if not ok:
            log.init.warning('Error while loading geometry.')
            self._set_default_geometry()

    def _connect_overlay_signals(self):
        if False:
            return 10
        'Connect the resize signal and resize everything once.'
        for (widget, signal, centered, padding) in self._overlays:
            signal.connect(functools.partial(self._update_overlay_geometry, widget, centered, padding))
            self._update_overlay_geometry(widget, centered, padding)

    def _set_default_geometry(self):
        if False:
            for i in range(10):
                print('nop')
        'Set some sensible default geometry.'
        self.setGeometry(QRect(50, 50, 800, 600))

    def _connect_signals(self):
        if False:
            for i in range(10):
                print('nop')
        'Connect all mainwindow signals.'
        mode_manager = modeman.instance(self.win_id)
        self.tabbed_browser.close_window.connect(self.close)
        mode_manager.entered.connect(hints.on_mode_entered)
        mode_manager.hintmanager.set_text.connect(self.status.set_text)
        mode_manager.entered.connect(self.status.on_mode_entered)
        mode_manager.left.connect(self.status.on_mode_left)
        mode_manager.left.connect(self.status.cmd.on_mode_left)
        mode_manager.left.connect(message.global_bridge.mode_left)
        mode_manager.keystring_updated.connect(self.status.keystring.on_keystring_updated)
        self.status.cmd.got_cmd[str].connect(self._commandrunner.run_safely)
        self.status.cmd.got_cmd[str, int].connect(self._commandrunner.run_safely)
        self.status.cmd.returnPressed.connect(self.tabbed_browser.on_cmd_return_pressed)
        self.status.cmd.got_search.connect(self._command_dispatcher.search)
        mode_manager.keystring_updated.connect(self._keyhint.update_keyhint)
        message.global_bridge.show_message.connect(self._messageview.show_message)
        message.global_bridge.flush()
        message.global_bridge.clear_messages.connect(self._messageview.clear_messages)
        self.tabbed_browser.current_tab_changed.connect(self.status.on_tab_changed)
        self.tabbed_browser.cur_progress.connect(self.status.prog.on_load_progress)
        self.tabbed_browser.cur_load_started.connect(self.status.prog.on_load_started)
        self.tabbed_browser.cur_scroll_perc_changed.connect(self.status.percentage.set_perc)
        self.tabbed_browser.widget.tab_index_changed.connect(self.status.tabindex.on_tab_index_changed)
        self.tabbed_browser.cur_url_changed.connect(self.status.url.set_url)
        self.tabbed_browser.cur_url_changed.connect(functools.partial(self.status.backforward.on_tab_cur_url_changed, tabs=self.tabbed_browser))
        self.tabbed_browser.cur_link_hovered.connect(self.status.url.set_hover_url)
        self.tabbed_browser.cur_load_status_changed.connect(self.status.url.on_load_status_changed)
        self.tabbed_browser.cur_search_match_changed.connect(self.status.search_match.set_match)
        self.tabbed_browser.cur_caret_selection_toggled.connect(self.status.on_caret_selection_toggled)
        self.tabbed_browser.cur_fullscreen_requested.connect(self._on_fullscreen_requested)
        self.tabbed_browser.cur_fullscreen_requested.connect(self.status.maybe_hide)
        self.tabbed_browser.cur_fullscreen_requested.connect(self._downloadview.on_fullscreen_requested)
        mode_manager.entered.connect(self.tabbed_browser.on_mode_entered)
        mode_manager.left.connect(self.tabbed_browser.on_mode_left)
        self.status.cmd.clear_completion_selection.connect(self._completion.on_clear_completion_selection)
        self.status.cmd.hide_completion.connect(self._completion.hide)

    def _set_decoration(self, hidden):
        if False:
            for i in range(10):
                print('nop')
        'Set the visibility of the window decoration via Qt.'
        if machinery.IS_QT5:
            window_flags = cast(Qt.WindowFlags, Qt.WindowType.Window)
        else:
            window_flags = Qt.WindowType.Window
        refresh_window = self.isVisible()
        if hidden:
            modifiers = Qt.WindowType.CustomizeWindowHint | Qt.WindowType.NoDropShadowWindowHint
            window_flags |= modifiers
        self.setWindowFlags(window_flags)
        if utils.is_mac and hidden and (not qtutils.version_check('6.3', compiled=False)):
            from ctypes import c_void_p
            from objc import objc_object
            from AppKit import NSWindowStyleMaskResizable
            win = objc_object(c_void_p=c_void_p(int(self.winId()))).window()
            win.setStyleMask_(win.styleMask() | NSWindowStyleMaskResizable)
        if refresh_window:
            self.show()

    @pyqtSlot(bool)
    def _on_fullscreen_requested(self, on):
        if False:
            i = 10
            return i + 15
        if not config.val.content.fullscreen.window:
            if on:
                self.state_before_fullscreen = self.windowState()
                self.setWindowState(Qt.WindowState.WindowFullScreen | self.state_before_fullscreen)
            elif self.isFullScreen():
                self.setWindowState(self.state_before_fullscreen)
        log.misc.debug('on: {}, state before fullscreen: {}'.format(on, debug.qflags_key(Qt, self.state_before_fullscreen)))

    @cmdutils.register(instance='main-window', scope='window')
    @pyqtSlot()
    def close(self):
        if False:
            print('Hello World!')
        'Close the current window.\n\n        //\n\n        Extend close() so we can register it as a command.\n        '
        super().close()

    def resizeEvent(self, e):
        if False:
            return 10
        "Extend resizewindow's resizeEvent to adjust completion.\n\n        Args:\n            e: The QResizeEvent\n        "
        super().resizeEvent(e)
        self._update_overlay_geometries()
        self._downloadview.updateGeometry()
        self.tabbed_browser.widget.tab_bar().refresh()

    def showEvent(self, e):
        if False:
            while True:
                i = 10
        'Extend showEvent to register us as the last-visible-main-window.\n\n        Args:\n            e: The QShowEvent\n        '
        super().showEvent(e)
        objreg.register('last-visible-main-window', self, update=True)

    def _confirm_quit(self):
        if False:
            for i in range(10):
                print('nop')
        'Confirm that this window should be closed.\n\n        Return:\n            True if closing is okay, False if a closeEvent should be ignored.\n        '
        tab_count = self.tabbed_browser.widget.count()
        window_count = len(objreg.window_registry)
        download_count = self._download_model.running_downloads()
        quit_texts = []
        if 'multiple-tabs' in config.val.confirm_quit and tab_count > 1:
            quit_texts.append('{} tabs are open.'.format(tab_count))
        if 'downloads' in config.val.confirm_quit and download_count > 0 and (window_count <= 1):
            quit_texts.append('{} {} running.'.format(download_count, 'download is' if download_count == 1 else 'downloads are'))
        if quit_texts or 'always' in config.val.confirm_quit:
            msg = jinja.environment.from_string('\n                <ul>\n                {% for text in quit_texts %}\n                   <li>{{text}}</li>\n                {% endfor %}\n                </ul>\n            '.strip()).render(quit_texts=quit_texts)
            confirmed = message.ask('Really quit?', msg, mode=usertypes.PromptMode.yesno, default=True)
            if not confirmed:
                log.destroy.debug('Cancelling closing of window {}'.format(self.win_id))
                return False
        return True

    def maybe_raise(self) -> None:
        if False:
            return 10
        'Raise the window if self.should_raise is set.'
        if self.should_raise:
            raise_window(self)
            self.should_raise = False

    def closeEvent(self, e):
        if False:
            return 10
        'Override closeEvent to display a confirmation if needed.'
        if crashsignal.crash_handler.is_crashing:
            e.accept()
            return
        if not self._confirm_quit():
            e.ignore()
            return
        e.accept()
        for key in ['last-visible-main-window', 'last-focused-main-window']:
            try:
                win = objreg.get(key)
                if self is win:
                    objreg.delete(key)
            except KeyError:
                pass
        sessions.session_manager.save_last_window_session()
        self._save_geometry()
        if self.is_private and len(objreg.window_registry) > 1 and (len([window for window in objreg.window_registry.values() if window.is_private]) == 1):
            log.destroy.debug('Wiping private data before closing last private window')
            websettings.clear_private_data()
        log.destroy.debug('Closing window {}'.format(self.win_id))
        self.tabbed_browser.shutdown()