import logging
import sys
from typing import Optional
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QSystemTrayIcon, QAction, QMenu
from feeluown.player import State
from feeluown.gui.helpers import elided_text, get_qapp
TOGGLE_APP_TEXT = ('激活主窗口', '隐藏主窗口')
TOGGLE_PLAYER_TEXT = ('播放', '暂停')
IS_MACOS = sys.platform == 'darwin'
logger = logging.getLogger(__name__)

class Tray(QSystemTrayIcon):

    def __init__(self, app):
        if False:
            for i in range(10):
                print('nop')
        '\n        type app: feeluown.app.App\n        '
        super().__init__()
        self._app = app
        self._app_old_state = None
        self._menu = QMenu()
        self._status_action = QAction('...')
        self._toggle_player_action = QAction(QIcon.fromTheme('media-play'), TOGGLE_PLAYER_TEXT[0])
        self._next_action = QAction(QIcon.fromTheme('media-skip-forward'), '下一首')
        self._prev_action = QAction(QIcon.fromTheme('media-skip-backward'), '上一首')
        self._quit_action = QAction(QIcon.fromTheme('exit'), '退出')
        if IS_MACOS:
            self._toggle_app_action: Optional[QAction] = QAction(QIcon.fromTheme('window'), TOGGLE_APP_TEXT[1])
        else:
            self._toggle_app_action = None
            self.activated.connect(self._on_activated)
        self._quit_action.triggered.connect(self._app.exit)
        self._toggle_player_action.triggered.connect(self._app.player.toggle)
        self._prev_action.triggered.connect(self._app.playlist.previous)
        self._next_action.triggered.connect(self._app.playlist.next)
        if self._toggle_app_action is not None:
            self._toggle_app_action.triggered.connect(self._toggle_app_state)
        self._app.player.state_changed.connect(self.on_player_state_changed)
        self._app.playlist.song_changed.connect(self.on_player_song_changed)
        self._app.theme_mgr.theme_changed.connect(self.on_theme_changed)
        get_qapp().applicationStateChanged.connect(self.on_app_state_changed)
        self._app.installEventFilter(self)
        self.setContextMenu(self._menu)
        self.setup_ui()

    def initialize(self):
        if False:
            print('Hello World!')
        self._set_icon()
        self._status_action.setIcon(self.icon())

    def setup_ui(self):
        if False:
            print('Hello World!')
        self._menu.addAction(self._status_action)
        self._menu.addAction(self._toggle_player_action)
        self._menu.addAction(self._prev_action)
        self._menu.addAction(self._next_action)
        self._menu.addSeparator()
        if self._toggle_app_action is not None:
            self._menu.addAction(self._toggle_app_action)
        self._menu.addAction(self._quit_action)
        self._status_action.setEnabled(False)

    def _on_activated(self, reason=QSystemTrayIcon.Unknown):
        if False:
            for i in range(10):
                print('nop')
        '\n        NOTE(cosven): Theoretically, we need not give default value for param reason.\n        However, we connect activated signal with `_toggle_app_state method` before,\n        and as you can see, `_toggle_app_state` does not accepts other parameters and it\n        works well. So we give an default value to avoid potential strange errors.\n        '
        if reason not in (QSystemTrayIcon.Context,):
            self._toggle_app_state()
        logger.info(f'tray icon activated, reason:{reason}')

    def _toggle_app_state(self):
        if False:
            i = 10
            return i + 15
        'activate/deactivate app'
        if self._app.isHidden():
            self._app.show()
            self._app.activateWindow()
        elif self._app.isMinimized():
            self._app.setWindowState(self._app_old_state)
        else:
            self._app.hide()

    def _set_icon(self):
        if False:
            while True:
                i = 10
        icon = QIcon.fromTheme('feeluown-tray', QIcon(self._app.theme_mgr.get_icon('tray')))
        self.setIcon(icon)

    def on_theme_changed(self, _):
        if False:
            i = 10
            return i + 15
        self._set_icon()

    def on_player_song_changed(self, song):
        if False:
            for i in range(10):
                print('nop')
        if song is not None:
            status = f'{song.title_display} - {song.artists_name_display}'
            if self._app.config.NOTIFY_ON_TRACK_CHANGED:
                self.showMessage(song.title_display, song.artists_name_display, msecs=self._app.config.NOTIFY_DURATION)
            self._status_action.setText(elided_text(status, 120))
            self._status_action.setToolTip(status)
            self.setToolTip(status)

    def on_player_state_changed(self, state):
        if False:
            for i in range(10):
                print('nop')
        if state == State.playing:
            self._toggle_player_action.setText(TOGGLE_PLAYER_TEXT[1])
            self._toggle_player_action.setIcon(QIcon.fromTheme('media-pause'))
            self._toggle_player_action.setEnabled(True)
        else:
            self._toggle_player_action.setText(TOGGLE_PLAYER_TEXT[0])
            self._toggle_player_action.setIcon(QIcon.fromTheme('media-play'))
            if state == State.stopped:
                self._toggle_player_action.setEnabled(False)
            else:
                self._toggle_player_action.setEnabled(True)

    def on_app_state_changed(self, state):
        if False:
            while True:
                i = 10
        if state == Qt.ApplicationActive:
            if IS_MACOS:
                self._app.show()
                self._app.activateWindow()
        elif state == Qt.ApplicationInactive:
            if self._toggle_app_action is not None:
                self._toggle_app_action.setText(TOGGLE_APP_TEXT[0])

    def eventFilter(self, obj, event):
        if False:
            return 10
        'event filter for app'
        app_text_idx = None
        if event.type() == QEvent.WindowStateChange:
            if obj.windowState() & Qt.WindowMinimized:
                self._app_old_state = event.oldState()
                app_text_idx = 0
            else:
                app_text_idx = 1
        elif event.type() == QEvent.Hide:
            app_text_idx = 0
        elif event.type() == QEvent.Show:
            app_text_idx = 1
        else:
            return False
        if app_text_idx is not None and self._toggle_app_action is not None:
            self._toggle_app_action.setText(TOGGLE_APP_TEXT[app_text_idx])
        return False