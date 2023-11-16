from PyQt5.QtCore import Qt, QRect, QEvent, QModelIndex, QItemSelectionModel
from PyQt5.QtWidgets import QWidget, QStackedLayout, QVBoxLayout, QHBoxLayout, QMenu, QApplication, QShortcut, QAbstractItemView
from PyQt5.QtGui import QColor, QLinearGradient, QPalette, QPainter, QKeySequence
from feeluown.player import PlaybackMode
from feeluown.gui.helpers import fetch_cover_wrapper
from feeluown.gui.widgets.textbtn import TextButton
from feeluown.gui.components import SongMenuInitializer
from feeluown.gui.widgets.tabbar import TabBar
from feeluown.gui.widgets.song_minicard_list import SongMiniCardListView, SongMiniCardListModel, SongMiniCardListDelegate
from feeluown.utils.reader import create_reader
PlaybackModeName = {PlaybackMode.one_loop: '单曲循环', PlaybackMode.sequential: '顺序播放', PlaybackMode.loop: '循环播放', PlaybackMode.random: '随机播放'}
PlaybackModes = list(PlaybackModeName.keys())

def acolor(s, a):
    if False:
        while True:
            i = 10
    "Create color with it's name and alpha"
    color = QColor(s)
    color.setAlpha(a)
    return color

class PlaylistOverlay(QWidget):

    def __init__(self, app, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self._app = app
        self._tabbar = TabBar(self)
        self._clear_playlist_btn = TextButton('清空播放队列')
        self._playback_mode_switch = PlaybackModeSwitch(app)
        self._goto_current_song_btn = TextButton('跳转到当前歌曲')
        self._btns = [self._clear_playlist_btn, self._playback_mode_switch, self._goto_current_song_btn]
        self._stacked_layout = QStackedLayout()
        self._shadow_width = 15
        self._player_playlist_model = PlayerPlaylistModel(self._app.playlist, fetch_cover_wrapper(self._app))
        self._tabbar.setAutoFillBackground(True)
        self._clear_playlist_btn.clicked.connect(self._app.playlist.clear)
        self._goto_current_song_btn.clicked.connect(self.goto_current_song)
        QShortcut(QKeySequence.Cancel, self).activated.connect(self.hide)
        q_app = QApplication.instance()
        assert q_app is not None
        q_app.focusChanged.connect(self.on_focus_changed)
        self._app.installEventFilter(self)
        self._tabbar.currentChanged.connect(self.show_tab)
        self.setup_ui()

    def setup_ui(self):
        if False:
            while True:
                i = 10
        self._layout = QVBoxLayout(self)
        self._btn_layout = QHBoxLayout()
        self._layout.setContentsMargins(self._shadow_width, 0, 0, 0)
        self._layout.setSpacing(0)
        self._btn_layout.setContentsMargins(7, 7, 7, 7)
        self._btn_layout.setSpacing(7)
        self._tabbar.setDocumentMode(True)
        self._tabbar.addTab('播放列表')
        self._tabbar.addTab('最近播放')
        self._layout.addWidget(self._tabbar)
        self._layout.addLayout(self._btn_layout)
        self._layout.addLayout(self._stacked_layout)
        self._btn_layout.addWidget(self._clear_playlist_btn)
        self._btn_layout.addWidget(self._playback_mode_switch)
        self._btn_layout.addWidget(self._goto_current_song_btn)
        self._btn_layout.addStretch(0)

    def on_focus_changed(self, _, new):
        if False:
            while True:
                i = 10
        '\n        Hide the widget when it loses focus.\n        '
        if not self.isVisible():
            return
        if new is None or new is self or new in self.findChildren(QWidget):
            return
        self.hide()

    def goto_current_song(self):
        if False:
            i = 10
            return i + 15
        view = self._stacked_layout.currentWidget()
        assert isinstance(view, PlayerPlaylistView)
        view.scroll_to_current_song()

    def show_tab(self, index):
        if False:
            return 10
        if not self.isVisible():
            return
        view_options = dict(row_height=60, no_scroll_v=False)
        if index == 0:
            self._show_btns()
            view: SongMiniCardListView = PlayerPlaylistView(self._app, **view_options)
            view.setModel(self._player_playlist_model)
        else:
            self._hide_btns()
            model = SongMiniCardListModel(create_reader(self._app.recently_played.list_songs()), fetch_cover_wrapper(self._app))
            view = SongMiniCardListView(**view_options)
            view.setModel(model)
        delegate = SongMiniCardListDelegate(view, card_min_width=self.width() - self.width() // 6, card_height=40, card_padding=(5 + SongMiniCardListDelegate.img_padding, 5, 0, 5), card_right_spacing=10)
        view.setItemDelegate(delegate)
        view.play_song_needed.connect(self._app.playlist.play_model)
        self._stacked_layout.addWidget(view)
        self._stacked_layout.setCurrentWidget(view)

    def _hide_btns(self):
        if False:
            for i in range(10):
                print('nop')
        for btn in self._btns:
            btn.hide()

    def _show_btns(self):
        if False:
            for i in range(10):
                print('nop')
        for btn in self._btns:
            btn.show()

    def paintEvent(self, e):
        if False:
            i = 10
            return i + 15
        super().paintEvent(e)
        painter = QPainter(self)
        painter.setPen(Qt.NoPen)
        painter.save()
        shadow_width = self._shadow_width
        rect = QRect(0, 0, shadow_width, self.height())
        gradient = QLinearGradient(rect.topRight(), rect.topLeft())
        gradient.setColorAt(0, acolor('black', 70))
        gradient.setColorAt(0.05, acolor('black', 60))
        gradient.setColorAt(0.1, acolor('black', 30))
        gradient.setColorAt(0.2, acolor('black', 5))
        gradient.setColorAt(1, acolor('black', 0))
        painter.setBrush(gradient)
        painter.drawRect(rect)
        painter.restore()
        painter.setBrush(self.palette().color(QPalette.Base))
        painter.drawRect(shadow_width, 0, self.width() - shadow_width, self.height())

    def showEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        super().showEvent(e)
        self.show_tab(self._tabbar.currentIndex())

    def eventFilter(self, obj, event):
        if False:
            while True:
                i = 10
        '\n        Hide myself when the app is resized.\n        '
        if obj is self._app and event.type() == QEvent.Resize:
            self.hide()
        return False

class PlayerPlaylistModel(SongMiniCardListModel):

    def __init__(self, playlist, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        reader = create_reader(playlist.list())
        super().__init__(reader, *args, **kwargs)
        self._playlist = playlist
        self._playlist.songs_added.connect(self.on_songs_added)
        self._playlist.songs_removed.connect(self.on_songs_removed)

    def flags(self, index):
        if False:
            i = 10
            return i + 15
        flags = super().flags(index)
        song = index.data(Qt.UserRole)[0]
        if self._playlist.is_bad(song):
            flags &= ~Qt.ItemIsEnabled
        return flags

    def on_songs_added(self, index, count):
        if False:
            for i in range(10):
                print('nop')
        self.beginInsertRows(QModelIndex(), index, index + count - 1)
        while count > 0:
            self._items.insert(index, self._playlist[index + count - 1])
            count -= 1
        self.endInsertRows()

    def on_songs_removed(self, index, count):
        if False:
            i = 10
            return i + 15
        self.beginRemoveRows(QModelIndex(), index, index + count - 1)
        while count > 0:
            self._items.pop(index + count - 1)
            count -= 1
        self.endRemoveRows()

class PlayerPlaylistView(SongMiniCardListView):

    def __init__(self, app, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self._app = app

    def contextMenuEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        index = self.indexAt(e.pos())
        if not index.isValid():
            return
        song = index.data(Qt.UserRole)[0]
        menu = QMenu()
        action = menu.addAction('从播放队列中移除')
        menu.addSeparator()
        SongMenuInitializer(self._app, song).apply(menu)
        action.triggered.connect(lambda : self._app.playlist.remove(song))
        menu.exec_(e.globalPos())

    def scroll_to_current_song(self):
        if False:
            return 10
        'Scroll to the current song, and select it to highlight it.'
        current_song = self._app.playlist.current_song
        songs = self._app.playlist.list()
        if current_song is not None:
            model = self.model()
            row = songs.index(current_song)
            index = model.index(row, 0)
            self.selectionModel().select(index, QItemSelectionModel.SelectCurrent)
            self.scrollTo(index, QAbstractItemView.PositionAtCenter)

class PlaybackModeSwitch(TextButton):

    def __init__(self, app, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self._app = app
        self.update_text()
        self.clicked.connect(self.switch_playback_mode)
        self._app.playlist.playback_mode_changed.connect(self.on_playback_mode_changed, aioqueue=True)
        self.setToolTip('修改播放模式')

    def switch_playback_mode(self):
        if False:
            print('Hello World!')
        playlist = self._app.playlist
        index = PlaybackModes.index(playlist.playback_mode)
        if index < len(PlaybackModes) - 1:
            new_index = index + 1
        else:
            new_index = 0
        playlist.playback_mode = PlaybackModes[new_index]

    def update_text(self):
        if False:
            i = 10
            return i + 15
        self.setText(PlaybackModeName[self._app.playlist.playback_mode])

    def on_playback_mode_changed(self, _):
        if False:
            print('Hello World!')
        self.update_text()