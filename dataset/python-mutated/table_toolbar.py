from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QComboBox, QWidget
from feeluown.models import AlbumType
from feeluown.gui.widgets import TextButton

class SongsTableToolbar(QWidget):
    play_all_needed = pyqtSignal()
    filter_albums_needed = pyqtSignal([list])
    filter_text_changed = pyqtSignal([str])

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self._tmp_buttons = []
        self.play_all_btn = TextButton('播放全部', self)
        self.play_all_btn.clicked.connect(self.play_all_needed.emit)
        self.play_all_btn.setObjectName('play_all')
        self.filter_albums_combobox = QComboBox(self)
        self.filter_albums_combobox.addItems(['所有专辑', '标准', '单曲与EP', '现场', '合辑'])
        self.filter_albums_combobox.currentIndexChanged.connect(self.on_albums_filter_changed)
        self.filter_albums_combobox.setMinimumContentsLength(8)
        self.filter_albums_combobox.hide()
        self._setup_ui()

    def albums_mode(self):
        if False:
            print('Hello World!')
        self._before_change_mode()
        self.filter_albums_combobox.show()

    def songs_mode(self):
        if False:
            for i in range(10):
                print('nop')
        self._before_change_mode()
        self.play_all_btn.show()

    def artists_mode(self):
        if False:
            i = 10
            return i + 15
        self._before_change_mode()

    def manual_mode(self):
        if False:
            for i in range(10):
                print('nop')
        "fully customized mode\n\n        .. versionadded:: 3.7.11\n           You'd better use this mode and add_tmp_button to customize toolbar.\n        "
        self._before_change_mode()

    def enter_state_playall_start(self):
        if False:
            return 10
        self.play_all_btn.setEnabled(False)
        self.play_all_btn.setText('获取所有歌曲...')

    def enter_state_playall_end(self):
        if False:
            while True:
                i = 10
        self.play_all_btn.setText('获取所有歌曲...done')
        self.play_all_btn.setEnabled(True)
        self.play_all_btn.setText('播放全部')

    def add_tmp_button(self, button):
        if False:
            print('Hello World!')
        'Append text button'
        if button not in self._tmp_buttons:
            index = len(self._tmp_buttons)
            if self.play_all_btn.isVisible():
                index = index + 1
            self._layout.insertWidget(index, button)
            self._tmp_buttons.append(button)

    def _setup_ui(self):
        if False:
            return 10
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 15, 30, 10)
        self._layout.addWidget(self.play_all_btn)
        self._layout.addStretch(0)
        self._layout.addWidget(self.filter_albums_combobox)

    def _before_change_mode(self):
        if False:
            return 10
        'filter all filter buttons'
        for button in self._tmp_buttons:
            self._layout.removeWidget(button)
            button.close()
        self._tmp_buttons.clear()
        self.filter_albums_combobox.hide()
        self.play_all_btn.hide()

    def on_albums_filter_changed(self, index):
        if False:
            while True:
                i = 10
        if index == 0:
            types = []
        elif index == 1:
            types = [AlbumType.standard]
        elif index == 2:
            types = [AlbumType.single, AlbumType.ep]
        elif index == 3:
            types = [AlbumType.live]
        else:
            types = [AlbumType.compilation, AlbumType.retrospective]
        self.filter_albums_needed.emit(types)