from PyQt5.QtCore import QTimer, QRect, Qt
from PyQt5.QtGui import QFontMetrics, QPainter, QPalette
from PyQt5.QtWidgets import QApplication, QLabel, QSizePolicy, QMenu
from feeluown.gui.components import SongMenuInitializer

class LineSongLabel(QLabel):
    """Show song info in one line (with limited width)."""
    default_text = '...'

    def __init__(self, app, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(text=self.default_text, parent=parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._app = app
        self._timer = QTimer()
        self._txt = self._raw_text = self.default_text
        self._font_metrics = QFontMetrics(QApplication.font())
        self._text_rect = self._font_metrics.boundingRect(self._raw_text)
        self._pos = 0
        self._timer.timeout.connect(self.change_text_position)
        self._app.player.metadata_changed.connect(self.on_metadata_changed, aioqueue=True)

    def on_metadata_changed(self, metadata):
        if False:
            i = 10
            return i + 15
        if not metadata:
            self.setText('')
            return
        text = metadata.get('title', '')
        if text:
            artists = metadata.get('artists', [])
            if artists:
                text += f" - {','.join(artists)}"
        self.setText(text)

    def change_text_position(self):
        if False:
            print('Hello World!')
        if not self.parent().isVisible():
            self._timer.stop()
            self._pos = 0
            return
        if self._text_rect.width() + self._pos > 0:
            self._pos -= 5
        else:
            self._pos = self.width()
        self.update()

    def setText(self, text):
        if False:
            print('Hello World!')
        self._txt = self._raw_text = text
        self._text_rect = self._font_metrics.boundingRect(self._raw_text)
        self._pos = 0
        self.update()

    def enterEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        if self._txt != self._raw_text:
            self._timer.start(150)

    def leaveEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        self._timer.stop()
        self._pos = 0
        self.update()

    def paintEvent(self, event):
        if False:
            return 10
        painter = QPainter(self)
        painter.setFont(QApplication.font())
        painter.setPen(self.palette().color(QPalette.Text))
        if self._timer.isActive():
            self._txt = self._raw_text
        else:
            self._txt = self._font_metrics.elidedText(self._raw_text, Qt.ElideRight, self.width())
        painter.drawText(QRect(self._pos, 0, self.width() - self._pos, self.height()), Qt.AlignLeft | Qt.AlignVCenter, self._txt)

    def contextMenuEvent(self, e):
        if False:
            print('Hello World!')
        song = self._app.playlist.current_song
        if song is None:
            return
        menu = QMenu()
        SongMenuInitializer(self._app, song).apply(menu)
        menu.exec(e.globalPos())