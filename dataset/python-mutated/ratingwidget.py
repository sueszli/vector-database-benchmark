from PyQt6 import QtCore, QtGui, QtWidgets
from picard import log
from picard.config import get_config

class RatingWidget(QtWidgets.QWidget):

    def __init__(self, parent, track):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self._track = track
        config = get_config()
        self._maximum = config.setting['rating_steps'] - 1
        try:
            self._rating = int(track.metadata['~rating'] or 0)
        except ValueError:
            self._rating = 0
        self._highlight = 0
        self._star_pixmap = QtGui.QPixmap(':/images/star.png')
        self._star_gray_pixmap = QtGui.QPixmap(':/images/star-gray.png')
        self._star_size = 16
        self._star_spacing = 2
        self._offset = 16
        self._width = self._maximum * (self._star_size + self._star_spacing) + self._offset
        self._height = self._star_size + 6
        self.setMaximumSize(self._width, self._height)
        self.setMinimumSize(self._width, self._height)
        self.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed))
        self.setMouseTracking(True)

    def sizeHint(self):
        if False:
            for i in range(10):
                print('nop')
        return QtCore.QSize(self._width, self._height)

    def _setHighlight(self, highlight):
        if False:
            i = 10
            return i + 15
        assert 0 <= highlight <= self._maximum
        if highlight != self._highlight:
            self._highlight = highlight
            self.update()

    def mousePressEvent(self, event):
        if False:
            print('Hello World!')
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            x = event.pos().x()
            if x < self._offset:
                return
            rating = self._getRatingFromPosition(x)
            if self._rating == rating:
                rating = 0
            self._rating = rating
            self._update_track()
            self.update()
            event.accept()

    def mouseMoveEvent(self, event):
        if False:
            i = 10
            return i + 15
        self._setHighlight(self._getRatingFromPosition(event.pos().x()))
        event.accept()

    def leaveEvent(self, event):
        if False:
            while True:
                i = 10
        self._setHighlight(0)
        event.accept()

    def _getRatingFromPosition(self, position):
        if False:
            for i in range(10):
                print('nop')
        rating = int((position - self._offset) / (self._star_size + self._star_spacing)) + 1
        if rating > self._maximum:
            rating = self._maximum
        return rating

    def _submitted(self, document, http, error):
        if False:
            while True:
                i = 10
        if error:
            self.tagger.window.set_statusbar_message(N_("Failed to submit rating for track '%(track_title)s' due to server error %(error)d"), {'track_title': self._track.metadata['title'], 'error': error}, echo=None)
            log.error('Failed to submit rating for %s (server HTTP error %d)', self._track, error)

    def _update_track(self):
        if False:
            i = 10
            return i + 15
        track = self._track
        rating = str(self._rating)
        track.metadata['~rating'] = rating
        for file in track.files:
            file.metadata['~rating'] = rating
        config = get_config()
        if config.setting['submit_ratings']:
            ratings = {('recording', track.id): self._rating}
            try:
                self.tagger.mb_api.submit_ratings(ratings, self._submitted)
            except ValueError:
                log.error('Failed to submit rating for recording %s', track.id, exc_info=True)

    def paintEvent(self, event=None):
        if False:
            for i in range(10):
                print('nop')
        painter = QtGui.QPainter(self)
        offset = self._offset
        for i in range(1, self._maximum + 1):
            if i <= self._rating or i <= self._highlight:
                pixmap = self._star_pixmap
            else:
                pixmap = self._star_gray_pixmap
            painter.drawPixmap(offset, 3, pixmap)
            offset += self._star_size + self._star_spacing