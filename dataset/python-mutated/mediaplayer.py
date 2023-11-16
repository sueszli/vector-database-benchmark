from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from MainWindow import Ui_MainWindow

def hhmmss(ms):
    if False:
        return 10
    s = round(ms / 1000)
    (m, s) = divmod(s, 60)
    (h, m) = divmod(m, 60)
    return '%d:%02d:%02d' % (h, m, s) if h else '%d:%02d' % (m, s)

class ViewerWindow(QMainWindow):
    state = pyqtSignal(bool)

    def closeEvent(self, e):
        if False:
            return 10
        self.state.emit(False)

class PlaylistModel(QAbstractListModel):

    def __init__(self, playlist, *args, **kwargs):
        if False:
            print('Hello World!')
        super(PlaylistModel, self).__init__(*args, **kwargs)
        self.playlist = playlist

    def data(self, index, role):
        if False:
            i = 10
            return i + 15
        if role == Qt.DisplayRole:
            media = self.playlist.media(index.row())
            return media.canonicalUrl().fileName()

    def rowCount(self, index):
        if False:
            for i in range(10):
                print('nop')
        return self.playlist.mediaCount()

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.player = QMediaPlayer()
        self.player.error.connect(self.erroralert)
        self.player.play()
        self.playlist = QMediaPlaylist()
        self.player.setPlaylist(self.playlist)
        self.viewer = ViewerWindow(self)
        self.viewer.setWindowFlags(self.viewer.windowFlags() | Qt.WindowStaysOnTopHint)
        self.viewer.setMinimumSize(QSize(480, 360))
        videoWidget = QVideoWidget()
        self.viewer.setCentralWidget(videoWidget)
        self.player.setVideoOutput(videoWidget)
        self.playButton.pressed.connect(self.player.play)
        self.pauseButton.pressed.connect(self.player.pause)
        self.stopButton.pressed.connect(self.player.stop)
        self.volumeSlider.valueChanged.connect(self.player.setVolume)
        self.viewButton.toggled.connect(self.toggle_viewer)
        self.viewer.state.connect(self.viewButton.setChecked)
        self.previousButton.pressed.connect(self.playlist.previous)
        self.nextButton.pressed.connect(self.playlist.next)
        self.model = PlaylistModel(self.playlist)
        self.playlistView.setModel(self.model)
        self.playlist.currentIndexChanged.connect(self.playlist_position_changed)
        selection_model = self.playlistView.selectionModel()
        selection_model.selectionChanged.connect(self.playlist_selection_changed)
        self.player.durationChanged.connect(self.update_duration)
        self.player.positionChanged.connect(self.update_position)
        self.timeSlider.valueChanged.connect(self.player.setPosition)
        self.open_file_action.triggered.connect(self.open_file)
        self.setAcceptDrops(True)
        self.show()

    def dragEnterEvent(self, e):
        if False:
            i = 10
            return i + 15
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        for url in e.mimeData().urls():
            self.playlist.addMedia(QMediaContent(url))
        self.model.layoutChanged.emit()
        if self.player.state() != QMediaPlayer.PlayingState:
            i = self.playlist.mediaCount() - len(e.mimeData().urls())
            self.playlist.setCurrentIndex(i)
            self.player.play()

    def open_file(self):
        if False:
            for i in range(10):
                print('nop')
        (path, _) = QFileDialog.getOpenFileName(self, 'Open file', '', 'mp3 Audio (*.mp3);;mp4 Video (*.mp4);;Movie files (*.mov);;All files (*.*)')
        if path:
            self.playlist.addMedia(QMediaContent(QUrl.fromLocalFile(path)))
        self.model.layoutChanged.emit()

    def update_duration(self, duration):
        if False:
            return 10
        self.timeSlider.setMaximum(duration)
        if duration >= 0:
            self.totalTimeLabel.setText(hhmmss(duration))

    def update_position(self, position):
        if False:
            i = 10
            return i + 15
        if position >= 0:
            self.currentTimeLabel.setText(hhmmss(position))
        self.timeSlider.blockSignals(True)
        self.timeSlider.setValue(position)
        self.timeSlider.blockSignals(False)

    def playlist_selection_changed(self, ix):
        if False:
            while True:
                i = 10
        i = ix.indexes()[0].row()
        self.playlist.setCurrentIndex(i)

    def playlist_position_changed(self, i):
        if False:
            for i in range(10):
                print('nop')
        if i > -1:
            ix = self.model.index(i)
            self.playlistView.setCurrentIndex(ix)

    def toggle_viewer(self, state):
        if False:
            i = 10
            return i + 15
        if state:
            self.viewer.show()
        else:
            self.viewer.hide()

    def erroralert(self, *args):
        if False:
            return 10
        print(args)
if __name__ == '__main__':
    app = QApplication([])
    app.setApplicationName('Failamp')
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    app.setStyleSheet('QToolTip { color: #ffffff; background-color: #2a82da; border: 1px solid white; }')
    window = MainWindow()
    app.exec_()