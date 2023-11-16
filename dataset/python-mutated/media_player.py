from PyQt5.QtCore import Qt, pyqtSignal, QObject, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QWidget

class MediaPlayerBase(QObject):
    """ Media player base class """
    mediaStatusChanged = pyqtSignal(QMediaPlayer.MediaStatus)
    playbackRateChanged = pyqtSignal(float)
    positionChanged = pyqtSignal(int)
    durationChanged = pyqtSignal(int)
    sourceChanged = pyqtSignal(QUrl)
    volumeChanged = pyqtSignal(int)
    mutedChanged = pyqtSignal(bool)

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent=parent)

    def isPlaying(self):
        if False:
            i = 10
            return i + 15
        ' Whether the media is playing '
        raise NotImplementedError

    def mediaStatus(self) -> QMediaPlayer.MediaStatus:
        if False:
            print('Hello World!')
        ' Return the status of the current media stream '
        raise NotImplementedError

    def playbackState(self) -> QMediaPlayer.State:
        if False:
            print('Hello World!')
        ' Return the playback status of the current media stream '
        raise NotImplementedError

    def duration(self):
        if False:
            print('Hello World!')
        ' Returns the duration of the current media in ms '
        raise NotImplementedError

    def position(self):
        if False:
            i = 10
            return i + 15
        ' Returns the current position inside the media being played back in ms '
        raise NotImplementedError

    def volume(self):
        if False:
            for i in range(10):
                print('nop')
        ' Return the volume of player '
        raise NotImplementedError

    def source(self) -> QUrl:
        if False:
            while True:
                i = 10
        ' Return the active media source being used '
        raise NotImplementedError

    def pause(self):
        if False:
            return 10
        ' Pause playing the current source '
        raise NotImplementedError

    def play(self):
        if False:
            for i in range(10):
                print('nop')
        ' Start or resume playing the current source '
        raise NotImplementedError

    def stop(self):
        if False:
            print('Hello World!')
        ' Stop playing, and reset the play position to the beginning '
        raise NotImplementedError

    def playbackRate(self) -> float:
        if False:
            i = 10
            return i + 15
        ' Return the playback rate of the current media '
        raise NotImplementedError

    def setPosition(self, position: int):
        if False:
            while True:
                i = 10
        ' Sets the position of media in ms '
        raise NotImplementedError

    def setSource(self, media: QUrl):
        if False:
            while True:
                i = 10
        ' Sets the current source '
        raise NotImplementedError

    def setPlaybackRate(self, rate: float):
        if False:
            i = 10
            return i + 15
        ' Sets the playback rate of player '
        raise NotImplementedError

    def setVolume(self, volume: int):
        if False:
            i = 10
            return i + 15
        ' Sets the volume of player '
        raise NotImplementedError

    def setMuted(self, isMuted: bool):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def videoOutput(self) -> QObject:
        if False:
            while True:
                i = 10
        ' Return the video output to be used by the media player '
        raise NotImplementedError

    def setVideoOutput(self, output: QObject) -> None:
        if False:
            i = 10
            return i + 15
        ' Sets the video output to be used by the media player '
        raise NotImplementedError

class MediaPlayer(QMediaPlayer):
    """ Media player """
    sourceChanged = pyqtSignal(QUrl)

    def __init__(self, parent=None):
        if False:
            return 10
        super().__init__(parent=parent)
        self.mediaChanged.connect(lambda i: i.canonicalUrl())
        self.setNotifyInterval(1000)

    def isPlaying(self):
        if False:
            while True:
                i = 10
        return self.state() == QMediaPlayer.PlayingState

    def source(self) -> QUrl:
        if False:
            while True:
                i = 10
        ' Return the active media source being used '
        return self.currentMedia().canonicalUrl()

    def setSource(self, media: QUrl):
        if False:
            return 10
        ' Sets the current source '
        self.setMedia(QMediaContent(media))