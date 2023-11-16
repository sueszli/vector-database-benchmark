from PyQt5.QtCore import Qt, pyqtSignal, QSize, QPropertyAnimation, QPoint
from PyQt5.QtGui import QPixmap, QPainter, QColor
from PyQt5.QtWidgets import QWidget, QGraphicsOpacityEffect, QHBoxLayout, QVBoxLayout
from ..common.icon import FluentIcon
from ..common.style_sheet import isDarkTheme, FluentStyleSheet
from ..components.widgets.button import TransparentToolButton
from ..components.widgets.tool_tip import ToolTipFilter
from ..components.widgets.slider import Slider
from ..components.widgets.label import CaptionLabel
from ..components.widgets.flyout import Flyout, FlyoutViewBase, PullUpFlyoutAnimationManager
from .media_player import MediaPlayer, MediaPlayerBase, QMediaPlayer

class MediaPlayBarButton(TransparentToolButton):
    """ Media play bar button """

    def _postInit(self):
        if False:
            i = 10
            return i + 15
        super()._postInit()
        self.installEventFilter(ToolTipFilter(self, 1000))
        self.setFixedSize(30, 30)
        self.setIconSize(QSize(16, 16))

class PlayButton(MediaPlayBarButton):
    """ Play button """

    def _postInit(self):
        if False:
            print('Hello World!')
        super()._postInit()
        self.setIconSize(QSize(14, 14))
        self.setPlay(False)

    def setPlay(self, isPlay: bool):
        if False:
            while True:
                i = 10
        if isPlay:
            self.setIcon(FluentIcon.PAUSE_BOLD)
            self.setToolTip(self.tr('Pause'))
        else:
            self.setIcon(FluentIcon.PLAY_SOLID)
            self.setToolTip(self.tr('Play'))

class VolumeView(FlyoutViewBase):
    """ Volume view """

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.muteButton = MediaPlayBarButton(FluentIcon.VOLUME, self)
        self.volumeSlider = Slider(Qt.Horizontal, self)
        self.volumeLabel = CaptionLabel('30', self)
        self.volumeSlider.setRange(0, 100)
        self.volumeSlider.setFixedWidth(208)
        self.setFixedSize(295, 64)
        h = self.height()
        self.muteButton.move(10, h // 2 - self.muteButton.height() // 2)
        self.volumeSlider.move(45, 21)

    def setMuted(self, isMute: bool):
        if False:
            while True:
                i = 10
        if isMute:
            self.muteButton.setIcon(FluentIcon.MUTE)
            self.muteButton.setToolTip(self.tr('Unmute'))
        else:
            self.muteButton.setIcon(FluentIcon.VOLUME)
            self.muteButton.setToolTip(self.tr('Mute'))

    def setVolume(self, volume: int):
        if False:
            while True:
                i = 10
        self.volumeSlider.setValue(volume)
        self.volumeLabel.setNum(volume)
        self.volumeLabel.adjustSize()
        tr = self.volumeLabel.fontMetrics().boundingRect(str(volume))
        self.volumeLabel.move(self.width() - 20 - tr.width(), self.height() // 2 - tr.height() // 2)

    def paintEvent(self, e):
        if False:
            print('Hello World!')
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        if isDarkTheme():
            painter.setBrush(QColor(46, 46, 46))
            painter.setPen(QColor(0, 0, 0, 20))
        else:
            painter.setBrush(QColor(248, 248, 248))
            painter.setPen(QColor(0, 0, 0, 10))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 8, 8)

class VolumeButton(MediaPlayBarButton):
    """ Volume button """
    volumeChanged = pyqtSignal(int)
    mutedChanged = pyqtSignal(bool)

    def _postInit(self):
        if False:
            i = 10
            return i + 15
        super()._postInit()
        self.volumeView = VolumeView(self)
        self.volumeFlyout = Flyout(self.volumeView, self.window(), False)
        self.setMuted(False)
        self.volumeFlyout.hide()
        self.volumeView.muteButton.clicked.connect(lambda : self.mutedChanged.emit(not self.isMuted))
        self.volumeView.volumeSlider.valueChanged.connect(self.volumeChanged)
        self.clicked.connect(self._showVolumeFlyout)

    def setMuted(self, isMute: bool):
        if False:
            while True:
                i = 10
        self.isMuted = isMute
        self.volumeView.setMuted(isMute)
        if isMute:
            self.setIcon(FluentIcon.MUTE)
        else:
            self.setIcon(FluentIcon.VOLUME)

    def setVolume(self, volume: int):
        if False:
            i = 10
            return i + 15
        self.volumeView.setVolume(volume)

    def _showVolumeFlyout(self):
        if False:
            print('Hello World!')
        if self.volumeFlyout.isVisible():
            return
        pos = PullUpFlyoutAnimationManager(self.volumeFlyout).position(self)
        self.volumeFlyout.exec(pos)

class MediaPlayBarBase(QWidget):
    """ Play bar base class """

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent=parent)
        self.player = None
        self.playButton = PlayButton(self)
        self.volumeButton = VolumeButton(self)
        self.progressSlider = Slider(Qt.Horizontal, self)
        self.opacityEffect = QGraphicsOpacityEffect(self)
        self.opacityAni = QPropertyAnimation(self.opacityEffect, b'opacity')
        self.opacityEffect.setOpacity(1)
        self.opacityAni.setDuration(250)
        self.setGraphicsEffect(self.opacityEffect)
        FluentStyleSheet.MEDIA_PLAYER.apply(self)

    def setMediaPlayer(self, player: MediaPlayerBase):
        if False:
            return 10
        ' set media player '
        self.player = player
        self.player.durationChanged.connect(self.progressSlider.setMaximum)
        self.player.positionChanged.connect(self._onPositionChanged)
        self.player.mediaStatusChanged.connect(self._onMediaStatusChanged)
        self.player.volumeChanged.connect(self.volumeButton.setVolume)
        self.player.mutedChanged.connect(self.volumeButton.setMuted)
        self.progressSlider.sliderMoved.connect(self.player.setPosition)
        self.progressSlider.clicked.connect(self.player.setPosition)
        self.playButton.clicked.connect(self.togglePlayState)
        self.volumeButton.volumeChanged.connect(self.player.setVolume)
        self.volumeButton.mutedChanged.connect(self.player.setMuted)
        self.player.setVolume(30)

    def fadeIn(self):
        if False:
            while True:
                i = 10
        self.opacityAni.setStartValue(self.opacityEffect.opacity())
        self.opacityAni.setEndValue(1)
        self.opacityAni.start()

    def fadeOut(self):
        if False:
            i = 10
            return i + 15
        self.opacityAni.setStartValue(self.opacityEffect.opacity())
        self.opacityAni.setEndValue(0)
        self.opacityAni.start()

    def play(self):
        if False:
            return 10
        self.player.play()

    def pause(self):
        if False:
            for i in range(10):
                print('nop')
        self.player.pause()

    def stop(self):
        if False:
            print('Hello World!')
        self.player.stop()

    def setVolume(self, volume: int):
        if False:
            print('Hello World!')
        ' Sets the volume of player '
        self.player.setVolume(volume)

    def setPosition(self, position: int):
        if False:
            for i in range(10):
                print('nop')
        ' Sets the position of media in ms '
        self.player.setPosition(position)

    def _onPositionChanged(self, position: int):
        if False:
            i = 10
            return i + 15
        self.progressSlider.setValue(position)

    def _onMediaStatusChanged(self, status):
        if False:
            while True:
                i = 10
        self.playButton.setPlay(self.player.isPlaying())

    def togglePlayState(self):
        if False:
            for i in range(10):
                print('nop')
        ' toggle the play state of media player '
        if self.player.isPlaying():
            self.player.pause()
        else:
            self.player.play()
        self.playButton.setPlay(self.player.isPlaying())

    def paintEvent(self, e):
        if False:
            return 10
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        if isDarkTheme():
            painter.setBrush(QColor(46, 46, 46))
            painter.setPen(QColor(0, 0, 0, 20))
        else:
            painter.setBrush(QColor(248, 248, 248))
            painter.setPen(QColor(0, 0, 0, 10))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 8, 8)

class SimpleMediaPlayBar(MediaPlayBarBase):
    """ simple media play bar """

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self.hBoxLayout = QHBoxLayout(self)
        self.hBoxLayout.setContentsMargins(10, 4, 10, 4)
        self.hBoxLayout.setSpacing(6)
        self.hBoxLayout.addWidget(self.playButton, 0, Qt.AlignLeft)
        self.hBoxLayout.addWidget(self.progressSlider, 1)
        self.hBoxLayout.addWidget(self.volumeButton, 0)
        self.setFixedHeight(48)
        self.setMediaPlayer(MediaPlayer(self))

    def addButton(self, button: MediaPlayBarButton):
        if False:
            for i in range(10):
                print('nop')
        ' add button to the right side of play bar '
        self.hBoxLayout.addWidget(button, 0)

class StandardMediaPlayBar(MediaPlayBarBase):
    """ Standard media play bar """

    def __init__(self, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self.vBoxLayout = QVBoxLayout(self)
        self.timeLayout = QHBoxLayout()
        self.buttonLayout = QHBoxLayout()
        self.leftButtonContainer = QWidget()
        self.centerButtonContainer = QWidget()
        self.rightButtonContainer = QWidget()
        self.leftButtonLayout = QHBoxLayout(self.leftButtonContainer)
        self.centerButtonLayout = QHBoxLayout(self.centerButtonContainer)
        self.rightButtonLayout = QHBoxLayout(self.rightButtonContainer)
        self.skipBackButton = MediaPlayBarButton(FluentIcon.SKIP_BACK, self)
        self.skipForwardButton = MediaPlayBarButton(FluentIcon.SKIP_FORWARD, self)
        self.currentTimeLabel = CaptionLabel('0:00:00', self)
        self.remainTimeLabel = CaptionLabel('0:00:00', self)
        self.__initWidgets()

    def __initWidgets(self):
        if False:
            return 10
        self.setFixedHeight(102)
        self.vBoxLayout.setSpacing(6)
        self.vBoxLayout.setContentsMargins(5, 9, 5, 9)
        self.vBoxLayout.addWidget(self.progressSlider, 1, Qt.AlignTop)
        self.vBoxLayout.addLayout(self.timeLayout)
        self.timeLayout.setContentsMargins(10, 0, 10, 0)
        self.timeLayout.addWidget(self.currentTimeLabel, 0, Qt.AlignLeft)
        self.timeLayout.addWidget(self.remainTimeLabel, 0, Qt.AlignRight)
        self.vBoxLayout.addStretch(1)
        self.vBoxLayout.addLayout(self.buttonLayout, 1)
        self.buttonLayout.setContentsMargins(0, 0, 0, 0)
        self.leftButtonLayout.setContentsMargins(4, 0, 0, 0)
        self.centerButtonLayout.setContentsMargins(0, 0, 0, 0)
        self.rightButtonLayout.setContentsMargins(0, 0, 4, 0)
        self.leftButtonLayout.addWidget(self.volumeButton, 0, Qt.AlignLeft)
        self.centerButtonLayout.addWidget(self.skipBackButton)
        self.centerButtonLayout.addWidget(self.playButton)
        self.centerButtonLayout.addWidget(self.skipForwardButton)
        self.buttonLayout.addWidget(self.leftButtonContainer, 0, Qt.AlignLeft)
        self.buttonLayout.addWidget(self.centerButtonContainer, 0, Qt.AlignHCenter)
        self.buttonLayout.addWidget(self.rightButtonContainer, 0, Qt.AlignRight)
        self.setMediaPlayer(MediaPlayer(self))
        self.skipBackButton.clicked.connect(lambda : self.skipBack(10000))
        self.skipForwardButton.clicked.connect(lambda : self.skipForward(30000))

    def skipBack(self, ms: int):
        if False:
            return 10
        ' Back up for specified milliseconds '
        self.player.setPosition(self.player.position() - ms)

    def skipForward(self, ms: int):
        if False:
            while True:
                i = 10
        ' Fast forward specified milliseconds '
        self.player.setPosition(self.player.position() + ms)

    def _onPositionChanged(self, position: int):
        if False:
            print('Hello World!')
        super()._onPositionChanged(position)
        self.currentTimeLabel.setText(self._formatTime(position))
        self.remainTimeLabel.setText(self._formatTime(self.player.duration() - position))

    def _formatTime(self, time: int):
        if False:
            for i in range(10):
                print('nop')
        time = int(time / 1000)
        s = time % 60
        m = int(time / 60)
        h = int(time / 3600)
        return f'{h}:{m:02}:{s:02}'