from PyQt5.QtCore import Qt, QPoint, pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QSlider, QPushButton, QWidget

class _Slider(QWidget):
    """A popup slider.

    TODO: this slide can become a independent component?
    TODO: draw border radius for widget
    NOTE: inherit from QWidget instead of QSlider since QSlider can not
    work with Qt.Popup window flag well. Currently, I don't know why.
    """
    about_to_hide = pyqtSignal()

    def __init__(self, parent=None, initial_value=100):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self._slider = QSlider(self)
        self._layout = QVBoxLayout(self)
        self._layout.addWidget(self._slider)
        self._layout.setSpacing(0)
        self.sliderMoved = self._slider.sliderMoved
        self.setValue = self._slider.setValue
        self._slider.setMinimum(0)
        self._slider.setMaximum(100)
        self._slider.setValue(initial_value)
        self.setWindowFlags(Qt.Popup)

    def is_mute(self):
        if False:
            print('Hello World!')
        return self._slider.value() <= 0

    def hideEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        super().hideEvent(event)
        self.about_to_hide.emit()

    def showEvent(self, event):
        if False:
            i = 10
            return i + 15
        parent = self.parent()
        if parent:
            assert isinstance(parent, QWidget)
            pgeom = parent.geometry()
            geom = self.geometry()
            x = (pgeom.width() - geom.width()) // 2
            y = -geom.height() - 10
            point = QPoint(x, y)
            self.move(parent.mapToGlobal(point))

class VolumeButton(QPushButton):
    UNMUTED_ICON = 0
    MUTED_ICON = 1
    change_volume_needed = pyqtSignal([int])

    def __init__(self, parent=None, *, icons=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self._icons = icons
        if self._icons:
            self._icon = VolumeButton.UNMUTED_ICON
            self.setIcon(self._icons['unmuted'])
        self.slider = _Slider(self)
        self.slider.hide()
        self.setCheckable(True)
        self.setMaximumWidth(40)
        self.slider.about_to_hide.connect(lambda : self.setChecked(False))
        self.slider.sliderMoved.connect(self.on_slider_moved)
        self.clicked.connect(self.slider.show)

    def on_volume_changed(self, value):
        if False:
            for i in range(10):
                print('nop')
        '(alpha)\n\n        .. versionadd:: 3.4\n        '
        self.slider.setValue(value)

    def on_slider_moved(self, value):
        if False:
            print('Hello World!')
        self.change_volume_needed.emit(value)
        if not self._icons:
            return
        if self.slider.is_mute():
            self.setIcon(self._icons['muted'])
            self._icon = VolumeButton.MUTED_ICON
        elif self._icon == VolumeButton.MUTED_ICON:
            self.setIcon(self._icons['unmuted'])
            self._icon = VolumeButton.UNMUTED_ICON