__license__ = 'GPL v3'
__copyright__ = '2010, Kovid Goyal <kovid@kovidgoyal.net>'
__docformat__ = 'restructuredtext en'
from qt.core import QToolButton, QSize, QPropertyAnimation, Qt, QMetaObject, pyqtProperty, QSizePolicy, QWidget, QIcon, QPainter, QStyleOptionToolButton, QStyle, QAbstractAnimation
from calibre.gui2 import config

class ThrobbingButton(QToolButton):

    @pyqtProperty(int)
    def icon_size(self):
        if False:
            print('Hello World!')
        return self._icon_size

    @icon_size.setter
    def icon_size(self, value):
        if False:
            i = 10
            return i + 15
        self._icon_size = value

    def __init__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        QToolButton.__init__(self, *args)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self._icon_size = -1
        QToolButton.setIcon(self, QIcon.ic('donate.png'))
        self.setText('\xa0')
        self.animation = QPropertyAnimation(self, b'icon_size', self)
        self.animation.setDuration(int(60 / 72.0 * 1000))
        self.animation.setLoopCount(4)
        self.animation.valueChanged.connect(self.value_changed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.animation.finished.connect(self.animation_finished)

    def animation_finished(self):
        if False:
            while True:
                i = 10
        self.icon_size = self.iconSize().width()

    def enterEvent(self, ev):
        if False:
            return 10
        self.start_animation()

    def leaveEvent(self, ev):
        if False:
            return 10
        self.stop_animation()

    def value_changed(self, val):
        if False:
            for i in range(10):
                print('nop')
        self.update()

    def start_animation(self):
        if False:
            print('Hello World!')
        if config['disable_animations']:
            return
        if self.animation.state() != QAbstractAnimation.State.Stopped or not self.isVisible():
            return
        size = self.iconSize().width()
        if size < 1:
            size = max(0, self.width() - 4)
        smaller = int(0.7 * size)
        self.animation.setStartValue(smaller)
        self.animation.setEndValue(size)
        QMetaObject.invokeMethod(self.animation, 'start', Qt.ConnectionType.QueuedConnection)

    def stop_animation(self):
        if False:
            for i in range(10):
                print('nop')
        self.animation.stop()
        self.animation_finished()

    def paintEvent(self, ev):
        if False:
            i = 10
            return i + 15
        size = self._icon_size if self._icon_size > 10 else self.iconSize().width()
        size = size or max(0, self.width() - 4)
        p = QPainter(self)
        opt = QStyleOptionToolButton()
        self.initStyleOption(opt)
        s = self.style()
        opt.iconSize = QSize(size, size)
        s.drawComplexControl(QStyle.ComplexControl.CC_ToolButton, opt, p, self)
if __name__ == '__main__':
    from qt.core import QApplication, QHBoxLayout
    app = QApplication([])
    w = QWidget()
    w.setLayout(QHBoxLayout())
    b = ThrobbingButton()
    b.setIcon(QIcon.ic('donate.png'))
    w.layout().addWidget(b)
    w.show()
    b.set_normal_icon_size(64, 64)
    b.start_animation()
    app.exec()