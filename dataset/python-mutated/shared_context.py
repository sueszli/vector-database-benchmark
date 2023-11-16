"""
This is a very simple example that demonstrates using a shared context
between two Qt widgets.
"""
from PyQt5 import QtWidgets, QtCore
from functools import partial
from vispy.app import Timer
from vispy.scene.visuals import Text
from vispy.scene.widgets import ViewBox
from vispy.scene import SceneCanvas

def on_resize(canvas, vb, event):
    if False:
        return 10
    vb.pos = (1, 1)
    vb.size = (canvas.size[0] - 2, canvas.size[1] - 2)

class Window(QtWidgets.QWidget):

    def __init__(self):
        if False:
            return 10
        super(Window, self).__init__()
        box = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight, self)
        self.resize(500, 200)
        self.setLayout(box)
        self.canvas_0 = SceneCanvas(bgcolor='w')
        self.vb_0 = ViewBox(parent=self.canvas_0.scene, bgcolor='r')
        self.vb_0.camera.rect = (-1, -1, 2, 2)
        self.canvas_0.events.initialize.connect(self.on_init)
        self.canvas_0.events.resize.connect(partial(on_resize, self.canvas_0, self.vb_0))
        box.addWidget(self.canvas_0.native)
        self.canvas_1 = SceneCanvas(bgcolor='w', shared=self.canvas_0.context)
        self.vb_1 = ViewBox(parent=self.canvas_1.scene, bgcolor='b')
        self.vb_1.camera.rect = (-1, -1, 2, 2)
        self.canvas_1.events.resize.connect(partial(on_resize, self.canvas_1, self.vb_1))
        box.addWidget(self.canvas_1.native)
        self.tick_count = 0
        self.timer = Timer(interval=1.0, connect=self.on_timer, start=True)
        self.setWindowTitle('Shared contexts')
        self.show()

    def on_init(self, event):
        if False:
            i = 10
            return i + 15
        self.text = Text('Initialized', font_size=40.0, anchor_x='left', anchor_y='top', parent=[self.vb_0.scene, self.vb_1.scene])

    def on_timer(self, event):
        if False:
            i = 10
            return i + 15
        self.tick_count += 1
        self.text.text = 'Tick #%s' % self.tick_count
        self.canvas_0.update()
        self.canvas_1.update()

    def keyPressEvent(self, event):
        if False:
            i = 10
            return i + 15
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
        elif event.key() == QtCore.Qt.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
if __name__ == '__main__':
    qt_app = QtWidgets.QApplication([])
    ex = Window()
    qt_app.exec_()