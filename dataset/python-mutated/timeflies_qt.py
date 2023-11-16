import sys
import reactivex
from reactivex import operators as ops
from reactivex.scheduler.mainloop import QtScheduler
from reactivex.subject import Subject
try:
    from PySide2 import QtCore
    from PySide2.QtWidgets import QApplication, QLabel, QWidget
except ImportError:
    try:
        from PyQt5 import QtCore
        from PyQt5.QtWidgets import QApplication, QLabel, QWidget
    except ImportError:
        raise ImportError('Please ensure either PySide2 or PyQt5 is available!')

class Window(QWidget):

    def __init__(self):
        if False:
            return 10
        QWidget.__init__(self)
        self.setWindowTitle('Rx for Python rocks')
        self.resize(600, 600)
        self.setMouseTracking(True)
        self.mousemove = Subject()

    def mouseMoveEvent(self, event):
        if False:
            i = 10
            return i + 15
        self.mousemove.on_next((event.x(), event.y()))

def main():
    if False:
        for i in range(10):
            print('nop')
    app = QApplication(sys.argv)
    scheduler = QtScheduler(QtCore)
    window = Window()
    window.show()
    text = 'TIME FLIES LIKE AN ARROW'

    def on_next(info):
        if False:
            while True:
                i = 10
        (label, (x, y), i) = info
        label.move(x + i * 12 + 15, y)
        label.show()

    def handle_label(label, i):
        if False:
            for i in range(10):
                print('nop')
        delayer = ops.delay(i * 0.1)
        mapper = ops.map(lambda xy: (label, xy, i))
        return window.mousemove.pipe(delayer, mapper)
    labeler = ops.flat_map_indexed(handle_label)
    mapper = ops.map(lambda c: QLabel(c, window))
    reactivex.from_(text).pipe(mapper, labeler).subscribe(on_next, on_error=print, scheduler=scheduler)
    sys.exit(app.exec_())
if __name__ == '__main__':
    main()