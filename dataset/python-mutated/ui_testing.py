import time
from pyqtgraph.Qt import QtCore, QtGui, QtTest, QtWidgets

def resizeWindow(win, w, h, timeout=2.0):
    if False:
        print('Hello World!')
    'Resize a window and wait until it has the correct size.\n    \n    This is required for unit testing on some platforms that do not guarantee\n    immediate response from the windowing system.\n    '
    QtWidgets.QApplication.processEvents()
    QtTest.QTest.qWaitForWindowExposed(win)
    win.resize(w, h)
    start = time.time()
    while True:
        (w1, h1) = (win.width(), win.height())
        if (w, h) == (w1, h1):
            return
        QtTest.QTest.qWait(10)
        if time.time() - start > timeout:
            raise TimeoutError('Window resize failed (requested %dx%d, got %dx%d)' % (w, h, w1, h1))

def mousePress(widget, pos, button, modifier=None):
    if False:
        while True:
            i = 10
    if isinstance(widget, QtWidgets.QGraphicsView):
        widget = widget.viewport()
    global_pos = QtCore.QPointF(widget.mapToGlobal(pos.toPoint()))
    if modifier is None:
        modifier = QtCore.Qt.KeyboardModifier.NoModifier
    event = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseButtonPress, pos, global_pos, button, QtCore.Qt.MouseButton.NoButton, modifier)
    QtWidgets.QApplication.sendEvent(widget, event)

def mouseRelease(widget, pos, button, modifier=None):
    if False:
        return 10
    if isinstance(widget, QtWidgets.QGraphicsView):
        widget = widget.viewport()
    global_pos = QtCore.QPointF(widget.mapToGlobal(pos.toPoint()))
    if modifier is None:
        modifier = QtCore.Qt.KeyboardModifier.NoModifier
    event = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseButtonRelease, pos, global_pos, button, QtCore.Qt.MouseButton.NoButton, modifier)
    QtWidgets.QApplication.sendEvent(widget, event)

def mouseMove(widget, pos, buttons=None, modifier=None):
    if False:
        while True:
            i = 10
    if isinstance(widget, QtWidgets.QGraphicsView):
        widget = widget.viewport()
    global_pos = QtCore.QPointF(widget.mapToGlobal(pos.toPoint()))
    if modifier is None:
        modifier = QtCore.Qt.KeyboardModifier.NoModifier
    if buttons is None:
        buttons = QtCore.Qt.MouseButton.NoButton
    event = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseMove, pos, global_pos, QtCore.Qt.MouseButton.NoButton, buttons, modifier)
    QtWidgets.QApplication.sendEvent(widget, event)

def mouseDrag(widget, pos1, pos2, button, modifier=None):
    if False:
        while True:
            i = 10
    mouseMove(widget, pos1)
    mousePress(widget, pos1, button, modifier)
    mouseMove(widget, pos2, button, modifier)
    mouseRelease(widget, pos2, button, modifier)

def mouseClick(widget, pos, button, modifier=None):
    if False:
        i = 10
        return i + 15
    mouseMove(widget, pos)
    mousePress(widget, pos, button, modifier)
    mouseRelease(widget, pos, button, modifier)