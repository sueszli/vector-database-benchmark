from os import path as op
import warnings
from vispy.testing import requires_application

@requires_application('pyqt4', has=['uic'])
def test_qt_designer():
    if False:
        print('Hello World!')
    'Embed Canvas via Qt Designer'
    from PyQt4 import QtGui, uic
    app = QtGui.QApplication.instance()
    if app is None:
        app = QtGui.QApplication([])
    fname = op.join(op.dirname(__file__), 'qt-designer.ui')
    with warnings.catch_warnings(record=True):
        (WindowTemplate, TemplateBaseClass) = uic.loadUiType(fname)

    class MainWindow(TemplateBaseClass):

        def __init__(self):
            if False:
                return 10
            TemplateBaseClass.__init__(self)
            self.ui = WindowTemplate()
            self.ui.setupUi(self)
    win = MainWindow()
    try:
        canvas = win.ui.canvas
        canvas.central_widget.add_view()
        win.show()
        app.processEvents()
    finally:
        win.close()
    return win
if __name__ == '__main__':
    win = test_qt_designer()
    win.show()