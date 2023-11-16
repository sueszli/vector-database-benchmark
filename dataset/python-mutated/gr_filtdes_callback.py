from gnuradio.filter import filter_design
import sys
try:
    from PyQt5 import Qt, QtCore, QtGui
except ImportError:
    print('Please install PyQt5 to run this script (http://www.riverbankcomputing.co.uk/software/pyqt/download)')
    raise SystemExit(1)
'\nCallback example\nFunction called when "design" button is pressed\nor pole-zero plot is changed\nlaunch function returns gr_filter_design mainwindow\nobject when callback is not None\n'

def print_params(filtobj):
    if False:
        while True:
            i = 10
    print('Filter Count:', filtobj.get_filtercount())
    print('Filter type:', filtobj.get_restype())
    print('Filter params', filtobj.get_params())
    print('Filter Coefficients', filtobj.get_taps())
app = Qt.QApplication(sys.argv)
main_win = filter_design.launch(sys.argv, print_params)
main_win.show()
app.exec_()