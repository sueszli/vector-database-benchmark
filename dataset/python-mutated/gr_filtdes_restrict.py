from gnuradio.filter import filter_design
import sys
try:
    from PyQt5 import Qt, QtCore, QtGui
except ImportError:
    print('Please install PyQt5 to run this script (http://www.riverbankcomputing.co.uk/software/pyqt/download)')
    raise SystemExit(1)
'\nCallback with restrict example\nFunction called when "design" button is pressed\nor pole-zero plot is changed\n'

def print_params(filtobj):
    if False:
        while True:
            i = 10
    print('Filter Count:', filtobj.get_filtercount())
    print('Filter type:', filtobj.get_restype())
    print('Filter params', filtobj.get_params())
    print('Filter Coefficients', filtobj.get_taps())
app = Qt.QApplication(sys.argv)
main_win = filter_design.launch(sys.argv, callback=print_params, restype='iir')
main_win.show()
app.exec_()