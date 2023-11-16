from gnuradio import gr
from gnuradio import blocks
from gnuradio import blocks
import sys
try:
    from gnuradio import qtgui
    from PyQt5 import QtWidgets, Qt
    import sip
except ImportError:
    print('Error: Program requires PyQt5 and gr-qtgui.')
    sys.exit(1)

class dialog_box(QtWidgets.QWidget):

    def __init__(self, display):
        if False:
            i = 10
            return i + 15
        QtWidgets.QWidget.__init__(self, None)
        self.setWindowTitle('PyQt Test GUI')
        self.boxlayout = QtWidgets.QBoxLayout(QtWidgets.QBoxLayout.LeftToRight, self)
        self.boxlayout.addWidget(display, 1)
        self.resize(800, 500)

class my_top_block(gr.top_block):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        gr.top_block.__init__(self)
        self.qapp = QtWidgets.QApplication(sys.argv)
        data0 = 10 * [0] + 40 * [1, 0] + 10 * [0]
        data0 += 10 * [0] + 40 * [0, 1] + 10 * [0]
        data1 = 20 * [0] + [0, 0, 0, 1, 1, 1, 0, 0, 0, 0] + 70 * [0]
        ncols = 100.25
        nrows = 100
        fs = 200
        src0 = blocks.vector_source_b(data0, True)
        src1 = blocks.vector_source_b(data1, True)
        thr = blocks.throttle(gr.sizeof_char, 50000)
        head = blocks.head(gr.sizeof_char, 10000000)
        self.snk1 = qtgui.time_raster_sink_b(fs, nrows, ncols, [], [], 'Time Raster Example', 2, None)
        self.connect(src0, thr, (self.snk1, 0))
        self.connect(src1, (self.snk1, 1))
        pyQt = self.snk1.qwidget()
        pyWin = sip.wrapinstance(pyQt, QtWidgets.QWidget)
        self.main_box = dialog_box(pyWin)
        self.main_box.show()
if __name__ == '__main__':
    tb = my_top_block()
    tb.start()
    tb.qapp.exec_()
    tb.stop()