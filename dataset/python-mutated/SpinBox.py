"""
This example demonstrates the SpinBox widget, which is an extension of 
QDoubleSpinBox providing some advanced features:

  * SI-prefixed units
  * Non-linear stepping modes
  * Bounded/unbounded values

"""
import ast
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
app = pg.mkQApp('SpinBox Example')
spins = [('Floating-point spin box, min=0, no maximum.<br>Non-finite values (nan, inf) are permitted.', pg.SpinBox(value=5.0, bounds=[0, None], finite=False)), ('Integer spin box, dec stepping<br>(1-9, 10-90, 100-900, etc), decimals=4', pg.SpinBox(value=10, int=True, dec=True, minStep=1, step=1, decimals=4)), ('Float with SI-prefixed units<br>(n, u, m, k, M, etc)', pg.SpinBox(value=0.9, suffix='V', siPrefix=True)), ('Float with SI-prefixed units,<br>dec step=0.1, minStep=0.1', pg.SpinBox(value=1.0, suffix='PSI', siPrefix=True, dec=True, step=0.1, minStep=0.1)), ('Float with SI-prefixed units,<br>dec step=0.5, minStep=0.01', pg.SpinBox(value=1.0, suffix='V', siPrefix=True, dec=True, step=0.5, minStep=0.01)), ('Float with SI-prefixed units,<br>dec step=1.0, minStep=0.001', pg.SpinBox(value=1.0, suffix='V', siPrefix=True, dec=True, step=1.0, minStep=0.001)), ('Float with SI prefix but no suffix', pg.SpinBox(value=1000000000.0, siPrefix=True)), ('Float with custom formatting', pg.SpinBox(value=23.07, format='${value:0.02f}', regex='\\$?(?P<number>(-?\\d+(\\.\\d+)?)|(-?\\.\\d+))$')), ('Int with suffix', pg.SpinBox(value=999, step=1, int=True, suffix='V')), ('Int with custom formatting', pg.SpinBox(value=4567, step=1, int=True, bounds=[0, None], format='0x{value:X}', regex='(0x)?(?P<number>[0-9a-fA-F]+)$', evalFunc=lambda s: ast.literal_eval('0x' + s))), ('Integer with bounds=[10, 20] and wrapping', pg.SpinBox(value=10, bounds=[10, 20], int=True, minStep=1, step=1, wrapping=True))]
win = QtWidgets.QMainWindow()
win.setWindowTitle('pyqtgraph example: SpinBox')
cw = QtWidgets.QWidget()
layout = QtWidgets.QGridLayout()
cw.setLayout(layout)
win.setCentralWidget(cw)
win.show()
changingLabel = QtWidgets.QLabel()
changedLabel = QtWidgets.QLabel()
changingLabel.setMinimumWidth(200)
font = changingLabel.font()
font.setBold(True)
font.setPointSize(14)
changingLabel.setFont(font)
changedLabel.setFont(font)
labels = []

def valueChanged(sb):
    if False:
        for i in range(10):
            print('nop')
    changedLabel.setText('Final value: %s' % str(sb.value()))

def valueChanging(sb, value):
    if False:
        return 10
    changingLabel.setText('Value changing: %s' % str(sb.value()))
for (text, spin) in spins:
    label = QtWidgets.QLabel(text)
    labels.append(label)
    layout.addWidget(label)
    layout.addWidget(spin)
    spin.sigValueChanged.connect(valueChanged)
    spin.sigValueChanging.connect(valueChanging)
layout.addWidget(changingLabel, 0, 1)
layout.addWidget(changedLabel, 2, 1)
if __name__ == '__main__':
    pg.exec()