HOST = 'localhost'
PORT = 4223
UID = 'XYZ'
from PyQt4 import QtGui, QtCore
import sys
from tinkerforge.ip_connection import IPConnection
from tinkerforge.bricklet_temperature_v2 import BrickletTemperatureV2

class Window(QtGui.QWidget):
    qtcb_temperature = QtCore.pyqtSignal(int)

    def __init__(self):
        if False:
            i = 10
            return i + 15
        QtGui.QWidget.__init__(self)
        self.button = QtGui.QPushButton('Refresh', self)
        self.button.clicked.connect(self.handle_button)
        self.label = QtGui.QLabel('TBD')
        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.button)
        layout.addWidget(self.label)
        self.ipcon = IPConnection()
        self.temperature = BrickletTemperatureV2(UID, self.ipcon)
        self.ipcon.connect(HOST, PORT)
        self.qtcb_temperature.connect(self.cb_temperature)
        self.temperature.register_callback(BrickletTemperatureV2.CALLBACK_TEMPERATURE, self.qtcb_temperature.emit)
        self.temperature.set_temperature_callback_period(1000)
        self.handle_button()

    def handle_button(self):
        if False:
            return 10
        self.cb_temperature(self.temperature.get_temperature())

    def cb_temperature(self, temperature):
        if False:
            i = 10
            return i + 15
        self.label.setText(u'Temperature: {0} °C'.format(temperature / 100.0))
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())