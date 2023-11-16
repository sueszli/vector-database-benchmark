"""
Created on 2018年11月6日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: SerialDebugAssistant
@description: 串口调试小助手
"""
from PyQt5.QtCore import pyqtSlot, QIODevice, QByteArray
from PyQt5.QtSerialPort import QSerialPortInfo, QSerialPort
from PyQt5.QtWidgets import QWidget, QMessageBox
from Lib.UiSerialPort import Ui_FormSerialPort

class Window(QWidget, Ui_FormSerialPort):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(Window, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self._serial = QSerialPort(self)
        self._serial.readyRead.connect(self.onReadyRead)
        self.getAvailablePorts()

    @pyqtSlot()
    def on_buttonConnect_clicked(self):
        if False:
            for i in range(10):
                print('nop')
        if self._serial.isOpen():
            self._serial.close()
            self.textBrowser.append('串口已关闭')
            self.buttonConnect.setText('打开串口')
            self.labelStatus.setProperty('isOn', False)
            self.labelStatus.style().polish(self.labelStatus)
            return
        name = self.comboBoxPort.currentText()
        if not name:
            QMessageBox.critical(self, '错误', '没有选择串口')
            return
        port = self._ports[name]
        self._serial.setPortName(port.systemLocation())
        self._serial.setBaudRate(getattr(QSerialPort, 'Baud' + self.comboBoxBaud.currentText()))
        self._serial.setParity(getattr(QSerialPort, self.comboBoxParity.currentText() + 'Parity'))
        self._serial.setDataBits(getattr(QSerialPort, 'Data' + self.comboBoxData.currentText()))
        self._serial.setStopBits(getattr(QSerialPort, self.comboBoxStop.currentText()))
        self._serial.setFlowControl(QSerialPort.NoFlowControl)
        ok = self._serial.open(QIODevice.ReadWrite)
        if ok:
            self.textBrowser.append('打开串口成功')
            self.buttonConnect.setText('关闭串口')
            self.labelStatus.setProperty('isOn', True)
            self.labelStatus.style().polish(self.labelStatus)
        else:
            self.textBrowser.append('打开串口失败')
            self.buttonConnect.setText('打开串口')
            self.labelStatus.setProperty('isOn', False)
            self.labelStatus.style().polish(self.labelStatus)

    @pyqtSlot()
    def on_buttonSend_clicked(self):
        if False:
            return 10
        if not self._serial.isOpen():
            print('串口未连接')
            return
        text = self.plainTextEdit.toPlainText()
        if not text:
            return
        text = QByteArray(text.encode('gb2312'))
        if self.checkBoxHexSend.isChecked():
            text = text.toHex()
        print('发送数据:', text)
        self._serial.write(text)

    def onReadyRead(self):
        if False:
            print('Hello World!')
        if self._serial.bytesAvailable():
            data = self._serial.readAll()
            if self.checkBoxHexView.isChecked():
                data = data.toHex()
            data = data.data()
            try:
                self.textBrowser.append('我收到了: ' + data.decode('gb2312'))
            except:
                self.textBrowser.append('我收到了: ' + repr(data))

    def getAvailablePorts(self):
        if False:
            print('Hello World!')
        self._ports = {}
        infos = QSerialPortInfo.availablePorts()
        infos.reverse()
        for info in infos:
            self._ports[info.portName()] = info
            self.comboBoxPort.addItem(info.portName())

    def closeEvent(self, event):
        if False:
            while True:
                i = 10
        if self._serial.isOpen():
            self._serial.close()
        super(Window, self).closeEvent(event)
if __name__ == '__main__':
    import sys
    import cgitb
    cgitb.enable(format='text')
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())