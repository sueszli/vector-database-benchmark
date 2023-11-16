from PyQt5 import uic
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtNetwork import QTcpSocket
from PyQt5.QtWidgets import QWidget

class ControlCar(QWidget):
    HOST = '127.0.0.1'
    PORT = 8888

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(ControlCar, self).__init__(*args, **kwargs)
        self._connCar = None
        uic.loadUi('carui.ui', self)
        self.resize(800, 600)
        self.buttonConnect.clicked.connect(self.doConnect)
        self.sliderForward.valueChanged.connect(self.doForward)
        self.sliderBackward.valueChanged.connect(self.doBackward)
        self.sliderLeft.valueChanged.connect(self.doLeft)
        self.sliderRight.valueChanged.connect(self.doRight)
        self.sliderForward.setEnabled(False)
        self.sliderBackward.setEnabled(False)
        self.sliderLeft.setEnabled(False)
        self.sliderRight.setEnabled(False)
        self._timer = QTimer(self, timeout=self.doGetImage)

    def _clearConn(self):
        if False:
            print('Hello World!')
        '清理连接'
        if self._connCar:
            self._connCar.close()
            self._connCar.deleteLater()
            del self._connCar
            self._connCar = None

    def closeEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        '窗口关闭事件'
        self._timer.stop()
        self._clearConn()
        super(ControlCar, self).closeEvent(event)

    def doConnect(self):
        if False:
            for i in range(10):
                print('nop')
        '连接服务器'
        self.buttonConnect.setEnabled(False)
        self._timer.stop()
        self._clearConn()
        self.browserResult.append('正在连接服务器')
        self._connCar = QTcpSocket(self)
        self._connCar.connected.connect(self.onConnected)
        self._connCar.disconnected.connect(self.onDisconnected)
        self._connCar.readyRead.connect(self.onReadyRead)
        self._connCar.error.connect(self.onError)
        self._connCar.connectToHost(self.HOST, self.PORT)

    def onConnected(self):
        if False:
            print('Hello World!')
        '连接成功'
        self.buttonConnect.setEnabled(False)
        self.sliderForward.setEnabled(True)
        self.sliderBackward.setEnabled(True)
        self.sliderLeft.setEnabled(True)
        self.sliderRight.setEnabled(True)
        self.browserResult.append('连接成功')
        self._timer.start(200)

    def onDisconnected(self):
        if False:
            while True:
                i = 10
        '丢失连接'
        self._timer.stop()
        self.buttonConnect.setEnabled(True)
        self.sliderForward.setEnabled(False)
        self.sliderBackward.setEnabled(False)
        self.sliderLeft.setEnabled(False)
        self.sliderRight.setEnabled(False)
        self.sliderForward.setValue(self.sliderForward.minimum())
        self.sliderBackward.setValue(self.sliderBackward.minimum())
        self.sliderLeft.setValue(self.sliderLeft.minimum())
        self.sliderRight.setValue(self.sliderRight.minimum())
        self.browserResult.append('丢失连接')

    def onReadyRead(self):
        if False:
            i = 10
            return i + 15
        '接收到数据'
        while self._connCar.bytesAvailable() > 0:
            try:
                data = self._connCar.readAll().data()
                if data and data.find(b'JFIF') > -1:
                    self.qlabel.setPixmap(QPixmap.fromImage(QImage.fromData(data)))
                else:
                    self.browserResult.append('接收到数据: ' + data.decode())
            except Exception as e:
                self.browserResult.append('解析数据错误: ' + str(e))

    def onError(self, _):
        if False:
            for i in range(10):
                print('nop')
        '连接报错'
        self._timer.stop()
        self.buttonConnect.setEnabled(True)
        self.browserResult.append('连接服务器错误: ' + self._connCar.errorString())

    def doForward(self, value):
        if False:
            while True:
                i = 10
        '向前'
        self.sendData('F:', str(value))

    def doBackward(self, value):
        if False:
            while True:
                i = 10
        '向后'
        self.sendData('B:', str(value))

    def doLeft(self, value):
        if False:
            for i in range(10):
                print('nop')
        '向左'
        self.sendData('L:', str(value))

    def doRight(self, value):
        if False:
            print('Hello World!')
        '向右'
        self.sendData('R:', str(value))

    def doGetImage(self):
        if False:
            for i in range(10):
                print('nop')
        self.sendData('getimage', '')

    def sendData(self, ver, data):
        if False:
            print('Hello World!')
        '发送数据'
        if not self._connCar or not self._connCar.isWritable():
            return self.browserResult.append('服务器未连接或不可写入数据')
        self._connCar.write(ver.encode() + str(data).encode() + b'\n')
if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = ControlCar()
    w.show()
    sys.exit(app.exec_())