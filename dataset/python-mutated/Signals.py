"""
Created on 2019年9月18日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: QtQuick.Signals
@description: 信号槽
"""
import sys
from time import time
try:
    from PyQt5.QtCore import QCoreApplication, Qt, pyqtSlot, pyqtSignal, QTimer
    from PyQt5.QtQml import QQmlApplicationEngine
    from PyQt5.QtWidgets import QApplication, QMessageBox, QWidget, QVBoxLayout, QPushButton, QTextBrowser
except ImportError:
    from PySide2.QtCore import QCoreApplication, Qt, Slot as pyqtSlot, Signal as pyqtSignal, QTimer
    from PySide2.QtQml import QQmlApplicationEngine
    from PySide2.QtWidgets import QApplication, QMessageBox, QWidget, QVBoxLayout, QPushButton, QTextBrowser
QML = 'import QtQuick 2.0\nimport QtQuick.Controls 1.6\nimport QtQuick.Layouts 1.3\n\nApplicationWindow {\n    visible: true\n    width: 400\n    height: 400\n    id: root\n    title: "editor"\n\n    // 定义信号槽\n    signal valueChanged(int value)\n    \n    Component.onCompleted: {\n        // 绑定信号槽到python中的函数\n        valueChanged.connect(_Window.onValueChanged)\n        // 绑定python中的信号到qml中的函数\n        _Window.timerSignal.connect(appendText)\n    }\n    \n    function appendText(text) {\n        // 定义添加文字函数\n        textArea.append(text)\n    }\n\n    ColumnLayout {\n        id: columnLayout\n        anchors.fill: parent\n\n        Button {\n            id: button\n            text: qsTr("Button")\n            Layout.fillWidth: true\n            onClicked: {\n                // 点击按钮调用python中的函数并得到返回值\n                var ret = _Window.testSlot("Button")\n                textArea.append("我调用了testSlot函数得到返回值: " + ret)\n            }\n        }\n\n        Slider {\n            id: sliderHorizontal\n            Layout.fillWidth: true\n            stepSize: 1\n            minimumValue: 0\n            maximumValue: 100\n            // 拉动条值改变时发送信号\n            onValueChanged: root.valueChanged(value)\n        }\n\n        TextArea {\n            id: textArea\n            Layout.fillWidth: true\n        }\n    }\n\n}\n'

class Window(QWidget):
    timerSignal = pyqtSignal(str)

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(Window, self).__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        layout.addWidget(QPushButton('Python调用qml中的函数', self, clicked=self.callQmlFunc))
        self.resultView = QTextBrowser(self)
        layout.addWidget(self.resultView)
        self._timer = QTimer(self, timeout=self.onTimeout)
        self._timer.start(2000)

    def onTimeout(self):
        if False:
            i = 10
            return i + 15
        self.timerSignal.emit('定时器发来:' + str(time()))

    def callQmlFunc(self):
        if False:
            print('Hello World!')
        engine.rootObjects()[0].appendText('我是被Python调用了')

    @pyqtSlot(int)
    def onValueChanged(self, value):
        if False:
            i = 10
            return i + 15
        self.resultView.append('拉动条值: %s' % value)

    @pyqtSlot(str, result=str)
    def testSlot(self, name):
        if False:
            return 10
        self.resultView.append('我被主动调用: %s' % name)
        return str(len(name))
if __name__ == '__main__':
    try:
        QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    except:
        pass
    app = QApplication(sys.argv)
    w = Window()
    w.resize(400, 400)
    w.show()
    w.move(400, 400)
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty('_Window', w)
    engine.objectCreated.connect(lambda obj, _: QMessageBox.critical(None, '错误', '运行失败，请检查') if not obj else 0)
    engine.loadData(QML.encode())
    sys.exit(app.exec_())