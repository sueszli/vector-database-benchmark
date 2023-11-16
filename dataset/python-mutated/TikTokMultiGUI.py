"""
@Description:GUI.py
@Date       :2022/08/18 14:39:03
@Author     :JohnserfSeed
@version    :1.0
@License    :MIT License
@Github     :https://github.com/johnserf-seed
@Mail       :johnserfseed@gmail.com
-------------------------------------------------
Change Log  :
2022/08/18 14:23:03 : Init
2022/08/18 14:39:03 : 添加多线程
2022/08/18 14:39:03 : 添加控制台显示
-------------------------------------------------
"""
import sys
from Util.Resource import *
from TikTokTool import Util
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QPoint, QThread, QMutex
from PyQt5.QtCore import QObject, QEventLoop, QTimer
from PyQt5.QtGui import QMouseEvent, QTextCursor

class PreventFastClickThreadMutex(QThread):
    qmut = QMutex()

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        self.qmut.lock()
        args = Util.Command().setting()
        args[0] = newuid
        profile = Util.Profile()
        profile.getProfile(args)
        self.qmut.unlock()

class Signal(QObject):
    text_update = QtCore.pyqtSignal(str)

    def write(self, text):
        if False:
            for i in range(10):
                print('nop')
        self.text_update.emit(str(text))

class Ui_Dialog(QtWidgets.QMainWindow):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.setupUi(self)
        sys.stdout = Signal()
        sys.stdout.text_update.connect(self.updatetext)

    def updatetext(self, text):
        if False:
            for i in range(10):
                print('nop')
        '更新textBrowser\n        Args:\n            text : 控制台文本\n        '
        cursor = self.textBrowser.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.textBrowser.append(text)
        self.textBrowser.setTextCursor(cursor)
        self.textBrowser.ensureCursorVisible()

    def setupUi(self, Dialog):
        if False:
            while True:
                i = 10
        Dialog.setObjectName('TikTokDownload')
        Dialog.resize(1030, 600)
        Dialog.setStyleSheet('')
        self.Label_Left = QtWidgets.QLabel(Dialog)
        self.Label_Left.setGeometry(QtCore.QRect(0, 0, 230, 600))
        self.Label_Left.setStyleSheet('background-color: #060716;\nborder-bottom-left-radius: 25px;\nborder-top-left-radius: 25px;\nfont: 10pt "微软雅黑";\ncolor: #FFFFFF;')
        self.Label_Left.setText('')
        self.Label_Left.setObjectName('Label_Left')
        self.Label_Right = QtWidgets.QLabel(Dialog)
        self.Label_Right.setGeometry(QtCore.QRect(230, 0, 800, 600))
        self.Label_Right.setStyleSheet('background-color: rgb(255, 255, 255);\nborder-top-right-radius: 25px;\nborder-bottom-right-radius: 25px;')
        self.Label_Right.setText('')
        self.Label_Right.setObjectName('Label_Right')
        self.Label_Logo = QtWidgets.QLabel(Dialog)
        self.Label_Logo.setGeometry(QtCore.QRect(30, 50, 161, 41))
        self.Label_Logo.setStyleSheet('background: url(:/img/logo-horizontal.svg) no-repeat;')
        self.Label_Logo.setText('')
        self.Label_Logo.setObjectName('Label_Logo')
        self.Label_Version = QtWidgets.QLabel(Dialog)
        self.Label_Version.setGeometry(QtCore.QRect(180, 90, 54, 12))
        self.Label_Version.setStyleSheet('color: rgb(255, 255, 255);\nfont: 9pt "微软雅黑";')
        self.Label_Version.setObjectName('Label_Version')
        self.Button_Close = QtWidgets.QPushButton(Dialog)
        self.Button_Close.setGeometry(QtCore.QRect(980, 20, 21, 21))
        self.Button_Close.setStyleSheet('border-radius: 10px;\nbackground-color: rgb(255, 81, 53);')
        self.Button_Close.setText('')
        self.Button_Close.setObjectName('Button_Close')
        self.Button_Max = QtWidgets.QPushButton(Dialog)
        self.Button_Max.setGeometry(QtCore.QRect(950, 20, 21, 21))
        self.Button_Max.setStyleSheet('border-radius: 10px;\nbackground-color: #FFC32D;')
        self.Button_Max.setText('')
        self.Button_Max.setObjectName('Button_Max')
        self.Button_Min = QtWidgets.QPushButton(Dialog)
        self.Button_Min.setGeometry(QtCore.QRect(920, 20, 21, 21))
        self.Button_Min.setStyleSheet('border-radius: 10px;\nbackground-color: #37C847;')
        self.Button_Min.setText('')
        self.Button_Min.setObjectName('Button_Min')
        self.plainTextEdit = QtWidgets.QPlainTextEdit(Dialog)
        self.plainTextEdit.setGeometry(QtCore.QRect(260, 90, 601, 41))
        self.plainTextEdit.setAutoFillBackground(False)
        self.plainTextEdit.setStyleSheet('background-color: #292B35;\nborder-radius: 10px;\nfont: 20pt "微软雅黑";\ncolor: rgb(255, 255, 255);')
        self.plainTextEdit.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.plainTextEdit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.plainTextEdit.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.plainTextEdit.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        self.plainTextEdit.setBackgroundVisible(False)
        self.plainTextEdit.setCenterOnScroll(False)
        self.plainTextEdit.setObjectName('plainTextEdit')
        self.Button_Go = QtWidgets.QPushButton(Dialog)
        self.Button_Go.setGeometry(QtCore.QRect(880, 90, 130, 41))
        self.Button_Go.setStyleSheet('#Button_Go {\n    border-radius: 10px;\n    font: 19pt "微软雅黑";\n    color: rgb(0, 0, 0);\n    background-color: #B9BAC7;\n}\n\n#Button_Go:hover {\n    color:#F72C51;\n}\n\n#Button_Go:pressed, QPushButton:checked {\n    background-color: #9d9d9d;\n}\n')
        self.Button_Go.setObjectName('Button_Go')
        self.Label_Background = QtWidgets.QLabel(Dialog)
        self.Label_Background.setGeometry(QtCore.QRect(230, 0, 800, 60))
        self.Label_Background.setStyleSheet('background-color: rgb(199, 199, 199);\nborder-top-right-radius: 25px;')
        self.Label_Background.setText('')
        self.Label_Background.setObjectName('Label_Background')
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(260, 190, 611, 351))
        self.widget.setObjectName('widget')
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(50, 390, 131, 61))
        self.pushButton.setStyleSheet('border-radius: 25px;\nfont: 16pt "微软雅黑";\ncolor: rgb(255, 255, 255);\nbackground-color: rgb(22, 23, 34);')
        self.pushButton.setObjectName('pushButton')
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(50, 490, 131, 61))
        self.pushButton_2.setStyleSheet('border-radius: 25px;\nfont: 16pt "微软雅黑";\ncolor: rgb(255, 255, 255);\nbackground-color: rgb(22, 23, 34);')
        self.pushButton_2.setObjectName('pushButton_2')
        self.Check_All = QtWidgets.QCheckBox(Dialog)
        self.Check_All.setGeometry(QtCore.QRect(260, 140, 71, 16))
        self.Check_All.setObjectName('Check_All')
        self.Check_All.setStyleSheet('font: 10pt "微软雅黑";\n')
        self.Check_Cover = QtWidgets.QCheckBox(Dialog)
        self.Check_Cover.setGeometry(QtCore.QRect(340, 140, 101, 16))
        self.Check_Cover.setObjectName('Check_Cover')
        self.Check_Cover.setStyleSheet('font: 10pt "微软雅黑";\n')
        self.Check_Music = QtWidgets.QCheckBox(Dialog)
        self.Check_Music.setGeometry(QtCore.QRect(440, 140, 101, 16))
        self.Check_Music.setObjectName('Check_Music')
        self.Check_Music.setStyleSheet('font: 10pt "微软雅黑";\n')
        self.textBrowser = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser.setGeometry(QtCore.QRect(260, 170, 750, 401))
        self.textBrowser.setObjectName('textBrowser')
        self.textBrowser.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.textBrowser.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.textBrowser.setStyleSheet('border-width:0;border-style:outset;')
        self.Label_Right.raise_()
        self.Label_Background.raise_()
        self.Label_Left.raise_()
        self.Label_Logo.raise_()
        self.Label_Version.raise_()
        self.Button_Close.raise_()
        self.Button_Max.raise_()
        self.Button_Min.raise_()
        self.plainTextEdit.raise_()
        self.Button_Go.raise_()
        self.widget.raise_()
        self.Check_All.raise_()
        self.pushButton.raise_()
        self.pushButton_2.raise_()
        self.Check_Cover.raise_()
        self.Check_Music.raise_()
        self.textBrowser.raise_()
        self.setMinimumHeight(600)
        self.setMinimumWidth(900)
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def mouseMoveEvent(self, e: QMouseEvent):
        if False:
            for i in range(10):
                print('nop')
        '# 重写移动事件\n\n        Args:\n            e (QMouseEvent): 鼠标事件\n        '
        try:
            self._endPos = e.pos() - self._startPos
            self.move(self.pos() + self._endPos)
        except:
            pass

    def mousePressEvent(self, e: QMouseEvent):
        if False:
            print('Hello World!')
        try:
            if e.button() == Qt.LeftButton:
                self._isTracking = True
                self._startPos = QPoint(e.x(), e.y())
        except:
            pass

    def mouseReleaseEvent(self, e: QMouseEvent):
        if False:
            return 10
        try:
            if e.button() == Qt.LeftButton:
                self._isTracking = False
                self._startPos = None
                self._endPos = None
        except:
            pass

    def btnClick(self):
        if False:
            for i in range(10):
                print('nop')
        '咻咻按钮事件\n        '
        global newuid
        newuid = self.plainTextEdit.toPlainText()
        self.thread_1 = PreventFastClickThreadMutex()
        self.thread_1.start()

    def retranslateUi(self, Dialog):
        if False:
            print('Hello World!')
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate('TikTokDownload', 'TikTokDownload'))
        self.Label_Version.setText(_translate('TikTokDownload', 'v2.1.1'))
        self.Button_Min.setToolTip('最小化')
        self.Button_Max.setToolTip('最大化')
        self.Button_Close.setToolTip('关闭')
        self.Button_Go.setText(_translate('TikTokDownload', '咻咻'))
        self.Check_All.setText(_translate('TikTokDownload', '全部下载'))
        self.pushButton.setText(_translate('TikTokDownload', '设置'))
        self.pushButton_2.setText(_translate('TikTokDownload', '关于'))
        self.Check_Cover.setText(_translate('TikTokDownload', '全部封面下载'))
        self.Check_Music.setText(_translate('TikTokDownload', '全部配乐下载'))
        self.plainTextEdit.setPlainText('https://v.douyin.com/efrHYf2/')
        self.Button_Go.clicked.connect(lambda : self.btnClick())
        self.Button_Max.clicked.connect(lambda : self.MaxButton())
        self.Button_Min.clicked.connect(lambda : self.MinButton())
        self.Button_Close.clicked.connect(lambda : self.CloseButton())

    def MaxButton(self):
        if False:
            print('Hello World!')
        '最大化与还原的切换\n        '
        if self.isMaximized():
            self.showNormal()
            self.Button_Max.setToolTip('最大化')
        else:
            self.showMaximized()
            self.Button_Max.setToolTip('还原')

    def MinButton(self):
        if False:
            return 10
        '最小化\n        '
        self.showMinimized()

    def CloseButton(self):
        if False:
            while True:
                i = 10
        '关闭\n        '
        sys.exit(0)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Ui_Dialog()
    w.setWindowFlags(Qt.FramelessWindowHint)
    w.setAttribute(Qt.WA_TranslucentBackground)
    w.show()
    sys.exit(app.exec_())