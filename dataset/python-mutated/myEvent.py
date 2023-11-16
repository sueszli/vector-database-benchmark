"""
@resource:http://blog.csdn.net/zzwdkxx/article/details/39338429
@description: 自定义QEvent事件,上面网址为C++版本原理解释,此篇为python改编
@Created on 2018年3月22日
@email: 625781186@qq.com
"""
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
MyEventType = QEvent.registerEventType(QEvent.User + 100)

class MyEvent(QEvent):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(MyEvent, self).__init__(*args, **kwargs)
        print(MyEventType)

        def QEvent(self, MyEventType):
            if False:
                i = 10
                return i + 15
            pass

class MySender(QCoreApplication):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(MySender, self).__init__(*args, **kwargs)

    def notify(self, receiver, event):
        if False:
            print('Hello World!')
        if event.type() == MyEventType:
            print('MyEventType is coming!')
        return QCoreApplication.notify(self, receiver, event)

class MyArmy(QWidget):

    def MyEventHandler(self, event):
        if False:
            return 10
        print('The event is being handled!')
        event.accept()

    def event(self, event):
        if False:
            return 10
        if event.type() == MyEventType:
            print('event() is dispathing MyEvent')
            self.MyEventHandler(event)
            if event.isAccepted():
                print('The event has been handled!')
                return True
        return QObject.event(self, event)

class MyWatcher(QObject):

    def eventFilter(self, watched, event):
        if False:
            print('Hello World!')
        if event.type() == MyEventType:
            print("I don't wanna filter MyEventType")
            return False
        return QObject.eventFilter(self, watched, event)
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    mySender = MySender(sys.argv)
    myArmy = MyArmy()
    myWatcher = MyWatcher()
    myArmy.installEventFilter(myWatcher)
    myEvent = MyEvent(MyEventType)
    mySender.sendEvent(myArmy, myEvent)
    mySender.exec()