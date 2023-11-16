import sys
import traceback
import win32api
import win32ui
from dde import *
from pywin.mfc import object

class DDESystemTopic(object.Object):

    def __init__(self, app):
        if False:
            return 10
        self.app = app
        object.Object.__init__(self, CreateServerSystemTopic())

    def Exec(self, data):
        if False:
            i = 10
            return i + 15
        try:
            self.app.OnDDECommand(data)
        except:
            (t, v, tb) = sys.exc_info()
            print('Error executing DDE command.')
            traceback.print_exception(t, v, tb)
            return 0

class DDEServer(object.Object):

    def __init__(self, app):
        if False:
            return 10
        self.app = app
        object.Object.__init__(self, CreateServer())
        self.topic = self.item = None

    def CreateSystemTopic(self):
        if False:
            for i in range(10):
                print('nop')
        return DDESystemTopic(self.app)

    def Shutdown(self):
        if False:
            return 10
        self._obj_.Shutdown()
        self._obj_.Destroy()
        if self.topic is not None:
            self.topic.Destroy()
            self.topic = None
        if self.item is not None:
            self.item.Destroy()
            self.item = None

    def OnCreate(self):
        if False:
            while True:
                i = 10
        return 1

    def Status(self, msg):
        if False:
            print('Hello World!')
        try:
            win32ui.SetStatusText(msg)
        except win32ui.error:
            pass