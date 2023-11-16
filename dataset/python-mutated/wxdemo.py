"""Demo of wxPython integration with Twisted."""
import sys
from wx import EVT_CLOSE, EVT_MENU, App, DefaultPosition, Frame, Menu, MenuBar, Size
from twisted.internet import wxreactor
from twisted.python import log
wxreactor.install()
from twisted.internet import reactor
ID_EXIT = 101

class MyFrame(Frame):

    def __init__(self, parent, ID, title):
        if False:
            for i in range(10):
                print('nop')
        Frame.__init__(self, parent, ID, title, DefaultPosition, Size(300, 200))
        menu = Menu()
        menu.Append(ID_EXIT, 'E&xit', 'Terminate the program')
        menuBar = MenuBar()
        menuBar.Append(menu, '&File')
        self.SetMenuBar(menuBar)
        EVT_MENU(self, ID_EXIT, self.DoExit)
        EVT_CLOSE(self, lambda evt: reactor.stop())

    def DoExit(self, event):
        if False:
            i = 10
            return i + 15
        reactor.stop()

class MyApp(App):

    def twoSecondsPassed(self):
        if False:
            print('Hello World!')
        print('two seconds passed')

    def OnInit(self):
        if False:
            while True:
                i = 10
        frame = MyFrame(None, -1, 'Hello, world')
        frame.Show(True)
        self.SetTopWindow(frame)
        reactor.callLater(2, self.twoSecondsPassed)
        return True

def demo():
    if False:
        i = 10
        return i + 15
    log.startLogging(sys.stdout)
    app = MyApp(0)
    reactor.registerWxApp(app)
    reactor.run()
if __name__ == '__main__':
    demo()