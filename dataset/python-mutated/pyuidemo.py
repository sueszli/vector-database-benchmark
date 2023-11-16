"""
Displays a frame with two buttons and a background image, using pyui library.

Run this example by typing in:
  python pyuidemo.py

Select "Quit" button to exit demo.
"""
import pyui
from twisted.internet import pyuisupport, reactor

def onButton(self):
    if False:
        for i in range(10):
            print('nop')
    print('got a button')

def onQuit(self):
    if False:
        for i in range(10):
            print('nop')
    reactor.stop()

def main():
    if False:
        i = 10
        return i + 15
    pyuisupport.install(args=(640, 480), kw={'renderer': '2d'})
    w = pyui.widgets.Frame(50, 50, 400, 400, 'clipme')
    b = pyui.widgets.Button('A button is here', onButton)
    q = pyui.widgets.Button('Quit!', onQuit)
    w.addChild(b)
    w.addChild(q)
    w.pack()
    w.setBackImage('pyui_bg.png')
    reactor.run()
if __name__ == '__main__':
    main()