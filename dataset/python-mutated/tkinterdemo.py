"""
An example of using Twisted with Tkinter.
Displays a frame with buttons that responds to mouse clicks.

Run this example by typing in:
 python tkinterdemo.py
"""
from tkinter import LEFT, Button, Frame, Tk
from twisted.internet import reactor, tksupport

class App:

    def onQuit(self):
        if False:
            while True:
                i = 10
        print('Quit!')
        reactor.stop()

    def onButton(self):
        if False:
            for i in range(10):
                print('nop')
        print('Hello!')

    def __init__(self, master):
        if False:
            print('Hello World!')
        frame = Frame(master)
        frame.pack()
        q = Button(frame, text='Quit!', command=self.onQuit)
        b = Button(frame, text='Hello!', command=self.onButton)
        q.pack(side=LEFT)
        b.pack(side=LEFT)
if __name__ == '__main__':
    root = Tk()
    tksupport.install(root)
    app = App(root)
    reactor.run()