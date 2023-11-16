from tkinter import *
import tkinter.messagebox as messagebox

class Application(Frame):

    def __init__(self, master=None):
        if False:
            i = 10
            return i + 15
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

    def createWidgets(self):
        if False:
            for i in range(10):
                print('nop')
        self.nameInput = Entry(self)
        self.nameInput.pack()
        self.alertButton = Button(self, text='Hello', command=self.hello)
        self.alertButton.pack()

    def hello(self):
        if False:
            return 10
        name = self.nameInput.get() or 'world'
        messagebox.showinfo('Message', 'Hello, %s' % name)
app = Application()
app.master.title('Hello World')
app.mainloop()